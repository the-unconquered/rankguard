"""RankGuard: commitment-bound rank verification for permutation watermarks.

RankGuard fixes two security problems in the reference PermuteGuard prototype:

1. Legacy metadata serializes the secret key and raw post-embedding fingerprint.
2. Legacy cosine verification is dominated by unchanged coordinates, so an
   unwatermarked checkpoint can be accepted when only a small fraction of
   weights is modified.

The rank-locked construction below uses the modified subset itself as the
watermark. During embedding, selected values are sorted and then written in a
key-derived permutation order. During verification, the rank order of the
suspect subset is compared with the expected permutation. An untouched model
therefore has an approximately random rank correlation with the target order,
while a watermarked model has near-perfect rank agreement.

This module uses only NumPy and the Python standard library so that the
controlled experiments can run without model checkpoints or GPU dependencies.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


def generate_master_key() -> bytes:
    return secrets.token_bytes(32)


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hmac(key: bytes, *parts: str) -> bytes:
    hm = hmac.new(key, digestmod=hashlib.sha256)
    for part in parts:
        b = part.encode("utf-8")
        hm.update(len(b).to_bytes(8, "big"))
        hm.update(b)
    return hm.digest()


def _seed(key: bytes, *parts: str) -> int:
    return int.from_bytes(_hmac(key, *parts)[:8], "big")


def _rng(key: bytes, *parts: str) -> np.random.Generator:
    return np.random.default_rng(_seed(key, *parts))


def _digest_array(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x, dtype=np.int64).tobytes()).hexdigest()


def _spearman_unit(observed_ranks: np.ndarray, expected_ranks: np.ndarray) -> float:
    """Map Spearman rank correlation from [-1, 1] to [0, 1]."""

    r = np.asarray(observed_ranks, dtype=np.float64)
    e = np.asarray(expected_ranks, dtype=np.float64)
    if r.size != e.size or r.size < 2:
        return 0.0
    # Centered Pearson correlation of ranks is Spearman correlation. This form
    # remains well-defined when an attack introduces ties.
    rc = r - r.mean()
    ec = e - e.mean()
    denom = float(np.linalg.norm(rc) * np.linalg.norm(ec))
    if denom <= 0 or not np.isfinite(denom):
        return 0.0
    rho = float(np.dot(rc, ec) / denom)
    if not np.isfinite(rho):
        return 0.0
    return float(np.clip((rho + 1.0) / 2.0, 0.0, 1.0))


def _ranks(values: np.ndarray) -> np.ndarray:
    """Return average ranks, robust to ties introduced by quantization/pruning."""

    x = np.asarray(values)
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(x.size, dtype=np.float64)
    i = 0
    while i < x.size:
        j = i + 1
        while j < x.size and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


@dataclass(frozen=True)
class RankGuardConfig:
    subset_fraction: float = 0.005
    verification_threshold: float = 0.80
    min_subset_size: int = 128


class RankGuard:
    """Rank-locked permutation watermark with a key-free public manifest."""

    def __init__(self, master_key: bytes, config: Optional[RankGuardConfig] = None):
        if not isinstance(master_key, (bytes, bytearray)) or len(master_key) < 32:
            raise ValueError("master_key must contain at least 32 bytes")
        self.master_key = bytes(master_key)
        self.config = config or RankGuardConfig()
        if not 0 < self.config.subset_fraction <= 1:
            raise ValueError("subset_fraction must be in (0, 1]")

    def _subset(self, layer_name: str, n: int) -> np.ndarray:
        k = max(self.config.min_subset_size, int(round(self.config.subset_fraction * n)))
        k = min(k, n)
        return _rng(self.master_key, "rankguard-subset", layer_name).permutation(n)[:k]

    def _permutation(self, layer_name: str, k: int) -> np.ndarray:
        return _rng(self.master_key, "rankguard-permutation", layer_name).permutation(k)

    def _commitment(self, model_id: str, salt: str, layer_name: str, subset: np.ndarray, permutation: np.ndarray) -> str:
        payload = {
            "model_id": model_id,
            "salt": salt,
            "layer": layer_name,
            "subset_sha256": _digest_array(subset),
            "permutation_sha256": _digest_array(permutation),
            "config": asdict(self.config),
        }
        return hmac.new(self.master_key, _canonical_json(payload).encode(), hashlib.sha256).hexdigest()

    def embed(
        self,
        weights: MutableMapping[str, np.ndarray],
        model_id: str,
        selected_layers: Optional[Sequence[str]] = None,
    ) -> Tuple[MutableMapping[str, np.ndarray], Dict[str, Any], Dict[str, Any]]:
        layers = list(selected_layers) if selected_layers is not None else list(weights.keys())
        salt = secrets.token_hex(16)
        public_layers: Dict[str, Any] = {}
        diagnostics: Dict[str, Any] = {}

        for layer in layers:
            w = weights[layer]
            flat = w.reshape(-1)
            subset = self._subset(layer, flat.size)
            permutation = self._permutation(layer, subset.size)
            original = flat[subset].copy()
            sorted_values = np.sort(original)
            watermarked_values = sorted_values[permutation]
            flat[subset] = watermarked_values.astype(flat.dtype, copy=False)
            commitment = self._commitment(model_id, salt, layer, subset, permutation)
            public_layers[layer] = {
                "numel": int(flat.size),
                "subset_size": int(subset.size),
                "commitment": commitment,
            }
            denom = float(np.linalg.norm(original.astype(np.float64))) + 1e-12
            diagnostics[layer] = {
                "subset_fraction_actual": float(subset.size / flat.size),
                "relative_subset_l2_change": float(
                    np.linalg.norm((watermarked_values - original).astype(np.float64)) / denom
                ),
                "multiset_preserved": bool(np.allclose(np.sort(watermarked_values), sorted_values)),
            }

        manifest = {
            "scheme": "RankGuard/1.0",
            "model_id": model_id,
            "salt": salt,
            "config": asdict(self.config),
            "selected_layers": layers,
            "layers": public_layers,
        }
        manifest["manifest_digest"] = hashlib.sha256(_canonical_json(manifest).encode()).hexdigest()
        private_record = {
            "scheme": manifest["scheme"],
            "model_id": model_id,
            "manifest_digest": manifest["manifest_digest"],
            "master_key_hex": self.master_key.hex(),
            "warning": "Private owner record. Do not distribute with the checkpoint.",
        }
        return weights, manifest, {"owner_record": private_record, "diagnostics": diagnostics}

    def verify(
        self,
        weights: Mapping[str, np.ndarray],
        manifest: Mapping[str, Any],
        private_bundle: Mapping[str, Any],
        candidate_key: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        key = self.master_key if candidate_key is None else bytes(candidate_key)
        # Derive the effective non-secret config from the (digest-authenticated)
        # manifest so verification works for any embedding-time parameterization,
        # not just this instance's defaults.
        mconfig = manifest.get("config", {}) or {}
        effective_config = RankGuardConfig(
            subset_fraction=float(mconfig.get("subset_fraction", self.config.subset_fraction)),
            verification_threshold=float(mconfig.get("verification_threshold", self.config.verification_threshold)),
            min_subset_size=int(mconfig.get("min_subset_size", self.config.min_subset_size)),
        )
        supplied_manifest = dict(manifest)
        supplied_digest = supplied_manifest.pop("manifest_digest", None)
        recomputed_digest = hashlib.sha256(_canonical_json(supplied_manifest).encode()).hexdigest()
        expected_digest = private_bundle.get("owner_record", {}).get("manifest_digest")
        manifest_ok = bool(
            supplied_digest
            and expected_digest
            and hmac.compare_digest(str(supplied_digest), str(expected_digest))
            and hmac.compare_digest(str(recomputed_digest), str(supplied_digest))
        )

        scores: Dict[str, float] = {}
        commitment_ok = True
        commitment_errors = []
        if manifest_ok:
            for layer in manifest["selected_layers"]:
                flat = np.asarray(weights[layer]).reshape(-1)
                subset = self._subset_with_key(key, layer, flat.size, effective_config)
                permutation = self._permutation_with_key(key, layer, subset.size)
                current_ranks = _ranks(flat[subset])
                scores[layer] = _spearman_unit(current_ranks, permutation)

                expected_commitment = self._commitment_with_key(
                    key,
                    manifest["model_id"],
                    manifest["salt"],
                    layer,
                    subset,
                    permutation,
                    effective_config,
                )
                actual = manifest["layers"][layer]["commitment"]
                if not hmac.compare_digest(expected_commitment, actual):
                    commitment_ok = False
                    commitment_errors.append(layer)
        else:
            commitment_ok = False
            commitment_errors.append("manifest_digest")
            # Still compute a behavioral score so a valid manifest with an
            # untouched model can be distinguished from a manifest mismatch.
            for layer in manifest["selected_layers"]:
                flat = np.asarray(weights[layer]).reshape(-1)
                subset = self._subset_with_key(key, layer, flat.size, effective_config)
                permutation = self._permutation_with_key(key, layer, subset.size)
                scores[layer] = _spearman_unit(_ranks(flat[subset]), permutation)

        confidence = float(np.mean(list(scores.values()))) if scores else 0.0
        verified = bool(manifest_ok and commitment_ok and confidence >= float(manifest["config"]["verification_threshold"]))
        return {
            "verified": verified,
            "confidence": confidence,
            "manifest_ok": manifest_ok,
            "commitment_ok": commitment_ok,
            "commitment_errors": commitment_errors,
            "layer_scores": scores,
            "threshold": float(manifest["config"]["verification_threshold"]),
        }

    @staticmethod
    def _subset_with_key(key: bytes, layer_name: str, n: int, config: Optional[RankGuardConfig] = None) -> np.ndarray:
        cfg = config or RankGuardConfig()
        k = max(cfg.min_subset_size, int(round(cfg.subset_fraction * n)))
        k = min(k, n)
        return _rng(key, "rankguard-subset", layer_name).permutation(n)[:k]

    @staticmethod
    def _permutation_with_key(key: bytes, layer_name: str, k: int) -> np.ndarray:
        return _rng(key, "rankguard-permutation", layer_name).permutation(k)

    def _commitment_with_key(self, key: bytes, model_id: str, salt: str, layer_name: str, subset: np.ndarray, permutation: np.ndarray, config: Optional[RankGuardConfig] = None) -> str:
        cfg = config or self.config
        payload = {
            "model_id": model_id,
            "salt": salt,
            "layer": layer_name,
            "subset_sha256": _digest_array(subset),
            "permutation_sha256": _digest_array(permutation),
            "config": asdict(cfg),
        }
        return hmac.new(key, _canonical_json(payload).encode(), hashlib.sha256).hexdigest()


def simulate_attack(
    weights: MutableMapping[str, np.ndarray],
    layers: Sequence[str],
    attack: str,
    rng: np.random.Generator,
    key: Optional[bytes] = None,
    config: Optional[RankGuardConfig] = None,
) -> MutableMapping[str, np.ndarray]:
    cfg = config or RankGuardConfig()
    for layer in layers:
        w = weights[layer]
        flat = w.reshape(-1)
        if attack == "clean":
            continue
        if attack == "pruning_10":
            flat[np.abs(flat) <= np.quantile(np.abs(flat), 0.10)] = 0
        elif attack == "pruning_20":
            flat[np.abs(flat) <= np.quantile(np.abs(flat), 0.20)] = 0
        elif attack == "quantization_8bit":
            lo, hi = float(flat.min()), float(flat.max())
            scale = (hi - lo) / 255.0
            if scale > 0:
                flat[:] = np.round((flat - lo) / scale) * scale + lo
        elif attack == "noise_0.002":
            flat += rng.normal(0.0, 0.002 * float(flat.std()), flat.size).astype(flat.dtype)
        elif attack == "noise_0.01":
            flat += rng.normal(0.0, 0.01 * float(flat.std()), flat.size).astype(flat.dtype)
        elif attack in {"shuffle_05", "shuffle_10"}:
            rate = 0.05 if attack == "shuffle_05" else 0.10
            idx = rng.permutation(flat.size)[: max(1, int(rate * flat.size))]
            flat[idx] = flat[idx[rng.permutation(idx.size)]]
        elif attack == "targeted_subset_overwrite":
            if key is None:
                # A blind attacker can only guess a comparable number of positions.
                count = max(cfg.min_subset_size, int(round(cfg.subset_fraction * flat.size)))
                idx = rng.permutation(flat.size)[: min(count, flat.size)]
            else:
                idx = RankGuard._subset_with_key(key, layer, flat.size, cfg)
            flat[idx] = rng.normal(0.0, float(flat.std()) + 1e-12, idx.size).astype(flat.dtype)
        else:
            raise ValueError(f"Unsupported attack: {attack}")
    return weights
