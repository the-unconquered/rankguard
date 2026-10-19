"""Leakage-resistant metadata and verification utilities for PermuteGuard-style experiments.

This module is a self-contained NumPy implementation used for the controlled
security experiments in the follow-up manuscript. It keeps the core permutation
embedding comparable to the original implementation while changing the security
architecture:

* the master key is never serialized to the public manifest;
* per-layer randomness is derived with HMAC-SHA256;
* the public manifest contains salted commitments rather than raw fingerprints;
* the private owner record contains the reference fingerprints required for
  robust forensic comparison;
* wrong-key and unrelated-model checks are evaluated rather than short-circuited.

The code is intended for research reproducibility. In production, store the
private owner record in an authenticated encrypted secrets store.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


def generate_master_key() -> bytes:
    """Generate a 256-bit master key."""

    return secrets.token_bytes(32)


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _hmac(key: bytes, *parts: str) -> bytes:
    hm = hmac.new(key, digestmod=hashlib.sha256)
    for part in parts:
        encoded = part.encode("utf-8")
        hm.update(len(encoded).to_bytes(8, "big"))
        hm.update(encoded)
    return hm.digest()


def _seed(key: bytes, *parts: str) -> int:
    return int.from_bytes(_hmac(key, *parts)[:8], "big")


def _rng(key: bytes, *parts: str) -> np.random.Generator:
    return np.random.default_rng(_seed(key, *parts))


def _fingerprint_indices(key: bytes, layer_name: str, n: int, k: int) -> np.ndarray:
    kk = min(int(k), int(n))
    return _rng(key, "fingerprint", layer_name).permutation(n)[:kk]


def _fingerprint(weight: np.ndarray, key: bytes, layer_name: str, k: int) -> np.ndarray:
    flat = np.asarray(weight, dtype=np.float32).reshape(-1)
    idx = _fingerprint_indices(key, layer_name, flat.size, k)
    return flat[idx].astype(np.float32, copy=False)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if not np.isfinite(denom) or denom <= 0:
        return 0.0
    value = float(np.dot(aa, bb) / denom)
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(abs(value), 0.0, 1.0))


def _fingerprint_bytes(vector: np.ndarray, quantization: int) -> bytes:
    """Quantize a reference fingerprint before commitment.

    Quantization limits the usefulness of any accidentally exposed commitment
    opening and avoids embedding full floating-point values in the public
    manifest. Robust matching is still performed against the private record.
    """

    q = np.asarray(vector, dtype=np.float32)
    if quantization == 1:
        return (q >= 0).astype(np.uint8).tobytes()
    lo = float(np.quantile(q, 0.01))
    hi = float(np.quantile(q, 0.99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(q.min()), float(q.max())
    if hi <= lo:
        return np.zeros(q.size, dtype=np.uint8).tobytes()
    scaled = np.clip((q - lo) / (hi - lo), 0.0, 1.0)
    max_level = max(1, int(quantization) - 1)
    return np.rint(scaled * max_level).astype(np.uint8).tobytes()


@dataclass(frozen=True)
class SecureConfig:
    permutation_strength: float = 0.002
    fingerprint_k: int = 4096
    verification_threshold: float = 0.80
    commitment_quantization: int = 16


class HardenedPermuteGuard:
    """Permutation embedding with split public/private metadata."""

    def __init__(self, master_key: bytes, config: Optional[SecureConfig] = None):
        if not isinstance(master_key, (bytes, bytearray)) or len(master_key) < 32:
            raise ValueError("master_key must contain at least 32 bytes")
        self.master_key = bytes(master_key)
        self.config = config or SecureConfig()

    def _embed_layer(self, weight: np.ndarray, layer_name: str) -> Dict[str, Any]:
        flat = weight.reshape(-1)
        n = int(flat.size)
        subset_size = max(1, int(round(self.config.permutation_strength * n)))
        subset_idx = _rng(self.master_key, "subset", layer_name).permutation(n)[:subset_size]
        reorder = _rng(self.master_key, "reorder", layer_name).permutation(subset_size)
        original = flat[subset_idx].copy()
        permuted = original[reorder]
        orig_norm = float(np.linalg.norm(original.astype(np.float64))) + 1e-12
        perm_norm = float(np.linalg.norm(permuted.astype(np.float64))) + 1e-12
        flat[subset_idx] = (permuted * (orig_norm / perm_norm)).astype(flat.dtype, copy=False)
        fingerprint = _fingerprint(weight, self.master_key, layer_name, self.config.fingerprint_k)
        return {
            "subset_size": int(subset_size),
            "numel": int(n),
            "fingerprint": fingerprint,
            "relative_l2_change": float(
                np.linalg.norm((permuted - original).astype(np.float64))
                / (np.linalg.norm(original.astype(np.float64)) + 1e-12)
            ),
        }

    @staticmethod
    def _layer_commitment(
        master_key: bytes,
        model_id: str,
        salt_hex: str,
        layer_name: str,
        layer_info: Mapping[str, Any],
        fingerprint: np.ndarray,
        quantization: int,
    ) -> str:
        payload = {
            "model_id": model_id,
            "salt": salt_hex,
            "layer": layer_name,
            "numel": int(layer_info["numel"]),
            "subset_size": int(layer_info["subset_size"]),
            "quantization": int(quantization),
            "fingerprint_sha256": hashlib.sha256(
                _fingerprint_bytes(fingerprint, quantization)
            ).hexdigest(),
        }
        return hmac.new(
            master_key, _canonical_json(payload).encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def embed(
        self,
        weights: MutableMapping[str, np.ndarray],
        model_id: str,
        selected_layers: Optional[Sequence[str]] = None,
    ) -> Tuple[MutableMapping[str, np.ndarray], Dict[str, Any], Dict[str, Any]]:
        """Embed the watermark and return (weights, public manifest, private record)."""

        layers = list(selected_layers) if selected_layers is not None else list(weights.keys())
        salt = secrets.token_hex(16)
        private_layers: Dict[str, Any] = {}
        public_layers: Dict[str, Any] = {}

        for layer_name in layers:
            info = self._embed_layer(weights[layer_name], layer_name)
            fp = info.pop("fingerprint")
            commitment = self._layer_commitment(
                self.master_key,
                model_id,
                salt,
                layer_name,
                info,
                fp,
                self.config.commitment_quantization,
            )
            public_layers[layer_name] = {**info, "commitment": commitment}
            private_layers[layer_name] = {
                "fingerprint": fp.astype(np.float32).tolist(),
                "commitment": commitment,
            }

        manifest = {
            "scheme": "PermuteGuard-Hardened/1.0",
            "model_id": model_id,
            "salt": salt,
            "config": asdict(self.config),
            "selected_layers": layers,
            "layers": public_layers,
        }
        manifest["manifest_digest"] = hashlib.sha256(
            _canonical_json(manifest).encode("utf-8")
        ).hexdigest()

        private_record = {
            "scheme": manifest["scheme"],
            "model_id": model_id,
            "salt": salt,
            "manifest_digest": manifest["manifest_digest"],
            "layers": private_layers,
            "warning": "Private owner record. Do not distribute with the model checkpoint.",
        }
        return weights, manifest, private_record

    def verify(
        self,
        weights: Mapping[str, np.ndarray],
        manifest: Mapping[str, Any],
        private_record: Mapping[str, Any],
        candidate_key: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """Verify a suspect model without short-circuiting wrong-key checks."""

        key = self.master_key if candidate_key is None else bytes(candidate_key)
        scores: Dict[str, float] = {}
        commitment_ok = True
        commitment_errors = []

        supplied_manifest = dict(manifest)
        supplied_digest = supplied_manifest.pop("manifest_digest", None)
        recomputed_digest = hashlib.sha256(
            _canonical_json(supplied_manifest).encode("utf-8")
        ).hexdigest()
        if supplied_digest != private_record.get("manifest_digest") or recomputed_digest != supplied_digest:
            commitment_ok = False
            commitment_errors.append("manifest_digest")

        for layer_name in manifest["selected_layers"]:
            reference = np.asarray(private_record["layers"][layer_name]["fingerprint"], dtype=np.float32)
            observed = _fingerprint(weights[layer_name], key, layer_name, self.config.fingerprint_k)
            scores[layer_name] = _cosine(observed, reference)

            expected = self._layer_commitment(
                key,
                manifest["model_id"],
                manifest["salt"],
                layer_name,
                manifest["layers"][layer_name],
                reference,
                self.config.commitment_quantization,
            )
            actual = manifest["layers"][layer_name]["commitment"]
            if not hmac.compare_digest(expected, actual):
                commitment_ok = False
                commitment_errors.append(layer_name)

        confidence = float(np.mean(list(scores.values()))) if scores else 0.0
        verified = bool(commitment_ok and confidence >= self.config.verification_threshold)
        return {
            "verified": verified,
            "confidence": confidence,
            "commitment_ok": bool(commitment_ok),
            "commitment_errors": commitment_errors,
            "layer_scores": scores,
            "threshold": self.config.verification_threshold,
        }


def simulate_attack(
    weights: MutableMapping[str, np.ndarray],
    layers: Sequence[str],
    attack: str,
    rng: np.random.Generator,
    key: Optional[bytes] = None,
    fingerprint_k: int = 4096,
) -> MutableMapping[str, np.ndarray]:
    """Apply a controlled attack to selected layers."""

    for layer in layers:
        w = weights[layer]
        flat = w.reshape(-1)
        if attack == "clean":
            continue
        if attack == "pruning_10":
            cutoff = np.quantile(np.abs(flat), 0.10)
            flat[np.abs(flat) <= cutoff] = 0
        elif attack == "pruning_20":
            cutoff = np.quantile(np.abs(flat), 0.20)
            flat[np.abs(flat) <= cutoff] = 0
        elif attack == "quantization_8bit":
            lo, hi = float(flat.min()), float(flat.max())
            scale = (hi - lo) / 255.0
            if scale > 0:
                flat[:] = np.round((flat - lo) / scale) * scale + lo
        elif attack == "noise_0.002":
            flat += rng.normal(0.0, 0.002 * float(flat.std()), flat.size).astype(flat.dtype)
        elif attack == "noise_0.01":
            flat += rng.normal(0.0, 0.01 * float(flat.std()), flat.size).astype(flat.dtype)
        elif attack == "shuffle_05":
            idx = rng.permutation(flat.size)[: max(1, int(0.05 * flat.size))]
            flat[idx] = flat[idx[rng.permutation(idx.size)]]
        elif attack == "shuffle_10":
            idx = rng.permutation(flat.size)[: max(1, int(0.10 * flat.size))]
            flat[idx] = flat[idx[rng.permutation(idx.size)]]
        elif attack == "targeted_fingerprint_overwrite":
            if key is None:
                idx = rng.permutation(flat.size)[: min(fingerprint_k, flat.size)]
            else:
                idx = _fingerprint_indices(key, layer, flat.size, fingerprint_k)
            scale = float(flat.std()) + 1e-12
            flat[idx] = rng.normal(0.0, scale, idx.size).astype(flat.dtype)
        else:
            raise ValueError(f"Unsupported attack: {attack}")
    return weights
