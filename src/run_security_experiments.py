#!/usr/bin/env python3
"""Run the controlled security experiments for the follow-up manuscript.

The experiments compare two detectors under identical synthetic tensors:

* Legacy-fingerprint: leakage-resistant metadata combined with the original
  post-embedding cosine fingerprint. This isolates whether metadata hardening
  alone is sufficient.
* RankGuard: the proposed rank-locked subset verifier, which uses the embedded
  permutation itself as the ownership signal.

No recovered plaintext keys are written to disk by the audit; only boolean
outcomes and short hashes of serialized keys are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional

import numpy as np
import pandas as pd

from hardened_permuteguard import (
    HardenedPermuteGuard,
    SecureConfig,
    generate_master_key,
    simulate_attack as simulate_legacy_attack,
)
from rankguard import RankGuard, RankGuardConfig, simulate_attack as simulate_rank_attack


MODEL_TYPES = ["gpt2", "tinyllama", "mistral", "falcon", "bloom", "opt"]
PREFIXES = ["PG", "PG7B", "SB"]


def audit_metadata(root: Path) -> pd.DataFrame:
    """Audit metadata artifacts without writing recovered secret strings."""

    paths = sorted(set(root.rglob("*metadata.json")) | set(root.rglob("*signbit.json")))
    rows: List[Dict[str, Any]] = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            rows.append({"file": str(path), "parse_error": str(exc)})
            continue
        config = data.get("config", {}) if isinstance(data.get("config"), dict) else {}
        key = config.get("watermark_key")
        model_id = str(data.get("model_id", ""))
        basename = os.path.basename(model_id.rstrip("/\\"))
        key_hash = data.get("key_hash")
        candidates = []
        for prefix in PREFIXES:
            for model_type in MODEL_TYPES:
                candidates.append(f"{prefix}::{model_type}::{basename}")
            candidates.extend([f"{prefix}::{basename}", f"{prefix}::{model_id}"])
        recovered = False
        if key_hash:
            for candidate in candidates:
                if hashlib.sha256(candidate.encode("utf-8")).hexdigest() == key_hash:
                    recovered = True
                    break
        layer_meta = data.get("layer_metadata", {}) if isinstance(data.get("layer_metadata"), dict) else {}
        stored_entries = []
        for payload in layer_meta.values():
            if isinstance(payload, dict):
                stored_entries.append(len(payload.get("fingerprint", payload.get("indices", []))))
        rows.append(
            {
                "file": str(path),
                "model_id": model_id,
                "layers": len(layer_meta),
                "stored_entries_total": int(sum(stored_entries)),
                "plaintext_key_present": key is not None,
                "default_or_predictable_pattern": bool(
                    isinstance(key, str) and key.startswith(tuple(f"{p}::" for p in PREFIXES))
                ),
                "dictionary_recovered_from_hash_only": recovered,
                "key_sha256_12": hashlib.sha256(str(key).encode()).hexdigest()[:12]
                if key is not None
                else None,
                "file_bytes": path.stat().st_size,
            }
        )
    return pd.DataFrame(rows)


def _copy_weights(weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {name: value.copy() for name, value in weights.items()}


def _record(rows: List[Dict[str, Any]], trial: int, scheme: str, condition: str, attacker_key: str, verdict: Dict[str, Any]) -> None:
    rows.append(
        {
            "trial": trial,
            "scheme": scheme,
            "condition": condition,
            "attacker_key": attacker_key,
            "confidence": float(verdict["confidence"]),
            "verified": bool(verdict["verified"]),
            "commitment_ok": bool(verdict["commitment_ok"]),
        }
    )


def _tamper_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    tampered = json.loads(json.dumps(manifest))
    tampered["config"]["verification_threshold"] = 0.99
    return tampered


def run_controlled_experiments(trials: int, seed: int, output: Path) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """Run paired synthetic-tensor experiments with GPT-2 projection-sized layers."""

    rng = np.random.default_rng(seed)
    shapes = [(768, 2304), (768, 2304), (768, 2304), (768, 2304)]
    layer_names = [f"h.{i}.attn.c_proj" for i in [0, 4, 7, 11]]
    global_attacks = [
        "clean",
        "pruning_10",
        "pruning_20",
        "quantization_8bit",
        "noise_0.002",
        "noise_0.01",
        "shuffle_05",
        "shuffle_10",
    ]
    rows: List[Dict[str, Any]] = []
    manifest_key_hits = {"legacy_fingerprint": [], "rankguard": []}
    raw_fingerprint_entries = {"legacy_fingerprint": [], "rankguard": []}

    for trial in range(1, trials + 1):
        legacy_key = generate_master_key()
        rank_key = generate_master_key()
        wrong_key = generate_master_key()
        base = {
            name: rng.normal(0.0, 0.02, size=shape).astype(np.float32)
            for name, shape in zip(layer_names, shapes)
        }
        unrelated = {
            name: rng.normal(0.0, 0.02, size=shape).astype(np.float32)
            for name, shape in zip(layer_names, shapes)
        }

        legacy = HardenedPermuteGuard(legacy_key, SecureConfig(permutation_strength=0.005))
        rank = RankGuard(rank_key, RankGuardConfig(subset_fraction=0.005))
        legacy_wm, legacy_manifest, legacy_private = legacy.embed(_copy_weights(base), "synthetic-gpt2", layer_names)
        rank_wm, rank_manifest, rank_bundle = rank.embed(_copy_weights(base), "synthetic-gpt2", layer_names)

        for scheme_name, manifest, key in [
            ("legacy_fingerprint", legacy_manifest, legacy_key),
            ("rankguard", rank_manifest, rank_key),
        ]:
            encoded = json.dumps(manifest, sort_keys=True).encode("utf-8")
            manifest_key_hits[scheme_name].append(1 if key in encoded or key.hex().encode() in encoded else 0)
            raw_fingerprint_entries[scheme_name].append(
                sum(1 for payload in manifest["layers"].values() if "fingerprint" in payload)
            )

        # Null controls evaluated before attacks.
        _record(rows, trial, "legacy_fingerprint", "unwatermarked_base", "correct_owner_key", legacy.verify(base, legacy_manifest, legacy_private))
        _record(rows, trial, "rankguard", "unwatermarked_base", "correct_owner_key", rank.verify(base, rank_manifest, rank_bundle))
        _record(rows, trial, "legacy_fingerprint", "unrelated_model", "correct_owner_key", legacy.verify(unrelated, legacy_manifest, legacy_private))
        _record(rows, trial, "rankguard", "unrelated_model", "correct_owner_key", rank.verify(unrelated, rank_manifest, rank_bundle))

        for attack in global_attacks:
            legacy_attacked = _copy_weights(legacy_wm)
            simulate_legacy_attack(
                legacy_attacked,
                layer_names,
                attack,
                np.random.default_rng(seed + 10_000 * trial),
                fingerprint_k=legacy.config.fingerprint_k,
            )
            _record(rows, trial, "legacy_fingerprint", attack, "none", legacy.verify(legacy_attacked, legacy_manifest, legacy_private))

            rank_attacked = _copy_weights(rank_wm)
            simulate_rank_attack(
                rank_attacked,
                layer_names,
                attack,
                np.random.default_rng(seed + 10_000 * trial),
                config=rank.config,
            )
            _record(rows, trial, "rankguard", attack, "none", rank.verify(rank_attacked, rank_manifest, rank_bundle))

        # Blind targeted removal uses a same-size guess at the protected coordinates.
        legacy_blind = _copy_weights(legacy_wm)
        simulate_legacy_attack(
            legacy_blind,
            layer_names,
            "targeted_fingerprint_overwrite",
            np.random.default_rng(seed + 20_000 * trial),
            key=None,
            fingerprint_k=legacy.config.fingerprint_k,
        )
        _record(rows, trial, "legacy_fingerprint", "blind_targeted_overwrite", "none", legacy.verify(legacy_blind, legacy_manifest, legacy_private))

        rank_blind = _copy_weights(rank_wm)
        simulate_rank_attack(
            rank_blind,
            layer_names,
            "targeted_subset_overwrite",
            np.random.default_rng(seed + 20_000 * trial),
            key=None,
            config=rank.config,
        )
        _record(rows, trial, "rankguard", "blind_targeted_overwrite", "none", rank.verify(rank_blind, rank_manifest, rank_bundle))

        # Leaked-key removal models the original public metadata failure.
        legacy_leaked = _copy_weights(legacy_wm)
        simulate_legacy_attack(
            legacy_leaked,
            layer_names,
            "targeted_fingerprint_overwrite",
            np.random.default_rng(seed + 30_000 * trial),
            key=legacy_key,
            fingerprint_k=legacy.config.fingerprint_k,
        )
        _record(rows, trial, "legacy_fingerprint", "leaked_key_targeted_overwrite", "leaked_master_key", legacy.verify(legacy_leaked, legacy_manifest, legacy_private))

        rank_leaked = _copy_weights(rank_wm)
        simulate_rank_attack(
            rank_leaked,
            layer_names,
            "targeted_subset_overwrite",
            np.random.default_rng(seed + 30_000 * trial),
            key=rank_key,
            config=rank.config,
        )
        _record(rows, trial, "rankguard", "leaked_key_targeted_overwrite", "leaked_master_key", rank.verify(rank_leaked, rank_manifest, rank_bundle))

        # Wrong-key and manifest-tampering controls are explicitly evaluated.
        _record(rows, trial, "legacy_fingerprint", "wrong_key", "wrong_key", legacy.verify(legacy_wm, legacy_manifest, legacy_private, candidate_key=wrong_key))
        _record(rows, trial, "rankguard", "wrong_key", "wrong_key", rank.verify(rank_wm, rank_manifest, rank_bundle, candidate_key=wrong_key))
        _record(rows, trial, "legacy_fingerprint", "tampered_manifest", "none", legacy.verify(legacy_wm, _tamper_manifest(legacy_manifest), legacy_private))
        _record(rows, trial, "rankguard", "tampered_manifest", "none", rank.verify(rank_wm, _tamper_manifest(rank_manifest), rank_bundle))

    df = pd.DataFrame(rows)
    by_scheme = {}
    for scheme_name, group in df.groupby("scheme"):
        by_scheme[scheme_name] = {
            "public_manifest_contains_master_key_rate": float(np.mean(manifest_key_hits[scheme_name])),
            "public_manifest_raw_fingerprint_entry_count_mean": float(np.mean(raw_fingerprint_entries[scheme_name])),
            "unwatermarked_base_verified_rate": float(group.loc[group.condition == "unwatermarked_base", "verified"].mean()),
            "unrelated_model_verified_rate": float(group.loc[group.condition == "unrelated_model", "verified"].mean()),
            "wrong_key_verified_rate": float(group.loc[group.condition == "wrong_key", "verified"].mean()),
            "tampered_manifest_verified_rate": float(group.loc[group.condition == "tampered_manifest", "verified"].mean()),
            "leaked_key_targeted_overwrite_verified_rate": float(group.loc[group.condition == "leaked_key_targeted_overwrite", "verified"].mean()),
        }
    summary = {
        "trials": trials,
        "layers_per_trial": len(layer_names),
        "tensor_shapes": shapes,
        "schemes": by_scheme,
    }
    return df, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20261013)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    if args.artifact_root is not None:
        audit = audit_metadata(args.artifact_root)
        audit.to_csv(args.output / "artifact_security_audit.csv", index=False)
    else:
        audit = pd.DataFrame()

    controlled, summary = run_controlled_experiments(args.trials, args.seed, args.output)
    controlled.to_csv(args.output / "controlled_security_trials.csv", index=False)
    condition_summary = (
        controlled.groupby(["scheme", "condition"])
        .agg(
            trials=("verified", "size"),
            confidence_mean=("confidence", "mean"),
            confidence_sd=("confidence", "std"),
            verified_rate=("verified", "mean"),
            commitment_ok_rate=("commitment_ok", "mean"),
        )
        .reset_index()
    )
    condition_summary.to_csv(args.output / "controlled_security_condition_summary.csv", index=False)
    summary_obj = {
        "controlled": summary,
        "artifact_audit": {
            "files": int(len(audit)),
            "plaintext_key_files": int(audit["plaintext_key_present"].sum()) if not audit.empty else 0,
            "predictable_key_files": int(audit["default_or_predictable_pattern"].sum()) if not audit.empty else 0,
            "dictionary_recovered_files": int(audit["dictionary_recovered_from_hash_only"].sum()) if not audit.empty else 0,
            "unique_plaintext_key_hashes": int(audit["key_sha256_12"].nunique()) if not audit.empty else 0,
            "total_metadata_mb": float(audit["file_bytes"].sum() / 1e6) if not audit.empty else 0.0,
            "median_metadata_kb": float(audit["file_bytes"].median() / 1e3) if not audit.empty else 0.0,
        },
    }
    (args.output / "security_experiment_summary.json").write_text(
        json.dumps(summary_obj, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary_obj, indent=2))


if __name__ == "__main__":
    main()
