#!/usr/bin/env python3
"""Minimal regression tests for the hardened metadata implementation."""

import json
import numpy as np

from hardened_permuteguard import (
    HardenedPermuteGuard,
    SecureConfig,
    generate_master_key,
    simulate_attack,
)


def main() -> None:
    rng = np.random.default_rng(123)
    key = generate_master_key()
    wrong = generate_master_key()
    wm = HardenedPermuteGuard(key, SecureConfig(permutation_strength=0.005))
    weights = {
        "layer0": rng.normal(0, 0.02, (768, 2304)).astype(np.float32),
        "layer1": rng.normal(0, 0.02, (768, 2304)).astype(np.float32),
    }
    original = {k: v.copy() for k, v in weights.items()}
    watermarked, manifest, private = wm.embed(weights, "unit-test", list(weights))
    public_bytes = json.dumps(manifest, sort_keys=True).encode()
    assert key not in public_bytes
    assert all("fingerprint" not in p for p in manifest["layers"].values())
    assert wm.verify(watermarked, manifest, private)["verified"]
    assert not wm.verify(watermarked, manifest, private, candidate_key=wrong)["verified"]
    # This is the documented legacy-detector failure, not a desired property:
    # a same-base unwatermarked model is still cosine-similar to the stored
    # post-embedding fingerprint because most sampled coordinates are unchanged.
    legacy_null = wm.verify(original, manifest, private)
    assert legacy_null["verified"], legacy_null
    tampered = json.loads(json.dumps(manifest))
    tampered["config"]["verification_threshold"] = 0.99
    assert not wm.verify(watermarked, tampered, private)["verified"]
    blind = {k: v.copy() for k, v in watermarked.items()}
    simulate_attack(blind, list(weights), "targeted_fingerprint_overwrite", rng, key=None)
    assert wm.verify(blind, manifest, private)["verified"]
    leaked = {k: v.copy() for k, v in watermarked.items()}
    simulate_attack(leaked, list(weights), "targeted_fingerprint_overwrite", rng, key=key)
    assert not wm.verify(leaked, manifest, private)["verified"]
    print("all hardened metadata tests passed; legacy null false-acceptance reproduced")


if __name__ == "__main__":
    main()
