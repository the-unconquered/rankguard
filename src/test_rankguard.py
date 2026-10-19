from copy import deepcopy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rankguard import RankGuard, generate_master_key, simulate_attack


def make_weights(seed=123):
    rng = np.random.default_rng(seed)
    return {
        "h.0.attn.c_proj": rng.normal(0, 0.02, (768, 2304)).astype(np.float32),
        "h.4.attn.c_proj": rng.normal(0, 0.02, (768, 2304)).astype(np.float32),
        "h.7.attn.c_proj": rng.normal(0, 0.02, (768, 2304)).astype(np.float32),
        "h.11.attn.c_proj": rng.normal(0, 0.02, (768, 2304)).astype(np.float32),
    }


original = make_weights()
key = generate_master_key()
wm = RankGuard(key)
watermarked, manifest, private = wm.embed(deepcopy(original), "synthetic-gpt2")

serialized = str(manifest)
assert key.hex() not in serialized
assert "master_key" not in serialized
assert private["owner_record"]["manifest_digest"] == manifest["manifest_digest"]

clean = wm.verify(watermarked, manifest, private)
assert clean["verified"], clean
assert clean["confidence"] > 0.999, clean["confidence"]

# The critical null control: the untouched base model must not be accepted.
null_result = wm.verify(original, manifest, private)
assert not null_result["verified"], null_result
assert null_result["confidence"] < 0.55, null_result["confidence"]

wrong = wm.verify(watermarked, manifest, private, candidate_key=generate_master_key())
assert not wrong["verified"], wrong
assert not wrong["commitment_ok"]

unrelated = wm.verify(make_weights(987), manifest, private)
assert not unrelated["verified"], unrelated
assert unrelated["confidence"] < 0.55, unrelated["confidence"]

tampered = deepcopy(manifest)
tampered["config"]["subset_fraction"] = 0.002
assert not wm.verify(watermarked, tampered, private)["verified"]

rng = np.random.default_rng(42)
for attack in ["pruning_10", "pruning_20", "quantization_8bit", "noise_0.002", "noise_0.01", "shuffle_05", "shuffle_10"]:
    attacked = simulate_attack(deepcopy(watermarked), manifest["selected_layers"], attack, rng)
    result = wm.verify(attacked, manifest, private)
    assert result["verified"], (attack, result)

blind = simulate_attack(deepcopy(watermarked), manifest["selected_layers"], "targeted_subset_overwrite", rng)
blind_result = wm.verify(blind, manifest, private)
assert blind_result["verified"], blind_result

leaked = simulate_attack(deepcopy(watermarked), manifest["selected_layers"], "targeted_subset_overwrite", rng, key=key)
leaked_result = wm.verify(leaked, manifest, private)
assert not leaked_result["verified"], leaked_result
assert leaked_result["confidence"] < 0.55, leaked_result["confidence"]

print("rankguard tests passed")
print("clean", clean["confidence"])
print("unwatermarked", null_result["confidence"])
print("unrelated", unrelated["confidence"])
print("leaked targeted", leaked_result["confidence"])
