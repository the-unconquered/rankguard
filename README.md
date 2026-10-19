# RankGuard Reproducibility Package

Sanitized companion package for the manuscript "RankGuard: Exposing and Repairing Key Leakage and Null-Model Acceptance in Parameter-Permutation Watermarking for Open-Weight Large Language Models".

**No secret material is included.** The artifact audit reports counts, boolean outcomes, and short (8-hex-digit) key-hash prefixes only; recovered key strings were never written to disk.

## Contents

- `src/rankguard.py` — RankGuard implementation (NumPy + standard library only): 256-bit master key, HMAC-SHA256-derived subset and rank permutation, multiset-preserving embedding, normalized Spearman rank verification, key-free public manifest with HMAC commitments and manifest digest, private owner record.
- `src/hardened_permuteguard.py` — the legacy cosine-fingerprint detector wrapped in hardened (key-free) metadata; used as the control showing that metadata hardening alone does not fix null-model acceptance.
- `src/run_security_experiments.py` — paired controlled experiment harness (30 trials per condition) comparing the legacy detector and RankGuard on GPT-2-shaped synthetic tensors; also performs the sanitized artifact audit.
- `src/test_rankguard.py`, `src/test_hardened_permuteguard.py` — regression tests, including the documented legacy null false-acceptance.
- `src/statistical_reanalysis.py` — repeated-trial reanalysis of the archived experiments (Wilson intervals, exact sign / Wilcoxon signed-rank / McNemar exact tests).
- `src/make_figures.py` — regenerates all manuscript figures from the result CSVs.
- `results/` — all derived statistics (CSV/JSON) reported in the manuscript.
- `figures/` — the five manuscript figures (PNG).

## Quick start

```bash
python3 src/test_rankguard.py            # expect: "rankguard tests passed"
python3 src/test_hardened_permuteguard.py  # expect: legacy null false-acceptance reproduced
python3 src/run_security_experiments.py  # regenerates results/controlled_*.csv + security_experiment_summary.json
python3 src/make_figures.py              # regenerates figures/
```

Only NumPy (and Matplotlib for figures) is required; no model checkpoints, GPUs, or deep-learning frameworks.

## Scope notes

- Controlled experiments use synthetic tensors with GPT-2 layer shapes (768x2304, 4 layers) because model checkpoints were unavailable in the analysis environment. The manuscript makes no downstream-utility claim for RankGuard.
- The repeated-trial reanalysis describes the legacy PermuteGuard pipeline on real LLMs from archived result files; those numbers do not transfer to RankGuard.
- To run RankGuard on a real checkpoint, load the model's state dict, call `RankGuard.embed(...)` on the target layers, and verify with `RankGuard.verify(...)`. Keep the returned private owner record secret.
