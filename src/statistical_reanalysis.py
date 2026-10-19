#!/usr/bin/env python3
"""Repeated-trial statistical reanalysis of the archived PermuteGuard results."""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon


def wilson_interval(successes: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def stage1(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for folder in sorted((root / "results").glob("*_v4")):
        files = list(folder.glob("results_*.json"))
        if not files:
            continue
        data = json.loads(files[0].read_text())
        model = data["experiment_info"]["model_type"]
        for attack, payload in data["attack_results"].items():
            n = int(payload["confidence"]["n"])
            verified_n = int(round(payload["verified_rate"] * n))
            practical_n = int(round(payload["practical_robust_rate"] * n))
            vlo, vhi = wilson_interval(verified_n, n)
            plo, phi = wilson_interval(practical_n, n)
            rows.append(
                {
                    "model": model,
                    "attack": attack,
                    "trials": n,
                    "confidence_mean": payload["confidence"]["mean"],
                    "confidence_std": payload["confidence"]["std"],
                    "verified_rate": payload["verified_rate"],
                    "verified_wilson95_low": vlo,
                    "verified_wilson95_high": vhi,
                    "utility_ok_rate": payload["utility_ok_rate"],
                    "practical_rate": payload["practical_robust_rate"],
                    "practical_wilson95_low": plo,
                    "practical_wilson95_high": phi,
                }
            )
    return pd.DataFrame(rows)


def stage2(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    folders = {
        "GPT-2": "gpt2_cmp_lora_v2",
        "TinyLlama-1.1B": "tinyllama_cmp_lora_v2",
        "Falcon-7B": "falcon_cmp_lora_v2",
        "Mistral-7B": "mistral_cmp_lora_v2",
    }
    paired: List[Dict[str, Any]] = []
    for model, folder in folders.items():
        file = glob.glob(str(root / "stage2_results" / folder / "stage2_comparison_*.json"))[0]
        data = json.loads(Path(file).read_text())
        attacks = data["methods"]["permutation"]["attacks"].keys()
        for attack in attacks:
            pg = data["methods"]["permutation"]["attacks"][attack]["trials"]
            sb = data["methods"]["signbit"]["attacks"][attack]["trials"]
            for a, b in zip(pg, sb):
                paired.append(
                    {
                        "model": model,
                        "attack": attack,
                        "trial": int(a["trial"]),
                        "pg_confidence": a["confidence"],
                        "sb_confidence": b["confidence"],
                        "confidence_diff": a["confidence"] - b["confidence"],
                        "pg_accuracy": a["macro_accuracy_over_subset"],
                        "sb_accuracy": b["macro_accuracy_over_subset"],
                        "accuracy_diff": a["macro_accuracy_over_subset"] - b["macro_accuracy_over_subset"],
                        "pg_invalid": a["invalid_examples"],
                        "sb_invalid": b["invalid_examples"],
                        "invalid_diff": a["invalid_examples"] - b["invalid_examples"],
                    }
                )
    paired_df = pd.DataFrame(paired)
    rows = []
    for metric in ["confidence_diff", "accuracy_diff", "invalid_diff"]:
        vals = paired_df[metric].to_numpy(float)
        nonzero = vals[np.abs(vals) > 1e-12]
        pos = int(np.sum(vals > 1e-12))
        neg = int(np.sum(vals < -1e-12))
        tie = int(len(vals) - pos - neg)
        sign_p = binomtest(min(pos, neg), pos + neg, 0.5).pvalue if pos + neg else 1.0
        wil_p = wilcoxon(nonzero).pvalue if len(nonzero) >= 5 else float("nan")
        rows.append(
            {
                "metric": metric,
                "n": len(vals),
                "mean": float(vals.mean()),
                "median": float(np.median(vals)),
                "positive": pos,
                "negative": neg,
                "ties": tie,
                "exact_sign_p": sign_p,
                "wilcoxon_nonzero_p": wil_p,
            }
        )
    return paired_df, pd.DataFrame(rows)


def _mcnemar(b: int, c: int) -> float:
    return binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0


def stage3(root: Path) -> pd.DataFrame:
    files = {
        "Mistral-7B": root / "stage3_results" / "mistral_utility_v2" / "stage3_utility_mistral-7b.json",
        "Falcon-7B": root / "stage3_results" / "falcon_utility_core_v1" / "stage3_utility_falcon-7b.json",
    }
    rows = []
    for model, path in files.items():
        data = json.loads(path.read_text())
        for condition in ["shuffle_05", "lora_medium"]:
            pg = {x["id"]: x for x in data["methods"]["permutation"]["conditions"][condition]["utility"]["details"]}
            sb = {x["id"]: x for x in data["methods"]["signbit"]["conditions"][condition]["utility"]["details"]}
            ids = sorted(set(pg) & set(sb))
            for metric in ["choice_correct", "generation_valid", "generation_correct"]:
                b = sum(bool(pg[i][metric]) and not bool(sb[i][metric]) for i in ids)
                c = sum((not bool(pg[i][metric])) and bool(sb[i][metric]) for i in ids)
                rows.append(
                    {
                        "model": model,
                        "condition": condition,
                        "metric": metric,
                        "paired_prompts": len(ids),
                        "pg_only_wins": b,
                        "signbit_only_wins": c,
                        "mcnemar_exact_p": _mcnemar(b, c),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    s1 = stage1(args.archive_root)
    s2_pairs, s2_tests = stage2(args.archive_root)
    s3 = stage3(args.archive_root)
    s1.to_csv(args.output / "stage1_repeated_trial_reanalysis.csv", index=False)
    s2_pairs.to_csv(args.output / "stage2_paired_trial_differences.csv", index=False)
    s2_tests.to_csv(args.output / "stage2_statistical_tests.csv", index=False)
    s3.to_csv(args.output / "stage3_mcnemar_tests.csv", index=False)
    print("stage1_rows", len(s1))
    print("stage2_paired_rows", len(s2_pairs))
    print(s2_tests.to_string(index=False))
    print(s3.to_string(index=False))


if __name__ == "__main__":
    main()
