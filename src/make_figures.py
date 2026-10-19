#!/usr/bin/env python3
"""Generate publication figures for the follow-up manuscript."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LEGACY_LABEL = "Hardened metadata +\nlegacy fingerprint"
RANK_LABEL = "RankGuard\n(proposed)"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_controlled(results: Path, out: Path) -> None:
    df = pd.read_csv(results / "controlled_security_trials.csv")
    order = [
        "clean",
        "unwatermarked_base",
        "unrelated_model",
        "wrong_key",
        "leaked_key_targeted_overwrite",
        "blind_targeted_overwrite",
        "tampered_manifest",
    ]
    labels = [
        "Watermarked\nclean",
        "Same base\nunwatermarked",
        "Unrelated\nmodel",
        "Wrong key",
        "Leaked-key\ntargeted",
        "Blind\ntargeted",
        "Tampered\nmanifest",
    ]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    width = 0.34
    x = np.arange(len(order))
    colors = {"legacy_fingerprint": "#B85C38", "rankguard": "#1F6F8B"}
    names = {"legacy_fingerprint": LEGACY_LABEL, "rankguard": RANK_LABEL}
    for j, scheme in enumerate(["legacy_fingerprint", "rankguard"]):
        means, sems = [], []
        for condition in order:
            values = df.loc[(df.scheme == scheme) & (df.condition == condition), "confidence"]
            means.append(values.mean())
            sems.append(values.std(ddof=1) / np.sqrt(values.size))
        ax.bar(x + (j - 0.5) * width, means, width, yerr=sems, capsize=2.5, label=names[scheme], color=colors[scheme], alpha=0.9)
    ax.axhline(0.80, color="#333333", linestyle="--", linewidth=1, label="Decision threshold")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Mean verification confidence")
    ax.set_ylim(0, 1.22)
    ax.set_title("Null controls separate checksum similarity from watermark evidence", pad=8)
    ax.legend(loc="upper center", ncols=3, frameon=False)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(out / "fig1_null_controls.png", bbox_inches="tight")
    plt.close(fig)

    attack_order = [
        "pruning_10",
        "pruning_20",
        "quantization_8bit",
        "noise_0.002",
        "noise_0.01",
        "shuffle_05",
        "shuffle_10",
    ]
    attack_labels = ["Prune\n10%", "Prune\n20%", "8-bit\nquant.", "Noise\n0.002", "Noise\n0.01", "Shuffle\n5%", "Shuffle\n10%"]
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    x = np.arange(len(attack_order))
    for j, scheme in enumerate(["legacy_fingerprint", "rankguard"]):
        means, sems = [], []
        for condition in attack_order:
            values = df.loc[(df.scheme == scheme) & (df.condition == condition), "confidence"]
            means.append(values.mean())
            sems.append(values.std(ddof=1) / np.sqrt(values.size))
        ax.bar(x + (j - 0.5) * width, means, width, yerr=sems, capsize=2.5, label=names[scheme], color=colors[scheme], alpha=0.9)
    ax.axhline(0.80, color="#333333", linestyle="--", linewidth=1)
    ax.set_xticks(x, attack_labels)
    ax.set_ylabel("Mean verification confidence")
    ax.set_ylim(0.84, 1.02)
    ax.set_title("Both detectors remain robust to blind global transformations")
    ax.legend(loc="lower left", frameon=False)
    ax.grid(axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(out / "fig2_attack_robustness.png", bbox_inches="tight")
    plt.close(fig)


def plot_audit(results: Path, out: Path) -> None:
    audit = pd.read_csv(results / "artifact_security_audit.csv")
    metrics = pd.Series(
        {
            "Plaintext key": audit["plaintext_key_present"].sum(),
            "Predictable pattern": audit["default_or_predictable_pattern"].sum(),
            "Hash-only recovery": audit["dictionary_recovered_from_hash_only"].sum(),
        }
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.2, 3.2), gridspec_kw={"width_ratios": [1.05, 1.4]})
    bars = ax1.bar(metrics.index, metrics.values, color=["#9E2A2B", "#C86B3D", "#5B4E77"])
    ax1.axhline(len(audit), color="#333333", linestyle="--", linewidth=1, label=f"All artifacts (n={len(audit)})")
    for bar in bars:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1, str(int(bar.get_height())), ha="center")
    ax1.set_ylim(0, len(audit) + 8)
    ax1.set_ylabel("Metadata files")
    ax1.set_title("Secret exposure in released artifacts")
    ax1.tick_params(axis="x", rotation=18)
    ax1.text(0.98, 0.94, "Dashed line: all artifacts (n=62)", transform=ax1.transAxes, ha="right", va="top", fontsize=8, color="#333333")
    ax1.grid(axis="y", alpha=0.2)

    sizes_kb = audit["file_bytes"] / 1000
    ax2.hist(sizes_kb, bins=18, color="#3A7CA5", edgecolor="white")
    ax2.axvline(sizes_kb.median(), color="#9E2A2B", linestyle="--", linewidth=1.2, label=f"Median = {sizes_kb.median():.1f} KB")
    ax2.set_xlabel("Metadata file size (KB)")
    ax2.set_ylabel("Count")
    ax2.set_title("Fingerprint-heavy metadata footprint")
    ax2.legend(frameon=False)
    ax2.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "fig3_artifact_audit.png", bbox_inches="tight")
    plt.close(fig)


def plot_protocol(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    ax.axis("off")

    boxes = [
        (0.04, 0.57, 0.20, 0.25, "Owner\n256-bit master key\n(private)", "#E8F3F8"),
        (0.32, 0.57, 0.22, 0.25, "Keyed subset\nand rank target\nHMAC-SHA256", "#D9EAD3"),
        (0.62, 0.57, 0.28, 0.25, "Checkpoint\nselected values sorted,\nthen rank-permuted", "#FCE5CD"),
        (0.04, 0.15, 0.24, 0.24, "Public manifest\nsalt, config, layer sizes,\nHMAC commitments", "#F4CCCC"),
        (0.38, 0.15, 0.24, 0.24, "Verifier\nrecompute subset + rank,\ncheck commitment", "#D9D2E9"),
        (0.72, 0.15, 0.22, 0.24, "Decision\nrank score ≥ τ and\nmanifest authenticated", "#CFE2F3"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#444444", linewidth=1.1, transform=ax.transAxes))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", transform=ax.transAxes, fontsize=9)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), xycoords=ax.transAxes, textcoords=ax.transAxes, arrowprops=dict(arrowstyle="->", lw=1.3, color="#444444"))

    arrow(0.24, 0.695, 0.32, 0.695)
    arrow(0.54, 0.695, 0.62, 0.695)
    arrow(0.32, 0.60, 0.24, 0.39)
    arrow(0.28, 0.27, 0.38, 0.27)
    arrow(0.62, 0.27, 0.72, 0.27)
    arrow(0.76, 0.57, 0.84, 0.39)
    ax.text(0.5, 0.93, "RankGuard: key-free public manifest and rank-locked ownership signal", ha="center", transform=ax.transAxes, fontsize=12, weight="bold")
    ax.text(0.5, 0.05, "Public artifacts do not contain the master key, raw weights, subset indices, or target permutation.", ha="center", transform=ax.transAxes, fontsize=8, color="#555555")
    fig.tight_layout()
    fig.savefig(out / "fig4_rankguard_protocol.png", bbox_inches="tight")
    plt.close(fig)


def plot_stage2(results: Path, out: Path) -> None:
    path = results / "stage2_paired_trial_differences.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    attack_order = ["none", "pruning_10", "quantization_8bit", "shuffle_05", "lora_light", "lora_medium"]
    labels = ["No attack", "Prune 10%", "8-bit quant.", "Shuffle 5%", "LoRA light", "LoRA medium"]
    data = [df.loc[df["attack"] == attack, "confidence_diff"].to_numpy() for attack in attack_order]
    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#B7D7E8")
        patch.set_edgecolor("#2F5D7C")
    for median in bp["medians"]:
        median.set_color("#9E2A2B")
        median.set_linewidth(1.4)
    ax.axhline(0, color="#333333", linestyle="--", linewidth=1)
    ax.set_ylabel("Paired confidence difference\n(PermuteGuard − sign-bit)")
    ax.set_title("Repeated-trial Stage 2 reanalysis")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "fig5_stage2_paired_difference.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    _style()
    plot_controlled(args.results, args.output)
    plot_audit(args.results, args.output)
    plot_protocol(args.output)
    plot_stage2(args.results, args.output)


if __name__ == "__main__":
    main()
