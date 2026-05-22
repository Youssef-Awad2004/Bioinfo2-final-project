"""
visualisation/data_stats.py
---------------------------
Summary stats for dataset CSV:
- SMILES length distribution
- Counts by type (anchors, positives, negatives, baselines)
- Positives/negatives per anchor stats

Usage:
  python visualisation/data_stats.py --csv cache/train.csv
  python visualisation/data_stats.py --csv cache/val.csv --plot
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CSV = "C:\\Users\\yousef\\Desktop\\College\\Bioinformatics_II\\Bioinfo2-final-project\\cache\\train.csv"


def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")
    return df[col]


def _print_len_stats(label: str, lengths: pd.Series) -> None:
    if lengths.empty:
        print(f"{label}: no rows")
        return

    print(f"{label}:")
    print(f"  count : {len(lengths)}")
    print(f"  min   : {int(lengths.min())}")
    print(f"  max   : {int(lengths.max())}")
    print(f"  mean  : {lengths.mean():.2f}")
    print(f"  median: {lengths.median():.2f}")


def _print_hist(lengths: pd.Series, bins: list[int]) -> None:
    if lengths.empty:
        print("  histogram: no data")
        return

    counts, edges = np.histogram(lengths.values, bins=bins)
    for i in range(len(counts)):
        lo = int(edges[i])
        hi = int(edges[i + 1])
        print(f"  {lo:4d}-{hi:4d}: {counts[i]}")


def _summary_counts(df: pd.DataFrame) -> None:
    print("\n-- Counts by type --")
    if "type" not in df.columns:
        print("Missing 'type' column; cannot summarize counts.")
        return

    counts = df["type"].value_counts(dropna=False)
    for t, c in counts.items():
        print(f"  {t:24s}: {int(c)}")


def _per_anchor_stats(anchors: pd.Series, positives: pd.DataFrame, negatives: pd.DataFrame) -> None:
    anchor_ids = set(anchors.dropna().astype(str).tolist())

    pos_counts = positives.groupby("anchor_id").size()
    neg_counts = negatives.groupby("anchor_id").size()

    def stats_for(label: str, series: pd.Series) -> None:
        if series.empty:
            print(f"  {label}: no rows")
            return
        print(f"  {label}: count={len(series)} min={series.min()} max={series.max()} \
mean={series.mean():.2f} median={series.median():.2f}")

    print("\n-- Positives/Negatives per anchor --")
    stats_for("positives/anchor", pos_counts)
    stats_for("negatives/anchor", neg_counts)

    pos_anchors = set(pos_counts.index.astype(str).tolist())
    neg_anchors = set(neg_counts.index.astype(str).tolist())

    missing_pos = sorted(anchor_ids - pos_anchors)
    missing_neg = sorted(anchor_ids - neg_anchors)

    print(f"  anchors total          : {len(anchor_ids)}")
    print(f"  anchors with positives : {len(pos_anchors)}")
    print(f"  anchors with negatives : {len(neg_anchors)}")
    print(f"  anchors missing pos    : {len(missing_pos)}")
    print(f"  anchors missing neg    : {len(missing_neg)}")


def _plot_stats(df: pd.DataFrame, outdir: Path, bins: list[int]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("\n[plot] matplotlib not available; skipping plots.")
        return

    outdir.mkdir(parents=True, exist_ok=True)
    out_file = outdir / "data_stats.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if "type" in df.columns:
        counts = df["type"].value_counts(dropna=False)
        axes[0].bar(counts.index.astype(str), counts.values)
        axes[0].set_title("Counts by type")
        axes[0].tick_params(axis="x", rotation=30, labelsize=8)
    else:
        axes[0].text(0.5, 0.5, "Missing type column", ha="center", va="center")
        axes[0].set_axis_off()

    lengths = df["smiles"].astype(str).str.len()
    axes[1].hist(lengths.values, bins=bins, color="#4C72B0", alpha=0.8)
    axes[1].set_title("SMILES length distribution")
    axes[1].set_xlabel("length")
    axes[1].set_ylabel("count")

    fig.tight_layout()
    fig.savefig(out_file, dpi=160)
    plt.close(fig)

    print(f"\n[plot] Wrote {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Path to train/val/test CSV")
    parser.add_argument("--plot", action="store_true", help="Save simple plots")
    parser.add_argument("--outdir", default="visualisation", help="Plot output directory")
    parser.add_argument("--bins", type=int, default=10, help="Number of bins for length histogram")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    smiles = _safe_series(df, "smiles").astype(str)
    lengths = smiles.str.len()

    print("=" * 60)
    print(f"CSV: {args.csv}")
    print(f"Rows: {len(df)}")
    print(f"Unique SMILES: {smiles.nunique()}")

    if "id" in df.columns:
        print(f"Unique IDs: {df['id'].nunique()}")

    _summary_counts(df)

    print("\n-- SMILES length stats (overall) --")
    _print_len_stats("overall", lengths)

    print("\n-- SMILES length histogram (overall) --")
    bins = np.linspace(lengths.min(), lengths.max(), max(args.bins, 2)).astype(int)
    bins = sorted(set(bins.tolist()))
    if len(bins) < 2:
        bins = [int(lengths.min()), int(lengths.max()) + 1]
    _print_hist(lengths, bins)

    if "type" in df.columns:
        for t, sub in df.groupby("type"):
            sub_lengths = sub["smiles"].astype(str).str.len()
            print(f"\n-- SMILES length stats ({t}) --")
            _print_len_stats(str(t), sub_lengths)

    if "type" in df.columns:
        anchors = df[df["type"] == "noncanonical_target"]["id"]
        positives = df[df["type"] == "positive_pair"]
        negatives = df[df["type"] == "hard_negative_pair"]
        _per_anchor_stats(anchors, positives, negatives)

    if args.plot:
        _plot_stats(df, Path(args.outdir), bins)

    print("=" * 60)


if __name__ == "__main__":
    main()
