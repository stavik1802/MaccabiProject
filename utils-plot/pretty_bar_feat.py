#!/usr/bin/env python3
"""
Flexible plotter for experiment counts.

It reads a CSV with columns like:
  - feature, multiplier, group, final_keep_count
  - OR feature, multiplier, group, conversions_to_keep
  - OR feature, multiplier, group, subs_count

It auto-detects which metric column exists and produces per-feature bar charts
with groups (G1..G4) on the x-axis and one bar per multiplier.

Usage:
  python plot_pretty_improved.py \
    --csv subs_onpitch_perturb_out/final_keep_counts_by_feature_multiplier.csv \
    --out_dir subs_onpitch_perturb_out

Optionally force a metric:
  --metric final_keep_count
  --metric conversions_to_keep
  --metric subs_count
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Display labels you prefer ----------
LABEL_MAP = {
    "inst_dist_m_sum":  "Distance (m)",
    "Speed (m/s)_sum":  "Speed (m/s)",
    "hsr_m_sum":        "High-speed running (m)",
    "vha_count_1s_sum": "Very high acceleration count",
    "avg_jerk_1s_sum":  "Avg jerk",
    "turns_per_sec_sum":"Turns count",
    "total_sprints_sum":"Total sprints",
    "sprint_attack_sum":"Sprints (attack)",
    "sprint_defense_sum":"Sprints (defense)",
    "dist_attack_sum":  "Distance (attack)",
    "dist_defense_sum": "Distance (defense)",
    "playerload_1s_sum":"PlayerLoad",
}

# Nicer group labels on x-axis
GROUP_NICE = {"G1":"G1 (CB)", "G2":"G2 (CM/DM)", "G3":"G3 (FB/W)", "G4":"G4 (AM/CF)"}

def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")

def pick_metric_col(df: pd.DataFrame, user_metric: str | None) -> str:
    if user_metric:
        if user_metric not in df.columns:
            raise ValueError(f"--metric '{user_metric}' not found in CSV. Available: {list(df.columns)}")
        return user_metric
    for cand in ("final_keep_count", "conversions_to_keep", "subs_count"):
        if cand in df.columns:
            return cand
    raise ValueError(
        "CSV must contain one of: 'final_keep_count', 'conversions_to_keep', or 'subs_count'. "
        f"Columns found: {list(df.columns)}"
    )

def titles_for_metric(metric_col: str) -> tuple[str, str, str]:
    """Return (title_prefix, y_label, subfolder) based on metric."""
    m = metric_col.lower()
    if "final_keep" in m:
        return ("KEEP counts vs perturbation — ", "# of KEEP (stay) decisions", "plots_keep")
    if "conversion" in m:
        return ("Fewer subs after perturbation — ", "Conversions to KEEP (SUB → KEEP)", "plots_conversions")
    return ("SUB recommendations per group vs perturbation — ", "# of SUB recommendations (total)", "plots_subs")

def plot_one_feature(df_feat: pd.DataFrame, feature_raw: str, metric_col: str, out_dir: Path):
    disp = LABEL_MAP.get(feature_raw, feature_raw)
    groups = ["G1","G2","G3","G4"]
    xticklabels = [GROUP_NICE.get(g, g) for g in groups]

    multipliers = sorted(df_feat["multiplier"].unique())
    x = np.arange(len(groups))
    width = 0.8 / max(1, len(multipliers))

    title_prefix, ylab, subfolder = titles_for_metric(metric_col)
    plot_dir = out_dir / subfolder
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    for i, m in enumerate(multipliers):
        vals = []
        for g in groups:
            sub = df_feat[(df_feat["group"] == g) & (df_feat["multiplier"] == m)]
            vals.append(int(sub[metric_col].sum()) if not sub.empty else 0)
        plt.bar(x + i*width, vals, width, label=f"x{m:g}")

    plt.xticks(x + (len(multipliers)-1)*width/2.0, xticklabels)
    plt.xlabel("Group")
    plt.ylabel(ylab)
    plt.title(f"{title_prefix}{disp}")
    plt.legend()

    out_path = plot_dir / f"{safe_name(metric_col)}_{safe_name(disp)}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to CSV from the experiment")
    ap.add_argument("--out_dir", required=True, help="Root output directory for plots")
    ap.add_argument("--metric", default=None,
                    help="Optional: explicitly choose the metric column "
                         "(final_keep_count | conversions_to_keep | subs_count)")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_root = Path(args.out_dir)

    df = pd.read_csv(csv_path)
    required_base = {"feature", "multiplier", "group"}
    if not required_base.issubset(df.columns):
        raise ValueError(f"CSV must include columns at least {required_base}. Found: {list(df.columns)}")

    metric_col = pick_metric_col(df, args.metric)
    print(f"Using metric column: {metric_col}")

    # Coerce types
    df["multiplier"] = pd.to_numeric(df["multiplier"], errors="coerce")
    df = df.dropna(subset=["multiplier"])
    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce").fillna(0).astype(int)

    feats = sorted(df["feature"].unique())
    print(f"Found {len(feats)} features to plot.")
    for f in feats:
        out = plot_one_feature(df[df["feature"] == f].copy(), f, metric_col, out_root)
        print(f"  📈 {out}")

    # Optional labeled CSV for reference
    labeled_csv = out_root / f"{safe_name(metric_col)}_labeled.csv"
    df_out = df.copy()
    df_out["feature_label"] = df_out["feature"].map(lambda x: LABEL_MAP.get(x, x))
    df_out.to_csv(labeled_csv, index=False)
    print(f"✅ Wrote labeled CSV: {labeled_csv}")

if __name__ == "__main__":
    main()
