#!/usr/bin/env python3
"""
Plot KEEP (stay) counts per starter group (G1..G4) for each feature × multiplier,
using the file produced by your on-pitch perturbation experiment:

  final_keep_counts_by_feature_multiplier.csv
    columns: feature, multiplier, group, final_keep_count

Outputs one chart per feature to:
  <out_dir>/plots_keep/keep_counts_<pretty_feature_name>.png

Usage:
  python plot_keep_counts.py \
    --csv subs_onpitch_perturb_out/final_keep_counts_by_feature_multiplier.csv \
    --out_dir subs_onpitch_perturb_out
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI needed
import matplotlib.pyplot as plt

# ---------- Pretty display names you want ----------
LABEL_MAP = {
    "inst_dist_m_sum":   "Distance (m)",
    "Speed (m/s)_sum":   "Speed (m/s)",
    "hsr_m_sum":         "High-speed running (m)",
    "vha_count_1s_sum":  "Very high acceleration count",
    "avg_jerk_1s_sum":   "Avg jerk",
    "turns_per_sec_sum": "Turns count",
    "total_sprints_sum": "Total sprints",
    "sprint_attack_sum": "Sprints (attack)",
    "sprint_defense_sum":"Sprints (defense)",
    "dist_attack_sum":   "Distance (attack)",
    "dist_defense_sum":  "Distance (defense)",
    "playerload_1s_sum": "PlayerLoad",
}

# Nicer group labels on x-axis
GROUP_NICE = {"G1":"G1 (CB)", "G2":"G2 (CM/DM)", "G3":"G3 (FB/W)", "G4":"G4 (AM/CF)"}

def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")

def plot_keep_for_feature(df_feat: pd.DataFrame, feature_raw: str, out_dir: Path) -> Path:
    disp = LABEL_MAP.get(feature_raw, feature_raw)
    groups = ["G1","G2","G3","G4"]
    xlabels = [GROUP_NICE.get(g, g) for g in groups]

    multipliers = sorted(df_feat["multiplier"].unique())
    x = np.arange(len(groups))
    width = 0.8 / max(1, len(multipliers))

    plt.figure(figsize=(10, 5))
    for i, m in enumerate(multipliers):
        vals = []
        for g in groups:
            sub = df_feat[(df_feat["group"] == g) & (df_feat["multiplier"] == m)]
            vals.append(int(sub["final_keep_count"].sum()) if not sub.empty else 0)
        plt.bar(x + i*width, vals, width, label=f"x{m:g}")

    plt.xticks(x + (len(multipliers)-1)*width/2.0, xlabels)
    plt.xlabel("Starter group")
    plt.ylabel("# of KEEP (stay) decisions")
    plt.title(f"KEEP counts vs perturbation — {disp}")
    plt.legend()

    plot_dir = out_dir / "plots_keep"
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / f"keep_counts_{safe_name(disp)}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",
        default="subs_onpitch_perturb_out/final_keep_counts_by_feature_multiplier.csv",
        help="Path to final_keep_counts_by_feature_multiplier.csv")
    ap.add_argument("--out_dir", default="subs_onpitch_perturb_out",
        help="Root output dir; plots go under plots_keep/")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_root = Path(args.out_dir)

    df = pd.read_csv(csv_path)
    needed = {"feature", "multiplier", "group", "final_keep_count"}
    if not needed.issubset(df.columns):
        raise ValueError(f"CSV must include columns: {needed}. Found: {list(df.columns)}")

    # normalize dtypes
    df["multiplier"] = pd.to_numeric(df["multiplier"], errors="coerce")
    df = df.dropna(subset=["multiplier"])
    df["final_keep_count"] = pd.to_numeric(df["final_keep_count"], errors="coerce").fillna(0).astype(int)

    feats = sorted(df["feature"].unique())
    print(f"Found {len(feats)} features.")
    for f in feats:
        out = plot_keep_for_feature(df[df["feature"] == f].copy(), f, out_root)
        print(f"  📈 {out}")

    # Optional: save a labeled copy for reference
    labeled_csv = out_root / "final_keep_counts_by_feature_multiplier_labeled.csv"
    df.assign(feature_label=df["feature"].map(lambda x: LABEL_MAP.get(x, x))).to_csv(labeled_csv, index=False)
    print(f"✅ Wrote labeled CSV: {labeled_csv}")

if __name__ == "__main__":
    main()
