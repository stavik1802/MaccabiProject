#!/usr/bin/env python3
"""
Plot SHAP importance for your grouped CatBoost model.

Inputs (CSVs from your SHAP analysis script):
- --agg_csv: shap_feature_importance_aggregated.csv
- --group_csv: shap_feature_importance_by_group.csv

Outputs:
- top_base_features_importance.png
- top_features_G1.png
- top_features_G2.png
- top_features_G3.png
- top_features_G4.png

Notes:
- No seaborn, one chart per figure, default colors.
- Use --exclude <name or substring> (repeatable) to filter features from plots.
"""

# Map raw column names -> display labels (edit as you like)
LABEL_MAP = {
    "inst_dist_m_sum": "Distance (m)",
    "Speed (m/s)_mean": "Speed (m/s)",
    "hsr_m_sum": "High-speed running (m)",
    "vha_count_1s_sum": "Very high accelaration count",
    "avg_jerk_1s_mean": "Avg jerk",
    "turns_per_sec_sum": "Turns count",
    "playerload_1s_sum": "PlayerLoad ",
    "walk_time_sum": "Walk time",
    "jog_time_sum": "Jog time",
    "run_time_sum": "Run time",
    "sprint_time_sum": "Sprint time",
    "total_sprints_sum": "Total sprints",
    "sprint_attack_sum": "Sprints (attack)",
    "sprint_defense_sum": "Sprints (defense)",
    "dist_attack_sum": "Distance (attack)",
    "dist_defense_sum": "Distance (defense)",
    "time_attack_sum": "Time (attack)",
    "time_defense_sum": "Time (defense)",
    "attacking_third_time_sum": "Time in attacking third",
    "middle_third_time_sum": "Time in middle third",
    "defending_third_time_sum": "Time in defending third",
}

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def short_name(name: str, max_len: int = 40) -> str:
    if len(name) <= max_len:
        return name
    return name[:max_len // 2 - 1] + "…" + name[-max_len // 2 + 2:]


def _exclude_mask(series: pd.Series, excludes: list[str]) -> pd.Series:
    """Return boolean mask: True if row should be excluded (matches any exclude item)."""
    if not excludes:
        return pd.Series(False, index=series.index)
    low = series.astype(str).str.lower()
    ex = [e.lower() for e in excludes]
    mask = pd.Series(False, index=series.index)
    for e in ex:
        mask |= low.str.contains(e, na=False)
    return mask


def _labelize(base_feature: str) -> str:
    """Map base feature name to a prettier label if available."""
    return LABEL_MAP.get(base_feature, base_feature)


def plot_top_aggregated(agg_csv: Path, out_dir: Path, top_n: int, excludes: list[str]) -> Path:
    df = pd.read_csv(agg_csv)
    if "base_feature" not in df.columns or "total_importance" not in df.columns:
        raise ValueError("Aggregated CSV must have 'base_feature' and 'total_importance' columns.")
    # apply exclusions on base_feature
    mask_ex = _exclude_mask(df["base_feature"], excludes)
    df = df[~mask_ex]

    top = df.sort_values("total_importance", ascending=False).head(top_n).copy()
    # Use pretty labels
    pretty = top["base_feature"].apply(_labelize).tolist()
    labels = [short_name(x) for x in pretty]
    vals = top["total_importance"].values

    plt.figure(figsize=(10, max(6, 0.35 * len(top))))
    y_pos = np.arange(len(top))
    plt.barh(y_pos, vals)
    plt.yticks(y_pos, labels)
    plt.gca().invert_yaxis()
    plt.xlabel("Total SHAP importance (sum of |SHAP| over groups)")
    plt.title(f"Top {len(top)} Base Physical Features by Importance")
    out_path = out_dir / "top_base_features_importance.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def plot_top_by_group(group_csv: Path, out_dir: Path, top_n: int, excludes: list[str]) -> list[Path]:
    df = pd.read_csv(group_csv)
    required = {"base_feature", "group", "mean_abs_shap"}
    if not required.issubset(df.columns):
        raise ValueError(f"Group CSV must include columns: {required}")
    # apply exclusions on base_feature
    mask_ex = _exclude_mask(df["base_feature"], excludes)
    df = df[~mask_ex]

    out_paths = []
    for g in ["G1", "G2", "G3", "G4"]:
        sub = df[df["group"] == g]
        agg = sub.groupby("base_feature", as_index=False)["mean_abs_shap"].sum()
        top = agg.sort_values("mean_abs_shap", ascending=False).head(top_n)
        # Use pretty labels
        pretty = top["base_feature"].apply(_labelize).tolist()
        labels = [short_name(x) for x in pretty]
        vals = top["mean_abs_shap"].values

        plt.figure(figsize=(10, max(6, 0.35 * len(top))))
        y_pos = np.arange(len(top))
        plt.barh(y_pos, vals)
        plt.yticks(y_pos, labels)
        plt.gca().invert_yaxis()
        plt.xlabel(f"SHAP importance within {g}")
        plt.title(f"Top {len(top)} Features by Importance – {g}")
        out_path = out_dir / f"top_features_{g}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        out_paths.append(out_path)
    return out_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg_csv", required=True, help="Path to shap_feature_importance_aggregated.csv")
    ap.add_argument("--group_csv", required=True, help="Path to shap_feature_importance_by_group.csv")
    ap.add_argument("--out_dir", default="feature_importance_out", help="Output directory for plots")
    ap.add_argument("--top_n", type=int, default=25, help="Top-N features to show")
    ap.add_argument("--exclude", action="append", default=[], help="Feature names or substrings to exclude (repeatable)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agg_path = Path(args.agg_csv)
    group_path = Path(args.group_csv)

    p1 = plot_top_aggregated(agg_path, out_dir, args.top_n, args.exclude)
    outs = plot_top_by_group(group_path, out_dir, args.top_n, args.exclude)

    print("Saved:")
    print(p1)
    for p in outs:
        print(p)


if __name__ == "__main__":
    main()
