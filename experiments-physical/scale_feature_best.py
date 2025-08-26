#!/usr/bin/env python3
# Rank features by prediction quality at 70' with scale differences handled via NMAE.
# Input CSV must have: feature, mae_at_70, nmae_at_70 (and optionally scale_used)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IN_CSV  = "experiments_feature_mae/feature_mae_at_70_normalized.csv"
OUT_DIR = "experiments_feature_mae"
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    df = pd.read_csv(IN_CSV)
    req = {"feature","mae_at_70","nmae_at_70"}
    if not req.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {req}, got {list(df.columns)}")

    # Ranks (lower is better). Use 'dense' so ties share the same rank number.
    df["rank_nmae"]    = df["nmae_at_70"].rank(method="dense", ascending=True).astype(int)
    df["rank_abs_mae"] = df["mae_at_70"].rank(method="dense", ascending=True).astype(int)

    # Optional combined rank (mostly NMAE, a little absolute MAE for tie-breaking)
    df["rank_combined"] = (0.8 * df["rank_nmae"] + 0.2 * df["rank_abs_mae"])

    # Sort by normalized rank as the main answer
    df_sorted = df.sort_values(["rank_nmae","nmae_at_70","mae_at_70"]).reset_index(drop=True)

    # Save ranked table
    out_csv = os.path.join(OUT_DIR, "feature_rankings_at_70.csv")
    cols = ["feature","mae_at_70","nmae_at_70","rank_nmae","rank_abs_mae","rank_combined"]
    df_sorted[cols].to_csv(out_csv, index=False)
    print(f"✅ Saved rankings: {out_csv}")

    # Bar chart ordered by NMAE (scale-aware). Lower is better.
    plt.figure(figsize=(12, 7))
    plt.barh(df_sorted["feature"], df_sorted["nmae_at_70"])
    plt.gca().invert_yaxis()
    plt.xlabel("Normalized MAE at 70:00 (by chosen scale, lower is better)")
    plt.title("Feature Ranking by Normalized Error @ 70’")
    # annotate with rank and value
    for i, (feat, n) in enumerate(zip(df_sorted["feature"], df_sorted["nmae_at_70"])):
        rank = df_sorted.loc[i, "rank_nmae"]
        plt.text(n, i, f"  #{rank}  ({n:.3f})", va="center")
    out_png = os.path.join(OUT_DIR, "feature_ranking_nmae_bar.png")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
    print(f"📈 Saved plot: {out_png}")

if __name__ == "__main__":
    main()
