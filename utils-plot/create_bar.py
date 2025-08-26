#!/usr/bin/env python3
# Make a bar plot of MAE@70 vs input window from the existing CSV.

import os
import pandas as pd
import matplotlib.pyplot as plt

IN_CSV  = "experiments_window/window_mae_at_70.csv"
OUT_PNG = "experiments_window/window_mae_at_70_bar.png"

def main():
    if not os.path.isfile(IN_CSV):
        raise FileNotFoundError(f"CSV not found: {IN_CSV}")

    df = pd.read_csv(IN_CSV)

    # Basic checks + sorting by window size
    if not {"window", "mae_at_70"}.issubset(df.columns):
        raise ValueError("CSV must contain 'window' and 'mae_at_70' columns.")

    df = df.sort_values("window").reset_index(drop=True)

    # Dynamic figure width based on number of bars
    n = len(df)
    fig_w = max(6, min(18, 0.4 * n + 6))
    plt.figure(figsize=(fig_w, 5))

    # Use strings for x-ticks to avoid awkward spacing
    x_labels = df["window"].astype(str)
    bars = plt.bar(x_labels, df["mae_at_70"])

    plt.xlabel("Input window size (chunks)")
    plt.ylabel("MAE at 70:00")
    plt.title("MAE at 70th Minute vs. Input Window Size (Bar Plot)")
    plt.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate each bar with its value
    for rect, val in zip(bars, df["mae_at_70"]):
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width()/2, height, f"{val:.4f}",
                 ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()
    print(f"✅ Saved bar plot to: {OUT_PNG}")

if __name__ == "__main__":
    main()
