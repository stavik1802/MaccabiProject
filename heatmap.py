import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                 # head‑less backend
import matplotlib.pyplot as plt

# ───────────────── 1. LOAD DATA ───────────────────────────────────────────
csv_path = Path("2024-08-17-xxlyAv/2024-08-17-xxIyAv-Entire-Session_second_half.csv")
df = pd.read_csv(csv_path)


# ───────────────── 2. CLEAN BAD GPS ROWS ──────────────────────────────────
df = df[(df['Lat'].abs() > 1) & (df['Lon'].abs() > 1)].reset_index(drop=True)
# ───────────────── 3. FIX UNITS IF NEEDED (µ° → °) ───────────────────────
if df['Lat'].abs().max() > 360:          # micro‑degrees?
    df[['Lat', 'Lon']] *= 1e-6

# ───────────────── 4. PROJECT TO METRES ───────────────────────────────────
def latlon_to_xy(lat, lon, lat0, lon0):
    R = 6_371_000          # Earth radius (m)
    x = math.radians(lon - lon0) * R * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * R
    return x, y

#lat0, lon0 = df.iloc[0][['Lat','Lon']]
lat0 ,lon0 = 32.7831263333333, 34.9653739999999
df[['x_m', 'y_m']] = df.apply(
    lambda r: latlon_to_xy(r['Lat'], r['Lon'], lat0, lon0),
    axis=1, result_type='expand'
)

# ───────────────── 5. BUILD HEAT‑MAP ──────────────────────────────────────
bins = 100
heat, xedges, yedges = np.histogram2d(df['x_m'], df['y_m'], bins=bins)

# ───────────────── 6. PLOT WITH WHITE TEXT ────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")   # white canvas
im = ax.imshow(
    heat.T,
    origin="lower",
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
    cmap="inferno"
)

ax.set_title("Player Position Heat‑map", color="black", pad=12)
ax.set_xlabel("X position (m)",  color="black")
ax.set_ylabel("Y position (m)",  color="black")
ax.tick_params(axis="both", colors="black")                 # black tick labels
for spine in ax.spines.values():                            # black border
    spine.set_color("black")

# colour‑bar in black
cbar = fig.colorbar(im, ax=ax, label="sample count")
cbar.ax.yaxis.label.set_color("black")
cbar.ax.tick_params(colors="black")

ax.set_aspect("equal")
fig.tight_layout()
fig.savefig("heatmapSecondHalf.png", dpi=150, facecolor="white")      # keep white bg

