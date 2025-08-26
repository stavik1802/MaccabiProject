import os
import shutil
from pathlib import Path

# Paths
possession_dir = Path("/home/stav.karasik/MaccabiProject/utils/norm_possession_1min")
games_dir = Path("/home/stav.karasik/MaccabiProject/scripts/1min_windows")

# Loop over all possession CSV files
for file in possession_dir.glob("*.csv"):
    # extract date part from filename (e.g. "2024-01-17_bnei-sakhnin_vs_maccabi.csv" -> "2024-01-17")
    date_str = file.name.split("_")[0]
    game_folder = games_dir / date_str

    if game_folder.exists():
        dest_file = game_folder / "poss.csv"
        shutil.copy(file, dest_file)   # copy the file into the folder as poss.csv
        print(f"✅ Copied {file.name} → {dest_file}")
    else:
        print(f"⚠️ No matching folder for {file.name}")
