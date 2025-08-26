# this script changes the data to be by date instead of by player

import os
import shutil
import re
from pathlib import Path

# === Config ===
player_data_root = Path("/home/stav.karasik/Maccabi/player_data")
output_root = Path("games")
output_root.mkdir(exist_ok=True)

# === Pattern for date extraction
date_pattern = re.compile(r'merged_features_(\d{4}-\d{2}-\d{2})\.csv')

# === Iterate over player folders
for player_folder in player_data_root.iterdir():
    if not player_folder.is_dir():
        continue
    player_code = player_folder.name
    for file in player_folder.glob("merged_features_*.csv"):
        match = date_pattern.match(file.name)
        if not match:
            continue
        game_date = match.group(1)
        output_dir = output_root / game_date
        output_dir.mkdir(parents=True, exist_ok=True)
        dest_file = output_dir / f"merged_features_{player_code}.csv"
        shutil.copy(file, dest_file)
        print(f"✅ Copied {file.name} to {dest_file}")

print("✅ All player files have been reorganized by game date.")
