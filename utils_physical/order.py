"""Reorder per-player feature files by half and standardized date pattern under player_data/."""
import os
import re
from pathlib import Path

root_folder = Path("player_data")
pattern = re.compile(r"(first|second)_half_features_(\d{4}-\d{2}-\d{2})\.csv")

# Traverse each player folder
for player_folder in root_folder.iterdir():
    if player_folder.is_dir():
        for file in player_folder.iterdir():
            match = pattern.match(file.name)
            if match:
                half, date = match.groups()
                new_name = f"{date}_{half}.csv"
                new_path = player_folder / new_name
                print(f"Renaming: {file} → {new_path}")
                file.rename(new_path)
