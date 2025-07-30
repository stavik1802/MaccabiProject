import os
import shutil
from pathlib import Path

# Paths
player_data_dir = Path("/home/stav.karasik/Maccabi/player_data")
games_output_dir = Path("/home/stav.karasik/MaccabiProject/scripts/games")

# Ensure output directory exists
games_output_dir.mkdir(parents=True, exist_ok=True)

# Loop through each player folder (e.g. AM_20, CB_3, etc.)
for player_folder in player_data_dir.iterdir():
    if player_folder.is_dir():
        player_code = player_folder.name  # e.g. AM_20

        # Loop through all CSV files for that player
        for csv_file in player_folder.glob("merged_features_*.csv"):
            # Extract game date from filename (after "merged_features_")
            filename_parts = csv_file.stem.split("_")
            game_date = filename_parts[-1]  # e.g. 2023-09-03

            # Create game folder if it doesn't exist
            game_folder = games_output_dir / game_date
            game_folder.mkdir(exist_ok=True)

            # Destination file name
            dest_file = game_folder / f"merged_features_{player_code}.csv"

            # Copy file
            shutil.copy2(csv_file, dest_file)
            print(f"✅ Copied {csv_file.name} → {dest_file}")
