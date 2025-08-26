"""Create per-game folders and copy raw files into a standardized games tree."""
import os
import shutil
from datetime import datetime

RAW_BASE = "/home/stav.karasik/Maccabi/23-24-run"
DEST_BASE = "/home/stav.karasik/MaccabiProject/scripts/games23"

os.makedirs(DEST_BASE, exist_ok=True)

for game_folder in os.listdir(RAW_BASE):
    game_path = os.path.join(RAW_BASE, game_folder)
    if not os.path.isdir(game_path):
        continue

    # ✅ Extract date from first 10 chars (YYYY-MM-DD)
    raw_date = game_folder[:10]
    try:
        dt = datetime.strptime(raw_date, "%Y-%m-%d")
        game_date = dt.strftime("%d-%m-%Y")
    except ValueError:
        print(f"❌ Skipping folder (no valid date): {game_folder}")
        continue

    halves_path = os.path.join(game_path, "filtered_data_halves")
    if not os.path.exists(halves_path):
        continue

    dest_game_folder = os.path.join(DEST_BASE, game_date)
    os.makedirs(dest_game_folder, exist_ok=True)

    # ✅ Copy subs file and rename
    subs_src = os.path.join(game_path, "filtered_subs.csv")
    if os.path.exists(subs_src):
        shutil.copy(subs_src, os.path.join(dest_game_folder, "subs.csv"))

    # ✅ Loop through players
    for player_folder in os.listdir(halves_path):
        player_path = os.path.join(halves_path, player_folder)
        if not os.path.isdir(player_path):
            continue

        merged_features_path = os.path.join(player_path, "merged_features.csv")
        if os.path.exists(merged_features_path):

            # ✅ Extract the player code (e.g. AM_22) from the folder name
            parts = player_folder.split("-")
            player_code = None
            for part in parts:
                if "_" in part and part[:2].isalpha():  # looks like AM_22, CB_5 etc
                    player_code = part
                    break

            if not player_code:  
                player_code = player_folder  # fallback if pattern fails

            # ✅ Copy file with correct name
            dest_file = os.path.join(dest_game_folder, f"merged_features_{player_code}.csv")
            shutil.copy(merged_features_path, dest_file)

            print(f"📂 {game_date} → Copied merged_features_{player_code}.csv")

print("✅ Done! All players processed correctly.")
