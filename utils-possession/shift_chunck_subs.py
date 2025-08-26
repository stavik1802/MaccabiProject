# shifts players features into independent chunks for model of possession prediction

from re import S
import pandas as pd
import os
from pathlib import Path

# === Setup ===
source_root = Path("/home/stav.karasik/MaccabiProject/scripts/games")
destination_root = source_root.parent / "games_shifted_chunks"
destination_root.mkdir(exist_ok=True)

# === Process each game folder ===
for game_folder in sorted(source_root.iterdir()):
    if not game_folder.is_dir():
        continue

    game_name = game_folder.name
    print(f"🔄 Processing game: {game_name}")
    new_game_path = destination_root / game_name
    new_game_path.mkdir(exist_ok=True)

    # --- Load substitutions ---
    subs_path = game_folder / "subs.csv"
    if not subs_path.exists():
        print(f"  ⚠️ No subs.csv found in {game_name}, skipping...")
        continue

    df_subs = pd.read_csv(subs_path)

    # Clean and parse substitution info
    sub_map = {}
    for _, row in df_subs.iterrows():
        in_player = row["In Player"]
        out_player = row["Out Player"]
        minute_str = str(row["Minute"])
        minute = int(''.join(filter(str.isdigit, minute_str)))  # Clean "89'" → 89
        fallback_chunk = (minute * 60) // 15
        sub_map[in_player] = {"after": out_player, "fallback_chunk": fallback_chunk}

    # --- Cache end chunk of subbed-out players ---
    out_end_chunks = {}
    for file in game_folder.glob("merged_features_*.csv"):
        player = file.stem.replace("merged_features_", "")
        df = pd.read_csv(file)
        if "chunk" in df.columns and not df.empty:
            out_end_chunks[player] = df["chunk"].max()

    # --- Shift and copy each player file ---
    for file in game_folder.glob("merged_features_*.csv"):
        player = file.stem.replace("merged_features_", "")
        df = pd.read_csv(file)

        if "chunk" not in df.columns or df.empty:
            df.to_csv(new_game_path / file.name, index=False)
            continue
        if player in sub_map:
            out_player = sub_map[player]["after"]
            if out_player in out_end_chunks:
                shift = out_end_chunks[out_player] + 1
            else:
                shift = sub_map[player]["fallback_chunk"]
            df["chunk"] = range(shift, shift + len(df))  # Overwrite original column
        else:
            df["chunk"] = df["chunk"]  # No change
        df.to_csv(new_game_path / file.name, index=False)

    # --- Copy subs.csv as-is ---
    (new_game_path / "subs.csv").write_bytes(subs_path.read_bytes())

print("✅ Done. All shifted games saved in:", destination_root)
