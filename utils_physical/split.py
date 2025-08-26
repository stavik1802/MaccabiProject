"""Split per-game CSVs into train and eval sets with an 80/20 split per game."""
import os
import shutil
import random
# split the games into train and eval sets each game contains 80% of the players in the train set and 20% in the eval set

source_root = "games"               # your folder with game subfolders
train_root = "games_train_new"
eval_root = "games_eval_new"
train_ratio = 0.8

os.makedirs(train_root, exist_ok=True)
os.makedirs(eval_root, exist_ok=True)

for game_folder in os.listdir(source_root):
    game_path = os.path.join(source_root, game_folder)
    if not os.path.isdir(game_path):
        continue

    files = [f for f in os.listdir(game_path) if f.endswith(".csv")]
    random.shuffle(files)
    split_idx = int(len(files) * train_ratio)

    train_files = files[:split_idx]
    eval_files = files[split_idx:]

    os.makedirs(os.path.join(train_root, game_folder), exist_ok=True)
    os.makedirs(os.path.join(eval_root, game_folder), exist_ok=True)

    for f in train_files:
        shutil.copy(os.path.join(game_path, f), os.path.join(train_root, game_folder, f))
    for f in eval_files:
        shutil.copy(os.path.join(game_path, f), os.path.join(eval_root, game_folder, f))

print("✅ Done: Split into 'games_train' and 'games_eval'.")
