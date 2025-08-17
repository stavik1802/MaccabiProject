# makes pertubated data for training the physical features prediction model

import os
import pandas as pd
import numpy as np

# === Input folder containing all player folders ===
root_dir = "games_train"  # change if your root folder has a different name

# === Go through each player folder ===
for player_code in os.listdir(root_dir):
    player_folder = os.path.join(root_dir, player_code)
    if not os.path.isdir(player_folder):
        continue

    # Iterate over all CSV files in the player's folder
    for file_name in os.listdir(player_folder):
        if not file_name.endswith(".csv"):
            continue

        file_path = os.path.join(player_folder, file_name)

        # Load CSV
        df = pd.read_csv(file_path)

        # Perturb numeric columns
        perturbed_df = df.copy()
        numeric_cols = perturbed_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            perturb_factors = np.random.uniform(0.9, 1.1, size=len(perturbed_df))
            perturbed_df[col] = perturbed_df[col] * perturb_factors

        # Create new filename with '_perturbed' before '.csv'
        name, ext = os.path.splitext(file_name)
        new_filename = f"{name}_perturbed{ext}"
        new_path = os.path.join(player_folder, new_filename)

        # Save perturbed file
        perturbed_df.to_csv(new_path, index=False)
        print(f"✅ Saved perturbed file to: {new_path}")
