import os
import pandas as pd
from pathlib import Path

# Path to your folder with original CSV files
input_folder = Path("games_possession_1min")
# Path to the output folder for updated files
output_folder = Path("norm_possession_1min")
output_folder.mkdir(parents=True, exist_ok=True)

# Column to update
target_column = "maccabi_haifa_possession_percent"

# Process each CSV file
for file in input_folder.glob("*.csv"):
    df = pd.read_csv(file)

    if target_column in df.columns:
        df[target_column] = df[target_column].apply(
            lambda x: x/100 if (isinstance(x, (int, float))) else 0.5
        )

    # Save updated file
    output_file = output_folder / file.name
    df.to_csv(output_file, index=False)

print("Done updating possession column in all files.")
