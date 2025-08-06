import pandas as pd
from typing import Dict, Union
from pathlib import Path

def rename_features_names(csv_file: str, rename_map: Dict[str, str]) -> pd.DataFrame:
    """
    Reads a CSV file, renames its columns based on the provided mapping dictionary, and returns the updated DataFrame.
    Args:
        csv_file (str): Path to the input CSV file.
        rename_map (Dict[str, str]): Dictionary mapping old column names to new column names.
    Returns:
        pd.DataFrame: DataFrame with renamed columns.
    """
    df = pd.read_csv(csv_file)
    df = df.rename(columns=rename_map)
    return df

def rename_game_feature_columns(player_folder: Union[str, Path], rename_map: Dict[str, str]):
    """
    Renames columns in all three feature files in a player folder using the provided mapping.
    This function is intended to be used on a player's directory inside filtered_data_halves.
    Args:
        player_folder (str or Path): Path to the player folder containing the feature files.
        rename_map (Dict[str, str]): Dictionary mapping old column names to new column names.
    """
    files = [
        "first_half_features.csv",
        "second_half_features.csv",
        "merged_features.csv"
    ]
    player_folder = Path(player_folder)
    for file in files:
        file_path = player_folder / file
        if file_path.exists():
            df = rename_features_names(str(file_path), rename_map)
            df.to_csv(file_path, index=False)
