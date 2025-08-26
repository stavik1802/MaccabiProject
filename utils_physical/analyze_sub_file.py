import re
import pandas as pd
from pathlib import Path

def analyze_subs_file(folder: Path,  top_player_ids: list[str], subs_csv_path: str = "filtered_subs.csv"):
    """
    use this function if you do subs file with sofa hunt
    Analyze substitution data to create player lists for a match.
    
    Args:
        subs_file_path: Path to the specific substitution CSV file
        top_player_ids: List of 10 starting players
    
    Returns:
        Dictionary containing:
        - full_game: Players who played the entire match
        - probable_subs: List of substitution pairs with timing
        - subbed_in_only: Players who only came in as substitutes
    """
    all_player_ids = set()
    for csv_file in folder.glob("*.csv"):
        m = re.match(r"basic_metrics_(\d{4}-\d{2}-\d{2})-([A-Z]+_\d+)-", csv_file.stem)
        if m :
            all_player_ids.add(m.group(2))
    # If nothing found, try rglob
    if not all_player_ids:
        for csv_file in folder.rglob("*.csv"):
            m = re.match(r"basic_metrics_(\d{4}-\d{2}-\d{2})-([A-Z]+_\d+)-", csv_file.stem)
            if m :
                all_player_ids.add(m.group(2))
    # 1. Load substitution data
    subs_df = pd.read_csv(subs_csv_path)
    
    # 2. Process substitutions
    subbed_out_players = set()
    subbed_in_players = set()
    probable_subs = []
    
    for _, row in subs_df.iterrows():
        out_player = row['Out Player']
        in_player = row['In Player']
        minute = row['Minute']
        
        subbed_out_players.add(out_player)
        subbed_in_players.add(in_player)
        probable_subs.append((out_player, in_player, minute))
    
    # 3. Determine full game players (starters who weren't subbed out)
    full_game = [pid for pid in top_player_ids if pid not in subbed_out_players]
    
    did_not_play = [pid for pid in all_player_ids if pid not in top_player_ids and pid not in subbed_in_players]
    
    return {
        "all_player_ids": list(all_player_ids),
        "full_game": full_game,
        "probable_subs": probable_subs,
        "did_not_play": did_not_play
    }

def print_player_lists(result):
    """Print the player lists in a formatted way"""
    print("=== PLAYER LISTS ===")
    print(f"Full Game Players ({len(result['full_game'])}): {result['full_game']}")
    print(f"Substitutions ({len(result['probable_subs'])}):")
    for out_player, in_player, minute in result['probable_subs']:
        print(f"  {minute}' - {out_player} → {in_player}")
    print(f"Subbed In Only ({len(result['subbed_in_only'])}): {result['subbed_in_only']}")

# Example usage
if __name__ == "__main__":
    # Example: Analyze a specific match
    subs_file = "sub/01-02-2025_H. Hadera_M. Haifa_substitutes.csv"
    top_player_ids = ["AM_20", "AM_21", "AM_22", "CB_5", "CB_6", "CF_35", "CF_38", "CM_18", "DM_15", "DM_16"]
    
    result = analyze_subs_file(subs_file, top_player_ids)
    print_player_lists(result)