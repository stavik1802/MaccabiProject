"""
Script: players_lists.py

Description:
    This script analyzes player participation data to identify which players played full games,
    were substituted, or did not play at all. It processes both GPS tracking data and
    substitution records to create a comprehensive view of player involvement in a match.
    Use it if you have full sofascore data excel file.

Key Features:
    - Identifies all players from GPS tracking data files
    - Determines players who played the full 90 minutes
    - Identifies players who did not participate
    - Matches substitution pairs (players subbed out and in)
    - Calculates substitution minutes based on playing time
    - Handles both direct folder and recursive file searches

Input:
    - Folder containing GPS tracking data files
    - Date of the match
    - List of top player IDs (typically starters)
    - Optional path to filtered substitutions CSV file

Output:
    Dictionary containing:
    - all_player_ids: List of all players with tracking data
    - full_game: Players who played the entire match
    - probable_subs: List of substitution pairs with timing
    - did_not_play: Players who didn't participate

Usage:
    analyze_subs(folder_path, match_date, top_player_ids, subs_csv_path="filtered_subs.csv")
    Example: analyze_subs(Path("./match_data"), "24.08.2024", ["AM_22", "CB_2"])
"""

import re
import pandas as pd
from pathlib import Path


def analyze_subs(folder: Path, date: str, top_player_ids: list[str], subs_csv_path: str = "filtered_subs.csv"):
    # 1. Get all player IDs from CSV filenames for the given date
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
    # 2. Load filtered_subs.csv
    subs_df = pd.read_csv(subs_csv_path)
    subs_df['Minutes played'] = subs_df['Minutes played'].str.replace("'", "").astype(int)
    subs_df = subs_df[subs_df['Date'] == date]

    # 3. Map Player_ID to Minutes played
    id_to_minutes = dict(zip(subs_df['Player_ID'], subs_df['Minutes played']))
    # 4. Full game players
    full_game = [pid for pid, mins in id_to_minutes.items() if mins == 90]

    # 5. Did not play
    did_not_play = list(all_player_ids - set(id_to_minutes.keys()))

    # 6. Probable subs (each player can only appear in one pair)
    probable_subs = []
    used_ids = set(full_game)  # full game players can't be subs
    # Create a list of candidates for subbed out and subbed in
    subbed_out_candidates = [pid for pid in id_to_minutes if pid in top_player_ids and id_to_minutes[pid] < 90 and id_to_minutes[pid] > 0]
    subbed_in_candidates = [pid for pid in id_to_minutes if pid not in top_player_ids and id_to_minutes[pid] < 90 and id_to_minutes[pid] > 0]
    # Sort by minutes played descending for subbed out, ascending for subbed in
    subbed_out_candidates.sort(key=lambda pid: -id_to_minutes[pid])
    subbed_in_candidates.sort(key=lambda pid: id_to_minutes[pid])
    for out_id in subbed_out_candidates:
        if id_to_minutes[out_id] == 90:
            used_ids.add(out_id)
            continue
        if out_id in used_ids:
            continue
        for in_id in subbed_in_candidates:
            if in_id in used_ids:
                continue
            if id_to_minutes[out_id] + id_to_minutes[in_id] == 90:
                sub_minute_out = id_to_minutes[out_id]
                sub_minute_in = 90 - id_to_minutes[in_id]
                probable_subs.append((out_id, in_id,sub_minute_out))
                used_ids.add(out_id)
                used_ids.add(in_id)
                break  # move to next out_idl

    # Filter out any players that are already in used_ids
    subbed_out_candidates = [pid for pid in subbed_out_candidates if pid not in used_ids]
    subbed_in_candidates = [pid for pid in subbed_in_candidates if pid not in used_ids]
    
    # Sort candidates by minutes played for new matching
    subbed_out_candidates.sort(key=lambda pid: id_to_minutes[pid])  # ascending order
    subbed_in_candidates.sort(key=lambda pid: -id_to_minutes[pid])  # descending order
    
    # Match pairs from highest in_candidates with lowest out_candidates
    for i in range(min(len(subbed_out_candidates), len(subbed_in_candidates))):
        out_id = subbed_out_candidates[i]
        in_id = subbed_in_candidates[i]
        sub_minute_out = id_to_minutes[out_id]
        probable_subs.append((out_id, in_id, sub_minute_out))
        used_ids.add(out_id)
        used_ids.add(in_id)

    return {
        "all_player_ids": list(all_player_ids),
        "full_game": full_game,
        "probable_subs": probable_subs,
        "did_not_play": did_not_play
    }
