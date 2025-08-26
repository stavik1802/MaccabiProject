#!/usr/bin/env python3
"""
Simple script to list all player codes found in the run1 directory
"""

import os
import re


def extract_date_from_folder_name(folder_name):
    """Extract date from folder name"""
    date_match = re.match(r'(\d{4}-\d{2}-\d{2})', folder_name)
    if date_match:
        return date_match.group(1)
    return None


def get_all_player_codes(run1_path):
    """Get all unique player codes from all games"""
    player_codes = set()
    
    for game_folder in os.listdir(run1_path):
        game_path = os.path.join(run1_path, game_folder)
        
        if not os.path.isdir(game_path):
            continue
            
        game_date = extract_date_from_folder_name(game_folder)
        if not game_date:
            continue
            
        filtered_data_path = os.path.join(game_path, "filtered_data_halves")
        if not os.path.exists(filtered_data_path):
            continue
            
        for folder_name in os.listdir(filtered_data_path):
            pattern = rf'basic_metrics_{game_date}-([A-Z]+_\d+)-Entire-Session'
            match = re.match(pattern, folder_name)
            if match:
                player_code = match.group(1)
                player_codes.add(player_code)
    
    return sorted(list(player_codes))


def main():
    run1_path = "run1"
    
    if not os.path.exists(run1_path):
        print(f"Error: Directory '{run1_path}' not found!")
        return
    
    print("Extracting all player codes from data...")
    player_codes = get_all_player_codes(run1_path)
    
    if not player_codes:
        print("No player codes found in the data!")
    else:
        print(f"Found {len(player_codes)} unique player codes:")
        print("-" * 40)
        for i, code in enumerate(player_codes, 1):
            print(f"{i:2d}. {code}")
        print("-" * 40)
        print(f"Total: {len(player_codes)} players")


if __name__ == "__main__":
    main() 