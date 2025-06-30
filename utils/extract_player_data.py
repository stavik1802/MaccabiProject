#!/usr/bin/env python3
"""
Player Data Extraction Script

This script extracts all merged_features.csv files for a specific player
across all games in the run1 directory and organizes them by player code.

Usage:
    python extract_player_data.py <player_code>
    python extract_player_data.py --all
    python extract_player_data.py --list-players

Example:
    python extract_player_data.py AM_20
    python extract_player_data.py --all
"""

import os
import sys
import shutil
import re
from pathlib import Path
from datetime import datetime


def extract_date_from_folder_name(folder_name):
    """
    Extract date from folder name like '2024-08-24-August 24, 2024-RawDataExtended'
    Returns date in YYYY-MM-DD format
    """
    # Extract the date part (YYYY-MM-DD) from the beginning of the folder name
    date_match = re.match(r'(\d{4}-\d{2}-\d{2})', folder_name)
    if date_match:
        return date_match.group(1)
    return None


def get_all_player_codes(run1_path):
    """
    Get all unique player codes from all games
    
    Args:
        run1_path (str): Path to the run1 directory
    
    Returns:
        set: Set of unique player codes
    """
    player_codes = set()
    
    # Iterate through all game directories
    for game_folder in os.listdir(run1_path):
        game_path = os.path.join(run1_path, game_folder)
        
        # Skip if not a directory
        if not os.path.isdir(game_path):
            continue
            
        # Extract date from folder name
        game_date = extract_date_from_folder_name(game_folder)
        if not game_date:
            continue
            
        # Look in filtered_data_halves directory
        filtered_data_path = os.path.join(game_path, "filtered_data_halves")
        if not os.path.exists(filtered_data_path):
            continue
            
        # Extract player codes from folder names
        for folder_name in os.listdir(filtered_data_path):
            # Pattern: basic_metrics_YYYY-MM-DD-PLAYER_CODE-Entire-Session
            pattern = rf'basic_metrics_{game_date}-([A-Z]+_\d+)-Entire-Session'
            match = re.match(pattern, folder_name)
            if match:
                player_code = match.group(1)
                player_codes.add(player_code)
    
    return sorted(list(player_codes))


def find_player_merged_features(run1_path, player_code):
    """
    Find all merged_features.csv files for a specific player across all games
    
    Args:
        run1_path (str): Path to the run1 directory
        player_code (str): Player code to search for (e.g., 'AM_20')
    
    Returns:
        list: List of tuples (game_date, source_file_path)
    """
    player_files = []
    
    # Iterate through all game directories
    for game_folder in os.listdir(run1_path):
        game_path = os.path.join(run1_path, game_folder)
        
        # Skip if not a directory
        if not os.path.isdir(game_path):
            continue
            
        # Extract date from folder name
        game_date = extract_date_from_folder_name(game_folder)
        if not game_date:
            print(f"Warning: Could not extract date from folder {game_folder}")
            continue
            
        # Look for the player's merged features file
        player_folder_name = f"basic_metrics_{game_date}-{player_code}-Entire-Session"
        player_folder_path = os.path.join(game_path, "filtered_data_halves", player_folder_name)
        
        # Check for merged_features.csv
        merged_features_path = os.path.join(player_folder_path, "merged_features.csv")
        if os.path.exists(merged_features_path):
            player_files.append((game_date, merged_features_path))
            print(f"Found merged features data for {player_code} on {game_date}")
        else:
            print(f"No merged features data found for {player_code} on {game_date}")
    
    return player_files


def copy_player_files(player_files, player_code, output_base_dir="player_data"):
    """
    Copy player files to a folder named after the player code
    
    Args:
        player_files (list): List of tuples (game_date, source_file_path)
        player_code (str): Player code
        output_base_dir (str): Base directory for output
    """
    # Create output directory
    player_output_dir = os.path.join(output_base_dir, player_code)
    os.makedirs(player_output_dir, exist_ok=True)
    
    print(f"\nCopying files to: {player_output_dir}")
    
    for game_date, source_file in player_files:
        # Create new filename with date
        new_filename = f"merged_features_{game_date}.csv"
        destination_file = os.path.join(player_output_dir, new_filename)
        
        try:
            shutil.copy2(source_file, destination_file)
            print(f"✓ Copied: {new_filename}")
        except Exception as e:
            print(f"✗ Error copying {source_file}: {e}")


def process_single_player(player_code, run1_path):
    """Process a single player"""
    print(f"\n{'='*60}")
    print(f"Processing player: {player_code}")
    print(f"{'='*60}")
    
    # Find all player files
    player_files = find_player_merged_features(run1_path, player_code)
    
    if not player_files:
        print(f"No data found for player {player_code}")
        return False
    
    print(f"\nFound {len(player_files)} merged features files for {player_code}")
    
    # Copy files to player directory
    copy_player_files(player_files, player_code)
    
    print(f"✓ Successfully extracted {len(player_files)} files for player {player_code}")
    return True


def process_all_players(run1_path):
    """Process all players found in the data"""
    print("Extracting all player codes from data...")
    player_codes = get_all_player_codes(run1_path)
    
    if not player_codes:
        print("No player codes found in the data!")
        return
    
    print(f"Found {len(player_codes)} unique player codes:")
    for code in player_codes:
        print(f"  - {code}")
    
    print(f"\nProcessing all {len(player_codes)} players...")
    
    successful_players = 0
    for player_code in player_codes:
        if process_single_player(player_code, run1_path):
            successful_players += 1
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: Successfully processed {successful_players}/{len(player_codes)} players")
    print(f"All data saved in: player_data/")
    print(f"{'='*60}")


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage:")
        print("  python extract_player_data.py <player_code>")
        print("  python extract_player_data.py --all")
        print("  python extract_player_data.py --list-players")
        print("\nExamples:")
        print("  python extract_player_data.py AM_20")
        print("  python extract_player_data.py --all")
        print("  python extract_player_data.py --list-players")
        sys.exit(1)
    
    run1_path = "run1"
    
    # Check if run1 directory exists
    if not os.path.exists(run1_path):
        print(f"Error: Directory '{run1_path}' not found!")
        sys.exit(1)
    
    if sys.argv[1] == "--list-players":
        print("Extracting all player codes from data...")
        player_codes = get_all_player_codes(run1_path)
        
        if not player_codes:
            print("No player codes found in the data!")
        else:
            print(f"Found {len(player_codes)} unique player codes:")
            for code in player_codes:
                print(f"  - {code}")
    
    elif sys.argv[1] == "--all":
        process_all_players(run1_path)
    
    else:
        player_code = sys.argv[1]
        process_single_player(player_code, run1_path)


if __name__ == "__main__":
    main() 