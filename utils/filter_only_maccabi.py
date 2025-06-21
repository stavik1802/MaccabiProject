#!/usr/bin/env python3
"""
Script: filter_only_maccabi.py
use this script if you have excel file with all the data from sofa score make sure you change to needed team
 ,the date is in the format DD.MM.YYYY , you change to the needed years and have the players played in the needed year.

Description:
    This script processes Maccabi Haifa's player data from the Israeli Premier League dataset.
    It filters and transforms player information, including handling date formatting and
    mapping player IDs to their position-based identifiers.

Key Features:
    - Filters data for Maccabi Haifa team only
    - Converts dates to a standardized format (DD.MM.YYYY)
    - Maps player IDs to position-based identifiers (e.g., AM_22, CB_2)
    - Excludes specific players (e.g., Sharif Kaiuf)
    - Handles special date formatting cases for the 2024-2025 season
    - Can process multiple game folders in batch

Input:
    - IsraeliPremierLeague_24-25.xlsx: Excel file containing league data
    - Date parameter for filtering specific matches
    - Optional: Parent folder containing game folders (yyyy-mm-dd format)

Output:
    - filtered_subs.csv: CSV file containing filtered player data with:
        - Player names
        - Position-based player IDs
        - Minutes played
        - Formatted dates

Usage:
    Single game: python filter_only_maccabi.py <date>
    Multiple games: python filter_only_maccabi.py --batch <parent_folder>
    where date is in the format DD.MM.YYYY
"""

import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime

def convert_folder_date_to_display(folder_name: str) -> str:
    """Convert folder date from yyyy-mm-dd to dd.mm.yyyy format."""
    try:
        date_obj = datetime.strptime(folder_name[:10], "%Y-%m-%d")
        return date_obj.strftime("%d.%m.%Y")
    except ValueError:
        raise ValueError(f"Invalid folder name format: {folder_name}. Expected format: yyyy-mm-dd-*")

def process_sofa_score(parent_folder: str):
    """
    Process all game folders in the parent directory.
    Each folder should be named in the format yyyy-mm-dd-*
    """
    parent_path = Path(parent_folder)
    if not parent_path.exists():
        raise ValueError(f"Parent folder not found: {parent_folder}")

    # Get all subfolders that match the date format
    game_folders = [f for f in parent_path.iterdir() if f.is_dir() and f.name[:10].replace("-", "").isdigit()]
    
    if not game_folders:
        print("❌ No valid game folders found.")
        return

    print(f"📂 Found {len(game_folders)} game folders to process")
    
    for game_folder in game_folders:
        try:
            # Get the date from folder name (first 10 characters)
            folder_date = game_folder.name[:10]
            display_date = convert_folder_date_to_display(game_folder.name)
            print(f"\n🔄 Processing game from {display_date}")
            
            # Save the filtered data in the game's own folder
            output_file = game_folder / f"filtered_subs.csv"
            
            # Process the game and save to its folder
            filter_maccabi_subs_by_date(display_date, output_file)
            
        except Exception as e:
            print(f"❌ Error processing {game_folder.name}: {e}")

def filter_maccabi_subs_by_date(dat, output_path="filtered_subs.csv"):
    def fix_date(s):
        s = str(s)
        if len(s) == 6:
            days = s[:1]
            month = s[1:3]
            year_prefix = s[3:]
            day = "0" + days
            if int(month) >= 7:
                full_year = '2024'
            else:
                full_year = '2025'

            return f"{day}.{month}.{full_year}"

        if len(s) != 7 or not s.endswith("202"):
            return s  # Skip invalid format
        day = s[:2]
        month = s[2:4]
        year_prefix = s[4:]

        if int(month) >= 7:
            full_year = '2024'
        else:
            full_year = '2025'

        return f"{day}.{month}.{full_year}"

    # --- Input and output filenames ---
    input_excel = "IsraeliPremierLeague_24-25.xlsx"
    team_name = "Maccabi Haifa"

    # --- Load Excel ---
    df = pd.read_excel(input_excel)

    # --- Filter rows ---
    df = df[df["Team"] == team_name]
    df = df[["Player","Player_ID", "Minutes played", "Date"]]
    df['Date'] = df['Date'].apply(fix_date)
    df = df[df['Player'] != 'Sharif Kaiuf']

    id_to_decrypt_name = {
        18911: "AM_22", # rephaelov 
        609926: "AM_21", #Sabia 
        186951: "CB_2", #goldberg 
        583896: "CB_3", #seck 
        976367: "CB_5", #syrota 
        305964: "CF_34", #david 
        943278: "CF_37", #pierrot 
        856468: "DM_15", #mohammad 
        1036096: "DM_16", #azulay, 
        594972: "LM_28", #Kenny Saief, 
        1481232: "RW_25", #Khalaili 
        1145681: "RB_8", #Feingold 
        808835: "LW_29", #HAZIZA 
        174557: "AM_20", #KINDA 
        905041: "CM_18",#JABBER 
        1167390: "DM_17", #HERMESH 
        787811: "LB_12", #NSIMBA 
        1804784: "CF_35", #DAHAN 
        1063936: "CF_38", # SHURANOV 
        931811: "CB_6" , #PEDRAO 
        375638: "LW_30", #NAHUEL 
        807071: "RB_11", #KANDIL
        1110159: "RW_24", #SAVERINA
        1435405: "RM_23", # HAGAG 
        1010066: "RB_10" #ELIMELECH
    }
    df['Player_ID'] = df['Player_ID'].map(id_to_decrypt_name)
    df = df[df["Date"] == dat]
    
    # --- Save to CSV ---
    df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(df)} filtered rows to '{output_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter Maccabi Haifa player data")
    parser.add_argument("--batch", type=str, help="Process all games in the specified folder")
    parser.add_argument("date", nargs="?", type=str, help="Date in DD.MM.YYYY format for single game processing")
    
    args = parser.parse_args()
    
    if args.batch:
        process_sofa_score(args.batch)
    elif args.date:
        filter_maccabi_subs_by_date(args.date)
    else:
        parser.print_help()


