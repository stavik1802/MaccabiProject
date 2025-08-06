"""
Script: pre_proccess.py

Description:
    This script performs comprehensive pre-processing of football match GPS tracking data.
    It orchestrates a complete data processing pipeline that transforms raw GPS data
    into analyzed, filtered, and structured match information.

Key Features:
    - Filters and processes Maccabi Haifa player data
    - Detects match periods (first half, halftime, second half)
    - Analyzes player substitutions and participation
    - Filters and splits data into first and second half segments
    - Infers field orientation and dimensions
    - Processes player metrics and performance data

Processing Pipeline:
    1. Maccabi Haifa player filtering
    2. GPS data processing for each player
    3. Match period detection
    4. Player substitution analysis
    5. Data filtering and segmentation
    6. Field orientation inference
    7. Player metrics calculation

Input:
    - Directory containing game folders with GPS tracking data
    - Each game folder should contain player CSV files with GPS coordinates

Output:
    - Processed and filtered player data
    - Match period information
    - Player substitution analysis
    - Field orientation data
    - Player performance metrics

Usage:
    python pre_proccess.py <directory_path>
    Example: python pre_proccess.py ./match_data
"""

from pathlib import Path
from filter_only_maccabi import process_sofa_score, filter_maccabi_subs_by_date
from update_data import process_match_basic_metrics
from find_playing_times import HalfTimeDetector
from find_ten_runners import top_runners
from players_lists import analyze_subs
from filter_data_all import main as filter_data_main
from infer_field import infer_field_transform_with_visualization
from minute_features import main as process_player_metrics
from analyze_sub_file import analyze_subs_file
import pandas as pd
from datetime import datetime
from extract_player_data import process_all_players
import json
import shutil
import calendar
import sys
sys.path.append('..')
from filter_subs_names import replace_players_in_folder

# Debug flags for controlling processing steps
DEBUG_1 = True  # Controls basic metrics processing
DEBUG = True    # Controls main processing pipeline
I = 1          # Counter for processing specific games

def parse_time(s):
    """
    Parse time string in HH:MM:SS.fff format to datetime.time object.
    
    This function handles time parsing for GPS data timestamps, ensuring
    proper formatting with microseconds for accurate time tracking.
    
    Args:
        s (str): Time string in HH:MM:SS or HH:MM:SS.fff format
        
    Returns:
        datetime.time: Parsed time object
        
    Raises:
        ValueError: If time format is invalid
        
    Example:
        >>> parse_time("18:58:00.123")
        datetime.time(18, 58, 0, 123000)
    """
    try:
        if '.' in s:
            secs, frac = s.strip().split('.')
            frac = (frac + '000000')[:6]  # Ensure 6-digit microseconds
            s = f"{secs}.{frac}"
        else:
            s = s.strip() + '.000000'  # Add microseconds if missing
        return datetime.strptime(s, "%H:%M:%S.%f").time()
    except ValueError:
        raise ValueError(f"Invalid time '{s}' – must be HH:MM:SS.fff (e.g., 18:58:00.0)")

def copy_lineup_files_to_games_data(lineups_extract_path: str, games_data_dir_path: str):
    """
    Copy substitutes CSV files from lineups_extract to matching game directories based on date.
    
    This function matches CSV files ending with '_substitutes.csv' in lineups_extract (format: dd-mm-yyyy_..._substitutes.csv) 
    with game directories (format: yyyy-mm-dd-Month dd, yyyy-RawDataExtended/) 
    and copies the files to the corresponding directories, renaming them to 'filtered_subs.csv'.
    
    Args:
        lineups_extract_path (str): Path to the lineups_extract directory
        games_data_dir_path (str): Path to the games data directory containing match directories
        
    Returns:
        None: Files are copied directly to target directories
        
    Example:
        >>> copy_lineup_files_to_games_data("./lineups_extract", "./run1")
        ✅ Copied 19-08-2023_M. Petach Tikva_M. Haifa_substitutes.csv to 2023-08-19-August 19, 2023-RawDataExtended/ as filtered_subs.csv
    """
    lineups_path = Path(lineups_extract_path)
    games_data_dir = Path(games_data_dir_path)
    
    if not lineups_path.exists():
        print(f"⚠️ Lineups extract path does not exist: {lineups_extract_path}")
        return
    
    if not games_data_dir.exists():
        print(f"⚠️ Games data directory does not exist: {games_data_dir_path}")
        return
    
    # Get only substitutes CSV files from lineups_extract
    csv_files = list(lineups_path.glob("*_substitutes.csv"))
    
    # Get all game directories from games data directory
    game_dirs = [d for d in games_data_dir.iterdir() if d.is_dir()]
    
    print(f"📁 Found {len(csv_files)} substitutes CSV files in lineups_extract")
    print(f"📁 Found {len(game_dirs)} game directories in games data directory")
    
    copied_count = 0
    
    for csv_file in csv_files:
        # Parse date from CSV filename (format: dd-mm-yyyy_...)
        try:
            date_part = csv_file.name.split('_')[0]  # Get "dd-mm-yyyy" part
            csv_date = datetime.strptime(date_part, "%d-%m-%Y")
            
            # Look for matching directory in games data directory
            target_dir = None
            for game_dir in game_dirs:
                # Parse date from directory name (format: yyyy-mm-dd-...)
                try:
                    dir_date_part = game_dir.name.split('-')[:3]  # Get ["yyyy", "mm", "dd"]
                    dir_date_str = "-".join(dir_date_part)  # "yyyy-mm-dd"
                    dir_date = datetime.strptime(dir_date_str, "%Y-%m-%d")
                    
                    if csv_date.date() == dir_date.date():
                        target_dir = game_dir
                        break
                except (ValueError, IndexError) as e:
                    continue  # Skip directories that don't match expected format
            
            if target_dir:
                # Copy the CSV file to the target directory with the new name 'filtered_subs.csv'
                target_file = target_dir / "filtered_subs.csv"
                shutil.copy2(csv_file, target_file)
                print(f"✅ Copied {csv_file.name} to {target_dir.name}/ as filtered_subs.csv")
                copied_count += 1
            else:
                print(f"⚠️ No matching directory found for {csv_file.name} (date: {csv_date.strftime('%Y-%m-%d')})")
                
        except (ValueError, IndexError) as e:
            print(f"❌ Error parsing date from {csv_file.name}: {e}")
    
    print(f"📊 Successfully copied {copied_count} out of {len(csv_files)} substitutes CSV files")

def pre_proccess(directory_path: str):
    """
    Pre-process all game data in the specified directory.
    
    This function orchestrates a complete data processing pipeline that transforms
    raw GPS tracking data into analyzed, filtered, and structured match information.
    The processing includes multiple stages of data transformation and analysis.
    
    Processing Stages:
        1. Maccabi Haifa Player Filtering: Identifies and filters data for Maccabi Haifa players
        2. GPS Data Processing: Processes raw GPS coordinates for each player
        3. Match Period Detection: Identifies first half, halftime, and second half periods
        4. Player Analysis: Analyzes substitutions and player participation
        5. Data Filtering: Filters and segments data into first and second half
        6. Field Inference: Determines field orientation and dimensions
        7. Metrics Processing: Calculates player performance metrics
    
    Args:
        directory_path (str): Path to the directory containing game folders.
                             Each folder should be named with a date format (YYYY-MM-DD)
                             and contain player GPS tracking data.
    
    Returns:
        None: All processed data is saved to the respective game folders
        
    Raises:
        Exception: If any stage of the processing pipeline fails
        
    Example:
        >>> pre_proccess("./match_data")
        ✅ Maccabi Haifa filtering completed successfully
        🔄 Processing GPS data for game: 2024-08-17-August 17
        ...
    """
    try:
        I = 1
        
        # Stage 0: Process lineups_extract directory
        print("🔄 Processing lineups_extract directory...")
        parent_path = Path(directory_path)
        lineups_extract_path = parent_path.parent / "lineups_extract"
        games_data_dir_path = parent_path.parent / "run1"
        
        # Replace player names in CSV files
        if lineups_extract_path.exists():
            print("📝 Replacing player names in lineups_extract...")
            replace_players_in_folder(str(lineups_extract_path))
            print("✅ Player name replacement completed")
            
            # Copy updated CSV files to matching game directories
            print("📋 Copying CSV files to matching game directories...")
            copy_lineup_files_to_games_data(str(lineups_extract_path), str(games_data_dir_path))
            print("✅ Substitutions CSV files copying completed")
        else:
            print(f"⚠️ Lineups extract directory not found: {lineups_extract_path}")
        
        # Stage 1: Maccabi Haifa player filtering
        print(directory_path)
        if (DEBUG):
            # process_sofa_score(directory_path)
            print("✅ Maccabi Haifa filtering completed successfully")

        # Stage 2: Process GPS data for each game folder
        parent_path = Path(directory_path)
        # Find game folders that start with a date (YYYY-MM-DD format)
        game_folders = [f for f in parent_path.iterdir() if f.is_dir() and f.name[:10].replace("-", "").isdigit()]
        
        for game_folder in game_folders:
            # Skip first 5 games (for testing/debugging purposes)
            # if (I ==1 or I == 2 or I == 3 or I == 4 or I==5):
            #     I = I + 1
            #     continue
            print(f"\n🔄 Processing GPS data for game: {game_folder.name}")
            
            # Stage 2a: Process basic metrics for all players
            if (DEBUG_1):
                # Create a processed_data directory in the game folder
                basic_metrics_dir = process_match_basic_metrics(game_folder)
            else:
                basic_metrics_dir = game_folder / "basic_metrics"
            
            # Stage 3: Detect match periods (first half, halftime, second half)
            if (DEBUG):
                print("🔍 Detecting match periods...")
                detector = HalfTimeDetector(str(basic_metrics_dir), {})
                first_half, second_half = detector.detect_match_periods()
                
                # Save match periods to a JSON file for reference
                periods_data = {
                    'first_half': {
                        'start': first_half.start.isoformat(),
                        'end': first_half.end.isoformat(),
                        'duration_minutes': (first_half.end - first_half.start).total_seconds() / 60
                    },
                    'halftime': {
                        'start': first_half.end.isoformat(),
                        'end': second_half.start.isoformat(),
                        'duration_minutes': (second_half.start - first_half.end).total_seconds() / 60
                    },
                    'second_half': {
                        'start': second_half.start.isoformat(),
                        'end': second_half.end.isoformat(),
                        'duration_minutes': (second_half.end - second_half.start).total_seconds() / 60
                    }
                }
                
                # Stage 4: Find top runners (most active players) for first half
                print("🏃 Finding top runners for first half...")
                first_half_runners = top_runners(basic_metrics_dir, first_half.start.strftime('%H:%M:%S.%f'))
                # Extract just the player IDs from the full filenames
                first_half_runners = [runner.split('-')[3] for runner in first_half_runners]
                
                # Stage 5: Analyze player substitutions and participation
                print("\n👥 Analyzing player participation and substitutions...")
                match_date = game_folder.name[:10]  # Get date from folder name
                # Convert date format from YYYY-MM-DD to DD.MM.YYYY for substitution analysis
                formatted_date = datetime.strptime(match_date, "%Y-%m-%d").strftime("%d.%m.%Y")
                subs_analysis = analyze_subs_file(
                    folder=basic_metrics_dir,
                    top_player_ids=first_half_runners,  # Use first half runners as top players
                    subs_csv_path=str(game_folder / "filtered_subs.csv")
                )
                
                # Print the analysis results for verification
                print("\n📊 Player Analysis Results:")
                print(f"All Players: {subs_analysis['all_player_ids']}")
                print(f"Full Game Players: {subs_analysis['full_game']}")
                print(f"Substitutions:")
                for out_id, in_id, minute in subs_analysis['probable_subs']:
                    print(f"  {out_id} → {in_id} at {minute}'")
                print(f"Did Not Play: {subs_analysis['did_not_play']}")

                # Stage 6: Filter and segment data into first and second half
                print("\n📊 Processing and filtering player data...")
                filtered_data_dir = game_folder / "filtered_data_halves"
                filtered_data_dir.mkdir(parents=True, exist_ok=True)
                
                filter_data_main(
                    str(basic_metrics_dir),
                    str(filtered_data_dir),
                    first_half.start.time(),
                    first_half.end.time(),
                    second_half.start.time(),
                    second_half.end.time(),
                    first_half_runners,
                    subs_analysis
                )
            
            # Stage 7: Infer field orientation and dimensions
            if (DEBUG):
                filtered_data_dir = game_folder / "filtered_data_halves"
                print("\n🏟️ Inferring field orientation and dimensions...")
                field_data = infer_field_transform_with_visualization(str(filtered_data_dir), show_plot=False)
                
                # Save field data to the game folder for future reference
                field_data_path = game_folder / "field_frame.json"
                with open(field_data_path, "w") as f:
                    json.dump(field_data, f, indent=2)
                print(f"✅ Saved field data to {field_data_path}")
            
            # Stage 8: Process player metrics and performance data
            print("\n📊 Processing player metrics...")
            try:
                # Find Center Forward (CF) and Center Back (CB) players for attack direction analysis
                cf_file: Path | None = None
                cb_file: Path | None = None
                for runner in first_half_runners:
                    if "CF" in runner:
                        cf_dir = filtered_data_dir / f"basic_metrics_{game_folder.name[:10]}-{runner}-Entire-Session"
                        cf_file = cf_dir / "first_half.csv"
                    elif "CB" in runner:
                        cb_dir = filtered_data_dir / f"basic_metrics_{game_folder.name[:10]}-{runner}-Entire-Session"
                        cb_file = cb_dir / "first_half.csv"
                if cf_file is None:
                     print("no cf file found, trying to find AM")
                     for runner in first_half_runners:
                        if "AM" in runner:
                            cf_dir = filtered_data_dir / f"basic_metrics_{game_folder.name[:10]}-{runner}-Entire-Session"
                            cf_file = cf_dir / "first_half.csv"
                print("this is the cf_file", cf_file)
                print("this is the cb_file", cb_file)
                # Process player metrics if both CF and CB files are available
                if cf_file and cf_file.exists() and cb_file and cb_file.exists():
                    process_player_metrics(
                        str(filtered_data_dir),
                        str(cf_file),
                        str(cb_file),
                        first_half.start.strftime('%H:%M:%S.%f'),
                        second_half.start.strftime('%H:%M:%S.%f')
                    )
                    print("✅ Player metrics processing completed")
                else:
                    print("⚠️ Could not find CF/CB files in first_half_runners")
                    if cf_file:
                        print(f"CF file exists: {cf_file.exists()}")
                    if cb_file:
                        print(f"CB file exists: {cb_file.exists()}")
            except Exception as e:
                print(f"❌ Error processing player metrics: {e}")
            
            # break  # Remove this if you want to process all games

        print("\n✅ All games processed successfully")
        process_all_players(directory_path)
        
    except Exception as e:
        print(f"❌ Error during pre-processing: {e}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python pre_proccess.py <directory_path>")
        print("Example: python pre_proccess.py ./match_data")
    else:
        pre_proccess(sys.argv[1])
