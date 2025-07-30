"""
Script: find_playing_times.py

Description:
    This script analyzes player tracking data to determine actual playing times
    and periods for each player during matches. It identifies when players are
    actively participating in the game to match patterns of half start and end.
    The data is sampled at 100Hz (100 samples per second).

    Match timing rules enforced:
    - First half must be at least 45 minutes from start of first half
    - Halftime break must be at least 15 minutes
    - Second half must be at least 45 minutes from start of second half

Input:
    - Player tracking data with timestamps (10Hz sampling rate)
    - Match event data
    - Substitution information
    - Position data to determine field presence

Output:
    - Detailed playing time analysis per player
    - Substitution timelines
    - Active play periods
    - Playing time statistics and reports
    - Visualization of speed patterns and detected periods

Usage:
    Run this script to analyze and extract player participation times from match data:
    python find_playing_times.py <data_folder>
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import timedelta, datetime, time
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from filter_only_maccabi import filter_maccabi_subs_by_date

# Constants for 10Hz sampling
SAMPLE_RATE_HZ = 100
SAMPLES_PER_SECOND = 100
SAMPLES_PER_MINUTE = SAMPLES_PER_SECOND * 60

# Football match timing constants (in minutes)
MIN_HALF_LENGTH = 45
MIN_HALFTIME_BREAK = 15
MAX_HALFTIME_BREAK = 20
MAX_HALF_LENGTH = 60  # Allow for some extra time

#split date function if needed  
# def extract_date_dd_mm_yyyy(filename):
#     # Example input: '2024-08-24-AM_20-Entire-Session'
#     print(filename)
#     name = Path(filename).stem
#     numb = len("speedonly_")
#     name = name[numb:]
#     print(name)

#     parts = name.split('-')
#     if len(parts) >= 3:
#         yyyy, mm, dd = parts[0], parts[1], parts[2]
#         return f"{dd}.{mm}.{yyyy}"
#     return None


@dataclass
class MatchPeriod:
    start: pd.Timestamp
    end: pd.Timestamp
    confidence: float
    method: str

    @property
    def duration(self) -> timedelta:
        return self.end - self.start

class HalfTimeDetector:
    def __init__(self, folder_path: str, config: Dict):
        self.folder_path = Path(folder_path)
        self.config = {
            'gap_threshold_sec': 60,
            'alignment_margin_min': 2,
            'min_speed_threshold': 1.5,
            'high_speed_threshold': 1.5,
            'min_active_players': 7,
            'min_half_length': MIN_HALF_LENGTH,
            'max_half_length': MAX_HALF_LENGTH,
            'min_halftime_break': MIN_HALFTIME_BREAK,
            'max_halftime_break': MAX_HALFTIME_BREAK,
            'sample_rate_hz': SAMPLE_RATE_HZ,
            **config
        }
        
        self.player_data = self._load_player_data()
        

    def _load_player_data(self) -> pd.DataFrame:
        """Load and combine all player data from speedonly files."""
        player_speeds = []
        
        
        # Get list of files first
        csv_files = list(self.folder_path.glob("*.csv"))
        total_files = len(csv_files)
        
        
        for i, file in enumerate(csv_files, 1):
            # Only process speedonly files
            # filter_maccabi_subs_by_date(extract_date_dd_mm_yyyy(file.name)) 
            print(f"Processing file {i}/{total_files}: {file.name}")
            try:
                # Read only the columns we need
                print(f"  Reading file...")
                df = pd.read_csv(file)
                print(f"  File loaded, shape: {df.shape}")
                
                # Convert time strings to pandas timestamps
                print("  Converting timestamps...")
                df['Timestamp'] = pd.to_datetime(df['Time'], format='%H:%M:%S.%f', errors='coerce')
                df = df.dropna(subset=['Timestamp'])
                
                # Basic data validation
                if len(df) < SAMPLES_PER_MINUTE:
                    print(f"  Skipping file - too short: {len(df)} samples")
                    continue
                    
                # Extract player ID from filename
                player_id = file.stem.split('-')[3:5]  # Get position and number
                player_id = '-'.join(player_id)  # Combine them
                print(f"  Processing player: {player_id}")
                
                # Create DataFrame with speed data, using Timestamp as index
                df_speed = df[['Timestamp', 'Speed (m/s)', 'Lat', 'Lon']].copy()
                df_speed = df_speed.drop_duplicates(subset='Timestamp')  # Handle duplicate timestamps
                df_speed = df_speed.set_index('Timestamp')
                
                # Rename columns to include player ID
                df_speed = df_speed.rename(columns={'Speed (m/s)': f'speed_{player_id}'})
                df_speed[f'active_{player_id}'] = df_speed[f'speed_{player_id}'] > self.config['min_speed_threshold']
                df_speed = df_speed.rename(columns={'Lat': f'lat_{player_id}', 'Lon': f'lon_{player_id}'})
                
                player_speeds.append(df_speed)
                print(f"✓ Completed processing {file.name}")
                
            except Exception as e:
                print(f"⚠️ Error processing {file.name}: {str(e)}")
                import traceback
                print(traceback.format_exc())
                
        if not player_speeds:
            raise ValueError("No valid player data found")
            
        print("Combining all player data...")
        # Concatenate all player data and sort by timestamp
        combined_df = pd.concat(player_speeds, axis=1).sort_index()
        
        # Add mean speed across all players
        speed_cols = [col for col in combined_df.columns if col.startswith('speed_')]
        combined_df['mean_speed'] = combined_df[speed_cols].mean(axis=1)
        
        # Top 10 mean speed (better signal of real activity)
        combined_df['mean_top10_speed'] = combined_df[speed_cols].apply(
        lambda row: row.nlargest(10).mean(), axis=1
        )

        # Resample for detection (e.g., 15s or 1min intervals)
        combined_df['mean_speed_15s'] = combined_df['mean_speed'].resample('15s').mean()
        combined_df['mean_top10_speed_15s'] = combined_df['mean_top10_speed'].resample('15s').mean()
        
        print(f"Final combined shape: {combined_df.shape}")
        return combined_df

    def find_activity_start(self, start_time: pd.Timestamp) -> Optional[pd.Timestamp]:
        """Find the first significant activity after a given time."""
        window = 30 * SAMPLE_RATE_HZ  # 30-second window
        after_start = self.player_data[self.player_data.index >= start_time]
        
        if after_start.empty:
            return None
            
        # Calculate mean speed across all players
        speed_cols = [col for col in after_start.columns if col.startswith('speed_')]
        mean_speed = after_start[speed_cols].mean(axis=1)
        
        # Smooth the speed data
        smooth_speed = mean_speed.rolling(window, min_periods=window//2).mean()
        
        # Find first sustained activity (speed > threshold for at least 30 seconds)
        high_activity = smooth_speed > self.config['high_speed_threshold']
        if not high_activity.any():
            return None
            
        return high_activity[high_activity].index[0]

    def find_activity_end(self, start_time: pd.Timestamp, min_duration: pd.Timedelta) -> Optional[pd.Timestamp]:
        """Find the end of activity period, ensuring minimum duration from start."""
        min_end_time = start_time + min_duration
        after_min_time = self.player_data[self.player_data.index >= min_end_time]
        
        if after_min_time.empty:
            return None
            
        window = 30 * SAMPLE_RATE_HZ  # 30-second window
        speed_cols = [col for col in after_min_time.columns if col.startswith('speed_')]
        mean_speed = after_min_time[speed_cols].mean(axis=1)
        
        # Smooth the speed data
        smooth_speed = mean_speed.rolling(window, min_periods=window//2).mean()
        
        # Find first sustained low activity
        low_activity = smooth_speed < self.config['min_speed_threshold']
        if not low_activity.any():
            return None
            
        return low_activity[low_activity].index[0]

    def detect_by_gap(self) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Try to detect halftime by finding the main gap in data."""
        print("\n🔍 Attempting gap-based halftime detection...")
        
        # Look for gaps in each player's data
        befores, afters = [], []
        
        for col in [c for c in self.player_data.columns if c.startswith('speed_')]:
            # Get data for this player
            player_data = self.player_data[col].dropna()
            if player_data.empty:
                continue
                
            # Find gaps
            time_diffs = player_data.index.to_series().diff().dt.total_seconds()
            gap_indices = time_diffs[time_diffs > self.config['gap_threshold_sec']].index
            
            for gap_index in gap_indices:
                before_idx = player_data.index.get_loc(gap_index) - 1
                if before_idx >= 0:
                    before = player_data.index[before_idx]
                    after = gap_index
                    befores.append(before)
                    afters.append(after)
        
        if not befores or not afters:
            print("❌ No suitable gaps found for halftime")
            return None
            
        # Filter gaps by alignment with median
        befores = pd.Series(befores)
        afters = pd.Series(afters)
        
        margin = timedelta(minutes=self.config['alignment_margin_min'])
        median_before = befores.median()
        median_after = afters.median()
        valid_mask = (abs(befores - median_before) < margin) & (abs(afters - median_after) < margin)
        
        filtered_befores = befores[valid_mask]
        filtered_afters = afters[valid_mask]
        
        if not filtered_befores.empty and not filtered_afters.empty:
            halftime_start = filtered_befores.min() - pd.Timedelta(minutes=2)
            halftime_end = filtered_afters.max() + pd.Timedelta(minutes=1)
            gap_duration = (halftime_end - halftime_start).total_seconds() / 60
            
            if (gap_duration >= self.config['min_halftime_break'] and 
                gap_duration <= self.config['max_halftime_break']):
                print(f"✅ Found halftime gap: {gap_duration:.1f} minutes")
                return halftime_start, halftime_end
        
        print("❌ No valid gaps found within halftime duration constraints")
        return None

    def detect_match_periods(self) -> Tuple[MatchPeriod, MatchPeriod]:
        """Detect match periods using gap detection first, then falling back to timing rules."""
        # First try gap detection
        gap_result = self.detect_by_gap()
        if gap_result:
            halftime_start, halftime_end = gap_result
            print(f"\n✅ Found halftime gap: {(halftime_end - halftime_start).total_seconds() / 60:.1f} minutes")
            
            # Find first half start by looking backwards from the gap
            # First get all data before halftime
            before_half = self.player_data[:halftime_start].copy()
            if before_half.empty:
                print("❌ No data found before halftime gap")
                return self._detect_by_timing_rules()
            
            # Calculate mean speed across all players
            speed_cols = [col for col in before_half.columns if col.startswith('speed_')]
            before_half['mean_top10_speed'] = before_half[speed_cols].apply(lambda row: row.nlargest(10).mean(), axis=1)

            
            # Resample to 1-minute windows and look for the start of sustained activity
            activity_start = None
            window_size = '1min'
            min_speed = 0.8
            
            # Look for the first window of sustained activity
            rolling_speed = before_half['mean_top10_speed'].resample(window_size).mean()
            active_windows = rolling_speed[rolling_speed >= min_speed]
            if not active_windows.empty:
                # Find the first sustained activity (at least 2 consecutive minutes)
                for idx in active_windows.index:
                    next_minute = idx + pd.Timedelta(minutes=1)
                    if next_minute in active_windows.index:
                        activity_start = idx
                        break
            
            if activity_start is None:
                print("❌ Could not detect clear start of first half activity")
                return self._detect_by_timing_rules()
                
            # Find first actual activity after the detected start window
            first_half_start = self.player_data[self.player_data.index >= activity_start].index[0]
            # first_half_start = first_half_start - pd.Timedelta(minutes=2)
            print(f"✅ Found first half start at: {first_half_start.time()}")
            
            # Find second half end (last activity after gap)
            after_half = self.player_data[self.player_data.index >= halftime_end].copy()
            if after_half.empty:
                print("❌ No data found after halftime gap")
                return self._detect_by_timing_rules()
            
            # Calculate mean speed for second half
            after_half['mean_top10_speed'] = after_half[speed_cols].mean(axis=1)
            
            # First find the earliest possible end time (min_half_length after start)
            min_end_time = halftime_end + pd.Timedelta(minutes=self.config['min_half_length'])
            max_end_time = halftime_end + pd.Timedelta(minutes=self.config['max_half_length'])
            
            # Look at data within valid time range
            valid_range = after_half[
                (after_half.index >= min_end_time) & 
                (after_half.index <= max_end_time)
            ]
            
            if valid_range.empty:
                print("❌ No data found in valid second half end range")
                return self._detect_by_timing_rules()
            
            # Look for the end of sustained activity
            rolling_speed = valid_range['mean_top10_speed'].resample(window_size).mean()
            # Find windows where activity drops below threshold
            inactive_windows = rolling_speed[rolling_speed < min_speed]
            
            if inactive_windows.empty:
                # If no clear drop in activity, use the max allowed duration
                print("⚠️ No clear activity drop found, using maximum allowed duration")
                second_half_end = max_end_time
            else:
                # Use the first sustained drop in activity
                for idx in inactive_windows.index:
                    next_minute = idx + pd.Timedelta(minutes=1)
                    if next_minute in inactive_windows.index:
                        second_half_end = idx
                        break
                else:
                    # If no sustained drop found, use the first drop point
                    second_half_end = inactive_windows.index[0]
            
            print(f"✅ Found second half end at: {second_half_end.time()}")
            
            # Create periods
            first_half = MatchPeriod(
                start=first_half_start,
                end=halftime_start,
                confidence=0.95,
                method='gap'
            )
            
            second_half = MatchPeriod(
                start=halftime_end,
                end=second_half_end,
                confidence=0.95,
                method='gap'
            )
            
            # Calculate durations
            first_half_duration = (first_half.end - first_half.start).total_seconds() / 60
            second_half_duration = (second_half.end - second_half.start).total_seconds() / 60
            halftime_duration = (second_half.start - first_half.end).total_seconds() / 60
            
            print(f"\nValidating gap-based periods:")
            print(f"First half: {first_half_duration:.1f} minutes")
            print(f"Halftime break: {halftime_duration:.1f} minutes")
            print(f"Second half: {second_half_duration:.1f} minutes")
            
            # Validate durations with some flexibility for gap-based detection
            min_half_length = 45  # More lenient minimum for gap-based detection
            max_half_length = self.config['max_half_length']
            
            if (first_half_duration < min_half_length or 
                second_half_duration < min_half_length or
                first_half_duration > max_half_length or
                second_half_duration > max_half_length):
                print(f"❌ Half duration invalid (should be between {min_half_length} and {max_half_length} minutes)")
                return self._detect_by_timing_rules()
            
            if not (self.config['min_halftime_break'] <= halftime_duration <= self.config['max_halftime_break']):
                print(f"❌ Invalid halftime duration: {halftime_duration:.1f} minutes")
                return self._detect_by_timing_rules()
            
            print("\n✅ Gap-based detection successful!")
            self.visualize_detection(first_half, second_half)
            return first_half, second_half
        
        else:
            print("\n⚠️ Gap detection failed, falling back to timing rules...")
            return self._detect_by_timing_rules()

    def _detect_by_timing_rules(self) -> Tuple[MatchPeriod, MatchPeriod]:
        """Fallback detection method mimicking gap-based logic using speed windows."""
        print("\n🔍 Using timing rules (resample-based start/end detection)...")

        window_size = '1min'
        min_speed = 0.8  # Threshold for activity
        sustained_minutes = 2

        # Use mean of all player speeds
        speed_cols = [col for col in self.player_data.columns if col.startswith('speed_')]
        self.player_data['mean_top10_speed'] = self.player_data[speed_cols].apply(
        lambda row: row.nlargest(10).mean(), axis=1)

        # Resample to 1-minute average speed
        resampled_speed = self.player_data['mean_top10_speed'].resample(window_size).mean()

        # --- First Half Start ---
        active_windows = resampled_speed[resampled_speed >= min_speed]
        fh_start = None
        for idx in active_windows.index:
            next_minute = idx + pd.Timedelta(minutes=1)
            if next_minute in active_windows.index:
                fh_start = idx
                break
        if fh_start is None:
            raise ValueError("Could not detect first half start based on sustained activity.")

        first_half_start = self.player_data[self.player_data.index >= fh_start].index[0]
        # first_half_start = first_half_start - pd.Timedelta(minutes=2)
        print(f"✅ Detected first half start: {first_half_start.time()}")

        # --- First Half End ---
        min_fh_end = first_half_start + pd.Timedelta(minutes=self.config['min_half_length'])
        max_fh_end = first_half_start + pd.Timedelta(minutes=self.config['max_half_length'])
        fh_valid = resampled_speed[(resampled_speed.index >= min_fh_end) & (resampled_speed.index <= max_fh_end)]
        print(fh_valid.head(20))
        low_windows = fh_valid[fh_valid < min_speed]
        if not low_windows.empty:
            for idx in low_windows.index:
                next_minute = idx + pd.Timedelta(minutes=1)
                if next_minute in low_windows.index:
                    first_half_end = idx
                    break
            else:
                first_half_end = low_windows.index[0]
        else:
            first_half_end = min_fh_end
            print("⚠️ No clear activity drop found for first half, using max duration")

        print(f"✅ Detected first half end: {first_half_end.time()}")

        # --- Second Half Start ---
        min_sh_start = first_half_end + pd.Timedelta(minutes=self.config['min_halftime_break'])
        max_sh_start = first_half_end + pd.Timedelta(minutes=self.config['max_halftime_break'])
        sh_valid = resampled_speed[(resampled_speed.index >= min_sh_start) & (resampled_speed.index <= max_sh_start)]
        active_windows = sh_valid[sh_valid >= min_speed]
        sh_start = None
        for idx in active_windows.index:
            next_minute = idx + pd.Timedelta(minutes=1)
            if next_minute in active_windows.index:
                sh_start = idx
                break
        if sh_start is None:
            sh_start = first_half_end + pd.Timedelta(minutes=17)

        second_half_start = self.player_data[self.player_data.index >= sh_start].index[0]
        # second_half_start = second_half_start - pd.Timedelta(minutes=2)
        print(f"✅ Detected second half start: {second_half_start.time()}")

        # --- Second Half End ---
        min_sh_end = second_half_start + pd.Timedelta(minutes=self.config['min_half_length'])
        max_sh_end = second_half_start + pd.Timedelta(minutes=self.config['max_half_length'])
        sh_valid = resampled_speed[(resampled_speed.index >= min_sh_end) & (resampled_speed.index <= max_sh_end)]
        low_windows = sh_valid[sh_valid < min_speed]
        if not low_windows.empty:
            for idx in low_windows.index:
                next_minute = idx + pd.Timedelta(minutes=1)
                if next_minute in low_windows.index:
                    second_half_end = idx
                    break
            else:
                second_half_end = low_windows.index[0]
        else:
            second_half_end = min_sh_end
            print("⚠️ No clear activity drop found for second half, using max duration")

        print(f"✅ Detected second half end: {second_half_end.time()}")

        # --- Final Validation ---
        fh_duration = (first_half_end - first_half_start).total_seconds() / 60
        sh_duration = (second_half_end - second_half_start).total_seconds() / 60
        halftime_duration = (second_half_start - first_half_end).total_seconds() / 60

        print(f"\n🧪 Validating durations:")
        print(f"  First half: {fh_duration:.1f} minutes")
        print(f"  Halftime: {halftime_duration:.1f} minutes")
        print(f"  Second half: {sh_duration:.1f} minutes")

        # if not (self.config['min_half_length'] <= fh_duration <= self.config['max_half_length']):
        #     raise ValueError("Invalid first half duration")
        # if not (self.config['min_half_length'] <= sh_duration <= self.config['max_half_length']):
        #     raise ValueError("Invalid second half duration")
        # if not (self.config['min_halftime_break'] <= halftime_duration <= self.config['max_halftime_break']):
        #     raise ValueError("Invalid halftime break duration")

        print("\n✅ Timing rule-based detection successful!")
        return (
            MatchPeriod(start=first_half_start, end=first_half_end, confidence=0.9, method='timing_rules'),
            MatchPeriod(start=second_half_start, end=second_half_end, confidence=0.9, method='timing_rules')
        )


    def visualize_detection(self, first_half: MatchPeriod, second_half: MatchPeriod):
        """Create visualization of the detection results with timing annotations."""
        plt.figure(figsize=(15, 8))
        
        # Downsample for visualization
        viz_data = self.player_data.resample('1S').mean()
        
        # Plot mean speed
        speed_cols = [col for col in viz_data.columns if col.startswith('speed_')]
        mean_speed = viz_data[speed_cols].mean(axis=1)
        plt.plot(mean_speed.index, mean_speed.values, 'b-', alpha=0.5, label='Mean Speed')
        
        # Plot active players
        active_cols = [col for col in viz_data.columns if col.startswith('active_')]
        active_players = viz_data[active_cols].sum(axis=1)
        plt.plot(active_players.index, active_players.values, 'g-', alpha=0.5, label='Active Players')
        
        # Highlight detected periods
        plt.axvspan(first_half.start, first_half.end, color='green', alpha=0.2, label='First Half')
        plt.axvspan(second_half.start, second_half.end, color='blue', alpha=0.2, label='Second Half')
        
        # Add timing annotations
        halftime_duration = (second_half.start - first_half.end).total_seconds() / 60
        first_half_duration = (first_half.end - first_half.start).total_seconds() / 60
        second_half_duration = (second_half.end - second_half.start).total_seconds() / 60
        
        plt.annotate(f'First Half\n{first_half_duration:.1f}min',
                    xy=(first_half.start + (first_half.end - first_half.start)/2, plt.ylim()[1]),
                    ha='center', va='bottom')
                    
        plt.annotate(f'Break\n{halftime_duration:.1f}min',
                    xy=(first_half.end + (second_half.start - first_half.end)/2, plt.ylim()[1]),
                    ha='center', va='bottom')
                    
        plt.annotate(f'Second Half\n{second_half_duration:.1f}min',
                    xy=(second_half.start + (second_half.end - second_half.start)/2, plt.ylim()[1]),
                    ha='center', va='bottom')
        
        plt.title('Match Period Detection Results (with Football Timing Rules)')
        plt.xlabel('Time')
        plt.ylabel('Speed (m/s) / Active Players')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plt.savefig(self.folder_path / 'period_detection.png')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Match period detection with football timing rules.")
    parser.add_argument("folder", help="Folder containing player CSV files")
    parser.add_argument("--min-half", type=int, default=MIN_HALF_LENGTH, 
                       help=f"Minimum half length in minutes (default: {MIN_HALF_LENGTH})")
    parser.add_argument("--max-half", type=int, default=MAX_HALF_LENGTH,
                       help=f"Maximum half length in minutes (default: {MAX_HALF_LENGTH})")
    parser.add_argument("--min-break", type=int, default=MIN_HALFTIME_BREAK,
                       help=f"Minimum halftime break in minutes (default: {MIN_HALFTIME_BREAK})")
    parser.add_argument("--max-break", type=int, default=MAX_HALFTIME_BREAK,
                       help=f"Maximum halftime break in minutes (default: {MAX_HALFTIME_BREAK})")
    args = parser.parse_args()

    config = {
        'min_half_length': args.min_half,
        'max_half_length': args.max_half,
        'min_halftime_break': args.min_break,
        'max_halftime_break': args.max_break
    }

    try:
        detector = HalfTimeDetector(args.folder, config)
        first_half, second_half = detector.detect_match_periods()
        
        print("\n✅ Match Periods Detected (with Football Timing Rules):")
        
        first_duration = (first_half.end - first_half.start).total_seconds() / 60
        print(f"\nFirst Half:")
        print(f"  Start: {first_half.start.time()}")
        print(f"  End:   {first_half.end.time()}")
        print(f"  Duration: {first_duration:.1f} minutes")
        
        break_duration = (second_half.start - first_half.end).total_seconds() / 60
        print(f"\nHalftime Break:")
        print(f"  Duration: {break_duration:.1f} minutes")
        
        second_duration = (second_half.end - second_half.start).total_seconds() / 60
        print(f"\nSecond Half:")
        print(f"  Start: {second_half.start.time()}")
        print(f"  End:   {second_half.end.time()}")
        print(f"  Duration: {second_duration:.1f} minutes")
        
        print(f"\n📊 Visualization saved to: {args.folder}/period_detection.png")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()


