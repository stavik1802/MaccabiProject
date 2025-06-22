import pandas as pd
import numpy as np
from pathlib import Path

# Parameters
SPEED_THRESHOLD = 2.0
WARMUP_SPEED_MIN = 1.5
WARMUP_SPEED_MAX = 3.0
MIN_ACTIVE_SECONDS = 30
SAMPLE_RATE = 10  # Hz

# Bench area box: (lat_min, lat_max, lon_min, lon_max) — adjust to your field layout
BENCH_AREA = {
    'lat_min': 32.7829309, 'lat_max': 32.7831894,
    'lon_min': 34.9649515, 'lon_max': 34.9649639
}

def is_in_bench_area(lat, lon):
    return (BENCH_AREA['lat_min'] <= lat <= BENCH_AREA['lat_max']) and \
           (BENCH_AREA['lon_min'] <= lon <= BENCH_AREA['lon_max'])

def bench_check(df):
    if {'Lat', 'Lon'}.issubset(df.columns):
        in_bench = df.apply(lambda row: is_in_bench_area(row['Lat'], row['Lon']), axis=1)
        still = df['Speed (m/s)'] < 0.5
        percent_in_bench = in_bench.mean()
        percent_still = still.mean()
        return percent_in_bench > 0.8 and percent_still > 0.9
    return False

def detect_warmup(df):
    if {'Lat', 'Lon', 'Speed (m/s)'}.issubset(df.columns):
        warmup_mask = (df['Speed (m/s)'] > WARMUP_SPEED_MIN) & (df['Speed (m/s)'] < WARMUP_SPEED_MAX)
        sideline_area = df['Lon'].between(BENCH_AREA['lon_min'], BENCH_AREA['lon_max'])
        warmup_time = (warmup_mask & sideline_area).sum() / SAMPLE_RATE
        return warmup_time >= 20  # seconds
    return False

def is_active(file_path: Path, half: str):
    if not file_path.exists():
        return False, False, False

    try:
        df = pd.read_csv(file_path)
        if df.empty or 'Speed (m/s)' not in df.columns:
            return False, False, False

        # Basic activity
        active_mask = df['Speed (m/s)'] > SPEED_THRESHOLD
        active_duration = active_mask.sum() / SAMPLE_RATE

        # Enhancements
        on_bench = bench_check(df)
        warmup = detect_warmup(df)

        actually_played = active_duration >= MIN_ACTIVE_SECONDS and not on_bench
        return actually_played, warmup, on_bench

    except Exception as e:
        print(f"⚠️ Error reading {file_path.name}: {e}")
        return False, False, False

def classify_status(first, second):
    first_active, first_warmup, _ = first
    second_active, second_warmup, _ = second

    if first_active and second_active:
        return "Full Match"
    elif first_active and not second_active:
        return "Subbed Off"
    elif not first_active and second_active:
        return "Subbed On"
    elif not first_active and not second_active and (first_warmup or second_warmup):
        return "Bench Warm-Up Only"
    else:
        return "Did Not Play"

def analyze_players(root_folder: str):
    root = Path(root_folder)
    result = []

    for player_folder in root.iterdir():
        if not player_folder.is_dir():
            continue

        player = player_folder.name
        first_half_path = player_folder / "first_half.csv"
        second_half_path = player_folder / "second_half.csv"

        first_info = is_active(first_half_path, "first")
        second_info = is_active(second_half_path, "second")

        status = classify_status(first_info, second_info)

        result.append({
            "Player": player,
            "FirstHalfActive": first_info[0],
            "SecondHalfActive": second_info[0],
            "FirstHalfWarmup": first_info[1],
            "SecondHalfWarmup": second_info[1],
            "OnBench": first_info[2] or second_info[2],
            "Status": status
        })

    return pd.DataFrame(result)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Classify player participation with warmup/bench check.")
    parser.add_argument("folder", help="Path to root folder containing player folders")
    parser.add_argument("--output", default="player_status.csv", help="Output CSV filename")
    args = parser.parse_args()

    df = analyze_players(args.folder)
    df.to_csv(args.output, index=False)
    print(f"✅ Player status saved to {args.output}")
