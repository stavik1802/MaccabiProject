"""
Script: find_bench.py
This script is used to find the bench area of a player. we don't use it but if you want to use it you can.
make sure the first and second half data is for both played and not played to find the bench area.
Description:
    This script analyzes GPS tracking data to automatically detect and identify the bench area
    where players remain stationary during a game. It uses a grid-based approach to find areas
    with high density of stationary points.

Input:
    - Root folder containing player subfolders
    - Each player folder should contain:
        * first_half.csv
        * second_half.csv
    - CSV files must have columns:
        * Speed (m/s)
        * Lat (latitude)
        * Lon (longitude)

Output:
    - Prints the coordinates of the detected bench area
    - Saves a visualization plot showing the detected bench area

Usage:
    python find_bench.py <folder_path>
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import argparse

# Parameters
SPEED_THRESHOLD = 0.5  # m/s
GRID_SIZE = 50  # Number of cells in each dimension
MIN_POINTS = 10  # Minimum points in a cell to consider it
EXPANSION_CELLS = 1  # Number of cells to expand around the densest cell

def process_chunk(chunk):
    """Process a single chunk of data and return stationary points."""
    if 'Speed (m/s)' not in chunk.columns or 'Lat' not in chunk.columns or 'Lon' not in chunk.columns:
        return None
    
    stationary = chunk[chunk['Speed (m/s)'] < SPEED_THRESHOLD]
    if stationary.empty:
        return None
        
    return stationary[['Lat', 'Lon']].dropna()

def create_grid(coords):
    """Create a grid and count points in each cell."""
    if len(coords) == 0:
        return None, None, None, None
    
    # Calculate grid boundaries
    lat_min, lat_max = coords['Lat'].min(), coords['Lat'].max()
    lon_min, lon_max = coords['Lon'].min(), coords['Lon'].max()
    
    # Add small padding
    lat_pad = (lat_max - lat_min) * 0.01
    lon_pad = (lon_max - lon_min) * 0.01
    lat_min -= lat_pad
    lat_max += lat_pad
    lon_min -= lon_pad
    lon_max += lon_pad
    
    # Create 2D histogram
    hist, lat_edges, lon_edges = np.histogram2d(
        coords['Lat'], 
        coords['Lon'],
        bins=GRID_SIZE,
        range=[[lat_min, lat_max], [lon_min, lon_max]]
    )
    
    return hist, lat_edges, lon_edges, (lat_min, lat_max, lon_min, lon_max)

def find_bench_area(hist, lat_edges, lon_edges):
    """Find the bench area based on point density."""
    if hist is None:
        return None, None
    
    # Find the cell with maximum density
    max_idx = np.unravel_index(hist.argmax(), hist.shape)
    if hist[max_idx] < MIN_POINTS:
        return None, None
    
    # Get the bounds of the densest region (including neighboring cells)
    lat_idx, lon_idx = max_idx
    lat_start = max(0, lat_idx - EXPANSION_CELLS)
    lat_end = min(len(lat_edges) - 1, lat_idx + EXPANSION_CELLS + 1)
    lon_start = max(0, lon_idx - EXPANSION_CELLS)
    lon_end = min(len(lon_edges) - 1, lon_idx + EXPANSION_CELLS + 1)
    
    bench_bounds = {
        'lat_min': lat_edges[lat_start],
        'lat_max': lat_edges[lat_end],
        'lon_min': lon_edges[lon_start],
        'lon_max': lon_edges[lon_end]
    }
    
    return bench_bounds, hist

def main():
    parser = argparse.ArgumentParser(description="Detect bench area from stationary points.")
    parser.add_argument("folder", help="Path to root folder containing player folders")
    args = parser.parse_args()
    
    root = Path(args.folder)
    all_stationary = []
    total_points = 0
    
    print("🔍 Processing tracking data...")
    
    # Process each player's data
    for player_folder in root.iterdir():
        if not player_folder.is_dir():
            continue
            
        for half in ["first_half.csv", "second_half.csv"]:
            file_path = player_folder / half
            try:
                # Process file in chunks
                for chunk in pd.read_csv(file_path, chunksize=10000):
                    stationary = process_chunk(chunk)
                    if stationary is not None:
                        all_stationary.append(stationary)
                        total_points += len(stationary)
                        
                        # Print progress
                        if total_points % 50000 == 0:
                            print(f"  Processed {total_points:,} points...")
                            
            except Exception as e:
                print(f"⚠️ Error processing {file_path}: {e}")
    
    if not all_stationary:
        print("❌ No valid stationary points found")
        return
    
    # Combine all stationary points
    print("📊 Creating density grid...")
    coords = pd.concat(all_stationary, ignore_index=True)
    hist, lat_edges, lon_edges, bounds = create_grid(coords)
    
    if hist is None:
        print("❌ Failed to create density grid")
        return
    
    # Find bench area
    bench_bounds, hist = find_bench_area(hist, lat_edges, lon_edges)
    if bench_bounds is None:
        print("❌ Could not detect bench area")
        return
    
    print("\n✅ Detected Bench Area:")
    print(f"Latitude:  {bench_bounds['lat_min']:.7f} to {bench_bounds['lat_max']:.7f}")
    print(f"Longitude: {bench_bounds['lon_min']:.7f} to {bench_bounds['lon_max']:.7f}")
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    
    # Plot density heatmap
    plt.imshow(
        hist.T,
        extent=[bounds[0], bounds[1], bounds[2], bounds[3]],
        origin='lower',
        aspect='auto',
        cmap='YlOrRd',
        alpha=0.6
    )
    
    # Highlight bench area
    plt.plot(
        [bench_bounds['lat_min'], bench_bounds['lat_max'], bench_bounds['lat_max'], bench_bounds['lat_min'], bench_bounds['lat_min']],
        [bench_bounds['lon_min'], bench_bounds['lon_min'], bench_bounds['lon_max'], bench_bounds['lon_max'], bench_bounds['lon_min']],
        'r-',
        linewidth=2,
        label='Detected Bench Area'
    )
    
    plt.colorbar(label='Number of stationary points')
    plt.title('Bench Area Detection')
    plt.xlabel('Latitude')
    plt.ylabel('Longitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_file = root / 'bench_detection.png'
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n📊 Visualization saved to: {out_file}")

if __name__ == "__main__":
    main()

