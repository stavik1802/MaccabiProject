"""
Script: find_field_bounds.py

Description:
    This script analyzes player tracking data to automatically determine the football field
    boundaries. It uses statistical analysis of player positions to identify the playing area,
    filtering out outliers and non-field positions (like bench areas). The boundaries are then
    adjusted to match standard football field dimensions (105m x 70m).

Input:
    - CSV files containing player tracking data with position information
    - Expected columns:
        * Lat (latitude)
        * Lon (longitude)
        * Speed (m/s) - used to filter stationary points

Output:
    - Dictionary with field boundaries (lat_min, lat_max, lon_min, lon_max)
    - Visualization of detected field boundaries
    - JSON file with boundary coordinates
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
from typing import Dict, Tuple, List
import gc
from math import radians, cos

# Standard football field dimensions in meters
FIELD_LENGTH = 105  # meters
FIELD_WIDTH = 70    # meters

def meters_to_degrees(meters: float, lat: float) -> float:
    """Convert meters to degrees at a given latitude."""
    # Earth's radius in meters
    earth_radius = 6371000
    
    # Convert to degrees (approximate)
    return (meters / earth_radius) * (180 / np.pi)

def adjust_boundaries_to_dimensions(boundaries: Dict[str, float]) -> Dict[str, float]:
    """
    Adjust the detected boundaries to match standard football field dimensions.
    
    Args:
        boundaries: Dictionary with detected lat/lon boundaries
    
    Returns:
        Dictionary with adjusted boundaries maintaining 105m x 70m proportions
    """
    # Calculate center point
    center_lat = (boundaries['lat_max'] + boundaries['lat_min']) / 2
    center_lon = (boundaries['lon_max'] + boundaries['lon_min']) / 2
    
    # Convert desired dimensions to degrees
    # For longitude, need to account for latitude (field gets narrower at higher latitudes)
    length_deg = meters_to_degrees(FIELD_LENGTH, center_lat)
    # Width needs to be adjusted for latitude
    width_deg = meters_to_degrees(FIELD_WIDTH, center_lat) / cos(radians(center_lat))
    
    # Calculate new boundaries maintaining the center point
    adjusted = {
        'lat_min': center_lat - length_deg / 2,
        'lat_max': center_lat + length_deg / 2,
        'lon_min': center_lon - width_deg / 2,
        'lon_max': center_lon + width_deg / 2
    }
    
    return adjusted

def process_chunk(chunk: pd.DataFrame, speed_threshold: float = 0.5) -> Dict[str, List[float]]:
    """Process a single chunk of data and return its statistics."""
    # Filter out stationary points
    chunk = chunk[chunk['Speed (m/s)'] > speed_threshold]
    
    if chunk.empty:
        return None
    
    # Calculate statistics for this chunk
    stats = {
        'Lat': [
            chunk['Lat'].quantile(0.01),
            chunk['Lat'].quantile(0.25),
            chunk['Lat'].quantile(0.75),
            chunk['Lat'].quantile(0.99)
        ],
        'Lon': [
            chunk['Lon'].quantile(0.01),
            chunk['Lon'].quantile(0.25),
            chunk['Lon'].quantile(0.75),
            chunk['Lon'].quantile(0.99)
        ]
    }
    
    return stats

def load_and_process_file(file_path: Path, chunk_size: int = 100000) -> List[Dict[str, List[float]]]:
    """Load and process a CSV file in chunks."""
    try:
        chunks_stats = []
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            if not {'Lat', 'Lon', 'Speed (m/s)'}.issubset(chunk.columns):
                print(f"Warning: Missing required columns in {file_path}")
                continue
                
            stats = process_chunk(chunk)
            if stats:
                chunks_stats.append(stats)
            
            # Force garbage collection
            del chunk
            gc.collect()
            
        return chunks_stats
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return []

def combine_stats(all_stats: List[Dict[str, List[float]]], padding_factor: float = 0.0001) -> Dict[str, float]:
    """Combine statistics from all chunks to determine final boundaries."""
    if not all_stats:
        raise ValueError("No valid statistics to process")
    
    # Combine all quantiles
    all_lat_stats = [stat for stats in all_stats for stat in stats['Lat']]
    all_lon_stats = [stat for stats in all_stats for stat in stats['Lon']]
    
    # Calculate initial boundaries
    boundaries = {
        'lat_min': min(all_lat_stats) - padding_factor,
        'lat_max': max(all_lat_stats) + padding_factor,
        'lon_min': min(all_lon_stats) - padding_factor,
        'lon_max': max(all_lon_stats) + padding_factor
    }
    
    # Adjust boundaries to match standard football field dimensions
    adjusted_boundaries = adjust_boundaries_to_dimensions(boundaries)
    
    return adjusted_boundaries

def plot_field_boundaries(boundaries: Dict[str, float], output_file: str):
    """Generate visualization of detected field boundaries."""
    plt.figure(figsize=(12, 8))
    
    # Plot field boundaries
    plt.plot([boundaries['lon_min'], boundaries['lon_max']], 
             [boundaries['lat_min'], boundaries['lat_min']], 'r-', label='Field Boundary')
    plt.plot([boundaries['lon_min'], boundaries['lon_max']], 
             [boundaries['lat_max'], boundaries['lat_max']], 'r-')
    plt.plot([boundaries['lon_min'], boundaries['lon_min']], 
             [boundaries['lat_min'], boundaries['lat_max']], 'r-')
    plt.plot([boundaries['lon_max'], boundaries['lon_max']], 
             [boundaries['lat_min'], boundaries['lat_max']], 'r-')
    
    # Add field dimensions text
    center_lat = (boundaries['lat_max'] + boundaries['lat_min']) / 2
    center_lon = (boundaries['lon_max'] + boundaries['lon_min']) / 2
    plt.text(center_lon, boundaries['lat_max'], f'Width: {FIELD_WIDTH}m', 
             horizontalalignment='center', verticalalignment='bottom')
    plt.text(boundaries['lon_max'], center_lat, f'Length: {FIELD_LENGTH}m', 
             horizontalalignment='left', verticalalignment='center', rotation=-90)
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Football Field Boundaries (Adjusted to Standard Dimensions)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_folder(folder_path: str) -> Dict[str, float]:
    """Analyze all tracking data in folder to find field boundaries."""
    folder = Path(folder_path)
    all_stats = []
    
    # Process each CSV file
    for csv_file in folder.rglob("*.csv"):
        print(f"Processing {csv_file.name}...")
        file_stats = load_and_process_file(csv_file)
        all_stats.extend(file_stats)
        
        # Force garbage collection
        gc.collect()
    
    if not all_stats:
        raise ValueError(f"No valid tracking data found in {folder}")
    
    # Combine all statistics to find boundaries
    boundaries = combine_stats(all_stats)
    
    return boundaries

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Detect football field boundaries from tracking data")
    parser.add_argument("folder", help="Path to folder containing tracking data CSV files")
    parser.add_argument("--output", default="field_bounds.json", 
                       help="Output JSON file for field boundaries")
    parser.add_argument("--plot", default="field_boundaries.png",
                       help="Output visualization file")
    parser.add_argument("--chunk-size", type=int, default=100000,
                       help="Number of rows to process at once")
    args = parser.parse_args()
    
    try:
        print(f"Analyzing tracking data in {args.folder}...")
        boundaries = analyze_folder(args.folder)
        
        # Save boundaries to JSON
        with open(args.output, 'w') as f:
            json.dump(boundaries, f, indent=4)
        print(f"✅ Field boundaries saved to {args.output}")
        
        # Generate visualization
        plot_field_boundaries(boundaries, args.plot)
        print(f"📊 Visualization saved to {args.plot}")
        
        # Print boundaries
        print("\nDetected Field Boundaries:")
        for key, value in boundaries.items():
            print(f"{key}: {value:.7f}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 