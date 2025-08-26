"""
Script: infer_field.py

Description:
    This script analyzes GPS tracking data from multiple players to infer the football field's
    orientation, dimensions, and boundaries. It uses Principal Component Analysis (PCA) to
    determine the field's alignment and scales the coordinates to match standard field dimensions
    (105m x 65m).

Key Features:
    - Processes GPS data from multiple players' first half movements
    - Uses PCA to determine field orientation and alignment
    - Scales coordinates to match standard field dimensions
    - Generates field corner coordinates in GPS (lat/lon) format
    - Creates a visualization of player movement heatmap
    - Outputs field boundaries and orientation data to JSON

Parameters:
    - TARGET_LENGTH: 105.0 meters (standard field length)
    - TARGET_WIDTH: 65.0 meters (standard field width)

Input:
    - Parent folder containing player subfolders
    - Each player folder should contain a "first_half.csv" with:
        - Time: timestamp
        - Lat: latitude
        - Lon: longitude

Output:
    - field_frame.json containing:
        - Field center coordinates
        - Field boundaries in meters
        - Corner coordinates in lat/lon
        - Rotation information
    - Optional visualization of player movement heatmap

Usage:
    python infer_field.py <parent_folder>
    Example: python infer_field.py ./match_data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import json

# Standard football field dimensions in meters
TARGET_LENGTH = 105.0  # meters
TARGET_WIDTH = 65.0    # meters

def infer_field_transform_with_visualization(parent_folder: str, show_plot=True):
    """
    Infer football field orientation and dimensions from player GPS tracking data.
    
    This function performs the following steps:
    1. Collects GPS coordinates from all players during the first half
    2. Uses PCA to find the main direction of player movement (field orientation)
    3. Rotates coordinates to align with the field's long axis
    4. Scales coordinates to match standard field dimensions
    5. Calculates field boundaries and corner coordinates
    6. Generates visualization and saves field data to JSON
    
    Args:
        parent_folder (str): Path to folder containing player subfolders
        show_plot (bool): Whether to display the heatmap visualization
        
    Returns:
        dict: Field configuration data including center, bounds, and corners
        
    Raises:
        RuntimeError: If no valid GPS points are found
    """
    all_coords = []

    # Step 1: Collect GPS coordinates from all players
    print("📊 Collecting GPS coordinates from all players...")
    for subfolder in Path(parent_folder).iterdir():
        file = subfolder / "first_half.csv"
        if not file.exists():
            continue
        try:
            df = pd.read_csv(file)
            # Parse timestamps and filter to first 40 minutes (avoiding warm-up and half-time)
            df["Timestamp"] = pd.to_datetime(df["Time"], format="%H:%M:%S.%f", errors="coerce")
            df = df.dropna(subset=["Timestamp"])
            # Use data from 5-40 minutes to avoid warm-up and half-time periods
            df = df[df["Timestamp"] <= df["Timestamp"].min() + pd.Timedelta(minutes=40)]
            df = df[df["Timestamp"] >= df["Timestamp"].min() + pd.Timedelta(minutes=5)]
            # Filter out invalid GPS coordinates
            df = df[(df["Lat"].abs() > 1) & (df["Lon"].abs() > 1)]
            all_coords.extend(zip(df["Lon"], df["Lat"]))  # X=Lon, Y=Lat
        except Exception as e:
            print(f"Error processing {file.name}: {e}")

    coords = np.array(all_coords)
    if len(coords) == 0:
        raise RuntimeError("No valid GPS points found.")

    print(f"✅ Collected {len(coords)} GPS points from all players")

    # Step 2: Calculate field center and center coordinates
    center = coords.mean(axis=0)
    coords_centered = coords - center
    print(f"📍 Field center: Lat {center[1]:.7f}, Lon {center[0]:.7f}")

    # Step 3: Use PCA to find field orientation
    print("🔄 Using PCA to determine field orientation...")
    pca = PCA(n_components=2)
    pca.fit(coords_centered)
    
    # The first principal component represents the direction of maximum variance
    # In football, this typically aligns with the field's long axis (goal-to-goal)
    angle_rad = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
    print(f"📐 Field rotation angle: {np.degrees(angle_rad):.1f}°")
    
    # Create rotation matrix to align field with coordinate system
    rotation_matrix = np.array([
        [np.cos(-angle_rad), -np.sin(-angle_rad)],
        [np.sin(-angle_rad),  np.cos(-angle_rad)]
    ])

    # Step 4: Rotate coordinates to align with field orientation
    rotated = coords_centered @ rotation_matrix.T

    # Step 5: Determine scaling factors to match standard field dimensions
    max_x, max_y = rotated.max(axis=0)
    min_x, min_y = rotated.min(axis=0)
    span_x = max_x - min_x
    span_y = max_y - min_y

    # Calculate scale factors to map observed field size to standard dimensions
    scale_x = TARGET_LENGTH / span_x
    scale_y = TARGET_WIDTH / span_y
    
    print(f"📏 Field spans: {span_x:.1f}m x {span_y:.1f}m")
    print(f"📏 Scaling factors: {scale_x:.3f} x {scale_y:.3f}")

    # Step 6: Apply scaling to get final field-aligned coordinates
    scaled_rotated = rotated * np.array([scale_x, scale_y])
    min_x, max_x = scaled_rotated[:, 0].min(), scaled_rotated[:, 0].max()
    min_y, max_y = scaled_rotated[:, 1].min(), scaled_rotated[:, 1].max()

    print("\n📆 Estimated Field Bounds (after rotation + scaling):")
    print(f"X: {min_x:.2f} m → {max_x:.2f} m (length: {max_x-min_x:.1f}m)")
    print(f"Y: {min_y:.2f} m → {max_y:.2f} m (width: {max_y-min_y:.1f}m)")

    def get_corners(rotated_coords, scale_x, scale_y):
        """
        Calculate field corner coordinates in GPS (lat/lon) format.
        
        This function:
        1. Takes the scaled and rotated coordinates
        2. Finds the bounding box corners
        3. Reverses the scaling and rotation transformations
        4. Converts back to GPS coordinates
        
        Args:
            rotated_coords: Scaled and rotated player coordinates
            scale_x, scale_y: Scaling factors used for transformation
            
        Returns:
            np.array: Corner coordinates in (lon, lat) format
        """
        # Find the four corners of the bounding box
        corners_xy = np.array([
            [rotated_coords[:, 0].min(), rotated_coords[:, 1].min()],  # Bottom-left
            [rotated_coords[:, 0].min(), rotated_coords[:, 1].max()],  # Top-left
            [rotated_coords[:, 0].max(), rotated_coords[:, 1].min()],  # Bottom-right
            [rotated_coords[:, 0].max(), rotated_coords[:, 1].max()]   # Top-right
        ]) / np.array([scale_x, scale_y])  # Reverse scaling
        
        # Reverse rotation transformation
        corners_rot = corners_xy @ rotation_matrix
        
        # Convert back to GPS coordinates by adding center offset
        corners_latlon = corners_rot + center
        return corners_latlon

    # Step 7: Calculate field corners for both possible orientations
    # We compute corners for both 0° and 90° orientations in case we need to flip the field
    corners_0 = get_corners(scaled_rotated, scale_x, scale_y)

    # For 90° rotation (if needed), rotate the scaled coordinates by 90 degrees
    scaled_rotated_90 = scaled_rotated @ np.array([[0, -1], [1, 0]])
    corners_90 = get_corners(scaled_rotated_90, scale_y, scale_x)

    # Default: use 0° version (you can choose 90° manually if needed)
    final_corners = corners_0
    used_rotation = "0° (default, no bench check)"

    print("\n📅 Field Corners (Lat, Lon):")
    corner_data = []
    for i, (lon, lat) in enumerate(final_corners):
        print(f"Corner {i+1}: Lat {lat:.7f}, Lon {lon:.7f}")
        corner_data.append({"corner": i + 1, "lat": float(lat), "lon": float(lon)})

    # Step 8: Create output JSON with field configuration
    output_json = {
        "center": {"lat": float(center[1]), "lon": float(center[0])},
        "field_bounds_m": {
            "x_min": float(min_x), "x_max": float(max_x),
            "y_min": float(min_y), "y_max": float(max_y)
        },
        "corners_latlon": corner_data,
        "rotation_used": used_rotation,
        "pca_components": pca.components_.tolist(),  # Store PCA components for reference
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist()  # How much variance each component explains
    }

    # Step 9: Save field configuration to JSON file
    with open("field_frame.json", "w") as f:
        json.dump(output_json, f, indent=2)
    print("\n📁 Saved field bounds to field_frame.json")

    # Step 10: Create visualization if requested
    if show_plot:
        plt.figure(figsize=(10, 6))
        # Create heatmap of player positions
        plt.hexbin(scaled_rotated[:, 0], scaled_rotated[:, 1], gridsize=75, cmap='hot')
        
        # Draw field boundary rectangle
        plt.gca().add_patch(plt.Rectangle(
            (min_x, min_y), TARGET_LENGTH, TARGET_WIDTH,
            linewidth=2, edgecolor='blue', facecolor='none', label="Field Frame (105x65)"
        ))
        
        plt.title("Player Heatmap in Aligned Field Frame (105x65 m)")
        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.colorbar(label="Log Density")
        plt.axis("equal")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return output_json

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python infer_field_pca.py <parent_folder>")
        print("Example: python infer_field_pca.py ./match_data")
    else:
        infer_field_transform_with_visualization(sys.argv[1])

