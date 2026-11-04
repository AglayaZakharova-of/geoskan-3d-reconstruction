"""
This script helps to view a single experiment's point cloud in Open3D.

Usage:
    python view_single_pcd.py <path_to_experiment_dir> [--voxel 0.05]

Example:
    python view_single_pcd.py results/hangar/rosbags_1/bag_1_surf_005
"""


import argparse
import sys
from pathlib import Path
import open3d as o3d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir", type=Path, help="Path to experiment folder (e.g., bag_1_surf_005)")
    ap.add_argument("--voxel", type=float, default=None, help="Optional voxel size for downsampling (e.g., 0.05)")
    args = ap.parse_args()

    pcd_dir = args.exp_dir / "Log" / "PCD"
    if not pcd_dir.is_dir():
        print(f"ERROR: PCD folder not found: {pcd_dir}", file=sys.stderr)
        return 1

    files = sorted(pcd_dir.glob("*.pcd"))
    if not files:
        print(f"ERROR: No .pcd files in {pcd_dir}", file=sys.stderr)
        return 1

    print(f"Loading {len(files)} .pcd files from {pcd_dir}...")
    merged = o3d.geometry.PointCloud()
    for f in files:
        merged += o3d.io.read_point_cloud(str(f))
    
    print(f"Total points: {len(merged.points)}")

    if args.voxel is not None:
        print(f"Downsampling with voxel size {args.voxel}...")
        merged = merged.voxel_down_sample(args.voxel)
        print(f"Points after downsampling: {len(merged.points)}")

    print("Showing window. Close it to exit.")
    o3d.visualization.draw_geometries([merged], window_name=args.exp_dir.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())