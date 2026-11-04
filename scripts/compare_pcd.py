"""
This script compares multiple FAST-LIVO2 PCD outputs.

Reads every <results_root>/<exp_name>/Log/PCD/*.pcd
Merges, downsamples (for fast visualization), computes stats,
saves side-by-side screenshots and a summary table.

Usage:
    python compare_pcd.py <results_root> [--voxel 0.1] [--save_dir <dir>]

Example:
    python compare_pcd.py results/hangar/rosbags_1
"""


import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


def load_merged(pcd_dir: Path) -> o3d.geometry.PointCloud:
    files = sorted(pcd_dir.glob("*.pcd"))
    if not files:
        raise FileNotFoundError(f"No .pcd in {pcd_dir}")
    merged = o3d.geometry.PointCloud()
    for f in files:
        merged += o3d.io.read_point_cloud(str(f))
    return merged


def stats(pcd: o3d.geometry.PointCloud) -> dict:
    pts = np.asarray(pcd.points)
    if len(pts) < 2:
        return {"n": len(pts)}
    # NN distance on a small random sample
    sample = pts[np.random.choice(len(pts), min(20000, len(pts)), replace=False)]
    nn_d = []
    kd = o3d.geometry.KDTreeFlann(pcd)
    for p in sample:
        _, _, d2 = kd.search_knn_vector_3d(p, 2)
        if len(d2) == 2:
            nn_d.append(np.sqrt(d2[1]))
    nn_d = np.array(nn_d)
    bbox = pcd.get_axis_aligned_bounding_box()
    return {
        "n": len(pts),
        "nn_mean": float(nn_d.mean()),
        "nn_median": float(np.median(nn_d)),
        "nn_p95": float(np.percentile(nn_d, 95)),
        "bbox_dx": float(bbox.get_extent()[0]),
        "bbox_dy": float(bbox.get_extent()[1]),
        "bbox_dz": float(bbox.get_extent()[2]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_root", type=Path,
                    help="Folder with <exp_name>/Log/PCD/*.pcd")
    ap.add_argument("--voxel", type=float, default=0.1,
                    help="Voxel for fast visualization (doesn't affect stats)")
    ap.add_argument("--save_dir", type=Path, default=None,
                    help="Where to save screenshots (default: <results_root>/_compare)")
    args = ap.parse_args()

    root = args.results_root
    if not root.is_dir():
        print(f"ERROR: {root} is not a directory", file=sys.stderr)
        return 1

    exps = sorted([d for d in root.iterdir()
                   if d.is_dir() and (d / "Log" / "PCD").is_dir()])
    if not exps:
        print(f"ERROR: no experiments found under {root}", file=sys.stderr)
        return 1

    save_dir = args.save_dir or (root / "_compare")
    save_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {}
    clouds_for_view = {}

    print(f"{'EXPERIMENT':<32} {'N_FULL':>10} {'N_VOXEL':>10} "
          f"{'NN_MEAN':>8} {'NN_MED':>8} {'NN_P95':>8}")
    print("-" * 82)

    for exp in exps:
        name = exp.name
        merged = load_merged(exp / "Log" / "PCD")
        st = stats(merged)
        down = merged.voxel_down_sample(args.voxel)
        st["n_voxel"] = len(down.points)
        all_stats[name] = st

        print(f"{name:<32} {st['n']:>10} {st['n_voxel']:>10} "
              f"{st['nn_mean']:>8.4f} {st['nn_median']:>8.4f} {st['nn_p95']:>8.4f}")

        # color per experiment for side-by-side view
        color_map = {
            0: [1, 0, 0],     # red
            1: [0, 0.7, 0],   # green
            2: [0, 0.4, 1],   # blue
            3: [1, 0.6, 0],   # orange
        }
        idx = list(exps).index(exp)
        c = color_map.get(idx, [0.5, 0.5, 0.5])
        down.paint_uniform_color(c)
        clouds_for_view[name] = down

    print()
    print("Bounding box (m):")
    print(f"{'EXPERIMENT':<32} {'dx':>8} {'dy':>8} {'dz':>8}")
    print("-" * 60)
    for name, st in all_stats.items():
        print(f"{name:<32} {st['bbox_dx']:>8.2f} "
              f"{st['bbox_dy']:>8.2f} {st['bbox_dz']:>8.2f}")

    # Show all clouds together (colors differentiate)
    print(f"\nVisualizing {len(clouds_for_view)} clouds (colors: "
          f"{', '.join(f'{n} -> {i}' for i, n in enumerate(clouds_for_view))})")
    print("Close the window to continue.")
    o3d.visualization.draw_geometries(
        list(clouds_for_view.values()),
        window_name=f"Compare: {', '.join(clouds_for_view.keys())}"
    )

    print(f"\nOK. Stats saved to {save_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())