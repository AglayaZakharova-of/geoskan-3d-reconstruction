"""
This code performs quantitative comparison of all meshes in a folder.
Works with either single files or directories for both inputs.


Usage:
    python evaluate_meshes.py \
        --mesh-input results/hangar/rosbags_1/mesh/p11_bestmix_poisson_d11.ply \
        --pcd-input  results/hangar/rosbags_1/bag_1_surf_005

    python evaluate_meshes.py \
        --mesh-input results/hangar/rosbags_1/mesh \
        --pcd-input  results/hangar/rosbags_1/bag_1_surf_005 \
        --csv        results/hangar/rosbags_1/mesh_metrics.csv
"""


import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def log(m): print(m, flush=True)


# --------------------------------------------------------------- reference cloud
def resolve_pcd_dir(path: Path) -> Path:
    """FAST-LIVO2 обычно кладёт PCD в подпапку PCD/ или Log/PCD/."""
    if list(path.glob("*.pcd")):
        return path
    candidates = [path / "PCD", path / "Log" / "PCD"]
    for c in candidates:
        if c.is_dir() and list(c.glob("*.pcd")):
            return c
    return path


def load_cloud(input_path: Path, sor_k: int, sor_std: float) -> o3d.geometry.PointCloud:
    """Загружает ОДИН файл облака (.pcd/.ply) ИЛИ объединяет все PCD из папки."""

    # 1) Один файл
    if input_path.is_file():
        if input_path.suffix.lower() not in (".pcd", ".ply"):
            raise ValueError(
                f"Unsupported point cloud format: {input_path.suffix}. "
                "Use .pcd or .ply"
            )
        log(f"Loading single cloud file: {input_path}")
        pcd = o3d.io.read_point_cloud(str(input_path))
        log(f"  -> {len(pcd.points):,} pts")

    # 2) Папка
    else:
        pcd_dir = resolve_pcd_dir(input_path)
        files = sorted(glob.glob(str(pcd_dir / "*.pcd")))
        if not files:
            raise FileNotFoundError(
                f"No .pcd files in {input_path} or its subdirs (PCD/, Log/PCD/)"
            )
        log(f"Loading {len(files)} PCD files from {pcd_dir}…")
        pts = np.vstack([
            np.asarray(o3d.io.read_point_cloud(f).points) for f in files
        ])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        log(f"  merged: {len(pcd.points):,} pts")

    # SOR
    if sor_k > 0:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=sor_k, std_ratio=sor_std
        )
        log(f"  after SOR(k={sor_k}, std={sor_std}): {len(pcd.points):,} pts")
    return pcd


# --------------------------------------------------------------- topology
def topology(mesh: o3d.geometry.TriangleMesh):
    tris = np.asarray(mesh.triangles)
    if len(tris) == 0:
        return 0, 0
    e = np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e.sort(axis=1)
    _, c = np.unique(e, axis=0, return_counts=True)
    return int((c == 1).sum()), int((c > 2).sum())


def components(mesh: o3d.geometry.TriangleMesh) -> int:
    if len(mesh.triangles) == 0:
        return 0
    _, sizes, _ = mesh.cluster_connected_triangles()
    return int(len(sizes))


# --------------------------------------------------------------- per-mesh eval
def evaluate_one(path: Path, ref_cloud: o3d.geometry.PointCloud,
                 n_samples: int, cloud_samples: int):
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    if len(mesh.triangles) == 0:
        log(f"  WARNING: {path.name} has no triangles, skipping")
        return None
    mesh.compute_vertex_normals()

    b, nm = topology(mesh)
    comp = components(mesh)

    # Uniform sample on the surface
    ms = mesh.sample_points_uniformly(number_of_points=n_samples)

    # Subsample reference for speed
    cp = np.asarray(ref_cloud.points)
    if cloud_samples > 0 and cp.shape[0] > cloud_samples:
        idx = np.random.default_rng(0).choice(
            cp.shape[0], cloud_samples, replace=False
        )
        cp = cp[idx]
    cs = o3d.geometry.PointCloud()
    cs.points = o3d.utility.Vector3dVector(cp)

    # Symmetric distances
    d_c2m = np.asarray(cs.compute_point_cloud_distance(ms))   # coverage
    d_m2c = np.asarray(ms.compute_point_cloud_distance(cs))   # accuracy

    return {
        "file": path.name,
        "verts": len(mesh.vertices),
        "tris": len(mesh.triangles),
        "boundary_edges": b,
        "nonmanifold_edges": nm,
        "components": comp,
        "cd_mm": float(d_c2m.mean() + d_m2c.mean()) * 1000,
        "hd_mm": float(max(d_c2m.max(), d_m2c.max())) * 1000,
        "mean_c2m_mm": float(d_c2m.mean()) * 1000,
        "mean_m2c_mm": float(d_m2c.mean()) * 1000,
        "max_c2m_mm": float(d_c2m.max()) * 1000,
        "max_m2c_mm": float(d_m2c.max()) * 1000,
    }


# --------------------------------------------------------------- table print
def print_table(rows):
    if not rows:
        print("No meshes evaluated.")
        return
    cols = ["file", "verts", "tris", "boundary_edges", "nonmanifold_edges",
            "components", "cd_mm", "hd_mm",
            "mean_c2m_mm", "mean_m2c_mm", "max_c2m_mm", "max_m2c_mm"]
    if HAS_PANDAS:
        df = pd.DataFrame(rows)[cols]
        for c in ("cd_mm", "hd_mm", "mean_c2m_mm", "mean_m2c_mm",
                  "max_c2m_mm", "max_m2c_mm"):
            df[c] = df[c].round(2)
        pd.set_option("display.width", 250)
        pd.set_option("display.max_columns", 50)
        print(df.to_string(index=False))
    else:
        widths = {c: max(len(c), max(
            len(f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]))
            for r in rows)) for c in cols}
        print(" | ".join(c.ljust(widths[c]) for c in cols))
        print("-+-".join("-" * widths[c] for c in cols))
        for r in rows:
            print(" | ".join(
                (f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]))
                .ljust(widths[c]) for c in cols))


# --------------------------------------------------------------- mesh discovery
def discover_meshes(input_path: Path, pattern: str) -> list[Path]:
    """Один файл -> [файл]. Папка -> glob по pattern."""
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        return []
    return sorted(input_path.glob(pattern))


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(
        description="Evaluate one or many meshes against a reference point cloud "
                    "(accepts single files or directories)."
    )

    # Меши: один файл ИЛИ папка
    ap.add_argument("--mesh-input", "--mesh-dir", dest="mesh_input",
                    required=True, type=Path,
                    help="Path to a single .ply mesh file OR a folder with meshes")
    # Облако: один файл ИЛИ папка
    ap.add_argument("--pcd-input", "--pcd-dir", dest="pcd_input",
                    required=True, type=Path,
                    help="Path to a single .pcd/.ply cloud OR a folder with PCDs")

    ap.add_argument("--pattern", default="*.ply",
                    help="Glob pattern for mesh discovery in a folder")
    ap.add_argument("--samples", type=int, default=100_000,
                    help="Points sampled on each mesh for CD/HD")
    ap.add_argument("--cloud-samples", type=int, default=500_000,
                    help="Subsample reference cloud (0 = all)")
    ap.add_argument("--sor-k", type=int, default=20)
    ap.add_argument("--sor-std", type=float, default=2.0)
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    mesh_input = args.mesh_input.expanduser().resolve()
    pcd_input = args.pcd_input.expanduser().resolve()

    if not mesh_input.exists():
        sys.exit(f"Mesh input not found: {mesh_input}")
    if not pcd_input.exists():
        sys.exit(f"PCD input not found: {pcd_input}")

    # 1) Загрузка референсного облака (папка или файл)
    ref = load_cloud(pcd_input, args.sor_k, args.sor_std)

    # 2) Список мешей (один файл или папка)
    files = discover_meshes(mesh_input, args.pattern)
    if not files:
        sys.exit(f"No meshes found in {mesh_input} (pattern: {args.pattern})")
    log(f"Will evaluate {len(files)} mesh(es)")

    # 3) Оценка
    rows = []
    for f in files:
        log(f"Evaluating {f.name} …")
        try:
            r = evaluate_one(f, ref, args.samples, args.cloud_samples)
            if r:
                rows.append(r)
        except Exception as e:
            log(f"  ERROR: {e}")

    print()
    print_table(rows)

    # 4) CSV
    if args.csv and rows:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        if HAS_PANDAS:
            pd.DataFrame(rows).to_csv(args.csv, index=False)
        else:
            with open(args.csv, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=rows[0].keys())
                w.writeheader()
                w.writerows(rows)
        log(f"\nCSV saved: {args.csv}")


if __name__ == "__main__":
    main()