"""
This script reconstruct mesh from FAST-LIVO2 point clouds using Open3D.
Can accept either a directory with multiple .pcd files or a single .pcd/.ply file.


Example:
    python build_mesh.py \
        --input-dir /data/results/hangar/rosbags_1/bag_1_surf_005 \
        --method both --poisson-depth 9
    
    python build_mesh.py --input my_single_scan.pcd \
        --method poisson \
        --poisson-depth 11 \
        --poisson-trim 0.05 \
        --poisson-scale 1.0 \
        --voxel-size 0.01 \
        --no-vis
"""


import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def resolve_pcd_dir(path: Path) -> Path:
    """FAST-LIVO2 обычно кладёт PCD в подпапку PCD/ или Log/PCD/."""
    if list(path.glob("*.pcd")):
        return path
    candidates = [path / "PCD", path / "Log" / "PCD"]
    for c in candidates:
        if c.is_dir() and list(c.glob("*.pcd")):
            return c
    return path


def load_input(input_path: Path) -> o3d.geometry.PointCloud:
    """Загружает либо один файл, либо объединяет все PCD из папки."""
    
    # 1. Если это один файл (pcd или ply)
    if input_path.is_file():
        if input_path.suffix.lower() not in [".pcd", ".ply"]:
            raise ValueError(f"Unsupported file format: {input_path.suffix}. Use .pcd or .ply")
        log(f"Loading single file: {input_path}")
        pcd = o3d.io.read_point_cloud(str(input_path))
        log(f"  -> loaded {len(pcd.points):,} points")
        return pcd

    # 2. Если это папка
    pcd_dir = resolve_pcd_dir(input_path)
    files = sorted(glob.glob(str(pcd_dir / "*.pcd")))
    if not files:
        raise FileNotFoundError(f"No .pcd files found in {pcd_dir} or its subdirs")
    
    log(f"Found {len(files)} PCD files in {pcd_dir}")

    all_points, all_colors = [], []
    has_color = True
    for i, f in enumerate(files):
        pcd = o3d.io.read_point_cloud(f)
        pts = np.asarray(pcd.points)
        all_points.append(pts)
        if pcd.has_colors() and np.asarray(pcd.colors).shape[0] == pts.shape[0]:
            all_colors.append(np.asarray(pcd.colors))
        else:
            has_color = False
        if (i + 1) % 20 == 0 or i == len(files) - 1:
            log(f"  loaded {i+1}/{len(files)}: {len(pts):,} pts")

    points = np.vstack(all_points)
    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(points)
    if has_color and len(all_colors) == len(files):
        merged.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
    log(f"Merged cloud: {len(points):,} points")
    return merged


def preprocess(pcd: o3d.geometry.PointCloud, voxel_size: float, sor_k: int, sor_std: float) -> o3d.geometry.PointCloud:
    if voxel_size > 0:
        log(f"Voxel downsample @ {voxel_size:.4f} m")
        pcd = pcd.voxel_down_sample(voxel_size)
        log(f"  -> {len(pcd.points):,} points")

    if sor_k > 0:
        log(f"Statistical outlier removal (k={sor_k}, std={sor_std})")
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=sor_k, std_ratio=sor_std)
        log(f"  -> {len(pcd.points):,} points kept")
    return pcd


def estimate_normals(pcd: o3d.geometry.PointCloud, k: int, radius: float, orient: str) -> o3d.geometry.PointCloud:
    log(f"Estimating normals (k={k}, radius={radius or 'auto'})")
    if radius > 0:
        search = o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=k)
    else:
        search = o3d.geometry.KDTreeSearchParamKNN(knn=k)
    pcd.estimate_normals(search_param=search)

    if orient == "mst":
        log(f"  orienting normals (consistent tangent plane, k={k}) — may take a while")
        t0 = time.time()
        pcd.orient_normals_consistent_tangent_plane(k)
        log(f"  orientation took {time.time() - t0:.1f}s")
    elif orient == "camera":
        log("  orienting normals towards origin (camera at 0,0,0)")
        pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))
    elif orient == "up":
        log("  orienting normals to align with +Z")
        pcd.orient_normals_to_align_with_direction(np.array([0.0, 0.0, 1.0]))
    elif orient == "none":
        log("  WARNING: normals not oriented — Poisson may produce artifacts")
    else:
        raise ValueError(f"Unknown orient: {orient}")
    return pcd


def poisson_reconstruct(pcd: o3d.geometry.PointCloud, depth: int, trim_quantile: float, linear_fit: bool, scale: float = 1.1) -> o3d.geometry.TriangleMesh:
    log(f"Poisson reconstruction (depth={depth}, scale={scale}, linear_fit={linear_fit})")
    t0 = time.time()
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, scale=scale, linear_fit=linear_fit
    )
    log(f"  Poisson took {time.time() - t0:.1f}s -> "
        f"{len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")

    if trim_quantile > 0:
        dens = np.asarray(densities)
        thr = np.quantile(dens, trim_quantile)
        mask = dens < thr
        mesh.remove_vertices_by_mask(mask)
        log(f"  trimmed {int(mask.sum()):,} low-density verts (q={trim_quantile}) "
            f"-> {len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")
    return mesh


def bpa_reconstruct(pcd: o3d.geometry.PointCloud, radii: list[float]) -> o3d.geometry.TriangleMesh:
    log(f"Ball Pivoting (radii={[round(r*1000, 1) for r in radii]} mm)")
    t0 = time.time()
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    log(f"  BPA took {time.time() - t0:.1f}s -> "
        f"{len(mesh.vertices):,} verts, {len(mesh.triangles):,} tris")
    return mesh


def auto_bpa_radii(pcd: o3d.geometry.PointCloud) -> list[float]:
    d = pcd.compute_nearest_neighbor_distance()
    m = float(np.mean(d))
    radii = [m * 1.5, m * 3.0, m * 6.0]
    log(f"  auto BPA radii (mean NN = {m*1000:.1f} mm): {[round(r*1000, 1) for r in radii]} mm")
    return radii


def save_mesh(mesh: o3d.geometry.TriangleMesh, path: Path) -> None:
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False, compressed=False)
    log(f"Saved: {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Build mesh from point clouds (dir or single file)")

    # I/O (--input может быть файлом или папкой, --input-dir оставлен для совместимости)
    ap.add_argument("--input", "--input-dir", dest="input_path", required=True, type=Path,
                    help="Path to a single .pcd/.ply file OR a directory with .pcd files")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Default: <input_dir>/../meshes or <input_file_dir>/meshes")
    ap.add_argument("--name", type=str, default=None,
                    help="Output basename")

    # preprocessing
    ap.add_argument("--voxel-size", type=float, default=0.0, help="Voxel downsample size in meters (0 = disabled)")
    ap.add_argument("--sor-k", type=int, default=20)
    ap.add_argument("--sor-std", type=float, default=2.0)

    # normals
    ap.add_argument("--normals-k", type=int, default=30)
    ap.add_argument("--normals-radius", type=float, default=0.0)
    ap.add_argument("--orient", choices=["mst", "camera", "up", "none"], default="mst")

    # reconstruction
    ap.add_argument("--method", choices=["poisson", "bpa", "both"], default="poisson")
    ap.add_argument("--poisson-depth", type=int, default=9)
    ap.add_argument("--poisson-trim", type=float, default=0.02, help="Quantile of low-density verts to remove (0 = off)")
    ap.add_argument("--poisson-scale", type=float, default=1.1, help="Scale factor for Poisson bbox (default 1.1)")
    ap.add_argument("--no-linear-fit", dest="linear_fit", action="store_false", help="Disable linear_fit for Poisson")
    ap.add_argument("--bpa-radii", type=float, nargs="+", default=None, help="BPA radii in meters; auto if omitted")

    # output
    ap.add_argument("--crop-to-bbox", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--save-cleaned", action="store_true", help="Also dump the cleaned point cloud as .ply")
    ap.add_argument("--no-vis", action="store_true")

    args = ap.parse_args()
    t_total = time.time()

    input_path = args.input_path.expanduser().resolve()
    if not input_path.exists():
        sys.exit(f"Error: Input path does not exist: {input_path}")

    # Определяем папку для сохранения
    if args.output_dir:
        output_dir = args.output_dir
    elif input_path.is_file():
        output_dir = input_path.parent / "meshes"
    else:
        output_dir = input_path.parent / "meshes"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Определяем базовое имя файла
    if args.name:
        name = args.name
    elif input_path.is_file():
        name = input_path.stem  # имя файла без расширения
    else:
        name = input_path.name  # имя папки

    # 1) load
    pcd = load_input(input_path)
    bbox = pcd.get_axis_aligned_bounding_box()
    log(f"Input bbox: min={np.round(bbox.min_bound, 2)}, max={np.round(bbox.max_bound, 2)}")

    # 2) preprocess
    pcd = preprocess(pcd, args.voxel_size, args.sor_k, args.sor_std)

    if args.save_cleaned:
        p = output_dir / f"{name}_cleaned.ply"
        o3d.io.write_point_cloud(str(p), pcd)
        log(f"Saved cleaned cloud: {p}")

    # 3) normals
    pcd = estimate_normals(pcd, args.normals_k, args.normals_radius, args.orient)

    # 4) reconstruct
    meshes_for_vis = []

    if args.method in ("poisson", "both"):
        mesh_p = poisson_reconstruct(pcd, args.poisson_depth, args.poisson_trim, 
                                     args.linear_fit, scale=args.poisson_scale)
        if args.crop_to_bbox:
            mesh_p = mesh_p.crop(bbox)
            log(f"  cropped to bbox -> {len(mesh_p.vertices):,} verts, {len(mesh_p.triangles):,} tris")
        out = output_dir / f"{name}_poisson_d{args.poisson_depth}.ply"
        save_mesh(mesh_p, out)
        meshes_for_vis.append((f"Poisson d={args.poisson_depth}", mesh_p))

    if args.method in ("bpa", "both"):
        radii = args.bpa_radii or auto_bpa_radii(pcd)
        mesh_b = bpa_reconstruct(pcd, list(radii))
        if args.crop_to_bbox:
            mesh_b = mesh_b.crop(bbox)
        out = output_dir / f"{name}_bpa.ply"
        save_mesh(mesh_b, out)
        meshes_for_vis.append(("BPA", mesh_b))

    log(f"DONE in {time.time() - t_total:.1f}s")

    # 5) optional visualization
    if not args.no_vis and meshes_for_vis:
        log("Visualizing (close window to exit)")
        geoms = []
        for _, m in meshes_for_vis:
            m.paint_uniform_color(np.random.rand(3))
            m.compute_vertex_normals()
            geoms.append(m)
        geoms.append(bbox)
        o3d.visualization.draw_geometries(
            geoms,
            window_name=f"{name} — {', '.join(n for n, _ in meshes_for_vis)}",
            width=1280, height=800,
            mesh_show_back_face=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())