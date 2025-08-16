import argparse
import os
import sys
import json
import math
import time
from datetime import datetime

import numpy as np
import pandas as pd
import open3d as o3d
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Try reading .e57 with pye57 or e57
_HAS_PYE57 = False
_HAS_E57 = False
try:
    import pye57
    _HAS_PYE57 = True
except ImportError:
    pass

if not _HAS_PYE57:
    try:
        import e57
        _HAS_E57 = True
    except ImportError:
        pass

# ----------- E57 LOADING -----------
def load_e57_points(path, want_color=True, want_intensity=True):
    if _HAS_PYE57:
        e = pye57.E57(path)
        pts_list, cols_list, ints_list = [], [], []
        for si in range(len(e.scan_count)):
            data = e.read_scan(si, intensity=want_intensity, colors=want_color)
            x, y, z = data["cartesianX"], data["cartesianY"], data["cartesianZ"]
            if x is None or y is None or z is None:
                continue
            pts_list.append(np.column_stack([x, y, z]))
            if want_color and all(k in data for k in ("colorRed", "colorGreen", "colorBlue")):
                cr = data["colorRed"] / 255.0
                cg = data["colorGreen"] / 255.0
                cb = data["colorBlue"] / 255.0
                cols_list.append(np.column_stack([cr, cg, cb]))
            if want_intensity and "intensity" in data and data["intensity"] is not None:
                ints_list.append(data["intensity"])
        if not pts_list:
            raise ValueError("No points found in E57 file.")
        result = {"points": np.vstack(pts_list)}
        if cols_list:
            result["color"] = np.vstack(cols_list)
        if ints_list:
            result["intensity"] = np.concatenate(ints_list)
        return result

    elif _HAS_E57:
        pc = e57.read_points(path)
        result = {"points": np.asarray(pc.points)}
        if hasattr(pc, "color") and pc.color is not None:
            result["color"] = np.asarray(pc.color)
        if hasattr(pc, "intensity") and pc.intensity is not None:
            result["intensity"] = np.asarray(pc.intensity)
        return result

    else:
        raise ImportError("Install 'pye57' or 'e57' to read E57 files.")

# ----------- PREPROCESSING -----------
def preprocess_pcd(pcd, voxel_size, nb_neighbors, std_ratio):
    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size)
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd

# ----------- KMEANS CLUSTERING -----------
def auto_select_k(X, k_min=2, k_max=12):
    best_k, best_score = None, -1
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init="auto")
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k or k_min

# ----------- ROTATION MATRIX TO EULER -----------
def rotation_matrix_to_euler_xyz(R):
    r20 = max(min(R[2,0], 1), -1)
    ry = math.asin(-r20)
    cy = math.cos(ry)
    if abs(cy) > 1e-8:
        rx = math.atan2(R[2,1] / cy, R[2,2] / cy)
        rz = math.atan2(R[1,0] / cy, R[0,0] / cy)
    else:
        rx, rz = 0, math.atan2(-R[0,1], R[1,1])
    return np.degrees([rx, ry, rz])

# ----------- COMPUTE OBB/AABB -----------
def compute_clusters_info(pcd, labels, min_points=200):
    results = []
    pts = np.asarray(pcd.points)
    for cid in np.unique(labels):
        if cid < 0:
            continue
        idx = np.where(labels == cid)[0]
        if len(idx) < min_points:
            continue
        cluster_points = pts[idx]
        cp = o3d.geometry.PointCloud()
        cp.points = o3d.utility.Vector3dVector(cluster_points)

        aabb = cp.get_axis_aligned_bounding_box()
        a_ext = np.array(aabb.get_extent())

        try:
            obb = o3d.geometry.OrientedBoundingBox.create_from_points(cp.points)
            center = obb.get_center()
            ext = np.array(obb.extent)
            R = np.asarray(obb.R)
            rot_xyz = rotation_matrix_to_euler_xyz(R)
        except Exception:
            center = aabb.get_center()
            ext = a_ext
            rot_xyz = [0, 0, 0]

        ext_sorted = np.sort(ext)[::-1]
        results.append({
            "asset_id": len(results)+1,
            "cluster_id": cid + 1,
            "center_x": center[0], "center_y": center[1], "center_z": center[2],
            "length": ext_sorted[0], "width": ext_sorted[1], "height": ext_sorted[2],
            "rot_x": rot_xyz[0], "rot_y": rot_xyz[1], "rot_z": rot_xyz[2]
        })
    return results

# ----------- EXPORT EXCEL -----------
def export_excel(results, metadata, path):
    df = pd.DataFrame(results)
    meta_df = pd.DataFrame([metadata])
    with pd.ExcelWriter(path) as writer:
        df.to_excel(writer, index=False, sheet_name="clusters")
        meta_df.to_excel(writer, index=False, sheet_name="metadata")

# ----------- OPTIONAL EXPORTS -----------
def export_ply(points, color, path):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if color is not None and len(color) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(path, pcd)

def export_py7(data, path, src_name):
    meta = {
        "source": src_name,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "point_count": len(data["points"]),
        "fields": list(data.keys())
    }
    np.savez_compressed(path, xyz=data["points"], rgb=data.get("color"), intensity=data.get("intensity"), meta_json=json.dumps(meta))

# ----------- MAIN -----------
def main():
    ap = argparse.ArgumentParser(description="E57 PointCloud Processing Pipeline")
    ap.add_argument("--input", required=True, help=".e57 point cloud file")
    ap.add_argument("--excel_out", required=True, help="Output Excel file")
    ap.add_argument("--voxel", type=float, default=0.02, help="Voxel size (m)")
    ap.add_argument("--nb_neighbors", type=int, default=20)
    ap.add_argument("--std_ratio", type=float, default=2.0)
    ap.add_argument("--k", type=int, default=0, help="If 0, auto-select k")
    ap.add_argument("--ply_out", help="Optional export to .ply")
    ap.add_argument("--py7_out", help="Optional export to .py7")
    args = ap.parse_args()

    t0 = time.time()
    print("[INFO] Loading E57...")
    io_data = load_e57_points(args.input)
    points = io_data["points"]
    colors = io_data.get("color")

    if args.ply_out:
        export_ply(points, colors, args.ply_out)
    if args.py7_out:
        export_py7(io_data, args.py7_out, os.path.basename(args.input))

    print("[INFO] Preprocessing...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd = preprocess_pcd(pcd, args.voxel, args.nb_neighbors, args.std_ratio)

    X = np.asarray(pcd.points)
    if args.k <= 0:
        print("[INFO] Selecting optimal k...")
        k = auto_select_k(X, 2, 12)
    else:
        k = args.k

    print(f"[INFO] Clustering with k={k}")
    km = KMeans(n_clusters=k, n_init="auto").fit(X)
    labels = km.labels_

    print("[INFO] Computing bounding boxes...")
    clusters_info = compute_clusters_info(pcd, labels)

    metadata = {
        "input_file": os.path.basename(args.input),
        "total_points": len(points),
        "processed_points": len(X),
        "clusters": len(clusters_info),
        "voxel_size": args.voxel,
        "nb_neighbors": args.nb_neighbors,
        "std_ratio": args.std_ratio,
        "k": k
    }
    export_excel(clusters_info, metadata, args.excel_out)
    print(f"[DONE] Exported Excel in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
