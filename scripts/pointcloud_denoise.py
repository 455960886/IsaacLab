import argparse
import numpy as np
import open3d as o3d

def denoise(input_path, output_path, nb_neighbors=50, std_ratio=5.0):
    pcd = o3d.io.read_point_cloud(input_path)
    print(f"Before: {len(pcd.points)} points")
    filtered, _ = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    print(f"After:  {len(filtered.points)} points")
    o3d.io.write_point_cloud(output_path, filtered)
    print(f"Saved to {output_path}")

def denoise_by_distance(input_path, output_path, max_distance):
    pcd = o3d.io.read_point_cloud(input_path)
    pts = np.asarray(pcd.points)
    centroid = pts.mean(axis=0)
    dists = np.linalg.norm(pts - centroid, axis=1)
    mask = dists <= max_distance
    print(f"Before: {len(pts)} points")
    print(f"Centroid: {centroid}")
    print(f"Max dist in cloud: {dists.max():.3f} m  —  removing points beyond {max_distance} m")
    filtered = pcd.select_by_index(np.where(mask)[0])
    print(f"After:  {len(filtered.points)} points  ({(~mask).sum()} removed)")
    o3d.io.write_point_cloud(output_path, filtered)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input .ply file")
    parser.add_argument("output", help="Output .ply file")
    parser.add_argument("--nb_neighbors", type=int, default=2, help="Neighbors to consider (default: 20)")
    parser.add_argument("--std_ratio", type=float, default=2.0, help="Std dev threshold (default: 2.0)")
    parser.add_argument("--max_distance", type=float, default=None,
                        help="Remove points farther than this distance (meters) from the centroid. "
                             "If set, skips statistical outlier removal.")
    args = parser.parse_args()
    if args.max_distance is not None:
        denoise_by_distance(args.input, args.output, args.max_distance)
    else:
        denoise(args.input, args.output, args.nb_neighbors, args.std_ratio)
