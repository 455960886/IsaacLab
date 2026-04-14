import argparse
import numpy as np
import open3d as o3d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input .ply file")
    parser.add_argument("output", help="Output .ply file")
    parser.add_argument("--n", type=int, default=10, help="Number of points in the added cluster.")
    parser.add_argument("--distance", type=float, default=1.0, help="Distance of the cluster center from the original centroid (meters).")
    parser.add_argument("--cluster_std", type=float, default=0.01, help="Std dev of the cluster spread (meters).")
    parser.add_argument("--direction", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                        help="Direction from centroid to place the cluster. Defaults to random unit vector.")
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.input)
    pts = np.asarray(pcd.points)
    centroid = pts.mean(axis=0)

    if args.direction is not None:
        direction = np.array(args.direction, dtype=float)
    else:
        direction = np.random.randn(3)
    direction /= np.linalg.norm(direction)

    cluster_center = centroid + direction * args.distance
    cluster_pts = cluster_center + np.random.randn(args.n, 3) * args.cluster_std

    new_pts = np.vstack([pts, cluster_pts])
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(new_pts)

    print(f"Before: {len(pts)} points")
    print(f"Centroid: {centroid}")
    print(f"Cluster center: {cluster_center}  (distance={args.distance} m, direction={direction})")
    print(f"After:  {len(new_pts)} points  (+{args.n} cluster points, spread std={args.cluster_std} m)")
    o3d.io.write_point_cloud(args.output, new_pcd)
    print(f"Saved to {args.output}")
