import argparse
import numpy as np
import open3d as o3d

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input .ply file")
    parser.add_argument("output", help="Output .ply file")
    parser.add_argument("--n", type=int, default=1, help="Number of points to add at the first point's location.")
    args = parser.parse_args()

    pcd = o3d.io.read_point_cloud(args.input)
    pts = np.asarray(pcd.points)

    first_point = pts[0]
    extra = np.tile(first_point, (args.n, 1))
    new_pts = np.vstack([pts, extra])

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(new_pts)

    print(f"Before: {len(pts)} points")
    print(f"First point location: {first_point}")
    print(f"After:  {len(new_pts)} points  (+{args.n} added at first point location)")
    o3d.io.write_point_cloud(args.output, new_pcd)
    print(f"Saved to {args.output}")
