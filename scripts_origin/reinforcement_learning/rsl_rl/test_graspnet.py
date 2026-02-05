#!/usr/bin/env python3
"""Minimal script to test GraspNet feature extraction from a PLY file."""

import sys
import argparse
import numpy as np
import torch
import open3d as o3d

# Add graspnet paths
GRASPNET_ROOT = "/home/roborock/IsaacLab/source/isaaclab/isaaclab/graspnet-baseline"
sys.path.insert(0, GRASPNET_ROOT)
sys.path.insert(0, f"{GRASPNET_ROOT}/models")
sys.path.insert(0, f"{GRASPNET_ROOT}/pointnet2")

from models.graspnet import GraspNet, pred_decode

def load_pointcloud(ply_path, num_points=20000):
    """Load PLY and sample to fixed number of points."""
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points).astype(np.float32)
    
    # Sample or pad to num_points
    if len(points) >= num_points:
        idxs = np.random.choice(len(points), num_points, replace=False)
    else:
        idxs = np.random.choice(len(points), num_points, replace=True)
    
    points = points[idxs]
    return points

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply", type=str, required=True, help="Path to PLY file")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint (optional)")
    parser.add_argument("--num_points", type=int, default=20000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load point cloud
    print(f"Loading point cloud from: {args.ply}")
    points = load_pointcloud(args.ply, args.num_points)
    print(f"Point cloud shape: {points.shape}")

    # Create model
    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                   cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04],
                   is_training=False)
    net = net.to(device)
    
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("Warning: No checkpoint loaded, using random weights")

    net.eval()

    # Prepare input
    end_points = {}
    end_points['point_clouds'] = torch.from_numpy(points).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)

    # Print available features
    print("\n=== Output Features ===")
    for key, val in end_points.items():
        if isinstance(val, torch.Tensor):
            print(f"{key}: shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"{key}: type={type(val)}")

    print("\nFeature extraction successful!")

if __name__ == "__main__":
    main()