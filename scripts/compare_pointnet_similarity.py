"""Compare cosine similarity of PointNet2 features between two pointclouds.

Uses the exact same PointNet2 encoder and preprocessing as image_features in observations.py:
  - Sample to 1024 points with exponential-decay distance weighting
  - Forward through sa1 → sa2 → sa3 (same checkpoint, same architecture)
  - Optionally L2-normalize features before cosine similarity

Usage:
    python scripts/compare_pointnet_similarity.py cloud1.ply cloud2.ply
    python scripts/compare_pointnet_similarity.py cloud1.ply cloud2.ply --no_normalize
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d

# ──────────────────────────────────────────────────────────────────────────────
# PointNet2 encoder (identical to _prepare_pointnet_model in observations.py)
# ──────────────────────────────────────────────────────────────────────────────

CKPT_PATH = "/home/roborock/IsaacLab/best_model.pth"

def build_encoder() -> nn.Module:
    import sys
    sys.path.insert(0, "/home/roborock/gitlab5/drl_manipulation/source/isaaclab/isaaclab/pointnet/log/classification/pointnet2_ssg_wo_normals")
    from pointnet2_cls_ssg import get_model as PointNet2ClsMsg

    classifier = PointNet2ClsMsg(num_class=40, normal_channel=False).cuda()
    checkpoint = torch.load(CKPT_PATH, map_location="cuda", weights_only=False)
    classifier.load_state_dict(checkpoint["model_state_dict"], strict=False)
    classifier.eval()

    class PointNet2Encoder(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.sa1 = base_model.sa1
            self.sa2 = base_model.sa2
            self.sa3 = base_model.sa3

        def forward(self, xyz):
            # xyz: [B, 3, N]
            B, _, _ = xyz.shape
            l1_xyz, l1_points = self.sa1(xyz, None)
            l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
            l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
            return l3_points.view(B, 1024)  # [B, 1024]

    return PointNet2Encoder(classifier).cuda().eval()


# ──────────────────────────────────────────────────────────────────────────────
# Pointcloud sampling (identical to depth_to_pointcloud_batch_gpu in observations.py)
# ──────────────────────────────────────────────────────────────────────────────

NUM_POINTS = 1024

def sample_pointcloud(pts: np.ndarray) -> np.ndarray:
    """Sample exactly NUM_POINTS from a pointcloud using exponential-decay weighting."""
    n = len(pts)
    if n == 0:
        raise ValueError("Empty pointcloud.")

    if n == NUM_POINTS:
        return pts  # Already the right size — no sampling needed

    if n > NUM_POINTS:
        # Weighted sampling: points closer to origin are preferred
        dist = np.linalg.norm(pts, axis=1)
        weights = np.exp(-dist / 1.0)
        weights /= weights.sum()
        idx = np.random.choice(n, NUM_POINTS, replace=False, p=weights)
    else:
        # Upsample with replacement
        idx = np.random.choice(n, NUM_POINTS, replace=True)

    return pts[idx]  # (NUM_POINTS, 3)


def load_and_sample(path: str, translate: list = None, save_path: str = None) -> torch.Tensor:
    """Load a PLY/PCD file, optionally translate, sample to NUM_POINTS, return [1, 3, NUM_POINTS] tensor."""
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    print(f"  Loaded {len(pts)} points from {path}")

    if translate is not None:
        pts = pts + np.array(translate, dtype=np.float32)
        print(f"  Translated by {translate}")

    if save_path is not None:
        out = o3d.geometry.PointCloud()
        out.points = o3d.utility.Vector3dVector(pts)
        o3d.io.write_point_cloud(save_path, out)
        print(f"  Saved translated cloud → {save_path}")

    pts = sample_pointcloud(pts)           # (1024, 3)
    t = torch.from_numpy(pts).cuda()       # (1024, 3)
    t = t.unsqueeze(0).permute(0, 2, 1)   # (1, 3, 1024)
    return t


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PointNet2 cosine similarity between two pointclouds.")
    parser.add_argument("cloud1", type=str, help="Path to first pointcloud (PLY/PCD).")
    parser.add_argument("cloud2", type=str, help="Path to second pointcloud (PLY/PCD). Can be same as cloud1.")
    parser.add_argument("--translate", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                        help="Translate cloud2 by (X Y Z) meters before comparison.")
    parser.add_argument("--save_translated", type=str, default=None, metavar="PATH",
                        help="Save the translated cloud2 to this PLY path.")
    parser.add_argument("--no_normalize", action="store_true",
                        help="Skip L2 normalization of features before cosine similarity.")
    args = parser.parse_args()

    print("\n[1] Building PointNet2 encoder...")
    encoder = build_encoder()
    print(f"    Checkpoint: {CKPT_PATH}")

    print("\n[2] Loading pointclouds...")
    pts1 = load_and_sample(args.cloud1)
    pts2 = load_and_sample(args.cloud2, translate=args.translate, save_path=args.save_translated)

    print("\n[3] Extracting features...")
    with torch.no_grad():
        feat1 = encoder(pts1)  # (1, 1024)
        feat2 = encoder(pts2)  # (1, 1024)

    if not args.no_normalize:
        feat1 = F.normalize(feat1, p=2, dim=1)
        feat2 = F.normalize(feat2, p=2, dim=1)
        print("    L2 normalization applied (same as observations.py)")

    similarity = F.cosine_similarity(feat1, feat2, dim=1).item()

    print(f"\n{'=' * 40}")
    print(f"  Cloud 1 : {args.cloud1}")
    print(f"  Cloud 2 : {args.cloud2}" + (f"  (translated {args.translate})" if args.translate else ""))
    print(f"  Feature dim : 1024")
    print(f"  Normalized  : {not args.no_normalize}")
    print(f"  Cosine similarity : {similarity:.6f}")
    print(f"{'=' * 40}\n")


if __name__ == "__main__":
    main()
