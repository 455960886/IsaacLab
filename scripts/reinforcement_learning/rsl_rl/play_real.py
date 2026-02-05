#!/usr/bin/env python3
"""
Minimal script to test exported policy with real robot observations.
Replicates ResNet18 + PointNet encoding pipeline from training.
"""

import torch
import cv2
import numpy as np
import open3d as o3d
from torchvision import models
import torch.nn as nn

# Import PointNet2 the same way as observations.py
from isaaclab.pointnet.log.classification.pointnet2_ssg_wo_normals.pointnet2_cls_ssg import get_model as PointNet2ClsMsg

# ===== EDIT THESE PATHS =====
IMAGE_PATH = "/home/roborock/real_test/20251223-164325.jpg"
PCD_PATH = "/home/roborock/real_test/frame_sampled.ply"
POLICY_PATH = "/home/roborock/IsaacLab/logs/rsl_rl/coarse_arm_lift/2025-12-22_20-27-34/exported2/12-23.pt"
POINTNET_CKPT = "/home/roborock/IsaacLab/best_model.pth"
DEVICE = "cuda"  # "cpu" or "cuda"
CROP_TOP = 120
# ============================


class PointNet2Encoder(nn.Module):
    """Feature extraction only (no classification head)"""
    def __init__(self, base_model):
        super().__init__()
        self.sa1 = base_model.sa1
        self.sa2 = base_model.sa2
        self.sa3 = base_model.sa3

    def forward(self, xyz):
        B, _, _ = xyz.shape
        norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        features = l3_points.view(B, 1024)
        return features


def load_resnet18_encoder(device):
    """Load ResNet18 feature extractor (removes classification head)"""
    model = models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
    model = nn.Sequential(*list(model.children())[:-1])  # Remove FC layer
    model = model.to(device).eval()
    return model


def load_pointnet_encoder(ckpt_path, device):
    """Load PointNet2 feature extractor"""
    # Use get_model function (same as observations.py)
    classifier = PointNet2ClsMsg(num_class=40, normal_channel=False).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    classifier.load_state_dict(checkpoint['model_state_dict'], strict=False)
    classifier.eval()

    encoder = PointNet2Encoder(classifier).to(device).eval()
    return encoder


def encode_image(img_path, resnet_model, device, crop_top=120):
    """Load and encode image with ResNet18 → 512 features"""
    # Load image
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Crop top (matching training)
    if crop_top > 0:
        img_rgb = img_rgb[crop_top:, :, :]

    # Convert to tensor and normalize [0, 1]
    img_float = img_rgb.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(device)

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    img_norm = (img_tensor - mean) / std

    # Extract features
    with torch.no_grad():
        features = resnet_model(img_norm)  # [1, 512, 1, 1]
        features = features.view(1, 512)    # [1, 512]

    return features


def encode_pointcloud(pcd_path, pointnet_model, device):
    """Load and encode point cloud with PointNet → 1024 features"""
    # Load PLY
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points, dtype=np.float32)

    # Ensure exactly 1024 points
    if len(points) != 1024:
        if len(points) > 1024:
            indices = np.random.choice(len(points), 1024, replace=False)
            points = points[indices]
        else:
            padding = np.zeros((1024 - len(points), 3), dtype=np.float32)
            points = np.vstack([points, padding])

    # Convert to tensor (1, 3, 1024) - note: PointNet expects (B, C, N)
    pcd_tensor = torch.from_numpy(points.T).unsqueeze(0).to(device)  # [1, 3, 1024]

    # Extract features
    with torch.no_grad():
        features = pointnet_model(pcd_tensor)  # [1, 1024]

    return features


def main():
    device = torch.device(DEVICE)

    print("Loading encoders...")
    resnet_encoder = load_resnet18_encoder(device)
    pointnet_encoder = load_pointnet_encoder(POINTNET_CKPT, device)

    print("Encoding image...")
    img_features = encode_image(IMAGE_PATH, resnet_encoder, device, CROP_TOP)

    print("Encoding point cloud...")
    pcd_features = encode_pointcloud(PCD_PATH, pointnet_encoder, device)

    # Normalize features (L2 norm)
    img_feat_norm = torch.nn.functional.normalize(img_features, p=2, dim=1)
    pc_feat_norm = torch.nn.functional.normalize(pcd_features, p=2, dim=1)

    # Concatenate: [1, 512] + [1, 1024] = [1, 1536]
    obs = torch.cat([img_feat_norm, pc_feat_norm], dim=1)

    print(f"Observation shape: {obs.shape}")  # Should be [1, 1536]

    # Load policy and inference
    print("Loading policy...")
    policy = torch.jit.load(POLICY_PATH, map_location=device)
    policy.eval()

    with torch.no_grad():
        actions = policy(obs)

    # Output
    print(f"\nActions (radians): {actions[0].cpu().numpy()}")
    print(f"Actions (degrees): {(actions[0].cpu() * 180.0 / np.pi).numpy()}")


if __name__ == "__main__":
    main()
