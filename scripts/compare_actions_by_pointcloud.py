"""Compare policy actions produced by different point clouds with a fixed RGB image.

Replicates the observation pipeline from image_features in observations.py.
Supports both:
  - Raw training checkpoints (model_XXXX.pt)  ← automatically detected
  - Exported TorchScript policies (policy.pt)

Observation layout (inferred from checkpoint obs_norm_state_dict shape):
  ResNet18 (512) + PointNet2 (1024) + joint_pos (remainder) = obs_dim

Usage:
    python scripts/compare_actions_by_pointcloud.py \\
        --image  /path/to/image.jpg \\
        --clouds cloud1.ply cloud2.ply cloud3.ply \\
        --policy /path/to/model_4040.pt \\
        --joint_pos 3.8 1.4 0.0 0.65 -0.65
"""

import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
import open3d as o3d
import cv2
from torchvision import models

# ── PointNet2 paths (same as observations.py) ─────────────────────────────────
POINTNET_SYS_PATH = (
    "/home/roborock/gitlab5/drl_manipulation/source/isaaclab/isaaclab"
    "/pointnet/log/classification/pointnet2_ssg_wo_normals"
)
POINTNET_CKPT = "/home/roborock/IsaacLab/best_model.pth"
CROP_TOP = 120


# ── Encoders ──────────────────────────────────────────────────────────────────

def build_resnet18(device: str) -> nn.Module:
    model = models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
    model = nn.Sequential(*list(model.children())[:-1])
    return model.to(device).eval()


def build_pointnet2(device: str) -> nn.Module:
    sys.path.insert(0, POINTNET_SYS_PATH)
    from pointnet2_cls_ssg import get_model as PointNet2ClsMsg

    classifier = PointNet2ClsMsg(num_class=40, normal_channel=False).to(device)
    ckpt = torch.load(POINTNET_CKPT, map_location=device, weights_only=False)
    classifier.load_state_dict(ckpt["model_state_dict"], strict=False)
    classifier.eval()

    class PointNet2Encoder(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.sa1 = base.sa1
            self.sa2 = base.sa2
            self.sa3 = base.sa3

        def forward(self, xyz):
            B = xyz.shape[0]
            l1_xyz, l1_pts = self.sa1(xyz, None)
            l2_xyz, l2_pts = self.sa2(l1_xyz, l1_pts)
            l3_xyz, l3_pts = self.sa3(l2_xyz, l2_pts)
            return l3_pts.view(B, 1024)

    return PointNet2Encoder(classifier).to(device).eval()


# ── Policy loading ─────────────────────────────────────────────────────────────

class EmpiricalNormalizer(nn.Module):
    """Mirrors RSL-RL's EmpiricalNormalization."""
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std.clamp(min=1e-6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


def _build_mlp(layer_sizes: list) -> nn.Sequential:
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


def load_policy(path: str):
    """
    Load policy from either a raw training checkpoint or an exported TorchScript file.
    Returns a callable (obs_tensor -> action_tensor) on CPU.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        # ── Raw training checkpoint ────────────────────────────────────────────
        sd = ckpt["model_state_dict"]
        nsd = ckpt["obs_norm_state_dict"]

        # Infer layer sizes from weight shapes
        actor_weights = [(k, v) for k, v in sd.items() if k.startswith("actor.") and k.endswith(".weight")]
        actor_weights.sort(key=lambda x: int(x[0].split(".")[1]))
        sizes = [actor_weights[0][1].shape[1]] + [w.shape[0] for _, w in actor_weights]

        actor = _build_mlp(sizes)
        # Load only actor weights (strip 'actor.' prefix is already absent in sd keys)
        actor_sd = {k[len("actor."):]: v for k, v in sd.items() if k.startswith("actor.")}
        actor.load_state_dict(actor_sd)
        actor.eval()

        normalizer = EmpiricalNormalizer(nsd["_mean"], nsd["_std"])

        obs_dim = sizes[0]
        print(f"    Loaded raw checkpoint  |  obs_dim={obs_dim}  |  arch={sizes}")

        def run_policy(obs: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return actor(normalizer(obs))

        return run_policy, obs_dim

    else:
        # ── TorchScript exported policy ────────────────────────────────────────
        policy = torch.jit.load(path, map_location="cpu").eval()

        # Infer obs_dim from normalizer buffer
        obs_dim = None
        for name, buf in policy.named_buffers():
            if "mean" in name.lower():
                obs_dim = buf.shape[-1]
                break

        print(f"    Loaded TorchScript policy  |  obs_dim={obs_dim}")

        def run_policy(obs: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                return policy(obs)

        return run_policy, obs_dim


# ── Feature extraction ────────────────────────────────────────────────────────

def encode_image(path: str, resnet: nn.Module, device: str) -> torch.Tensor:
    """RGB image → 512-dim ResNet18 features (no L2 norm, matches training)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[CROP_TOP:, :, :]
    
    t = torch.from_numpy(img.astype(np.float32) / 255.0)
    t = t.permute(2, 0, 1).unsqueeze(0).to(device)                # [1, 3, H, W]

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    t = (t - mean) / std

    with torch.no_grad():
        feat = resnet(t)                                           # [1, 512, 1, 1]
    return feat.view(1, 512)


def encode_pointcloud(path: str, pnet: nn.Module, device: str) -> torch.Tensor:
    """PLY → 1024-dim PointNet2 features (no L2 norm)."""
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    n = len(pts)
    if n == 0:
        raise ValueError(f"Empty point cloud: {path}")

    if n == 1024:
        pass
    elif n > 1024:
        idx = np.random.choice(n, 1024, replace=False)
        pts = pts[idx]
    else:
        idx = np.random.choice(n, 1024, replace=True)
        pts = pts[idx]

    t = torch.from_numpy(pts).to(device)
    xyz = t.unsqueeze(0).permute(0, 2, 1).contiguous()   # [1, 3, 1024]
    with torch.no_grad():
        feat = pnet(xyz)                                   # [1, 1024]
    return feat


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare policy actions across different point clouds (fixed RGB)."
    )
    parser.add_argument("--image",  required=True, help="Path to RGB image.")
    parser.add_argument("--clouds", required=True, nargs="+", help="PLY point cloud files.")
    parser.add_argument("--policy", required=True, help="Training checkpoint or exported policy (.pt).")
    parser.add_argument(
        "--joint_pos", type=float, nargs="+", default=None,
        help="Joint positions appended to observation. Number of values must fill obs_dim - 1536.",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = args.device

    print("\n[1] Loading policy...")
    run_policy, obs_dim = load_policy(args.policy)

    joint_dim = obs_dim - 1536  # obs_dim - (512 img + 1024 pc)
    print(f"    joint_pos dim expected: {joint_dim}")

    if args.joint_pos is None:
        joint_pos_vals = [0.0] * joint_dim
        print(f"    --joint_pos not provided, defaulting to zeros ({joint_dim} values)")
    else:
        joint_pos_vals = args.joint_pos
        if len(joint_pos_vals) != joint_dim:
            raise ValueError(
                f"--joint_pos requires {joint_dim} values for this checkpoint "
                f"(got {len(joint_pos_vals)})"
            )

    joint_pos = torch.tensor(joint_pos_vals, dtype=torch.float32).unsqueeze(0)  # [1, joint_dim]

    print("\n[2] Loading encoders...")
    resnet = build_resnet18(device)
    pnet   = build_pointnet2(device)

    print("\n[3] Encoding RGB image (fixed)...")
    img_feat = encode_image(args.image, resnet, device).cpu()   # [1, 512]

    print(f"\n[4] Running policy for {len(args.clouds)} point cloud(s)...\n")
    results = []
    for path in args.clouds:
        pc_feat = encode_pointcloud(path, pnet, device).cpu()   # [1, 1024]
        obs = torch.cat([img_feat, pc_feat, joint_pos], dim=-1) # [1, obs_dim]
        action = run_policy(obs)
        results.append((path, action))
        print(f"  {path.split('/')[-1]}: action = {action[0].numpy()}")

    # ── Comparison table ───────────────────────────────────────────────────────
    num_actions = results[0][1].shape[1]
    action_names = [f"a{i}" for i in range(num_actions)]
    col_w = 12
    label_w = 45

    print(f"\n{'=' * (label_w + col_w * num_actions)}")
    print(f"{'Point Cloud':<{label_w}}" + "".join(f"{n:>{col_w}}" for n in action_names))
    print(f"{'=' * (label_w + col_w * num_actions)}")
    for path, action in results:
        label = path.split("/")[-1][:label_w - 1]
        row = f"{label:<{label_w}}" + "".join(f"{v:>{col_w}.4f}" for v in action[0].numpy())
        print(row)
    print(f"{'=' * (label_w + col_w * num_actions)}")

    if len(results) > 1:
        print(f"\n[Δ vs '{results[0][0].split('/')[-1]}']")
        base = results[0][1][0]
        for path, action in results[1:]:
            label = path.split("/")[-1][:label_w - 1]
            diff = (action[0] - base).numpy()
            row = f"{label:<{label_w}}" + "".join(f"{d:>+{col_w}.4f}" for d in diff)
            print(row)
        print()


if __name__ == "__main__":
    main()
