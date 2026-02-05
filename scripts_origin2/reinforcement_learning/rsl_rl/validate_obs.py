"""Script to verify RGB + Point Cloud observations."""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify RGB + Point Cloud observations.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
# parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on.")
parser.add_argument("--save_pcd", action="store_true", help="Save point cloud to file.")
parser.add_argument("--output_dir", type=str, default="./pcd_output", help="Directory to save point clouds.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import numpy as np
import os
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def save_point_cloud_to_ply(points, filename, colors=None):
    """Save point cloud to PLY format.

    Args:
        points: (N, 3) numpy array of XYZ coordinates
        filename: Output filename
        colors: Optional (N, 3) numpy array of RGB colors (0-255)
    """
    import struct

    # Create output directory
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Remove invalid points (NaN, Inf)
    valid_mask = np.all(np.isfinite(points), axis=1)
    points = points[valid_mask]

    if colors is not None:
        colors = colors[valid_mask]

    print(f"Saving {len(points)} valid points to {filename}")

    # Write PLY file
    with open(filename, 'wb') as f:
        # Header
        header = f"""ply
format binary_little_endian 1.0
element vertex {len(points)}
property float x
property float y
property float z
"""
        if colors is not None:
            header += """property uchar red
property uchar green
property uchar blue
"""
        header += "end_header\n"

        f.write(header.encode('ascii'))

        # Vertex data
        for i in range(len(points)):
            f.write(struct.pack('fff', points[i, 0], points[i, 1], points[i, 2]))
            if colors is not None:
                f.write(struct.pack('BBB', 
                    int(colors[i, 0]), 
                    int(colors[i, 1]), 
                    int(colors[i, 2])))

    print(f"✓ Point cloud saved to {filename}")


def save_rgb_image(image, filename):
    """Save RGB image to PNG."""
    from PIL import Image

    # Create output directory
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Convert to numpy and scale to 0-255
    if torch.is_tensor(image):
        image = image.cpu().numpy()

    # If normalized [0, 1], scale to [0, 255]
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    # Save using PIL
    img_pil = Image.fromarray(image)
    img_pil.save(filename)
    print(f"✓ RGB image saved to {filename}")


def main():
    """Verify observation shapes and data types."""

    print(f"\n{'='*80}")
    print(f"Task: {args_cli.task}")
    print(f"Num Envs: {args_cli.num_envs}")
    print(f"{'='*80}\n")

    # Parse environment config
    env_cfg = parse_env_cfg(
        args_cli.task, 
        device=args_cli.device, 
        num_envs=args_cli.num_envs, 
        use_fabric=not args_cli.disable_fabric
    )

    # Create environment with config
    env = gym.make(args_cli.task, cfg=env_cfg)

    print("Environment created successfully!")

    # Get action dimension correctly
    if hasattr(env.unwrapped, 'num_actions'):
        num_actions = env.unwrapped.num_actions
    else:
        num_actions = env.action_space.shape[-1] if len(env.action_space.shape) > 0 else env.action_space.n

    print(f"Action space: {env.action_space}")
    print(f"Number of actions: {num_actions}\n")

    # Reset environment
    print("Resetting environment...")
    obs_dict, _ = env.reset()
    print("Reset successful!\n")

    # Check observation structure
    print(f"{'Observation Structure':-^80}")
    print(f"Type: {type(obs_dict)}")

    if isinstance(obs_dict, dict):
        print(f"Top-level keys: {list(obs_dict.keys())}\n")

        if "policy" in obs_dict:
            policy_obs = obs_dict["policy"]
            print(f"Policy Observations Type: {type(policy_obs)}")

            if isinstance(policy_obs, dict):
                print(f"Policy keys: {list(policy_obs.keys())}\n")

                print(f"{'Detailed Observation Info':-^80}\n")
                for key, value in policy_obs.items():
                    print(f"{key}:")
                    print(f"  Shape:  {value.shape}")
                    print(f"  Dtype:  {value.dtype}")
                    print(f"  Device: {value.device}")

                    if torch.is_floating_point(value):
                        has_nan = torch.isnan(value).any().item()
                        has_inf = torch.isinf(value).any().item()
                        print(f"  Has NaN: {has_nan}")
                        print(f"  Has Inf: {has_inf}")

                        if not has_nan and not has_inf:
                            print(f"  Min: {value.min().item():.3f}")
                            print(f"  Max: {value.max().item():.3f}")
                            print(f"  Mean: {value.mean().item():.3f}")
                    else:
                        print(f"  Min: {value.min().item()}")
                        print(f"  Max: {value.max().item()}")
                    print()

                # Expected shapes (depth camera: 400x300)
                expected_pcd_shape = (args_cli.num_envs, 120000, 3)  # 300*400 points
                expected_rgb_shape = (args_cli.num_envs, 300, 400, 3)  # H, W, C

                print(f"{'Shape Verification':-^80}\n")

                if "point_cloud" in policy_obs:
                    actual = policy_obs["point_cloud"].shape
                    match = actual == expected_pcd_shape
                    status = "✓ CORRECT" if match else f"✗ MISMATCH (Expected: {expected_pcd_shape})"
                    print(f"Point Cloud: {actual}")
                    print(f"  Status: {status}\n")
                else:
                    print(f"Point Cloud: ✗ NOT FOUND\n")

                if "rgb_image" in policy_obs:
                    actual = policy_obs["rgb_image"].shape
                    match = actual == expected_rgb_shape
                    status = "✓ CORRECT" if match else f"✗ MISMATCH (Expected: {expected_rgb_shape})"
                    print(f"RGB Image: {actual}")
                    print(f"  Status: {status}\n")
                else:
                    print(f"RGB Image: ✗ NOT FOUND\n")

                # Save point cloud and RGB image if requested
                if args_cli.save_pcd and "point_cloud" in policy_obs:
                    print(f"{'Saving Point Cloud and RGB Image':-^80}\n")

                    # Save for each environment (usually just env 0)
                    for env_idx in range(min(args_cli.num_envs, 3)):  # Save max 3 envs
                        pcd = policy_obs["point_cloud"][env_idx].cpu().numpy()  # (120000, 3)

                        # Save point cloud
                        pcd_filename = os.path.join(args_cli.output_dir, f"pointcloud_env{env_idx}.ply")
                        save_point_cloud_to_ply(pcd, pcd_filename)

                        # Save RGB image
                        if "rgb_image" in policy_obs:
                            rgb = policy_obs["rgb_image"][env_idx].cpu().numpy()  # (300, 400, 3)
                            rgb_filename = os.path.join(args_cli.output_dir, f"rgb_image_env{env_idx}.png")
                            save_rgb_image(rgb, rgb_filename)

                    print()

            else:
                print(f"\n✗ ERROR: Policy obs is a concatenated tensor: {policy_obs.shape}")
                print(f"  Set concatenate_terms=False in RgbPcdObservationCfg!\n")
        else:
            print(f"\n✗ ERROR: 'policy' key not found in observations!\n")
    else:
        print(f"\n✗ ERROR: Observations is a single tensor: {obs_dict.shape}\n")

    # Take test steps
    print(f"{'Taking Test Steps':-^80}\n")
    for step in range(20):
        actions = torch.zeros(args_cli.num_envs, num_actions, device=env.unwrapped.device)
        obs_dict, rewards, terminated, truncated, info = env.step(actions)

        print(f"Step {step + 1}: ✓")
        if isinstance(obs_dict, dict) and "policy" in obs_dict and isinstance(obs_dict["policy"], dict):
            for key in ["point_cloud", "rgb_image"]:
                if key in obs_dict["policy"]:
                    val = obs_dict["policy"][key]
                    print(f"  {key}: shape={val.shape}, dtype={val.dtype}")

        # Save point cloud from step 1
        if args_cli.save_pcd and isinstance(obs_dict, dict) and "policy" in obs_dict:
            if "point_cloud" in obs_dict["policy"]:
                pcd = obs_dict["policy"]["point_cloud"][0].cpu().numpy()
                pcd_filename = os.path.join(args_cli.output_dir, f"pointcloud_step{step+1}.ply")
                save_point_cloud_to_ply(pcd, pcd_filename)

                if "rgb_image" in obs_dict["policy"]:
                    rgb = obs_dict["policy"]["rgb_image"][0].cpu().numpy()
                    rgb_filename = os.path.join(args_cli.output_dir, f"rgb_image_step{step+1}.png")
                    save_rgb_image(rgb, rgb_filename)

    print(f"\n{'='*80}")
    print("✓ Verification Complete!")
    if args_cli.save_pcd:
        print(f"✓ Point clouds saved to: {args_cli.output_dir}")
    print(f"{'='*80}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()