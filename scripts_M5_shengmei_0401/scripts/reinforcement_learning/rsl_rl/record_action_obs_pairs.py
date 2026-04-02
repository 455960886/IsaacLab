# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to record action-observation pairs (RGB images and point clouds) during policy playback."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Record action-observation pairs with RGB images and point clouds.")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to record.")
parser.add_argument("--output_dir", type=str, default="recorded_data", help="Directory to save recorded data.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint from Nucleus.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
import numpy as np
import csv
import open3d as o3d
from PIL import Image

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


def save_rgb_image(rgb_tensor, filepath):
    """Save RGB image tensor to PNG file.

    Args:
        rgb_tensor: Torch tensor of shape (H, W, 3) with values in [0, 255] or [0, 1]
        filepath: Output file path
    """
    img_array = rgb_tensor.cpu().numpy()

    # Normalize to [0, 255] if needed
    if img_array.max() <= 1.0:
        img_array = (img_array * 255).astype(np.uint8)
    else:
        img_array = img_array.astype(np.uint8)

    image = Image.fromarray(img_array)
    image.save(filepath)


def save_point_cloud_ply(points_tensor, filepath):
    """Save point cloud tensor to PLY file.

    Args:
        points_tensor: Torch tensor of shape (N, 3) with XYZ coordinates
        filepath: Output file path
    """
    points_np = points_tensor.cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    o3d.io.write_point_cloud(filepath, pcd)


def main():
    """Record action-observation pairs."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # get checkpoint path
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # load model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # create output directory structure
    os.makedirs(args_cli.output_dir, exist_ok=True)
    rgb_dir = os.path.join(args_cli.output_dir, "rgb_images")
    pcd_dir = os.path.join(args_cli.output_dir, "point_clouds")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(pcd_dir, exist_ok=True)

    # create CSV file for action-observation metadata
    csv_file = os.path.join(args_cli.output_dir, "action_obs_data.csv")
    csv_headers = ["episode", "step", "rgb_filename", "pcd_filename"] + [f"action_{i}" for i in range(5)]

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    # tracking variables
    current_episode = 0
    current_step = 0
    total_steps_saved = 0

    # initial reset
    obs, _ = env.get_observations()

    print(f"\n{'='*60}")
    print(f"Starting recording: {args_cli.num_episodes} episodes")
    print(f"Output directory: {args_cli.output_dir}")
    print(f"{'='*60}\n")

    while simulation_app.is_running() and current_episode < args_cli.num_episodes:
        with torch.inference_mode():
            # get action from policy
            actions = policy(obs)

            # Extract raw RGB image from gripper camera
            try:
                # The gripper camera is typically "tiled_camera" in the lift task
                rgb_sensor = env.unwrapped.scene.sensors.get("tiled_camera")
                if rgb_sensor is not None:
                    rgb_image = rgb_sensor.data.output["rgb"][0]  # [0] for first env
                else:
                    print("[WARNING] Could not find 'tiled_camera' sensor")
                    rgb_image = None
            except Exception as e:
                print(f"[WARNING] Error accessing RGB image: {e}")
                rgb_image = None

            # Extract point cloud from cache
            # The point cloud is cached in env.point_cloud_cache by the image_features observation term
            try:
                if hasattr(env.unwrapped, 'point_cloud_cache'):
                    point_cloud = env.unwrapped.point_cloud_cache[0]  # [0] for first env
                else:
                    print("[WARNING] No point_cloud_cache found in environment")
                    point_cloud = None
            except Exception as e:
                print(f"[WARNING] Error accessing point cloud: {e}")
                point_cloud = None

            # Save data for this step
            rgb_filename = f"ep{current_episode:03d}_step{current_step:04d}_rgb.png"
            pcd_filename = f"ep{current_episode:03d}_step{current_step:04d}_pcd.ply"

            # Save RGB image
            if rgb_image is not None:
                save_rgb_image(rgb_image, os.path.join(rgb_dir, rgb_filename))

            # Save point cloud
            if point_cloud is not None:
                save_point_cloud_ply(point_cloud, os.path.join(pcd_dir, pcd_filename))

            # Save to CSV
            action_list = actions[0].cpu().numpy().tolist()
            with open(csv_file, 'a', newline='') as f:
                writer = csv.writer(f)
                row = [current_episode, current_step, rgb_filename, pcd_filename] + action_list
                writer.writerow(row)

            total_steps_saved += 1

            if current_step == 0:
                print(f"[Episode {current_episode}] Started recording...")

            # step environment
            obs, rewards, dones, _ = env.step(actions)
            current_step += 1

            # check for episode completion
            if torch.any(dones):
                print(f"[Episode {current_episode}] Completed - {current_step} steps recorded")
                current_episode += 1
                current_step = 0

    # Final summary
    print(f"\n{'='*60}")
    print(f"Recording complete!")
    print(f"- Total episodes: {current_episode}")
    print(f"- Total steps saved: {total_steps_saved}")
    print(f"- RGB images saved to: {rgb_dir}")
    print(f"- Point clouds saved to: {pcd_dir}")
    print(f"- Action-observation data saved to: {csv_file}")
    print(f"{'='*60}\n")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
