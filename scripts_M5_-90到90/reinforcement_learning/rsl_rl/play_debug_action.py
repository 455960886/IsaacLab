# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL with debug saving."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip
import cv2  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with RSL-RL and save debug data.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--debug_output_dir", type=str, default="debug_data", help="Directory to save debug data.")
parser.add_argument("--save_interval", type=int, default=1, help="Save data every N steps (default: 1 = every step).")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import numpy as np
import csv
import open3d as o3d
from PIL import Image

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)


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
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # specify directory for logging experiments
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

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported2")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy828.onnx"
    )

    # Setup debug saving directories
    debug_dir = args_cli.debug_output_dir
    rgb_dir = os.path.join(debug_dir, "rgb_images")
    pcd_dir = os.path.join(debug_dir, "point_clouds")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(pcd_dir, exist_ok=True)

    # Create CSV file for action-observation metadata
    csv_file = os.path.join(debug_dir, "action_data.csv")
    # Determine number of actions (assume first env)
    num_actions = env.unwrapped.action_manager.action.shape[1]
    csv_headers = ["step", "rgb_filename", "pcd_filename"] + [f"action_{i}" for i in range(num_actions)]

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    print(f"\n{'='*60}")
    print(f"Debug mode enabled - saving data to: {debug_dir}")
    print(f"Save interval: every {args_cli.save_interval} step(s)")
    print(f"{'='*60}\n")

    dt = env.unwrapped.step_dt

    # reset environment
    obs, _ = env.get_observations()
    timestep = 0
    saved_count = 0

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            print(actions/3.14*180)

            # Save debug data at specified interval
            if timestep % args_cli.save_interval == 0:
                # Extract raw RGB image from gripper camera
                try:
                    rgb_sensor = env.unwrapped.scene.sensors.get("tiled_camera")
                    if rgb_sensor is not None:
                        rgb_image = rgb_sensor.data.output["rgb"][0]  # [0] for first env
                    else:
                        rgb_image = None
                except Exception as e:
                    print(f"[WARNING] Error accessing RGB image: {e}")
                    rgb_image = None

                # Extract point cloud from cache
                try:
                    if hasattr(env.unwrapped, 'point_cloud_cache'):
                        point_cloud = env.unwrapped.point_cloud_cache[0]  # [0] for first env
                    else:
                        point_cloud = None
                except Exception as e:
                    print(f"[WARNING] Error accessing point cloud: {e}")
                    point_cloud = None

                # Save files
                rgb_filename = f"step_{timestep:06d}_rgb.png"
                pcd_filename = f"step_{timestep:06d}_pcd.ply"

                if rgb_image is not None:
                    save_rgb_image(rgb_image, os.path.join(rgb_dir, rgb_filename))

                if point_cloud is not None:
                    save_point_cloud_ply(point_cloud, os.path.join(pcd_dir, pcd_filename))

                # Save to CSV
                action_list = actions[0].cpu().numpy().tolist()
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    row = [timestep, rgb_filename, pcd_filename] + action_list
                    writer.writerow(row)

                saved_count += 1
                if saved_count % 10 == 0:
                    print(f"[DEBUG] Saved {saved_count} samples (step {timestep})")

            # env stepping
            obs, _, _, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        else:
            timestep += 1

        # time delay for real-time evaluation
        # sleep_time = dt - (time.time() - start_time)
        # if args_cli.real_time and sleep_time > 0:
        #     time.sleep(sleep_time)

    # Final summary
    print(f"\n{'='*60}")
    print(f"Debug data saved!")
    print(f"- Total steps: {timestep}")
    print(f"- Samples saved: {saved_count}")
    print(f"- Output directory: {debug_dir}")
    print(f"{'='*60}\n")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
