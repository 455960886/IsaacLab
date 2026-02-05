#!/usr/bin/env python3
# Simple script to test joint movement with pendulum pattern

import argparse
import numpy as np
import torch

from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Test joint movement with pendulum pattern")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-CoarseArm-v0", help="Task name")
parser.add_argument("--joint", type=str, required=True, help="Joint to move (M3 or M4)")
parser.add_argument("--angle", type=float, required=True, help="Swing angle in degrees")
parser.add_argument("--period", type=float, default=2.0, help="Period in seconds")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Enable cameras
args_cli.enable_cameras = True

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Rest of imports
import gymnasium as gym
import isaaclab_tasks  # noqa: F401


def main():
    """Main function."""

    from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
        CoarseArmCubeLiftEnvCfg,
    )

    # Create environment config (same as manual_control.py)
    env_cfg = CoarseArmCubeLiftEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env_cfg.episode_length_s = 1000.0  # Disable timeout

    print(f"[INFO] Creating environment: {args_cli.task}")
    print(f"[INFO] Number of environments: {env_cfg.scene.num_envs}")

    # Create environment (same as manual_control.py)
    env = gym.make(args_cli.task, cfg=env_cfg)
    device = env.unwrapped.device

    # Reset environment (same as manual_control.py)
    obs, _ = env.reset()
    print("[INFO] Environment reset complete")

    # Determine joint index (based on manual_control.py mapping)
    # Index 0 = M0 (Base), Index 1 = M3, Index 2 = M4
    if args_cli.joint == "M3":
        joint_idx = 1
    elif args_cli.joint == "M4":
        joint_idx = 2
    else:
        raise ValueError(f"Joint must be M3 or M4, got {args_cli.joint}")

    print(f"[INFO] Joint: {args_cli.joint} (action index {joint_idx})")
    print(f"[INFO] Swing: ±{args_cli.angle}° (switching every action step)")

    # Calculate action dt based on environment decimation
    action_dt = env.unwrapped.step_dt
    print(f"[INFO] Action dt: {action_dt}s (matches training decimation)")
    print(f"[INFO] Starting pendulum motion...\n")

    # Pendulum state
    angle_deg = args_cli.angle
    angle_rad = np.deg2rad(angle_deg)  # Convert to radians for the action
    current_target_rad = angle_rad
    timestep = 0

    # Main loop (same as manual_control.py)
    try:
        while simulation_app.is_running():
            # Switch direction every step
            current_target_rad = -current_target_rad
            print(f"[{timestep}] Action: {np.rad2deg(current_target_rad):.1f}° ({current_target_rad:.3f} rad)")

            # Create action tensor (4 arm joints + 1 gripper)
            action = torch.zeros(env_cfg.scene.num_envs, 5, device=device)
            action[:, joint_idx] = current_target_rad

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            timestep += 1

            # Update simulation
            simulation_app.update()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        env.close()
        print("[INFO] Environment closed")


if __name__ == "__main__":
    main()
    simulation_app.close()
