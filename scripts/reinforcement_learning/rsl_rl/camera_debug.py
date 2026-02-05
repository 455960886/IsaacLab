"""Script to extract camera local position from robot base."""

import argparse
import torch
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Extract camera local position.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Lift-Cube-Franka-IK-Rel-v0", help="Name of the task.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym

from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
    CoarseArmCubeLiftEnvCfg,
)


def extract_camera_local_transform(env):
    """Extract camera's local position and rotation relative to robot base."""

    print("\n" + "="*70)
    print("EXTRACTING CAMERA LOCAL TRANSFORM")
    print("="*70)

    robot = env.scene["robot"]
    camera = env.scene.sensors["depth_camera"]
    device = env.device

    # Test at different base angles to verify transform
    test_angles = [0.0, np.pi/4, np.pi/2, np.pi, -np.pi/2]  # 0°, 45°, 90°, 180°, -90°

    results = []

    for angle in test_angles:
        # Set robot to specific joint configuration
        # M0 (base) at test angle, other joints at default
        joint_pos = torch.tensor([[
            angle,      # M0 - base rotation
            1.57,       # M1
            3.9,        # M3 (skip M2 based on your config)
            1.4,        # M4
            0.0,        # M5
            0.0,        # M6_1
            0.0         # M6_2
        ]], device=device)

        # Write joint state and update (with zero velocities)
        joint_vel = torch.zeros_like(joint_pos)
        env_ids = torch.tensor([0], device=device)
        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # Step simulation multiple times to let it settle
        for _ in range(10):
            env.sim.step()
        env.scene.update(env.step_dt)

        # Get camera and base positions
        camera_pos_w = camera.data.pos_w[0].cpu().numpy()
        camera_quat_w_ros = camera.data.quat_w_ros[0].cpu().numpy()  # (x, y, z, w)
        base_pos_w = robot.data.root_pos_w[0].cpu().numpy()

        # Calculate relative position
        relative_pos = camera_pos_w - base_pos_w

        # Store results
        results.append({
            'angle_deg': np.degrees(angle),
            'angle_rad': angle,
            'camera_pos_w': camera_pos_w,
            'base_pos_w': base_pos_w,
            'relative_pos': relative_pos,
            'camera_quat_ros': camera_quat_w_ros
        })

        print(f"\nBase angle: {np.degrees(angle):6.1f}°")
        print(f"  Camera world pos:    {camera_pos_w}")
        print(f"  Base world pos:      {base_pos_w}")
        print(f"  Relative position:   {relative_pos}")
        print(f"  Camera quat (ROS):   {camera_quat_w_ros}")

    # Calculate camera local position by inverse rotation
    # When base is at 0°, the relative position IS the local position
    camera_local_pos_at_zero = results[0]['relative_pos']

    print("\n" + "="*70)
    print("CAMERA LOCAL POSITION (relative to base at 0°):")
    print(f"  X: {camera_local_pos_at_zero[0]:.6f}")
    print(f"  Y: {camera_local_pos_at_zero[1]:.6f}")
    print(f"  Z: {camera_local_pos_at_zero[2]:.6f}")
    print("="*70)

    # Verify: Check if rotating this local position matches observed positions
    print("\nVERIFICATION: Rotating local position should match observed relative positions")
    print("-"*70)

    for result in results[1:]:  # Skip 0° as that's our reference
        angle = result['angle_rad']

        # Rotate local position by base angle (Z-axis rotation)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        R_z = np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1]
        ])

        predicted_relative = R_z @ camera_local_pos_at_zero
        observed_relative = result['relative_pos']
        error = np.linalg.norm(predicted_relative - observed_relative)

        print(f"\nBase angle: {result['angle_deg']:6.1f}°")
        print(f"  Predicted relative pos: {predicted_relative}")
        print(f"  Observed relative pos:  {observed_relative}")
        print(f"  Error: {error:.6f} m")

        if error < 0.001:
            print(f"  ✓ GOOD - Error < 1mm")
        else:
            print(f"  ✗ WARNING - Error >= 1mm, camera may have additional transforms")

    print("\n" + "="*70)
    print("COPY THESE VALUES TO YOUR CODE:")
    print("="*70)
    print(f"camera_local_pos = torch.tensor([{camera_local_pos_at_zero[0]:.6f}, "
          f"{camera_local_pos_at_zero[1]:.6f}, {camera_local_pos_at_zero[2]:.6f}], "
          f"device=device)")
    print("="*70 + "\n")

    return camera_local_pos_at_zero


def main():
    # Create environment
    env_cfg = CoarseArmCubeLiftEnvCfg()
    env_cfg.scene.num_envs = 1  # Only need 1 environment
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else "cuda:0"

    print(f"[INFO] Creating environment: {args_cli.task}")

    env = gym.make(args_cli.task, cfg=env_cfg)

    # Reset environment
    obs, _ = env.reset()
    print("[INFO] Environment reset complete\n")

    # Extract camera transform
    try:
        camera_local_pos = extract_camera_local_transform(env.unwrapped)

        print("\n[INFO] Extraction complete!")
        print("[INFO] Use the printed values in your gripper_transform_corrected.py file")

    except Exception as e:
        print(f"\n[ERROR] Failed to extract camera transform: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Close environment
        env.close()
        print("\n[INFO] Environment closed")


if __name__ == "__main__":
    main()
    simulation_app.close()