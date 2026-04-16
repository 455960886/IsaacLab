import argparse
import os
import torch
import numpy as np
import json

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect depth camera data.")
parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to run.")
parser.add_argument("--output_dir", type=str, default="depth_image_processing/depth_data", help="Directory to save depth data.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg


def main():
    """Collect depth camera data."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=True
    )

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # create output directory
    os.makedirs(args_cli.output_dir, exist_ok=True)

    # tracking variables
    current_episode = 0
    current_step = 0
    total_steps_saved = 0

    # initial reset
    obs, info = env.reset()
    print(f"Initial obs keys: {list(obs.keys())}")
    if 'policy' in obs:
        print(f"Policy obs type: {type(obs['policy'])}, shape: {getattr(obs['policy'], 'shape', 'No shape')}")

    print(f"Starting depth data collection for {args_cli.num_episodes} episodes...")

    while simulation_app.is_running() and current_episode < args_cli.num_episodes:
        with torch.inference_mode():
            # Access depth data from the policy observations
            if 'policy' in obs:
                policy_obs = obs['policy']
                
                # policy_obs should be your depth tensor
                if torch.is_tensor(policy_obs):
                    depth_array = policy_obs[0].cpu().numpy()
                    
                    # Save raw depth data
                    np.save(os.path.join(args_cli.output_dir, f"ep{current_episode:02d}_step{current_step:03d}.npy"), depth_array)
                    total_steps_saved += 1
                    
                    if current_step % 10 == 0:
                        print(f"Episode {current_episode}, Step {current_step}: Saved depth data, shape {depth_array.shape}")
                else:
                    print(f"Policy obs is not a tensor: {type(policy_obs)}")
            else:
                print(f"No policy key found. Available keys: {list(obs.keys())}")

            # Step with random actions
            random_actions = torch.randn(args_cli.num_envs, 5).to(env.unwrapped.device)
            obs, rewards, dones, truncated, info = env.step(random_actions)
                
            current_step += 1

            # check for episode completion
            if torch.any(dones):
                print(f"Completed episode {current_episode} ({current_step} steps)")
                current_episode += 1
                current_step = 0
                if current_episode < args_cli.num_episodes:
                    obs, info = env.reset()

    print(f"\nDepth data collection complete!")
    print(f"- Total depth frames saved: {total_steps_saved}")
    print(f"- Data saved to: {args_cli.output_dir}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()