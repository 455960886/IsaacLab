import argparse
import os
import torch
import numpy as np
from PIL import Image
import json
import csv

from isaaclab.app import AppLauncher

# local imports
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Validate observation-action pairs for sim-to-real analysis.")
parser.add_argument("--num_episodes", type=int, default=4, help="Number of episodes to run (will record episodes 1 to num_episodes-1).")
parser.add_argument("--output_dir", type=str, default="validation_data", help="Directory to save validation data.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
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
from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
import isaaclab_tasks
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


def main():
    """Validate observation-action pairs."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # get checkpoint path
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # load model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # create output directories
    os.makedirs(args_cli.output_dir, exist_ok=True)
    images_dir = os.path.join(args_cli.output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # create CSV file for observation-action pairs
    csv_file = os.path.join(args_cli.output_dir, "observation_action_pairs.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # adjust column headers based on your action space
        writer.writerow(["episode", "step", "image_filename", "action_0", "action_1", "action_2", "action_3"])

    # tracking variables
    current_episode = 0
    current_step = 0
    total_steps_saved = 0

    # initial reset
    obs, _ = env.get_observations()

    print(f"Starting validation: will run {args_cli.num_episodes} episodes, recording episodes 1-{args_cli.num_episodes-1}")

    while simulation_app.is_running() and current_episode < args_cli.num_episodes:
        with torch.inference_mode():
            # convert observation for policy
            policy_obs = obs.permute(0, 3, 1, 2) if len(obs.shape) == 4 and obs.shape[-1] == 3 else obs

            # get action from policy
            actions = policy(policy_obs)

            # save observation and action (only for episodes >= 1)
            if current_episode >= 1:
                # save observation image
                if torch.is_tensor(obs):
                    img_array = obs[0].cpu().numpy()
                    if img_array.max() <= 1.0:
                        img_array = (img_array * 255).astype(np.uint8)
                    else:
                        img_array = img_array.astype(np.uint8)

                    image = Image.fromarray(img_array)
                    img_filename = f"ep{current_episode:03d}_step{current_step:04d}.png"
                    image.save(os.path.join(images_dir, img_filename))

                # save to CSV
                action_list = actions[0].cpu().numpy().tolist()
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    row = [current_episode, current_step, img_filename] + action_list
                    writer.writerow(row)

                total_steps_saved += 1

                if current_step == 0:
                    print(f"Started recording episode {current_episode}")

            # step environment
            obs, rewards, dones, _ = env.step(actions)
            current_step += 1

            # check for episode completion
            if torch.any(dones):
                if current_episode >= 1:
                    print(f"Finished recording episode {current_episode} ({current_step} steps)")
                else:
                    print(f"Completed episode {current_episode} (not recorded)")

                current_episode += 1
                current_step = 0

    # save all actions to JSON
    print(f"Validation complete:")
    print(f"- Total steps saved: {total_steps_saved}")
    print(f"- Images saved to: {images_dir}")
    print(f"- Data saved to: {csv_file}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()