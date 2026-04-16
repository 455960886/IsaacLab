import sys
print("sys.argv =", sys.argv)

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

import warnings                                                                                                                                                                                                               
warnings.filterwarnings("ignore", message=".*Ill-formed SdfPath.*")        

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
# 处理命令行参数
# 输入--help会显示description的内容
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2001, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--bc_checkpoint", type=str, default=None, help="Path to a BC pre-trained checkpoint to warm-start PPO actor weights.")
# 是否用分布式训练（多 GPU 或多机）
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
# 加入 RSL-RL 库中定义的一些标准训练参数（比如 policy 网络结构、优化器配置等）。
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
# 添加关于模拟器（Isaac Sim）启动的参数，比如是否开启 GUI，是否启用相机等。
AppLauncher.add_app_launcher_args(parser)

# args_cli是所有你通过命令行显式指定的参数，存储为一个 Namespace
# hydra_args 是多余的参数，后续会传给 Hydra（一个高级配置系统）
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
try:
    installed_version = metadata.version("rsl-rl-lib")
except metadata.PackageNotFoundError:
    try:
        installed_version = metadata.version("rsl_rl")
    except metadata.PackageNotFoundError:
        installed_version = RSL_RL_VERSION  # metadata unavailable; skip version check
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import gymnasium as gym
import git
import os
import pathlib
import torch
from datetime import datetime

# RSL-RL 的训练循环逻辑（rsl_rl/runners/on_policy_runner.py）
from rsl_rl.runners import OnPolicyRunner
import rsl_rl.runners.on_policy_runner as rsl_on_policy_runner

from positive_m5_actor_critic import PositiveM5ActorCritic

rsl_on_policy_runner.PositiveM5ActorCritic = PositiveM5ActorCritic

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)


def _store_code_state_safe(logdir: str, repositories: list[str]) -> list[str]:
    """Store git state without failing on surrogate bytes returned by GitPython."""
    git_log_dir = os.path.join(logdir, "git")
    os.makedirs(git_log_dir, exist_ok=True)
    file_paths = []

    for repository_file_path in repositories:
        try:
            repo = git.Repo(repository_file_path, search_parent_directories=True)
            commit_tree = repo.head.commit.tree
        except Exception:
            print(f"Could not find git repository in {repository_file_path}. Skipping.")
            continue

        repo_name = pathlib.Path(repo.working_dir).name
        diff_file_name = os.path.join(git_log_dir, f"{repo_name}.diff")
        if os.path.isfile(diff_file_name):
            continue

        print(f"Storing git diff for '{repo_name}' in: {diff_file_name}")
        content = f"--- git status ---\n{repo.git.status()} \n\n\n--- git diff ---\n{repo.git.diff(commit_tree)}"
        with open(diff_file_name, "w", encoding="utf-8", errors="backslashreplace") as file:
            file.write(content)
        file_paths.append(diff_file_name)

    return file_paths


rsl_on_policy_runner.store_code_state = _store_code_state_safe

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# hydra_task_config 会从配置文件加载环境 & agent 配置。
@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    # 👉 env_cfg = 环境配置
    # 👉 agent_cfg = RSL-RL 训练配置

    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.algorithm.class_name == "Distillation":
        # For distillation, load_run is the teacher experiment name (sibling dir, not a subdir of student).
        # Build the teacher log path: logs/rsl_rl/{load_run}
        teacher_log_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.load_run))
        resume_path = get_checkpoint_path(teacher_log_path, run_dir=".*", checkpoint=agent_cfg.load_checkpoint)
    elif agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    
    # env.reset()
    # for i in range(10):
    #     actions = torch.zeros(env.unwrapped.num_envs, env.unwrapped.action_manager.total_action_dim, device=env.unwrapped.device)
    #     obs, _, _, _,_ = env.step(actions)

    # from step1_test_fingers import step1_simple_check, step1_visualize_fingers_and_pointcloud
    # step1_simple_check(env.unwrapped)
    # step1_visualize_fingers_and_pointcloud(env.unwrapped, env_id=0, output_path="test.ply")
    # env.reset()

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)
    elif args_cli.bc_checkpoint is not None:
        # Load BC pre-trained weights.
        # We do NOT use runner.load() here because:
        #   1. We don't want to load the BC optimizer into the PPO optimizer.
        #   2. runner.load() with empirical_normalization=True expects obs_norm_state_dict
        #      in the standard RSL-RL format which runner.save() writes — our BC checkpoint
        #      stores it separately so we handle it manually below.
        print(f"[INFO]: Loading BC pre-trained weights from: {args_cli.bc_checkpoint}")
        bc_ckpt = torch.load(args_cli.bc_checkpoint, map_location=agent_cfg.device, weights_only=False)

        # 1. Load actor (+ critic) weights
        missing, unexpected = runner.alg.policy.load_state_dict(bc_ckpt["model_state_dict"], strict=False)
        if missing:
            print(f"[BC load] Missing keys (random-init): {missing}")
        if unexpected:
            print(f"[BC load] Unexpected keys (ignored): {unexpected}")

        # 2. Restore obs normalizer so PPO sees the same normalized obs that BC was trained on.
        #    Both obs_normalizer and privileged_obs_normalizer receive the same state because
        #    your setup has no privileged observations (privileged_obs falls back to obs).
        if "obs_norm_state_dict" in bc_ckpt and agent_cfg.empirical_normalization:
            runner.obs_normalizer.load_state_dict(bc_ckpt["obs_norm_state_dict"])
            runner.privileged_obs_normalizer.load_state_dict(bc_ckpt["obs_norm_state_dict"])
            print("[BC load] Obs normalizer initialized from BC dataset statistics.")
        else:
            print("[BC load] Warning: obs_norm_state_dict not found — normalizer starts from scratch.")

        print("[BC load] Done. PPO optimizer and iteration counter start fresh.")

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
