# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""
from isaaclab.app import AppLauncher
import argparse

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
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
parser.add_argument("--sleep", type=float, default=0.0,
                    help="Extra sleep seconds after each step (e.g., 0.05 means 50ms).")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
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
import copy
import torch
from rsl_rl.runners import OnPolicyRunner
import rsl_rl.runners.on_policy_runner as rsl_on_policy_runner

from positive_m5_actor_critic import PositiveM5ActorCritic

rsl_on_policy_runner.PositiveM5ActorCritic = PositiveM5ActorCritic
import numpy as np

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
import torch.nn as nn
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)

# ===== local debug switches =====
# Set these values directly when you want to enable/disable play-time debug logs.
DEBUG_PRINT_OBS = True
DEBUG_PRINT_LAST_ACTION = True
DEBUG_LAST_ACTION_EVERY_N_STEPS = 1
DEBUG_ENV_INDEX = 0
# =================================


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

    # ===== debug: print checkpoint expected obs_dim =====
    ckpt = torch.load(resume_path, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)

    # rsl_rl 的 key 有时会带前缀，做个稳健查找
    actor_w_key = None
    for k in sd.keys():
        if k.endswith("actor.0.weight"):
            actor_w_key = k
            break
    if actor_w_key is None and "actor.0.weight" in sd:
        actor_w_key = "actor.0.weight"

    if actor_w_key is not None:
        w = sd[actor_w_key]
        print(f"[CKPT] actor.0.weight key='{actor_w_key}' expects obs_dim = {w.shape[1]}")
    else:
        # 兜底：把所有含 actor/weight 的 key 打出来，方便你定位
        cand = [k for k in sd.keys() if ("actor" in k and "weight" in k)]
        print("[CKPT] ❌ cannot find 'actor.0.weight' in state_dict. Candidate keys:")
        for k in cand[:50]:
            print("  ", k)
    # ================================================

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

    env.reset()    

    def print_obs_group_details(env, group_name: str, env_idx: int = 0, preview_vals: int = 8):
        base_env = env.unwrapped
        om = getattr(base_env, "observation_manager", None)
        if om is None:
            print("[OBS] env.unwrapped.observation_manager not found.")
            return

        if group_name not in om.active_terms:
            print(f"[OBS] group '{group_name}' not in active_terms: {list(om.active_terms.keys())}")
            return

        names = om.active_terms[group_name]
        dims_list = om.group_obs_term_dim[group_name]
        concat = om.group_obs_concatenate[group_name]
        env_index = max(0, min(int(env_idx), env.unwrapped.num_envs - 1))

        print("=" * 100)
        print(
            f"[OBS] group='{group_name}' env={env_index} concat={concat} "
            f"group_obs_dim={om.group_obs_dim[group_name]}"
        )
        if group_name == "policy":
            print("[OBS] note: play.py 里真正喂给 actor/policy 的就是这一组。")
        elif group_name == "critic":
            print("[OBS] note: 这一组是 privileged obs，play 推理时通常不喂给 actor，只用于训练时 value/critic。")

        group_obs = om.compute_group(group_name)
        if not torch.is_tensor(group_obs):
            print(f"[OBS] group '{group_name}' is not concatenated, skip slice layout.")
            return

        start = 0
        total_flat = 0
        sample = group_obs[env_index].detach().float().cpu()
        for i, (name, dims) in enumerate(zip(names, dims_list)):
            flat = int(np.prod(dims))
            seg = sample[start : start + flat]
            seg_min = float(seg.min().item()) if flat > 0 else 0.0
            seg_max = float(seg.max().item()) if flat > 0 else 0.0
            seg_mean = float(seg.mean().item()) if flat > 0 else 0.0
            preview = seg[:preview_vals].tolist()
            preview_suffix = ""
            if flat > preview_vals:
                preview_suffix = f" ... ({flat} values total)"
            print(
                f"[OBS] {i:02d}: {group_name}.{name:<28} "
                f"shape={tuple(dims)!s:<12} flat={flat:<5} slice=[{start}:{start + flat}] "
                f"min={seg_min:.6f} max={seg_max:.6f} mean={seg_mean:.6f}"
            )
            print(f"[OBS]      values[:{preview_vals}] = {preview}{preview_suffix}")
            start += flat
            total_flat += flat

        print(f"[OBS] total_flat={total_flat}")
        print("=" * 100)

    def get_obs_term_slice(env, group_name: str, term_name: str):
        base_env = env.unwrapped
        om = getattr(base_env, "observation_manager", None)
        if om is None or group_name not in om.active_terms:
            return None, None

        start = 0
        for name, dims in zip(om.active_terms[group_name], om.group_obs_term_dim[group_name]):
            flat = int(np.prod(dims))
            if name == term_name:
                return slice(start, start + flat), tuple(dims)
            start += flat
        return None, None

    def print_action_layout(env):
        action_manager = getattr(env.unwrapped, "action_manager", None)
        if action_manager is None:
            print("[LAST_ACTION] action_manager not found.")
            return

        labels = []
        index = 0
        for term_name in action_manager.active_terms:
            term = action_manager.get_term(term_name)
            joint_names = getattr(term, "_joint_names", [])
            for term_index in range(term.action_dim):
                if term_index < len(joint_names):
                    label = f"{term_name}.{joint_names[term_index]}"
                else:
                    label = f"{term_name}[{term_index}]"
                labels.append(f"{index}:{label}")
                index += 1
        print("[LAST_ACTION] action layout =", ", ".join(labels))

    def print_last_action_compare(tag: str, obs_tensor: torch.Tensor, expected_action: torch.Tensor, obs_slice, env_idx: int):
        if obs_slice is None:
            return
        obs_last_action = obs_tensor[env_idx, obs_slice].detach().float().cpu()
        expected = expected_action[env_idx].detach().float().cpu()
        diff = obs_last_action - expected
        max_abs_diff = float(diff.abs().max().item()) if diff.numel() > 0 else 0.0
        is_same = bool(torch.allclose(obs_last_action, expected, atol=1.0e-5, rtol=1.0e-5))
        print(
            f"[LAST_ACTION] {tag} env={env_idx} "
            f"obs.policy.last_action={obs_last_action.tolist()} | "
            f"expected_action={expected.tolist()} | "
            f"same={is_same} max_abs_diff={max_abs_diff:.8f}"
        )

    # 启动时打印 obs 结构和样本值
    if DEBUG_PRINT_OBS:
        try:
            print(env.unwrapped.observation_manager)
        except Exception as e:
            print(f"[OBS] print(observation_manager) failed: {e}")

        print_obs_group_details(env, group_name="policy", env_idx=DEBUG_ENV_INDEX, preview_vals=8)
        print_obs_group_details(env, group_name="critic", env_idx=DEBUG_ENV_INDEX, preview_vals=8)

    obs_dbg, obs_extras_dbg = env.get_observations()
    if DEBUG_PRINT_OBS:
        print("[OBS] actor input tensor shape =", tuple(obs_dbg.shape))
    critic_dbg = obs_extras_dbg.get("observations", {}).get("critic")
    if DEBUG_PRINT_OBS and torch.is_tensor(critic_dbg):
        print("[OBS] critic tensor shape      =", tuple(critic_dbg.shape))
    if DEBUG_PRINT_OBS:
        print("[OBS] action dim                =", env.num_actions)
    last_action_slice, last_action_dims = get_obs_term_slice(env, group_name="policy", term_name="last_action")
    if DEBUG_PRINT_LAST_ACTION:
        if last_action_slice is None:
            print("[LAST_ACTION] policy.last_action term not found; per-step comparison will be skipped.")
        else:
            print(
                f"[LAST_ACTION] policy.last_action slice=[{last_action_slice.start}:{last_action_slice.stop}] "
                f"shape={last_action_dims}"
            )
        print_action_layout(env)
    # ================================================


    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    print("before creating runner")
    print("6666")
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    print("7777")
    ppo_runner.load(resume_path)
    # 让 normalizer 和 policy 同设备（跟 env 一致）
    device = env.unwrapped.device  # 通常是 cuda:0
    ppo_runner.obs_normalizer.to(device)
    # rsl_rl>=2.3: alg.policy; older: alg.actor_critic
    if hasattr(ppo_runner.alg, "policy"):
        ppo_runner.alg.policy.to(device)
    elif hasattr(ppo_runner.alg, "actor_critic"):
        ppo_runner.alg.actor_critic.to(device)

    print("after creating runner")
    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
        print("1111")
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # export policy to onnx/jit
    # export_model_dir = os.path.join(os.path.dirname(resume_path), "exported2")
    # export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    # export_policy_as_onnx(
    #     policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy1905.onnx"
    # )
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported2")
    os.makedirs(export_model_dir, exist_ok=True)

    # 关键：用环境真实输出的 obs 作为 example input（shape 会是 [1, 1541]）
    obs, _ = env.get_observations()
    obs = obs.to(device)

    dummy = obs[0:1].detach().to("cpu")
    print("[EXPORT] dummy obs shape =", tuple(dummy.shape))  # 应该是 (1, 1541)

    # ⚠️ 关键：不要对 runner 里的对象原地 .to("cpu")，否则后面推理会 device mismatch
    # 用 deepcopy 拷贝一份做导出即可
    policy_nn_cpu = copy.deepcopy(policy_nn).to("cpu").eval()
    normalizer_cpu = copy.deepcopy(ppo_runner.obs_normalizer).to("cpu").eval()

    class ActorWithNorm(nn.Module):
        def __init__(self, policy, normalizer):
            super().__init__()
            self.policy = policy
            self.normalizer = normalizer

        def forward(self, x):
            x = self.normalizer(x)
            # policy 可能是 ActorCritic(有 .actor)，也可能本身就是 actor
            if hasattr(self.policy, "actor"):
                return self.policy.actor(x)
            return self.policy(x)

    export_module = ActorWithNorm(policy_nn_cpu, normalizer_cpu).eval()

    onnx_path = os.path.join(export_model_dir, "policy1905.onnx")
    torch.onnx.export(
        export_module,
        dummy,
        onnx_path,
        input_names=["obs"],
        output_names=["actions"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
    )
    print("[INFO] Exported ONNX to:", onnx_path)

    # 如果你还想导出 jit：
    jit_path = os.path.join(export_model_dir, "policy.pt")
    traced = torch.jit.trace(export_module, dummy)
    traced.save(jit_path)
    print("[INFO] Exported JIT to:", jit_path)


    dt = env.unwrapped.step_dt
    # img_bgr = cv2.imread('/home/roborock/下载/9.png')
    # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1)
    # obs = img_tensor.unsqueeze(0)
    # 推理阶段继续使用与 env 一致的 device（不要重新覆盖）
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = "/home/roborock/下载/"
    # reset environment
    obs, _ = env.get_observations()
    obs = obs.to(device)
    num_envs = env.unwrapped.num_envs  # 或者 env.num_envs（看 wrapper）
    global_step = 0
    last_env_action = env.unwrapped.action_manager.action.detach().clone().to(device)

    # 每个环境各自的 episode 计数和 step 计数
    ep_id = np.zeros(num_envs, dtype=np.int64)
    ep_step = np.zeros(num_envs, dtype=np.int64)

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            env0 = max(0, min(DEBUG_ENV_INDEX, num_envs - 1))
            should_print_last_action = (
                DEBUG_PRINT_LAST_ACTION and global_step % max(1, DEBUG_LAST_ACTION_EVERY_N_STEPS) == 0
            )
            if should_print_last_action:
                print_last_action_compare(
                    "before_policy",
                    obs,
                    last_env_action,
                    last_action_slice,
                    env_idx=env0,
                )
            actions = policy(obs)

            # 你现在把 actions 当 numpy 用了（np.pi），建议用 torch.pi 或转 cpu

            # 原始动作（rad/原始单位）
            a0_rad = actions[env0].detach().cpu().numpy()

            # 转换成 deg（如果你确定动作是 rad）
            actions_deg = actions * (180.0 / torch.pi)
            a0_deg = actions_deg[env0].detach().cpu().numpy()

            print(
                f"[PLAY] env={env0} ep={ep_id[env0]} ep_step={ep_step[env0]} global_step={global_step} | "
                f"action(rad)={a0_rad} | action(deg)={a0_deg}"
            )

            action_sent_to_env = actions
            if getattr(env, "clip_actions", None) is not None:
                action_sent_to_env = torch.clamp(actions, -env.clip_actions, env.clip_actions)

            obs, _, dones, infos = env.step(actions)
            last_env_action = env.unwrapped.action_manager.action.detach().clone().to(device)
            if should_print_last_action:
                print_last_action_compare(
                    "after_step",
                    obs,
                    last_env_action,
                    last_action_slice,
                    env_idx=env0,
                )
            if should_print_last_action and last_action_slice is not None:
                obs_after_last_action = obs[env0, last_action_slice].detach().float().cpu()
                action_sent_cpu = action_sent_to_env[env0].detach().float().cpu()
                action_diff = obs_after_last_action - action_sent_cpu
                print(
                    f"[LAST_ACTION] after_step_vs_policy_action env={env0} "
                    f"action_sent={action_sent_cpu.tolist()} | "
                    f"max_abs_diff={float(action_diff.abs().max().item()):.8f} | "
                    f"done={bool(dones[env0].item())}"
                )
            if DEBUG_PRINT_OBS:
                tail10 = obs[env0, -10:].detach().float().cpu().numpy()
                print(f"[OBS] env={env0} last10 = {tail10}")

            # step 计数更新：每一步所有 env 的 ep_step 都 +1
            ep_step += 1
            global_step += 1

            # 如果某些 env done，打印并对这些 env 重置 ep_step，ep_id+1
            # if torch.any(dones):
            #     done_ids = torch.nonzero(dones, as_tuple=False).squeeze(-1).cpu().numpy()
            #     for eid in done_ids:
            #         print(f"[PLAY] ✅ env={eid} episode结束：ep={ep_id[eid]} 总步数={ep_step[eid]}")
            #         ep_id[eid] += 1
            #         ep_step[eid] = 0

        # # time delay for real-time evaluation
        # time.sleep(0.6)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
