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
import cv2
from rsl_rl.runners import OnPolicyRunner
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

    def print_obs_concat_layout(env, group_name: str = "policy", env_idx: int = 0, preview_vals: int = 0):
        base_env = env.unwrapped
        om = getattr(base_env, "observation_manager", None)
        if om is None:
            print("[OBS] env.unwrapped.observation_manager not found.")
            return

        # 1) 打印 manager 自带的表格（你日志里那个）
        try:
            print(om)
        except Exception as e:
            print(f"[OBS] print(observation_manager) failed: {e}")

        # 2) 打印拼接切片顺序
        if group_name not in om.active_terms:
            print(f"[OBS] group '{group_name}' not in active_terms: {list(om.active_terms.keys())}")
            return

        names = om.active_terms[group_name]
        dims_list = om.group_obs_term_dim[group_name]
        concat = om.group_obs_concatenate[group_name]

        print(f"[OBS] group='{group_name}' concat={concat} group_obs_dim={om.group_obs_dim[group_name]}")
        idx = 0
        for i, (name, dims) in enumerate(zip(names, dims_list)):
            flat = int(np.prod(dims))
            print(f"[OBS] {i:02d}: {group_name}.{name} dims={tuple(dims)} flat={flat} slice=[{idx}:{idx+flat}]")
            idx += flat
        print(f"[OBS] total_flat={idx}")

        # 3) 可选：把当前 obs 按 slice 拆开，预览每段前 N 个值
        if preview_vals and concat:
            obs_tensor, _ = env.get_observations()
            # obs_tensor: (num_envs, obs_dim)
            one = obs_tensor[env_idx].detach().to("cpu")
            idx = 0
            for name, dims in zip(names, dims_list):
                flat = int(np.prod(dims))
                seg = one[idx:idx+flat]
                print(f"[OBS] preview {group_name}.{name}: first {preview_vals} = {seg[:preview_vals].tolist()}")
                idx += flat

    # 调用：只看 slice
    print_obs_concat_layout(env, group_name="policy", env_idx=0, preview_vals=0)

    # 再额外打印一次实际 obs tensor shape（最直观）
    obs_dbg, _ = env.get_observations()
    print("[OBS] env.get_observations() shape =", tuple(obs_dbg.shape))

    # 或者：顺便预览每段前 8 个值（image 会很大，只看前几个即可）
    # print_obs_concat_layout(env, group_name="policy", env_idx=0, preview_vals=8)
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

    # 每个环境各自的 episode 计数和 step 计数
    ep_id = np.zeros(num_envs, dtype=np.int64)
    ep_step = np.zeros(num_envs, dtype=np.int64)

    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            actions = policy(obs)

            # 你现在把 actions 当 numpy 用了（np.pi），建议用 torch.pi 或转 cpu
            env0 = 0

            # 原始动作（rad/原始单位）
            a0_rad = actions[env0].detach().cpu().numpy()

            # 转换成 deg（如果你确定动作是 rad）
            actions_deg = actions * (180.0 / torch.pi)
            a0_deg = actions_deg[env0].detach().cpu().numpy()

            print(
                f"[PLAY] env={env0} ep={ep_id[env0]} ep_step={ep_step[env0]} global_step={global_step} | "
                f"action(rad)={a0_rad} | action(deg)={a0_deg}"
            )

            obs, _, dones, infos = env.step(actions)

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