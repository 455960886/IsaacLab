"""Script to manually control robot and test rewards using keyboard input."""

import argparse
import torch
import numpy as np
import weakref
import os
import re
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，适合在 Isaac 里跑
import matplotlib.pyplot as plt

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Manual control for reward testing.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--reward_term", type=str, default=None, help="Specific reward term to display (e.g., 'reaching_object')")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import carb
import omni.appwindow


# ============================================================
# ✅ 中文调试开关：保证日志“明显中文”，且可控不刷屏
# ============================================================
DEBUG_CN_LOG = True          # 总开关：True=打印中文调试
LOG_EVERY = 1               # 每隔多少步打印一次（1=每步；10=每10步）
LOG_ENV_ID = 0              # 只打印哪个环境（多环境时避免刷屏）
LOG_TO_FILE = True          # 是否写入文件
LOG_FILE = "/tmp/手动调试_action_obs.log"


def cn_log(msg: str):
    if not DEBUG_CN_LOG:
        return
    line = f"{msg}\n"
    print(line, end="", flush=True)
    if LOG_TO_FILE:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)


def try_extract_policy_tensor(obs):
    """
    兼容不同返回结构，尽量拿到 policy 组的拼接向量 tensor：
    - dict 且 obs["policy"] 是 tensor -> 返回该 tensor
    - obs 直接是 tensor -> 返回 obs
    否则返回 None
    """
    if isinstance(obs, dict):
        pol = obs.get("policy", None)
        if torch.is_tensor(pol):
            return pol
        return None
    if torch.is_tensor(obs):
        return obs
    return None


def try_extract_joint_tail5_from_policy(obs):
    """从 policy 拼接向量里取末尾 5 维（通常是 joint_pos: M3,M4,M5,M6_1,M6_2）。"""
    pol = try_extract_policy_tensor(obs)
    if pol is None or pol.numel() == 0 or pol.ndim != 2 or pol.shape[1] < 5:
        return None
    return pol[LOG_ENV_ID, -5:].detach().cpu().numpy()


def try_extract_m0_pos_from_obs(obs):
    """
    尝试从 env.reset()/env.step() 返回的 obs 里取出 m0_pos。
    兼容两种形式：
      1) obs 是 dict，且 obs["policy"] 是拼接后的 tensor -> 默认最后一维是 m0_pos
      2) obs 是 dict，且 obs["policy"] 是 dict -> 直接找 key "m0_pos"
      3) obs 直接就是 tensor -> 默认最后一维是 m0_pos（不常见，但顺手兼容）
    取第 0 个 env 的值（多 env 时也能跑）。
    """
    try:
        # case A: dict obs
        if isinstance(obs, dict):
            pol = obs.get("policy", None)
            if pol is None:
                return None

            # A1: policy 是 dict（未 concatenate_terms）
            if isinstance(pol, dict):
                if "m0_pos" not in pol:
                    return None
                t = pol["m0_pos"]
                if torch.is_tensor(t):
                    if t.numel() == 0:
                        return None
                    return float(t.reshape(-1)[0].item())
                # numpy / list fallback
                arr = np.array(t).reshape(-1)
                return float(arr[0])

            # A2: policy 是 tensor（concatenate_terms=True）
            if torch.is_tensor(pol):
                if pol.numel() == 0:
                    return None
                # 默认最后一维就是 m0_pos（因为你在 cfg 里把 m0_pos 放在 image 后面）
                return float(pol[0, -1].item())

        # case B: obs 直接是 tensor
        if torch.is_tensor(obs):
            if obs.numel() == 0:
                return None
            return float(obs[0, -1].item())

    except Exception as e:
        print(f"[m0_debug] extract m0_pos failed: {e}")
        return None

    return None


def print_m0_pos_from_obs(obs, prefix="[m0_debug]"):
    """打印 m0_pos（rad 和 deg），用于确认 M0 角度进了观测。"""
    m0 = try_extract_m0_pos_from_obs(obs)
    if m0 is None:
        print(f"{prefix} 没有在 obs 里找到 m0_pos（可能 cfg 没生效，或 policy 观测不是 dict/tensor 预期结构）")
        return
    m0_deg = m0 * 180.0 / np.pi
    print(f"{prefix} m0_pos = {m0:+.6f} rad  ({m0_deg:+.2f} deg)")


def plot_spawn_distribution(env, out_dir="spawn_debug"):
    """
    在当前目录下的 spawn_debug/ 里保存一次散点图。
    点来自 env.unwrapped._spawn_debug_xy（在 reset_object_pool_state_uniform 里记录）。
    - 如果没记录到任何点，就什么也不画，只打印一句提示。
    - 每次调用都会生成一个新文件 spawn_xy_0000.png, spawn_xy_0001.png, ...
    """
    # 保护：没有这个属性或者为空就直接返回
    base_env = env.unwrapped
    if not hasattr(base_env, "_spawn_debug_xy") or len(base_env._spawn_debug_xy) == 0:
        print("[spawn_debug] 暂无记录的 spawn 点（先多 reset 几次再尝试画图）")
        return

    pts = np.array(base_env._spawn_debug_xy, dtype=np.float32)  # [N, 2]，每一行是 (x_world, y_world)
    xs = pts[:, 0]
    ys = pts[:, 1]

    os.makedirs(out_dir, exist_ok=True)
    # 简单的计数器：每次调用 +1，文件名自增
    if not hasattr(base_env, "_spawn_plot_counter"):
        base_env._spawn_plot_counter = 0
    idx = base_env._spawn_plot_counter
    base_env._spawn_plot_counter += 1

    fname = os.path.join(out_dir, f"spawn_xy_{idx:04d}.png")

    plt.figure(figsize=(6, 6))
    plt.scatter(xs, ys, s=6, alpha=0.5)
    plt.axhline(0.0, linestyle="--")
    plt.axvline(0.0, linestyle="--")
    plt.gca().set_aspect("equal", "box")
    plt.xlabel("X (world)")
    plt.ylabel("Y (world)")
    plt.title("Object spawn distribution (world XY)")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"[spawn_debug] 保存散点图到: {fname}")


class KeyboardController:
    """Simple keyboard controller for robot manipulation."""

    def __init__(self, num_arm_joints=4, num_gripper_joints=2):
        self.num_arm_joints = num_arm_joints
        self.num_gripper_joints = num_gripper_joints
        self.action_scale = 0.5

        # action buffers
        self.arm_action = np.zeros(num_arm_joints)
        self.gripper_open = False
        self._should_exit = False
        self._should_reset = False

        # keyboard interface
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

        # Key mappings
        self._key_mapping = {
            "LEFT": (0, 1),    
            "RIGHT": (0, -1),  
            "UP": (1, -1),     
            "DOWN": (1, 1),    
            "F": (2, 1),       
            "V": (2, -1),
            "Q": (3, 1),
            "E": (3, -1),
        }

        print("\n" + "="*60)
        print("KEYBOARD CONTROLS:")
        print("="*60)
        print("Arm Joint Controls:")
        print("  LEFT/RIGHT: Joint M0 (Base)")
        print("  UP/DOWN: Joint M3")
        print("  F/V: Joint M4")
        print("  Q/E: Joint M5 (Wrist)")
        print("\nGripper Controls:")
        print("  SPACE: Toggle Gripper (Open/Close)")
        print("\nOther:")
        print("  R: Reset robot to default pose")
        print("  ESC: Exit")
        print("="*60 + "\n")

    def __del__(self):
        """Clean up keyboard subscription."""
        if hasattr(self, '_keyboard_sub') and self._keyboard_sub is not None:
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._keyboard_sub)

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Callback for keyboard events."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # Gripper toggle
            if event.input.name == "SPACE":
                self.gripper_open = not self.gripper_open
                print(f"Gripper: {'OPEN' if self.gripper_open else 'CLOSED'}")
            # Reset robot
            elif event.input.name == "R":
                self._should_reset = True
                print("Resetting robot to default pose...")
            # Exit
            elif event.input.name == "ESCAPE":
                self._should_exit = True
            # Arm joints - press
            elif event.input.name in self._key_mapping:
                joint_idx, direction = self._key_mapping[event.input.name]
                self.arm_action[joint_idx] = direction * self.action_scale

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # Arm joints - release
            if event.input.name in self._key_mapping:
                joint_idx, _ = self._key_mapping[event.input.name]
                self.arm_action[joint_idx] = 0.0

    def get_action_tensor(self, num_envs, device):
        """Convert numpy actions to tensor for multiple environments."""
        if self._should_exit:
            return None, True

        arm_tensor = torch.tensor(self.arm_action, dtype=torch.float32, device=device).repeat(num_envs, 1)
        gripper_action = 1.0 if self.gripper_open else -1.0
        gripper_tensor = torch.tensor([[gripper_action]], dtype=torch.float32, device=device).repeat(num_envs, 1)
        action = torch.cat([arm_tensor, gripper_tensor], dim=-1)

        return action, False

    def should_reset(self):
        """Check if reset was requested and clear the flag."""
        if self._should_reset:
            self._should_reset = False
            return True
        return False


def main():

    from isaaclab_tasks.manager_based.manipulation.lift.config.franka.joint_pos_env_cfg import (
        CoarseArmCubeLiftEnvCfg,
    )

    env_cfg = CoarseArmCubeLiftEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else "cuda:0"

    print(f"[INFO] Creating environment: {args_cli.task}")
    print(f"[INFO] Number of environments: {env_cfg.scene.num_envs}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    device = env.unwrapped.device

    # ====== 环境/空间信息：帮助确认 action/obs 维度 ======
    try:
        cn_log(f"【手动调试】已开启中文日志。日志文件：{LOG_FILE}")
        cn_log(f"【手动调试】env.device = {device}")
        cn_log(f"【手动调试】env.action_space = {env.action_space}")
        cn_log(f"【手动调试】env.observation_space = {env.observation_space}")
    except Exception as e:
        cn_log(f"【手动调试】读取 action/observation space 失败（不影响运行）：{e}")

    # ====== 尝试解析 robot 的关节名与 M6 索引（用于打印实际关节状态/target）======
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    joint_names_all = list(getattr(robot, "joint_names", []))

    def _find_joint_ids(pattern: str):
        reg = re.compile(pattern)
        return [i for i, n in enumerate(joint_names_all) if reg.fullmatch(n)]

    m6_ids = _find_joint_ids(r"M6_.*")
    m345_ids = _find_joint_ids(r"M3|M4|M5")
    m0_ids = _find_joint_ids(r"M0")
    cn_log(f"【手动调试】robot关节总数={len(joint_names_all)} | M0 ids={m0_ids} | M3/4/5 ids={m345_ids} | M6 ids={m6_ids}")

    obs, _ = env.reset()
    print("[INFO] Environment reset complete")
    print_m0_pos_from_obs(obs, prefix="[m0_debug][startup]")

    # reset 后打印一次 obs 的 tail5（如果能拿到）
    tail5 = try_extract_joint_tail5_from_policy(obs)
    if tail5 is not None:
        cn_log(f"【手动调试】reset后 | policy输入obs末尾5维(joint_pos)={tail5}（通常顺序=M3,M4,M5,M6_1,M6_2）")
    else:
        cn_log("【手动调试】reset后 | 未能从 obs 里解析 policy 拼接向量末尾5维（可能当前 obs 不是拼接结构/版本差异）")

    if hasattr(env.unwrapped, 'reward_manager'):
        available_terms = env.unwrapped.reward_manager.active_terms
        print(f"[INFO] Available reward terms: {', '.join(available_terms)}")
        print(f"[DEBUG] Reward manager attributes: {[attr for attr in dir(env.unwrapped.reward_manager) if not attr.startswith('_')]}")

        if args_cli.reward_term:
            if args_cli.reward_term in available_terms:
                print(f"[INFO] Displaying only: {args_cli.reward_term}")
            else:
                print(f"[WARNING] Requested term '{args_cli.reward_term}' not found!")
        else:
            print("[INFO] Displaying all reward terms")
        print()

    # Create keyboard controller
    controller = KeyboardController(num_arm_joints=4, num_gripper_joints=1)

    total_reward = 0.0
    step_count = 0
    episode_count = 0

    print("\n[INFO] Starting manual control loop...")
    print("[INFO] Press keys to control the robot. Press ESC to exit.\n")

    try:
        while simulation_app.is_running():
            if controller.should_reset():
                # 每次手动按 R reset 之后，画一张当前所有 spawn 点的散点图
                # 注意：reset() 里会调用 reset_object_pool_state_uniform 并往 _spawn_debug_xy 里追加一个点
                # 所以这里的图包含“到目前为止所有 reset 的分布”
                obs, _ = env.reset()
                total_reward = 0.0
                step_count = 0
                print_m0_pos_from_obs(obs, prefix="[m0_debug][after_reset_R]")
                # plot_spawn_distribution(env)
                print("Robot reset complete\n")
                continue

            # Get action from keyboard
            action, should_exit = controller.get_action_tensor(env_cfg.scene.num_envs, device)

            if should_exit:
                print("\n[INFO] Exit requested by user")
                break
   
            # ====== (A) 打印“你按键产生的 action”（送进 env.step 的 action）======
            if DEBUG_CN_LOG and (step_count % LOG_EVERY == 0):
                a0 = action[LOG_ENV_ID].detach().cpu().numpy()
                cn_log(f"【手动调试】步={step_count} | 键盘送入env.step的action={a0} | gripper_open={controller.gripper_open}")

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # reward information
            step_count += 1
            reward_value = reward[0].item()  # Get reward from first environment
            
            # ====== (B) 打印“环境最终执行的 action / 夹爪项 raw/processed / 关节target”等 ======
            if DEBUG_CN_LOG and ((step_count - 1) % LOG_EVERY == 0):
                env0 = LOG_ENV_ID
                try:
                    am = base_env.action_manager
                    if hasattr(am, "action"):
                        final_act = am.action[env0].detach().cpu().numpy()
                        cn_log(f"【手动调试】步={step_count-1} | ActionManager最终动作向量={final_act}")
                    else:
                        cn_log(f"【手动调试】步={step_count-1} | ActionManager没有 action 字段（版本差异）")

                    # 夹爪 action term 细节（字段名因版本不同）
                    try:
                        term = am.get_term("gripper_action")
                        if hasattr(term, "raw_actions"):
                            cn_log(f"【手动调试】步={step_count-1} | gripper_action.raw_actions={term.raw_actions[env0].detach().cpu().numpy()}")
                        if hasattr(term, "processed_actions"):
                            cn_log(f"【手动调试】步={step_count-1} | gripper_action.processed_actions={term.processed_actions[env0].detach().cpu().numpy()}")
                        if hasattr(term, "actions"):
                            cn_log(f"【手动调试】步={step_count-1} | gripper_action.actions={term.actions[env0].detach().cpu().numpy()}")
                    except Exception as e:
                        cn_log(f"【手动调试】步={step_count-1} | 无法读取 gripper_action term 细节（版本差异）：{e}")

                except Exception as e:
                    cn_log(f"【手动调试】步={step_count-1} | 读取 ActionManager 信息失败：{e}")

                # 打印机器人真实关节状态（M6 当前 joint_pos / target）
                try:
                    if m6_ids:
                        jp = robot.data.joint_pos[env0, m6_ids].detach().cpu().numpy()
                        cn_log(f"【手动调试】步={step_count-1} | 机器人当前M6 joint_pos={jp}")
                        if hasattr(robot.data, "joint_pos_target"):
                            jt = robot.data.joint_pos_target[env0, m6_ids].detach().cpu().numpy()
                            cn_log(f"【手动调试】步={step_count-1} | 机器人当前M6 joint_pos_target={jt}")
                except Exception as e:
                    cn_log(f"【手动调试】步={step_count-1} | 读取机器人M6关节状态失败：{e}")

                # ====== (C) 打印“policy obs 里 joint_pos 的末尾5维”（你要对齐的输入）======
                tail5 = try_extract_joint_tail5_from_policy(obs)
                if tail5 is not None:
                    cn_log(f"【手动调试】步={step_count-1} | policy输入obs末尾5维(joint_pos)={tail5}（通常顺序=M3,M4,M5,M6_1,M6_2）")
                else:
                    cn_log(f"【手动调试】步={step_count-1} | 未解析到 policy obs 末尾5维（结构/版本差异）")

            print(f"Step {step_count:4d} | Reward: {reward_value:+.4f} | Total: {total_reward:+.4f}", end="")

            if hasattr(env.unwrapped, 'reward_manager'):
                reward_components = []

                if args_cli.reward_term:
                    if args_cli.reward_term in env.unwrapped.reward_manager.active_terms:
                        term_idx = env.unwrapped.reward_manager.active_terms.index(args_cli.reward_term)

                        # Try different ways to access the reward value
                        term_value = None
                        if hasattr(env.unwrapped.reward_manager, '_term_buffer'):
                            term_value = env.unwrapped.reward_manager._term_buffer[0, term_idx].item()
                        elif hasattr(env.unwrapped.reward_manager, 'get_term'):
                            term_value = env.unwrapped.reward_manager.get_term(args_cli.reward_term)[0].item()
                        elif hasattr(env.unwrapped.reward_manager, '_episode_sums'):
                            term_value = env.unwrapped.reward_manager._episode_sums[args_cli.reward_term][0].item()

                        if term_value is not None:
                            reward_components.append(f"{args_cli.reward_term}: {term_value:+.3f}")
                        elif step_count == 1:
                            print(f"\n[DEBUG] Could not access reward value for '{args_cli.reward_term}'", end="")
                else:
                    for term_idx, term_name in enumerate(env.unwrapped.reward_manager.active_terms):
                        term_value = None
                        if hasattr(env.unwrapped.reward_manager, '_term_buffer'):
                            term_value = env.unwrapped.reward_manager._term_buffer[0, term_idx].item()
                        elif hasattr(env.unwrapped.reward_manager, '_episode_sums'):
                            if term_name in env.unwrapped.reward_manager._episode_sums:
                                term_value = env.unwrapped.reward_manager._episode_sums[term_name][0].item()

                        if term_value is not None:
                            reward_components.append(f"{term_name}: {term_value:+.3f}")

                if reward_components:
                    print(f" | {' | '.join(reward_components)}", end="")

            print()  # New line

            if terminated.any() or truncated.any():
                episode_count += 1
                print(f"\n{'='*60}")
                print(f"Episode {episode_count} ended")
                print(f"Total steps: {step_count}")
                print(f"Total reward: {total_reward:.4f}")
                print(f"Average reward: {total_reward/step_count:.4f}")
                print(f"{'='*60}\n")

                # Episode 结束时也顺便画一张当前 spawn 分布
                plot_spawn_distribution(env)

                # Reset（同样会在 reset 中记录新的 spawn 点）
                obs, _ = env.reset()
                print_m0_pos_from_obs(obs, prefix="[m0_debug][after_episode_reset]")
                total_reward = 0.0
                step_count = 0

            # Update simulation
            simulation_app.update()

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        print(f"\n[INFO] Final Statistics:")
        print(f"  Total episodes: {episode_count}")
        print(f"  Total steps: {step_count}")
        if step_count > 0:
            print(f"  Average reward per step: {total_reward/step_count:.4f}")

        # Close environment
        env.close()
        print("[INFO] Environment closed")


if __name__ == "__main__":
    main()
    simulation_app.close()