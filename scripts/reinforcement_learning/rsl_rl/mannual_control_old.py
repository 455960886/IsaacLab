"""Script to manually control robot and test rewards using keyboard input."""

import argparse
import torch
import numpy as np
import weakref

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

from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg


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
        }

        print("\n" + "="*60)
        print("KEYBOARD CONTROLS:")
        print("="*60)
        print("Arm Joint Controls:")
        print("  LEFT/RIGHT: Joint M0 (Base)")
        print("  UP/DOWN: Joint M3")
        print("  F/V: Joint M4")
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

    obs, _ = env.reset()
    print("[INFO] Environment reset complete")

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
                obs, _ = env.reset()
                total_reward = 0.0
                step_count = 0
                print("Robot reset complete\n")
                continue

            # Get action from keyboard
            action, should_exit = controller.get_action_tensor(env_cfg.scene.num_envs, device)

            if should_exit:
                print("\n[INFO] Exit requested by user")
                break

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            #reward information
            step_count += 1
            reward_value = reward[0].item()  # Get reward from first environment
            total_reward += reward_value

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

                # Reset
                obs, _ = env.reset()
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