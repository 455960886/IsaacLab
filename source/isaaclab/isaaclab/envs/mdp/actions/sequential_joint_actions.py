  # Copyright (c) 2022-2025, The Isaac Lab Project Developers.
  # All rights reserved.
  #
  # SPDX-License-Identifier: BSD-3-Clause

  """Sequential joint position actions for matching real robot behavior."""

  from __future__ import annotations

  import torch
  from collections.abc import Sequence
  from typing import TYPE_CHECKING

  import isaaclab.utils.math as math_utils
  import isaaclab.utils.string as string_utils
  from isaaclab.assets.articulation import Articulation
  from isaaclab.managers.action_manager import ActionTerm

  if TYPE_CHECKING:
      from isaaclab.envs import ManagerBasedEnv
      from .actions_cfg import JointPositionActionCfg


  class SequentialJointPositionAction(ActionTerm):
      """Sequential joint position action that moves joints one at a time.
      
      This action term applies joint commands sequentially to match real robot behavior 
      where joints cannot move simultaneously. The environment decimation is split into 
      N phases (where N is the number of joints), with only one joint moving per phase.
      
      For example, with 3 joints (M3, M4, M6) and decimation=20:
      - Steps 0-6: Only M3 moves
      - Steps 7-13: Only M4 moves  
      - Steps 14-19: Only M6 moves
      
      The policy still outputs all joint actions simultaneously, but they are applied
      sequentially during simulation.
      """

      cfg: JointPositionActionCfg
      """The configuration of the action term."""
      _asset: Articulation
      """The articulation asset on which the action term is applied."""
      _scale: torch.Tensor
      """The scaling factor applied to the input action."""
      _offset: torch.Tensor
      """The offset applied to the input action."""

      def __init__(self, cfg: JointPositionActionCfg, env: ManagerBasedEnv):
          # initialize the action term
          super().__init__(cfg, env)

          # resolve the joints over which the action term is applied
          self._joint_ids, self._joint_names = self._asset.find_joints(self.cfg.joint_names)
          self._num_joints = len(self._joint_ids)

          # log the joint names and indices
          print(f"[SequentialJointPositionAction] Joints: {self._joint_names} (IDs: {self._joint_ids})")

          # create tensors for raw and processed actions
          self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
          self._processed_actions = torch.zeros_like(self.raw_actions)

          # Track which phase we're in (0 to num_joints-1)
          self._current_phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
          self._phase_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

          # Calculate steps per phase based on decimation
          self._decimation = env.cfg.decimation
          self._steps_per_phase = max(1, self._decimation // self._num_joints)

          # Store target positions for each joint
          self._target_positions = torch.zeros(self.num_envs, self._num_joints, device=self.device)

          print(f"[SequentialJointPositionAction] Decimation: {self._decimation}, "
                f"Steps per phase: {self._steps_per_phase}, Num joints: {self._num_joints}")

          # create the action buffers
          # -- scale
          if isinstance(cfg.scale, (float, int)):
              self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device) * cfg.scale
          elif isinstance(cfg.scale, dict):
              self._scale = torch.ones_like(self._raw_actions)
              index_list, _, value_list = string_utils.resolve_matching_names_values(cfg.scale, self._joint_names)
              self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
          else:
              raise ValueError(f"Unsupported scale type: {type(cfg.scale)}")
          # -- offset
          if isinstance(cfg.offset, (float, int)):
              self._offset = torch.ones(self.num_envs, self.action_dim, device=self.device) * cfg.offset
          elif isinstance(cfg.offset, dict):
              self._offset = torch.zeros_like(self._raw_actions)
              index_list, _, value_list = string_utils.resolve_matching_names_values(cfg.offset, self._joint_names)
              self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
          else:
              raise ValueError(f"Unsupported offset type: {type(cfg.offset)}")

      """
      Properties.
      """

      @property
      def action_dim(self) -> int:
          return self._num_joints

      @property
      def raw_actions(self) -> torch.Tensor:
          return self._raw_actions

      @property
      def processed_actions(self) -> torch.Tensor:
          return self._processed_actions

      """
      Operations.
      """

      def process_actions(self, actions: torch.Tensor):
          # store the raw actions
          self._raw_actions[:] = actions
          # apply the affine transformations
          self._processed_actions = self._raw_actions * self._scale + self._offset

          # Store target positions for sequential execution
          # Get current joint positions
          current_positions = self._asset.data.joint_pos[:, self._joint_ids]

          # Calculate target positions (relative or absolute based on config)
          if self.cfg.use_relative_mode:
              self._target_positions = current_positions + self._processed_actions
          else:
              self._target_positions = self._processed_actions

          # Clamp targets to joint limits if specified
          if self.cfg.use_limits:
              limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
              self._target_positions = torch.clamp(self._target_positions, limits[:, :, 0], limits[:, :, 1])

      def apply_actions(self):
          """Apply actions sequentially based on current phase.
          
          This is called every simulation step (not decimated).
          We use internal phase tracking to determine which joint moves.
          """
          # Get current joint positions
          current_positions = self._asset.data.joint_pos[:, self._joint_ids].clone()

          # Create command with only the active joint moving
          joint_pos_target = current_positions.clone()

          for env_idx in range(self.num_envs):
              phase = self._current_phase[env_idx]
              # Set target for the active joint in this phase
              joint_pos_target[env_idx, phase] = self._target_positions[env_idx, phase]

          # Set the joint position targets
          self._asset.set_joint_position_target(joint_pos_target, joint_ids=self._joint_ids)

          # Increment phase step counter
          self._phase_step += 1

          # Check if we should advance to next phase
          phase_complete = self._phase_step >= self._steps_per_phase

          # Advance phase where needed (cycle through 0 to num_joints-1)
          self._current_phase = torch.where(
              phase_complete,
              (self._current_phase + 1) % self._num_joints,
              self._current_phase
          )

          # Reset step counter where phase advanced
          self._phase_step = torch.where(phase_complete, torch.zeros_like(self._phase_step), self._phase_step)

      def reset(self, env_ids: Sequence[int] | None = None) -> None:
          # Reset phase tracking for specified environments
          if env_ids is None:
              env_ids = slice(None)

          self._current_phase[env_ids] = 0
          self._phase_step[env_ids] = 0
          self._raw_actions[env_ids] = 0.0
          self._target_positions[env_ids] = self._asset.data.joint_pos[env_ids, self._joint_ids]
