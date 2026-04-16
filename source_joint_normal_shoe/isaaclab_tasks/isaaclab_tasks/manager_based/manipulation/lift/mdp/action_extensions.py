# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions import joint_actions
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class EMAJointPositionAction(joint_actions.JointPositionAction):
    """对已经映射到真实关节角的目标做 EMA 平滑。

    适用于当前 lift 任务这种：
    policy 先输出 [-1, 1]，环境侧再映射到你手动指定的 M3/M4/M5 真实角度范围。
    在这个基础上做 EMA，比直接换成 ToLimitsAction 更贴合当前项目。
    """

    cfg: "EMAJointPositionActionCfg"

    def __init__(self, cfg: "EMAJointPositionActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        if not 0.0 <= cfg.alpha <= 1.0:
            raise ValueError(f"EMA alpha 必须在 [0, 1]，当前收到: {cfg.alpha}")
        self._alpha = float(cfg.alpha)
        self._prev_applied_actions = torch.zeros_like(self.processed_actions)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # reset 时把 EMA 历史直接对齐到当前关节角，避免 episode 开始时目标突跳。
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)
        # 这里不能直接写 joint_pos[env_ids, self._joint_ids]。
        # 当 env_ids 和 self._joint_ids 都是索引张量/列表时，PyTorch 会走高级索引，
        # 试图广播 [num_envs] 和 [num_joints]，从而报 shape mismatch。
        # 先按环境切片，再按关节切片，才能稳定得到 [num_envs, action_dim]。
        if isinstance(self._joint_ids, slice):
            current_joint_pos = self._asset.data.joint_pos[env_ids, self._joint_ids]
        else:
            current_joint_pos = self._asset.data.joint_pos[env_ids][:, self._joint_ids]
        self._prev_applied_actions[env_ids, :] = current_joint_pos

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)

        ema_targets = self._alpha * self._processed_actions
        ema_targets += (1.0 - self._alpha) * self._prev_applied_actions

        # 再做一次 clip，确保平滑后的目标不会越出当前自定义关节范围。
        if self.cfg.clip is not None:
            ema_targets = torch.clamp(
                ema_targets,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )

        self._processed_actions[:] = ema_targets
        self._prev_applied_actions[:] = self._processed_actions


@configclass
class EMAJointPositionActionCfg(JointPositionActionCfg):
    """EMA 平滑版 JointPositionAction 配置。"""

    class_type: type[ActionTerm] = EMAJointPositionAction
    alpha: float = 1.0
    """EMA 系数。越小越平滑，越大越跟手。"""
