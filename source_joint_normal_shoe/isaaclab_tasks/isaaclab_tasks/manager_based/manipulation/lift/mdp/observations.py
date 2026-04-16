# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.envs.mdp.observations import (
    joint_pos_with_binary_m6_latched as _joint_pos_with_binary_m6_latched,
)
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.string as string_utils
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b


def active_object_yaw(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
) -> torch.Tensor:
    """Return the active object's yaw as a temporary low-dimensional orientation signal."""
    from isaaclab.assets import RigidObjectCollection

    object_collection: RigidObjectCollection = env.scene[object_cfg.name]
    all_quat_w = object_collection.data.object_quat_w

    if hasattr(env, "active_object_indices"):
        active_indices = env.active_object_indices
    else:
        # Observation terms are shape-checked before startup events initialize the active object selection.
        active_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    env_indices = torch.arange(env.num_envs, device=env.device)
    active_quat_w = all_quat_w[env_indices, active_indices]

    w = active_quat_w[:, 0]
    x = active_quat_w[:, 1]
    y = active_quat_w[:, 2]
    z = active_quat_w[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    return yaw.unsqueeze(-1)


def joint_pos_normalized_with_binary_m6_latched(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    action_name: str = "gripper_action",
    action_index: int = 0,
    m6_open_value: float = 1.0,
    m6_close_value: float = 0.0,
    toggle_threshold: float = 0.0,
    debug: bool = False,
    debug_every: int = 200,
) -> torch.Tensor:
    """返回归一化后的关节观测。

    处理逻辑：
    1. 先复用现有逻辑，得到 [M3, M4, M5, M6_1, M6_2]，其中 M6 已经被替换成二值 latch 观测；
    2. 再把 M3/M4/M5 按当前 arm_action 的 scale/offset 逆变换回 [-1, 1]；
    3. M6_1/M6_2 保持二值，不做归一化。

    这样可以让：
      - 动作侧：policy 输出 [-1, 1]
      - 观测侧：M3/M4/M5 也落在 [-1, 1]
    两边数值尺度一致，更接近 inhand 任务那套标准处理方式。
    """

    q = _joint_pos_with_binary_m6_latched(
        env=env,
        asset_cfg=asset_cfg,
        action_name=action_name,
        action_index=action_index,
        m6_open_value=m6_open_value,
        m6_close_value=m6_close_value,
        toggle_threshold=toggle_threshold,
        debug=debug,
        debug_every=debug_every,
    )

    arm_action_cfg = getattr(env.cfg.actions, "arm_action", None)
    if arm_action_cfg is None:
        raise RuntimeError("未找到 env.cfg.actions.arm_action，无法按动作定义对关节观测做归一化。")
    if not isinstance(arm_action_cfg.scale, dict) or not isinstance(arm_action_cfg.offset, dict):
        raise TypeError("当前 joint 观测归一化要求 arm_action 的 scale/offset 使用 dict 形式配置。")

    robot = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    if joint_ids is None:
        raise ValueError("asset_cfg.joint_ids 为空，无法确定当前观测对应的关节顺序。")
    if isinstance(joint_ids, slice):
        joint_ids = list(range(robot.num_joints))[joint_ids]

    observed_joint_names = [robot.joint_names[joint_id] for joint_id in joint_ids]

    scale_by_name = {name: 1.0 for name in observed_joint_names}
    offset_by_name = {name: 0.0 for name in observed_joint_names}

    _, scale_names, scale_values = string_utils.resolve_matching_names_values(arm_action_cfg.scale, observed_joint_names)
    _, offset_names, offset_values = string_utils.resolve_matching_names_values(arm_action_cfg.offset, observed_joint_names)

    for name, value in zip(scale_names, scale_values):
        scale_by_name[name] = float(value)
    for name, value in zip(offset_names, offset_values):
        offset_by_name[name] = float(value)

    # 只归一化 M3/M4/M5，M6_1/M6_2 仍然保持二值观测。
    for col, joint_name in enumerate(observed_joint_names):
        if joint_name in {"M3", "M4", "M5"}:
            scale = scale_by_name[joint_name]
            offset = offset_by_name[joint_name]
            if scale == 0.0:
                raise ValueError(f"关节 {joint_name} 的 scale 为 0，无法做归一化。")
            q[:, col] = (q[:, col] - offset) / scale
            q[:, col] = q[:, col].clamp(-1.0, 1.0)

    return q

