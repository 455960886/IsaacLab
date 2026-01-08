# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING
import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def modify_reward_weight(env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, weight: float, num_steps: int):
    """Curriculum that modifies a reward weight a given number of steps.

    Args:
        env: The learning environment.
        env_ids: Not used since all environments are affected.
        term_name: The name of the reward term.
        weight: The weight of the reward term.
        num_steps: The number of steps after which the change should be applied.
    """
    # print(env.common_step_counter)
    if env.common_step_counter > num_steps:
        # obtain term settings
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        # update term settings
        term_cfg.weight = weight
        env.reward_manager.set_term_cfg(term_name, term_cfg)


import torch


def curriculum_reset_pose_range(
    env,
    env_ids=None,
    axis: str = "y",
    start=(-0.02, 0.02),
    end=(-0.10, 0.10),
    duration_steps: int = 6_000_000,
):
    """Write current reset pose_range override into env cache.
    This is called by CurriculumManager.
    """
    # IsaacLab 通常有 common_step_counter；没有的话就退化为 0
    step = int(getattr(env, "common_step_counter", 0))
    if duration_steps <= 0:
        p = 1.0
    else:
        p = max(0.0, min(1.0, step / float(duration_steps)))

    lo = float(start[0] + p * (end[0] - start[0]))
    hi = float(start[1] + p * (end[1] - start[1]))

    # 写入缓存（reset 时读取覆盖）
    if not hasattr(env, "_curriculum_reset_pose_range"):
        env._curriculum_reset_pose_range = {}
    env._curriculum_reset_pose_range[axis] = (lo, hi)

    # 可选：给 logger 用
    return torch.tensor([p], device=env.device) if hasattr(env, "device") else p


def widen_reset_y(
    env,
    env_ids,
    axis: str = "y",
    start: float = 0.0,
    end: float = 0.10,
    duration_steps: int = 200_000,
):
    step = int(getattr(env, "common_step_counter", 0))
    alpha = min(step / float(max(duration_steps, 1)), 1.0)
    cur = start + alpha * (end - start)

    if not hasattr(env, "_curriculum_reset_pose_range"):
        env._curriculum_reset_pose_range = {}
    env._curriculum_reset_pose_range[axis] = (-cur, cur)

    # 给 runner 读（可选，但很实用）
    env._curriculum_debug = {
        "step": step,
        "axis": axis,
        "alpha": alpha,
        "lo": -cur,
        "hi": cur,
    }

    # 返回一个 tensor（可选：CurriculumManager 可能拿这个做日志）
    if hasattr(env, "device"):
        return torch.tensor([cur], device=env.device)
    return cur
