# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# -----------------------------------------------------------------------------
# Logging helpers (single-process friendly)
# -----------------------------------------------------------------------------

def _log(msg: str, level: str = "warn") -> None:
    """Log through carb if available (Isaac Sim runtime), otherwise fallback to print."""
    try:
        import carb  # available in Isaac Sim / Kit runtime
        if level == "info":
            carb.log_info(msg)
        elif level == "error":
            carb.log_error(msg)
        else:
            carb.log_warn(msg)
    except Exception:
        print(msg, flush=True)


def _step(env) -> int:
    return int(getattr(env, "common_step_counter", 0))


def _env_ids_sample(env_ids: Sequence[int] | torch.Tensor, k: int = 5) -> list[int]:
    if isinstance(env_ids, torch.Tensor):
        return [int(x) for x in env_ids.detach().cpu().tolist()[:k]]
    return [int(x) for x in list(env_ids)[:k]]


def _format_origins_y(env, sample_env_ids: list[int]) -> str:
    """Best-effort fetch of env.scene.env_origins[:, 1] to prove per-env origin correctness."""
    try:
        origins = env.scene.env_origins  # (num_envs, 3) on device
        idx = torch.tensor(sample_env_ids, device=origins.device, dtype=torch.long)
        oy = origins[idx, 1].detach().cpu().tolist()
        oy = [float(v) for v in oy]
        return f" sample_env_origins(y)={oy}"
    except Exception as e:
        return f" (env_origins unavailable: {e})"


# -----------------------------------------------------------------------------
# Curriculum terms
# -----------------------------------------------------------------------------

def curriculum_expand_object_spawn_y_range_linear(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int] | torch.Tensor,
    y_range_start: tuple[float, float],
    y_range_end: tuple[float, float],
    start_step: int,
    end_step: int,
    update_every_steps: int = 200,      # 只控制“写 override”的频率（建议别太小）
    debug: bool = True,
    debug_every_steps: int = 2000,      # 打印频率（建议 >= 2000）
    store_attr: str = "_curriculum_reset_pose_range_override",
) -> None:
    """
    Gradually expand the object's spawn/reset y-range from small to large.

    How it works:
    - Write an override dict on env: env._curriculum_reset_pose_range_override["y"] = (y_min, y_max)
    - Your events.py::reset_object_pool_state_uniform merges this override into pose_range.

    Notes:
    - y_range is delta-y relative to env.scene.env_origins[env_id]
    - Works with parallel env origins naturally.
    """
    step = _step(env)

    # --------- Sanity checks ---------
    if end_step <= start_step:
        raise ValueError(
            f"[CURR][y_range] end_step must be > start_step, got start_step={start_step}, end_step={end_step}"
        )
    if y_range_start[0] > y_range_start[1] or y_range_end[0] > y_range_end[1]:
        raise ValueError(f"[CURR][y_range] invalid ranges: start={y_range_start}, end={y_range_end}")

    # --------- First-call probe (always log once) ---------
    if not getattr(env, "_curriculum_obj_y_first_call_logged", False):
        setattr(env, "_curriculum_obj_y_first_call_logged", True)
        sample_ids = _env_ids_sample(env_ids, 5)
        _log(f"[CURR][y_range] FIRST_CALL step={step} env_ids(sample)={sample_ids}", level="warn")

    # --------- Compute progress ---------
    if step <= start_step:
        progress = 0.0
    elif step >= end_step:
        progress = 1.0
    else:
        progress = (step - start_step) / float(end_step - start_step)

    # Linear interpolation
    y_min = y_range_start[0] + (y_range_end[0] - y_range_start[0]) * progress
    y_max = y_range_start[1] + (y_range_end[1] - y_range_start[1]) * progress
    y_range_now = (float(y_min), float(y_max))

    # --------- Throttle override update ---------
    last_update_step = int(getattr(env, "_curriculum_obj_y_last_update_step", -10**9))
    should_update = (step - last_update_step) >= int(update_every_steps) or step in (start_step, end_step)

    if should_update:
        override = getattr(env, store_attr, None)
        if override is None or not isinstance(override, dict):
            override = {}
            setattr(env, store_attr, override)

        override["y"] = y_range_now
        env._curriculum_obj_y_last_update_step = step

    # --------- Debug logs (single-process, no rank gating) ---------
    if debug:
        last_print_step = int(getattr(env, "_curriculum_obj_y_last_print_step", -10**9))
        should_print = (step - last_print_step) >= int(debug_every_steps) or step in (start_step, end_step)

        if should_print:
            env._curriculum_obj_y_last_print_step = step
            sample_ids = _env_ids_sample(env_ids, 5)
            origins_info = _format_origins_y(env, sample_ids)
            _log(
                "[CURR][y_range] "
                f"step={step} progress={progress:.4f} "
                f"y_range_now={y_range_now} "
                f"updated={should_update} "
                f"store_attr={store_attr} "
                f"env_ids(sample)={sample_ids}"
                f"{origins_info}",
                level="warn",
            )


def modify_reward_weight(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int] | torch.Tensor,
    term_name: str,
    weight: float,
    num_steps: int,
    debug: bool = False,
    debug_every_steps: int = 5000,
) -> None:
    """Curriculum that modifies a reward weight after a given number of steps."""
    step = _step(env)

    if step > int(num_steps):
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        if float(term_cfg.weight) != float(weight):
            term_cfg.weight = float(weight)
            env.reward_manager.set_term_cfg(term_name, term_cfg)
            _log(f"[CURR][reward] step={step} set {term_name}.weight={weight}", level="warn")

    if debug:
        last = int(getattr(env, "_curriculum_reward_last_print_step", -10**9))
        if (step - last) >= int(debug_every_steps):
            env._curriculum_reward_last_print_step = step
            _log(f"[CURR][reward] step={step} watching term={term_name} target_weight={weight}", level="info")
