# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Helpers for object-pool scale randomization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import omni.usd
from pxr import Gf, Sdf

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCollection
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _normalize_base_scale(base_scale) -> tuple[float, float, float]:
    """Normalize scale config into a 3D tuple."""
    if base_scale is None:
        return (1.0, 1.0, 1.0)
    if isinstance(base_scale, (float, int)):
        value = float(base_scale)
        return (value, value, value)
    if len(base_scale) != 3:
        raise ValueError(f"scale 必须是长度为 3 的 tuple，当前收到: {base_scale}")
    return tuple(float(v) for v in base_scale)


def _sample_scale_factors(
    num_samples: int,
    scale_factor_range: tuple[float, float] | dict[str, tuple[float, float]],
) -> torch.Tensor:
    """Sample multiplicative scale factors on CPU."""
    if isinstance(scale_factor_range, dict):
        invalid_keys = set(scale_factor_range.keys()) - {"x", "y", "z"}
        if invalid_keys:
            raise ValueError(
                f"scale_factor_range 只支持 x/y/z 三个轴，收到非法键: {sorted(invalid_keys)}"
            )
        range_tensor = torch.tensor(
            [scale_factor_range.get(axis, (1.0, 1.0)) for axis in ("x", "y", "z")],
            dtype=torch.float32,
            device="cpu",
        )
        rand = torch.rand((num_samples, 3), dtype=torch.float32, device="cpu")
        lower = range_tensor[:, 0].unsqueeze(0)
        upper = range_tensor[:, 1].unsqueeze(0)
        return lower + rand * (upper - lower)

    lower, upper = scale_factor_range
    rand = torch.rand((num_samples, 1), dtype=torch.float32, device="cpu")
    return (lower + rand * (upper - lower)).repeat(1, 3)


def randomize_object_pool_scale_prestartup(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    scale_factor_range: tuple[float, float] | dict[str, tuple[float, float]],
    log_per_env_scales: bool = False,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
):
    """Randomize object-pool scale once before simulation starts.

    发生在 prestartup 阶段，每个 env 的尺寸会在整个训练过程中保持固定。
    """
    if env.sim.is_playing():
        raise RuntimeError("物体尺寸随机化必须在仿真开始前执行，请将该事件配置为 mode='prestartup'。")

    object_collection = env.scene[asset_cfg.name]
    if not isinstance(object_collection, RigidObjectCollection):
        raise TypeError(
            f"{asset_cfg.name} 不是 RigidObjectCollection，当前类型为: {type(object_collection)}"
        )

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    object_pool_cfg = getattr(env.cfg.scene, asset_cfg.name)
    rigid_object_cfgs = object_pool_cfg.rigid_objects
    object_names = list(rigid_object_cfgs.keys())
    env.object_pool_randomized_scale_object_names = object_names
    env.object_pool_randomized_scales = torch.full(
        (env.scene.num_envs, len(object_names), 3),
        float("nan"),
        dtype=torch.float32,
        device="cpu",
    )
    stage = omni.usd.get_context().get_stage()

    print(
        f"[尺寸随机化] 开始为 {asset_cfg.name} 执行启动期尺寸随机化："
        f"env数量={len(env_ids)}，物体数量={len(rigid_object_cfgs)}。",
        flush=True,
    )
    print(
        "[尺寸随机化] 注意：这里修改的是几何尺寸，不会自动同步修改配置里显式写死的质量参数。",
        flush=True,
    )

    with Sdf.ChangeBlock():
        for object_idx, (object_name, object_cfg) in enumerate(rigid_object_cfgs.items()):
            prim_paths = sim_utils.find_matching_prim_paths(object_cfg.prim_path)
            if len(prim_paths) < env.scene.num_envs:
                raise RuntimeError(
                    f"物体 {object_name} 匹配到的 prim 数量不足，期望至少 {env.scene.num_envs} 个，"
                    f"实际只有 {len(prim_paths)} 个。"
                )

            raw_base_scale = getattr(object_cfg.spawn, "scale", None)
            base_scale = _normalize_base_scale(raw_base_scale)
            base_scale_tensor = torch.tensor(base_scale, dtype=torch.float32, device="cpu").unsqueeze(0)

            if raw_base_scale is None:
                print(
                    f"[尺寸随机化] 物体 {object_name} 未显式配置基础 scale，默认按 (1.0, 1.0, 1.0) 作为基准。",
                    flush=True,
                )

            sampled_factors = _sample_scale_factors(len(env_ids), scale_factor_range)
            sampled_scales = sampled_factors * base_scale_tensor

            scale_min = sampled_scales.min(dim=0).values.tolist()
            scale_max = sampled_scales.max(dim=0).values.tolist()
            print(
                f"[尺寸随机化] 物体 {object_name}：基础scale={base_scale}，"
                f"随机后范围 x=[{scale_min[0]:.4f}, {scale_max[0]:.4f}]，"
                f"y=[{scale_min[1]:.4f}, {scale_max[1]:.4f}]，"
                f"z=[{scale_min[2]:.4f}, {scale_max[2]:.4f}]",
                flush=True,
            )

            env.object_pool_randomized_scales[env_ids.long(), object_idx] = sampled_scales

            if log_per_env_scales:
                for sample_id, env_id in enumerate(env_ids.tolist()):
                    final_scale = sampled_scales[sample_id].tolist()
                    print(
                        f"[尺寸随机化][逐环境] env_{env_id:03d} | 物体={object_name} | "
                        f"最终scale=({final_scale[0]:.6f}, {final_scale[1]:.6f}, {final_scale[2]:.6f})",
                        flush=True,
                    )

            for sample_id, env_id in enumerate(env_ids.tolist()):
                prim_path = prim_paths[env_id]
                prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

                scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
                if scale_spec is None:
                    scale_spec = Sdf.AttributeSpec(
                        prim_spec,
                        prim_path + ".xformOp:scale",
                        Sdf.ValueTypeNames.Double3,
                    )

                scale_spec.default = Gf.Vec3f(*sampled_scales[sample_id].tolist())

    print(
        "[尺寸随机化] object_pool 尺寸随机化完成。后续 reset 只会随机位置/姿态，不会再次改变尺寸。",
        flush=True,
    )


def log_active_object_pool_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
):
    """Print only the final scale of the active object in each env."""
    if not hasattr(env, "active_object_indices"):
        print("[尺寸随机化][警告] 未找到 active_object_indices，无法打印 active 物体的 scale。", flush=True)
        return

    if not hasattr(env, "object_pool_randomized_scales"):
        print("[尺寸随机化][警告] 未找到启动期缓存的 scale，无法打印 active 物体的 scale。", flush=True)
        return

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    object_names = env.object_pool_randomized_scale_object_names
    cached_scales = env.object_pool_randomized_scales

    print(f"[尺寸随机化] 开始打印每个环境当前 active 物体的最终 scale：env数量={len(env_ids)}。", flush=True)
    for env_id in env_ids.tolist():
        active_idx = int(env.active_object_indices[env_id].item())
        active_name = object_names[active_idx]
        final_scale = cached_scales[env_id, active_idx].tolist()
        print(
            f"[尺寸随机化][active] env_{env_id:03d} | 物体={active_name} | "
            f"最终scale=({final_scale[0]:.6f}, {final_scale[1]:.6f}, {final_scale[2]:.6f})",
            flush=True,
        )
