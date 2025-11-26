# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the functions that are specific to the lift environments."""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .observations import *  # noqa: F401, F403
from .rewards import *  # noqa: F401, F403
from .terminations import *  # noqa: F401, F403

import torch

def debug_print_semantic_ids(env, max_envs: int = 1):
    """ 打印depth_camera的语义分割图的语义 ID 和 idToLabels 映射，用于调试 """
    scene = env.scene
    if "depth_camera" not in scene.sensors:
        print("[debug_print_semantic_ids] No depth_camera found in the scene sensors.")
        return
    cam = scene.sensors["depth_camera"]
    data = cam.data
    print("========== [debug_print_semantic_ids] CameraData introspection ==========")
    print("type(cam.data):", type(data))

    # 1. 打印data对象的属性名称
    attrs = [a for a in dir(data) if not a.startswith("_")]
    print("cam.data attributes:", attrs)

    # 2. 如果有 output
    if hasattr(data, "output"):
        try:
            keys = list(data.output.keys())
        except Exception:
            keys = []
        print("cam.data.output keys:", keys)
    else:
        print("cam.data has no attribute 'output'")
    
    # 3. 如果有 info，打印 info 的键
    if hasattr(data, "info"):
        try:
            info_keys = list(data.info.keys())
        except Exception:
            info_keys = []
        print("cam.data.info keys:", info_keys)
    else:
        print("cam.data has no attribute 'info'")

    # --- 尝试抓 semantic_segmentation 缓冲 ---
    seg = None

    # 3.1 优先看是不是有 data.semantic_segmentation 这个属性
    if hasattr(data, "semantic_segmentation"):
        seg = data.semantic_segmentation
        print("[debug] found data.semantic_segmentation")
    # 3.2 其次看 output 字典里有没有这个 key
    elif hasattr(data, "output") and isinstance(data.output, dict):
        for k in data.output.keys():
            if "semantic" in k.lower():   # 容忍 semanticSegmentation / semantic_segmentation 等
                seg = data.output[k]
                print(f"[debug] found semantic buffer in data.output['{k}']")
                break

    if seg is None:
        print("[debug_print_semantic_ids] ❌ semantic segmentation buffer not found yet.")
        print("==============================================================")
        return

    print("[debug] seg tensor shape:", seg.shape, "dtype:", getattr(seg, "dtype", None))

    # 统一处理最后一维
    seg_tensor = torch.as_tensor(seg)
    if seg_tensor.dim() == 4 and seg_tensor.shape[-1] > 1:
        seg_ids = seg_tensor[..., 0].to(torch.int64)  # (N, H, W)
    elif seg_tensor.dim() == 4 and seg_tensor.shape[-1] == 1:
        seg_ids = seg_tensor[..., 0].to(torch.int64)  # (N, H, W)
    elif seg_tensor.dim() == 3:
        seg_ids = seg_tensor.to(torch.int64)
    else:
        print("[debug_print_semantic_ids] Unexpected seg tensor shape:", seg_tensor.shape)
        print("==============================================================")
        return

    num_envs = min(env.num_envs, max_envs)
    for i in range(num_envs):
        seg_i = seg_ids[i]  # (H, W)
        unique_ids = torch.unique(seg_i)
        print(f"[Env {i}] unique semantic IDs:", unique_ids.tolist())

    # info 里面有 ID ↔ label 映射
    if hasattr(data, "info"):
        info = data.info.get("semantic_segmentation", {})
        id_to_labels = info.get("idToLabels", {})
        print("\nidToLabels:")
        for k, v in id_to_labels.items():
            print(f"  {k}: {v}")
    else:
        print("[debug_print_semantic_ids] data.info not available.")

    print("==============================================================")


def debug_print_semantic_ids_on_reset(env, env_ids, max_envs: int = 2, print_every: int = 1):
    """作为 EventTerm 在每次 reset 时调用。
    env_ids 是这次需要 reset 的环境索引（IsaacLab 自动传的），这里其实可以不用。
    print_every 控制：每多少次 reset 打印一次。
    """
    counter = getattr(env, "_semantic_debug_counter", 0)

    if counter % print_every == 0:
        print(f"[debug_semantic_on_reset] reset #{counter}, env_ids={env_ids}")
        debug_print_semantic_ids(env, max_envs=max_envs)

    env._semantic_debug_counter = counter + 1