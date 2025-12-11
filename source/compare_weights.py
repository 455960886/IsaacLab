#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
对比两个 checkpoint 里各层参数的变化情况。

用法示例：
    python source/compare_weights.py \
        --ckpt_a logs/rsl_rl/coarse_arm_lift/2025-12-09_21-59-43/model_0.pt \
        --ckpt_b logs/rsl_rl/coarse_arm_lift/2025-12-09_21-59-43/model_3640.pt \
        --top_k 40 \
        --filter rgb_backbone,pcd_backbone,actor,critic
"""

import argparse
import torch
import numpy as np


def load_state_dict(path):
    print(f"[INFO] Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        # 常见结构：{ 'model_state_dict': ..., ... }
        if "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
            print("[INFO] Using key 'model_state_dict'")
        else:
            sd = ckpt
            print("[WARN] 'model_state_dict' not found, using checkpoint directly as state_dict")
    else:
        raise RuntimeError(f"Unsupported checkpoint type: {type(ckpt)}")
    return sd


def tensor_stats(t: torch.Tensor):
    """返回 mean, std，都是 float"""
    return t.mean().item(), t.std().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_a", type=str, required=True, help="第一个 checkpoint 路径（通常是 model_0.pt）")
    parser.add_argument("--ckpt_b", type=str, required=True, help="第二个 checkpoint 路径（例如 model_3640.pt）")
    parser.add_argument("--top_k", type=int, default=30, help="打印变化最大的前 K 个参数")
    parser.add_argument(
        "--filter",
        type=str,
        default="",
        help="只保留名字里包含这些子串的参数，多项用逗号分隔，例如: rgb_backbone,pcd_backbone,actor,critic",
    )
    args = parser.parse_args()

    sd_a = load_state_dict(args.ckpt_a)
    sd_b = load_state_dict(args.ckpt_b)

    # 需要对比的 key = 两个 state_dict 的交集
    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())
    common_keys = sorted(list(keys_a & keys_b))

    print(f"[INFO] ckpt A 参数个数: {len(keys_a)}")
    print(f"[INFO] ckpt B 参数个数: {len(keys_b)}")
    print(f"[INFO] 交集参数个数: {len(common_keys)}")

    # 处理 filter
    filter_substrings = []
    if args.filter.strip():
        filter_substrings = [s.strip() for s in args.filter.split(",") if s.strip()]
        print(f"[INFO] 过滤关键词: {filter_substrings}")

    rows = []

    for name in common_keys:
        # 根据 filter 过滤
        if filter_substrings:
            if not any(sub in name for sub in filter_substrings):
                continue

        wa = sd_a[name]
        wb = sd_b[name]

        # 只看 tensor 参数
        if not isinstance(wa, torch.Tensor) or not isinstance(wb, torch.Tensor):
            continue
        if wa.shape != wb.shape:
            # 理论上不该发生，有的话打印出来提醒一下
            print(f"[WARN] shape mismatch for {name}: {wa.shape} vs {wb.shape}")
            continue

        # 🚩 新增：只对浮点类型做比较，跳过 Long / Int 等
        if not wa.is_floating_point() or not wb.is_floating_point():
            # 例如 batchnorm 的 num_batches_tracked 就是 Long 类型，这类我们不关心
            # print(f"[INFO] skip non-float param: {name} ({wa.dtype})")
            continue

        # 展平为 1D，方便计算 L2 差
        diff = (wa - wb).view(-1)

        l2_diff = diff.norm().item()  # 绝对变化量
        # 归一化一下：除以参数自身的 L2 范数，避免层规模太大时差值自然变大
        denom = (wa.view(-1).norm().item() + 1e-8)
        rel_l2_diff = l2_diff / denom

        mean_a, std_a = tensor_stats(wa)
        mean_b, std_b = tensor_stats(wb)

        rows.append(
            {
                "name": name,
                "l2_diff": l2_diff,
                "rel_l2_diff": rel_l2_diff,
                "mean_a": mean_a,
                "std_a": std_a,
                "mean_b": mean_b,
                "std_b": std_b,
            }
        )

    if not rows:
        print("[WARN] 没有任何参数通过 filter 筛选，请检查 --filter 或 checkpoint 路径是否正确。")
        return

    # 按相对变化量排序（rel_l2_diff 大的说明“变得更多”）
    rows.sort(key=lambda r: r["rel_l2_diff"], reverse=True)

    top_k = min(args.top_k, len(rows))
    print(f"\n[RESULT] 变化最大的前 {top_k} 个参数：\n")
    header = f"{'name':70s} | {'rel_l2':>10s} | {'l2_diff':>10s} | {'std_a':>8s} -> {'std_b':>8s}"
    print(header)
    print("-" * len(header))

    for r in rows[:top_k]:
        print(
            f"{r['name'][:70]:70s} | "
            f"{r['rel_l2_diff']:10.4e} | "
            f"{r['l2_diff']:10.4e} | "
            f"{r['std_a']:8.4f} -> {r['std_b']:8.4f}"
        )

    print("\n[INFO] 提示：")
    print("  - rel_l2 越大，说明这一层从 ckpt_a 到 ckpt_b 的变化越明显。")
    print("  - std_a / std_b 方便你看这一层的参数分布有没有变宽 / 变窄。")
    print("  - 你可以用 --filter 来只看 rgb_backbone / pcd_backbone / actor / critic 等子模块。")


if __name__ == "__main__":
    main()
