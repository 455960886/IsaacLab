#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import torch

PT_OLD = "/home/robo/下载/compare/freeze_model_410.pt"
PT_NEW = "/home/robo/下载/compare/sa3_model_540.pt"

# True: 同时把结果写到文件
SAVE_TO_FILE = True
OUT_PATH = "/home/robo/下载/compare/compare_summary.txt"


def load_state_dict(path):
    ckpt = torch.load(path, map_location="cpu")
    if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        return ckpt["model_state_dict"]
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if isinstance(ckpt, dict) and all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt
    raise ValueError(f"在 {path} 里没找到 state_dict / model_state_dict")


def is_trainable_param(name: str) -> bool:
    # 只看真正训练会更新的 weight / bias
    return name.endswith(".weight") or name.endswith(".bias") or name == "std"


def get_group(name: str) -> str:
    if name.startswith("point_encoder.sa1."):
        return "pointnet.sa1"
    if name.startswith("point_encoder.sa2."):
        return "pointnet.sa2"
    if name.startswith("point_encoder.sa3."):
        return "pointnet.sa3"
    if name.startswith("actor."):
        return "actor"
    if name.startswith("critic."):
        return "critic"
    if name == "std":
        return "std"
    return "other"


def summarize_group(rows):
    """
    rows: [(name, mean_abs, max_abs, l2, numel)]
    """
    if not rows:
        return {
            "tensor_count": 0,
            "numel": 0,
            "mean_abs": 0.0,
            "max_abs": 0.0,
            "l2": 0.0,
            "top_name": "-",
            "top_mean_abs": 0.0,
        }

    total_numel = sum(r[4] for r in rows)
    weighted_mean_abs = sum(r[1] * r[4] for r in rows) / total_numel
    global_max_abs = max(r[2] for r in rows)
    global_l2 = math.sqrt(sum((r[3] ** 2) for r in rows))

    top_row = max(rows, key=lambda x: x[1])  # 按 mean_abs 最大找变化最明显的层

    return {
        "tensor_count": len(rows),
        "numel": total_numel,
        "mean_abs": weighted_mean_abs,
        "max_abs": global_max_abs,
        "l2": global_l2,
        "top_name": top_row[0],
        "top_mean_abs": top_row[1],
    }


def build_report(sd_old, sd_new):
    groups = {
        "pointnet.sa1": [],
        "pointnet.sa2": [],
        "pointnet.sa3": [],
        "actor": [],
        "critic": [],
        "std": [],
    }

    common_keys = sorted(set(sd_old.keys()) & set(sd_new.keys()))

    for name in common_keys:
        if not is_trainable_param(name):
            continue

        t_old = sd_old[name]
        t_new = sd_new[name]

        if not (torch.is_tensor(t_old) and torch.is_tensor(t_new)):
            continue
        if t_old.shape != t_new.shape:
            continue

        group = get_group(name)
        if group == "other":
            continue

        delta = (t_new - t_old).float()
        abs_delta = delta.abs()

        mean_abs = abs_delta.mean().item()
        max_abs = abs_delta.max().item()
        l2 = torch.norm(delta, p=2).item()
        numel = delta.numel()

        groups[group].append((name, mean_abs, max_abs, l2, numel))

    summary = {g: summarize_group(rows) for g, rows in groups.items()}
    return summary


def format_line(name, s):
    return (
        f"{name:<12} | "
        f"层数 {s['tensor_count']:>2} | "
        f"参数量 {s['numel']:>8} | "
        f"平均变化 {s['mean_abs']:.8f} | "
        f"最大变化 {s['max_abs']:.8f}"
    )


def main():
    sd_old = load_state_dict(PT_OLD)
    sd_new = load_state_dict(PT_NEW)

    summary = build_report(sd_old, sd_new)

    lines = []
    lines.append("===== checkpoint 参数变化简洁总结 =====")
    lines.append(f"旧参数: {PT_OLD}")
    lines.append(f"新参数: {PT_NEW}")
    lines.append("")
    lines.append("说明：")
    lines.append("1. 这里只比较真正训练会更新的 weight / bias / std")
    lines.append("2. 已自动跳过 BN 的 running_mean / running_var / num_batches_tracked")
    lines.append("3. 平均变化 = 所有参数元素的 |new-old| 平均值")
    lines.append("4. 最大变化 = 该组里单个参数元素的最大 |new-old|")
    lines.append("")

    lines.append("---- PointNet ----")
    lines.append(format_line("pointnet.sa1", summary["pointnet.sa1"]))
    lines.append(format_line("pointnet.sa2", summary["pointnet.sa2"]))
    lines.append(format_line("pointnet.sa3", summary["pointnet.sa3"]))
    lines.append("")

    lines.append("---- Actor-Critic ----")
    lines.append(format_line("actor", summary["actor"]))
    lines.append(format_line("critic", summary["critic"]))
    lines.append(format_line("std", summary["std"]))
    lines.append("")

    # 找 pointnet 和 actor_critic 各自变化最大的一块
    pointnet_best = max(
        ["pointnet.sa1", "pointnet.sa2", "pointnet.sa3"],
        key=lambda k: summary[k]["mean_abs"]
    )
    ac_best = max(
        ["actor", "critic", "std"],
        key=lambda k: summary[k]["mean_abs"]
    )

    lines.append("---- 一句话结论 ----")
    lines.append(
        f"PointNet 里变化最大的是 {pointnet_best}，"
        f"其中最明显的层是 {summary[pointnet_best]['top_name']} "
        f"(平均变化 {summary[pointnet_best]['top_mean_abs']:.8f})"
    )
    lines.append(
        f"Actor-Critic 里变化最大的是 {ac_best}，"
        f"其中最明显的层是 {summary[ac_best]['top_name']} "
        f"(平均变化 {summary[ac_best]['top_mean_abs']:.8f})"
    )

    report = "\n".join(lines)
    print(report)

    if SAVE_TO_FILE:
        with open(OUT_PATH, "w", encoding="utf-8") as f:
            f.write(report + "\n")
        print(f"\n已写入: {OUT_PATH}")


if __name__ == "__main__":
    main()