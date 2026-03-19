#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import torch

PT_PATH = "/home/robo/下载/compare/sa3_model_410.pt"
OUT_PATH = "/home/robo/下载/compare/sa3_first6.txt"

TARGET_KEYS = ("sa1", "sa2", "sa3")

# 这些名字一律认为是 BN / Norm，直接排除
NORM_PATTERNS = (
    "bn", "batchnorm", "norm", "ln", "layernorm", "gn", "groupnorm"
)

# 这些名字更像 Conv / Linear 层，优先保留
CONV_LINEAR_HINTS = (
    "conv", "fc", "linear", "mlp"
)


def find_state_dict(obj):
    """尽量从 checkpoint 中找到真正的 state_dict"""
    if isinstance(obj, dict):
        for key in ["model_state_dict", "state_dict", "model", "policy", "actor_critic"]:
            if key in obj and isinstance(obj[key], dict):
                return obj[key]

        if any(torch.is_tensor(v) for v in obj.values()):
            return obj

    raise ValueError("没能在 checkpoint 中找到可用的 state_dict。")


def is_sa_param(name: str) -> bool:
    lower = name.lower()
    return (
        lower.startswith(TARGET_KEYS)
        or ".sa1." in lower
        or ".sa2." in lower
        or ".sa3." in lower
    )


def is_weight_or_bias(name: str) -> bool:
    return name.endswith(".weight") or name.endswith(".bias")


def is_norm_layer(layer_name: str) -> bool:
    lower = layer_name.lower()
    parts = re.split(r"[._]", lower)
    for p in parts:
        if p in NORM_PATTERNS:
            return True
    for pat in NORM_PATTERNS:
        if f".{pat}." in lower or lower.endswith(f".{pat}") or lower.startswith(f"{pat}."):
            return True
    return False


def looks_like_conv_or_linear(layer_name: str, tensor: torch.Tensor, kind: str) -> bool:
    """
    在只拿到 state_dict 的情况下做近似判断：
    1) 层名带 conv/fc/linear/mlp，优先认为是 Conv/Linear
    2) weight:
       - ndim >= 2 通常可能是 conv/linear 的 weight
    3) bias:
       - ndim == 1，且层名不能像 norm
    """
    lower = layer_name.lower()

    if any(h in lower for h in CONV_LINEAR_HINTS):
        return True

    if kind == "weight":
        return tensor.ndim >= 2
    elif kind == "bias":
        return tensor.ndim == 1 and not is_norm_layer(layer_name)

    return False


def sort_key(layer_name: str):
    lower = layer_name.lower()
    if "sa1" in lower:
        group = 1
    elif "sa2" in lower:
        group = 2
    elif "sa3" in lower:
        group = 3
    else:
        group = 99
    return (group, layer_name)


def first_n_values(t: torch.Tensor, n=6):
    flat = t.detach().cpu().reshape(-1)
    vals = flat[:n].tolist()
    return vals


def main():
    print(f"Loading: {PT_PATH}")
    ckpt = torch.load(PT_PATH, map_location="cpu")
    state_dict = find_state_dict(ckpt)

    grouped = {}

    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            continue
        if not is_sa_param(name):
            continue
        if not is_weight_or_bias(name):
            continue

        if name.endswith(".weight"):
            layer_name = name[:-7]
            kind = "weight"
        else:
            layer_name = name[:-5]
            kind = "bias"

        # 排除 BN / Norm
        if is_norm_layer(layer_name):
            continue

        # 只保留 Conv / Linear
        if not looks_like_conv_or_linear(layer_name, tensor, kind):
            continue

        grouped.setdefault(layer_name, {})
        grouped[layer_name][kind] = tensor.detach().cpu()

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(f"Loading: {PT_PATH}\n")
        f.write(f"state_dict 参数总数: {len(state_dict)}\n\n")

        if not grouped:
            f.write("没有找到符合条件的参数。\n")
            f.write("条件：sa1/sa2/sa3 + 非BN + Conv/Linear + weight/bias\n")
            print(f"没有找到符合条件的参数，结果已写入: {OUT_PATH}")
            return

        f.write("筛选条件：sa1/sa2/sa3 + 非BN + Conv/Linear + weight/bias\n")
        f.write("每个 tensor 只打印前 6 个参数值（flatten 后）\n\n")

        for layer_name in sorted(grouped.keys(), key=sort_key):
            f.write("=" * 120 + "\n")
            f.write(f"Layer: {layer_name}\n")

            if "weight" in grouped[layer_name]:
                w = grouped[layer_name]["weight"]
                f.write(f"[weight] shape = {tuple(w.shape)}\n")
                f.write(f"[weight] first 6 = {first_n_values(w, 6)}\n")

            if "bias" in grouped[layer_name]:
                b = grouped[layer_name]["bias"]
                f.write(f"[bias] shape = {tuple(b.shape)}\n")
                f.write(f"[bias] first 6 = {first_n_values(b, 6)}\n")

            f.write("\n")

    print(f"已写入: {OUT_PATH}")


if __name__ == "__main__":
    main()