#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

OLD_PT = "/home/robo/下载/compare/freeze_model_410.pt"
NEW_PT = "/home/robo/下载/compare/sa3_model_410.pt"


def load_state_dict(path):
    ckpt = torch.load(path, map_location="cpu")
    if "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
        return ckpt["model_state_dict"]
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    if isinstance(ckpt, dict) and all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt
    raise ValueError(f"在 {path} 里没找到 state_dict")


def diff_stats(a, b):
    d = (b - a).float().reshape(-1)
    ad = d.abs()
    return {
        "mean_abs": ad.mean().item(),
        "max_abs": ad.max().item(),
        "first6": d[:6].tolist(),
    }


def print_one(sd_old, sd_new, name):
    s = diff_stats(sd_old[name], sd_new[name])
    print(f"{name}")
    print(f"  mean_abs = {s['mean_abs']:.8f}")
    print(f"  max_abs  = {s['max_abs']:.8f}")
    print(f"  first6   = {s['first6']}")
    print()


def main():
    sd_old = load_state_dict(OLD_PT)
    sd_new = load_state_dict(NEW_PT)

    print("===== 只看最关键的 3 类变化 =====\n")

    print("---- 1) sa3 的 BN running stats ----")
    bn_keys = [
        "point_encoder.sa3.mlp_bns.0.running_mean",
        "point_encoder.sa3.mlp_bns.0.running_var",
        "point_encoder.sa3.mlp_bns.0.num_batches_tracked",
        "point_encoder.sa3.mlp_bns.1.running_mean",
        "point_encoder.sa3.mlp_bns.1.running_var",
        "point_encoder.sa3.mlp_bns.1.num_batches_tracked",
        "point_encoder.sa3.mlp_bns.2.running_mean",
        "point_encoder.sa3.mlp_bns.2.running_var",
        "point_encoder.sa3.mlp_bns.2.num_batches_tracked",
    ]
    for k in bn_keys:
        print_one(sd_old, sd_new, k)

    print("---- 2) 动作探索强度 std ----")
    print(f"std old = {sd_old['std'].tolist()}")
    print(f"std new = {sd_new['std'].tolist()}")
    print(f"std diff= {(sd_new['std'] - sd_old['std']).tolist()}")
    print()

    print("---- 3) actor 最后一层输出层 ----")
    for k in ["actor.8.weight", "actor.8.bias"]:
        print_one(sd_old, sd_new, k)

    print("===== 一句话判断 =====")
    print("如果 sa3 的 BN running_mean/running_var 变化远大于 sa3 的 weight/bias，")
    print("那么 reward 的大断层更像是 BN 统计量漂移引起的，而不是权重本身突变。")


if __name__ == "__main__":
    main()