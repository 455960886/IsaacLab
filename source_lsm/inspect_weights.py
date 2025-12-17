import torch
from pathlib import Path

# ===== 1. 修改成你要看的那个 checkpoint 路径 =====
# 例如你说的：
# /home/robo/code/IsaacLab/logs/rsl_rl/coarse_arm_lift/2025-12-09_21-59-43/model_100.pt
ckpt_path = Path("/home/robo/code/IsaacLab/logs/rsl_rl/coarse_arm_lift/2025-12-09_21-59-43/model_0.pt")

print(f"Loading checkpoint from: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location="cpu")

print(">>> ckpt type:", type(ckpt))
if isinstance(ckpt, dict):
    print(">>> ckpt keys:", list(ckpt.keys())[:20])

# ===== 2. 自动猜测 state_dict 在哪里 =====
state_dict = None

# 常见几种保存方式
for key in [
    "model_state_dict",
    "policy_state_dict",
    "actor_critic_state_dict",
    "state_dict",
]:
    if isinstance(ckpt, dict) and key in ckpt:
        state_dict = ckpt[key]
        print(f"Using state_dict from key: {key}")
        break

# 有些直接就是 state_dict
if state_dict is None and isinstance(ckpt, dict):
    # 简单判断：key 看起来像 'mlp.0.weight' 这种
    sample_keys = list(ckpt.keys())
    if all(isinstance(k, str) for k in sample_keys):
        print("No nested key found, treating ckpt itself as state_dict.")
        state_dict = ckpt

assert state_dict is not None, "没找到 state_dict，请看上面的 ckpt.keys() 自己确认结构"

# ===== 3. 打印每一层权重的统计量 =====
print("\n=== All weight layers stats ===")
for name, tensor in state_dict.items():
    if not isinstance(tensor, torch.Tensor):
        continue
    # 先只关心真正的权重，不看 bias / BN 的 running_* 参数
    if name.endswith(".bias") or "running_mean" in name or "running_var" in name:
        continue

    w = tensor.float()
    print(
        f"{name:60s} | shape={tuple(w.shape)!s:18s} "
        f"mean={w.mean().item(): .4f} std={w.std().item(): .4f} "
        f"min={w.min().item(): .4f} max={w.max().item(): .4f}"
    )

# ===== 4. 专门看 ResNet 和 PCD（按名字筛选，你自己看 state_dict 里叫什么） =====
print("\n=== RGB / ResNet 相关的权重（name 里包含 'rgb' 或 'resnet'） ===")
for name, tensor in state_dict.items():
    if not isinstance(tensor, torch.Tensor):
        continue
    lname = name.lower()
    if ("rgb" in lname or "resnet" in lname) and "weight" in lname \
       and "running" not in lname and not lname.endswith("bias"):
        w = tensor.float()
        print(
            f"{name:60s} | mean={w.mean().item(): .4f} std={w.std().item(): .4f}"
        )

print("\n=== PCD / point cloud 相关的权重（name 里包含 'pcd' 或 'point'） ===")
for name, tensor in state_dict.items():
    if not isinstance(tensor, torch.Tensor):
        continue
    lname = name.lower()
    if ("pcd" in lname or "point" in lname) and "weight" in lname \
       and "running" not in lname and not lname.endswith("bias"):
        w = tensor.float()
        print(
            f"{name:60s} | mean={w.mean().item(): .4f} std={w.std().item(): .4f}"
        )
