import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights
import numpy as np
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cosine_similarity(x, y, eps=1e-8):
    dot = (x * y).sum(dim=-1)
    norm_x = x.norm(dim=-1)
    norm_y = y.norm(dim=-1)
    return dot / (norm_x * norm_y + eps)


# ---------- Blur kernels ----------
def _box_blur_avgpool_chw(img_chw: torch.Tensor, k: int, passes: int) -> torch.Tensor:
    """
    img_chw: (3,H,W) float in [0,1]
    Use avg_pool2d with stride=1 + padding to keep size. Repeat 2~3 passes.
    """
    assert img_chw.ndim == 3 and img_chw.shape[0] == 3
    x = img_chw.unsqueeze(0)  # (1,3,H,W)
    for _ in range(passes):
        x = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
    return x.squeeze(0)


# --------- 仿真图像“丑化”函数：在 HWC、uint8、RGB 上操作 ---------
def uglify_sim_image_gaussian(img_rgb: np.ndarray, seed: int | None = None, add_noise: bool = True) -> np.ndarray:
    """随机高斯模糊 + 随机高斯噪声（逻辑对齐 _apply_domain_randomization）"""
    assert img_rgb.ndim == 3 and img_rgb.shape[2] == 3, f"Expect HWC RGB, got {img_rgb.shape}"
    rng = np.random.default_rng(seed)

    img = img_rgb.astype(np.float32) / 255.0

    # Random Gaussian blur
    k = int(rng.integers(3, 12))  # 3..11
    if k % 2 == 0:
        k += 1
        if k > 11:
            k = 11
    sigma = float(rng.random() * 3.0 + 0.5)  # 0.5..3.5

    img = cv2.GaussianBlur(
        img, ksize=(k, k), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT101
    )

    if add_noise:
        noise_std = float(rng.random() * 0.08 + 0.02)  # 0.02..0.10 in [0,1]
        noise = rng.normal(0.0, noise_std, size=img.shape).astype(np.float32)
        img = np.clip(img + noise, 0.0, 1.0)
    else:
        img = np.clip(img, 0.0, 1.0)

    return (img * 255.0 + 0.5).astype(np.uint8)


def uglify_sim_image_box(img_rgb: np.ndarray, seed: int | None = None, add_noise: bool = True) -> np.ndarray:
    """
    Box blur = avg_pool2d 重复 2~3 次 + 随机高斯噪声（仍在 [0,1] 浮点域）
    """
    assert img_rgb.ndim == 3 and img_rgb.shape[2] == 3, f"Expect HWC RGB, got {img_rgb.shape}"
    rng = np.random.default_rng(seed)

    # uint8 HWC -> float [0,1] CHW torch
    img = img_rgb.astype(np.float32) / 255.0
    img_chw = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # (3,H,W)

    # 选一个“等价强度”的 box blur：kernel_size 也用 3..11 odd，passes 2..3
    k = int(rng.integers(3, 12))  # 3..11
    if k % 2 == 0:
        k += 1
        if k > 11:
            k = 11
    passes = int(rng.integers(2, 4))  # 2 or 3

    img_chw = _box_blur_avgpool_chw(img_chw, k=k, passes=passes)

    # back to numpy HWC for adding noise (or也可以用torch加噪)
    img = img_chw.permute(1, 2, 0).cpu().numpy()

    if add_noise:
        noise_std = float(rng.random() * 0.08 + 0.02)
        noise = rng.normal(0.0, noise_std, size=img.shape).astype(np.float32)
        img = np.clip(img + noise, 0.0, 1.0)
    else:
        img = np.clip(img, 0.0, 1.0)

    return (img * 255.0 + 0.5).astype(np.uint8)


def load_image(img_path, augment: str = "none", seed: int | None = None, add_noise: bool = True):
    img = cv2.imread(img_path)
    assert img is not None, f"Failed to read image: {img_path}"
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if augment == "gaussian":
        img = uglify_sim_image_gaussian(img, seed=seed, add_noise=add_noise)
    elif augment == "box":
        img = uglify_sim_image_box(img, seed=seed, add_noise=add_noise)
    elif augment == "none":
        pass
    else:
        raise ValueError(f"Unknown augment mode: {augment}")

    # HWC -> CHW, [0,1]
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0

    # ImageNet normalize
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = (img - mean) / std

    img = np.expand_dims(img, axis=0).astype(np.float32)  # (1,3,H,W)
    return img


# ====== 加载 real & sim，并分别对 sim 做 gaussian / box ======
real_path = "/home/robo/下载/0107_real.png"
sim_path  = "/home/robo/下载/0107_sim.png"

seed = 42  # 固定 seed：保证 gaussian/box 各自可复现（且每次运行一致）
add_noise = True  # 如果你只想“比较 blur”，先改成 False

img_real = load_image(real_path, augment="none")
img_sim_gauss = load_image(sim_path, augment="gaussian", seed=seed, add_noise=add_noise)
img_sim_box   = load_image(sim_path, augment="box",     seed=seed, add_noise=add_noise)


def run_compare(img_a_np, img_b_np, tag: str, model, features, blocks):
    # pixel cosine
    a_flat = torch.from_numpy(img_a_np.reshape(1, -1)).float().to(device)
    b_flat = torch.from_numpy(img_b_np.reshape(1, -1)).float().to(device)
    sim_pix = cosine_similarity(a_flat, b_flat).item()

    # global feature cosine
    a = torch.from_numpy(img_a_np).to(device)
    b = torch.from_numpy(img_b_np).to(device)
    with torch.no_grad():
        fa = features(a).squeeze(-1).squeeze(-1)
        fb = features(b).squeeze(-1).squeeze(-1)
    sim_feat = cosine_similarity(fa, fb).item()

    # per-layer cosine
    def forward_collect_feats(x, blocks):
        feats = {}
        out = x
        for name, layer in blocks:
            out = layer(out)
            feats[name] = out.flatten(1)
        return feats

    with torch.no_grad():
        feats1 = forward_collect_feats(a, blocks)
        feats2 = forward_collect_feats(b, blocks)

    print(f"\n===== {tag} =====")
    print("pixel cosine similarity:", sim_pix)
    print("global feature cosine similarity:", sim_feat)
    print("==== per-layer cosine similarity ====")
    for name in feats1.keys():
        s = cosine_similarity(feats1[name], feats2[name]).item()
        print(f"{name:7s}: {s:.4f}")


# 1) ResNet18 backbone
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
model.eval()
features = nn.Sequential(*list(model.children())[:-1]).to(device)

blocks = [
    ("conv1",   model.conv1),
    ("bn1",     model.bn1),
    ("relu",    model.relu),
    ("maxpool", model.maxpool),
    ("layer1",  model.layer1),
    ("layer2",  model.layer2),
    ("layer3",  model.layer3),
    ("layer4",  model.layer4),
    ("avgpool", model.avgpool),
]

# 2) 分别对比 real vs (sim+gaussian) 和 real vs (sim+box)
run_compare(img_real, img_sim_gauss, "REAL vs SIM+Gaussian", model, features, blocks)
run_compare(img_real, img_sim_box,   "REAL vs SIM+Box(avg_pool2d x2~3)", model, features, blocks)

# （可选）再看 gaussian 与 box 两种增强结果彼此有多像
run_compare(img_sim_gauss, img_sim_box, "SIM+Gaussian vs SIM+Box", model, features, blocks)
