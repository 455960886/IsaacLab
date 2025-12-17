import torch
import torch.nn as nn
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


def uglify_sim_image(img_rgb: np.ndarray) -> np.ndarray:
    """
    对仿真图做模糊 + 降对比度 + 加少量噪声，
    让它的观感更接近真实相机拍出来的效果。
    img_rgb: H x W x 3, uint8, RGB
    """
    img = img_rgb.copy().astype(np.float32)

    # 1) 高斯模糊，模拟对焦不准 / 模糊
    img = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=1.2)

    # 2) 降一点对比度，稍微提一点亮度，接近 real 的“灰一点”的感觉
    # alpha = 0.85  # 对比度 (<1 更灰)
    # beta = 8.0    # 亮度 (+常数)
    # img = img * alpha + beta

    # 3) 加一点固定随机噪声（为了可重复，设一个固定种子）
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(loc=0.0, scale=5.0, size=img.shape)  # 标准差 5，比较温和
    img = img + noise

    # 4) clip 回合法像素范围再转回 uint8
    img = np.clip(img, 0, 255).astype(np.uint8)

    # 如果想看“丑化后的图”，可以临时保存一下：
    cv2.imwrite("/home/robo/下载/sim_ugly_debug_lego_1.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    return img


def load_image(img_path, augment: bool = False):
    # 读取图片（BGR）
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转 RGB

    # === 对仿真图做“丑化”增强 ===
    if augment:
        img = uglify_sim_image(img)

    # HWC -> CHW, 归一化到 [0,1]
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0

    # ResNet ImageNet 预处理
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = (img - mean) / std

    # 增加 batch 维度 (1,3,H,W)
    img = np.expand_dims(img, axis=0).astype(np.float32)

    return img


# ====== 加载真实图 & 仿真图（仿真图加 augment=True） ======
img = load_image("/home/robo/下载/1.png")
imgsim = load_image(
    "/home/robo/.local/share/ov/data/documents/Kit/shared/screenshots/capture.2025-12-04 18.15.12.png",
    augment=True,
)

# ------ 像素级余弦相似度（只是看看） ------
img_flat = img.reshape(1, -1)
imgsim_flat = imgsim.reshape(1, -1)
img_flat = torch.from_numpy(img_flat).float().to(device)
imgsim_flat = torch.from_numpy(imgsim_flat).float().to(device)
similarity_png = cosine_similarity(img_flat, imgsim_flat)
print("pixel cosine similarity:", similarity_png.item())

# 转成 tensor 做 ResNet 前向
input_tensor = torch.from_numpy(img).to(device)
input_tensorsim = torch.from_numpy(imgsim).to(device)

# 1️⃣ 加载预训练 ResNet18（去掉最后 FC）
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.eval()
features = nn.Sequential(*list(model.children())[:-1])
model.to(device)

# 2️⃣ 特征余弦相似度
with torch.no_grad():
    op = features(input_tensor)
    opsim = features(input_tensorsim)
op = op.squeeze(-1).squeeze(-1)
opsim = opsim.squeeze(-1).squeeze(-1)
similarity = cosine_similarity(op, opsim)
print("global feature cosine similarity:", similarity.item())

# ====== 分层看 ResNet18 的相似度 ======
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


def forward_collect_feats(x, blocks):
    feats = {}
    out = x
    for name, layer in blocks:
        out = layer(out)
        feats[name] = out.flatten(1)
    return feats


with torch.no_grad():
    feats1 = forward_collect_feats(input_tensor, blocks)
    feats2 = forward_collect_feats(input_tensorsim, blocks)

print("==== per-layer cosine similarity ====")
for name in feats1.keys():
    sim = cosine_similarity(feats1[name], feats2[name])
    print(f"{name:7s}: {sim.item():.4f}")
