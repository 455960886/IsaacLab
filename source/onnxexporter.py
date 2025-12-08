
import torch
import torch.nn as nn
# from PIL import Image
from torchvision import models
from torchvision.models import ResNet18_Weights
import numpy as np
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# test = cv2.imread('/home/roborock/Downloads/usb_img.png')
# test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
# cropped = test[120:, :]
# cv2.imwrite('/home/roborock/Downloads/usb_img_120.png',cropped)


def cosine_similarity(x, y, eps=1e-8):
    dot = (x * y).sum(dim=-1)
    norm_x = x.norm(dim=-1)
    norm_y = y.norm(dim=-1)
    return dot / (norm_x * norm_y + eps)


def load_image(img_path):
    # 读取图片
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize 到模型的输入尺寸
    # img = cv2.resize(img, input_size)
    # cv2.imwrite('/home/roborock/docker_images_v1.8.x/docker_bushu/sim1.png',img)
    # HWC -> CHW
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0  # 归一化到 [0,1]

    # ResNet ImageNet 预处理
    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
    img = (img - mean) / std

    # 增加 batch 维度 (1,3,H,W)
    img = np.expand_dims(img, axis=0).astype(np.float32)

    return img


img = load_image("/home/robo/.local/share/ov/data/documents/Kit/shared/screenshots/real1.png")
imgsim = load_image("/home/robo/.local/share/ov/data/documents/Kit/shared/screenshots/capture.2025-12-03 15.19.51.png")
# imgsim = load_image("/home/robo/下载/c747ee93-a7e0-47c5-ac1a-fec11f8c6e51.png")

img_flat = img.reshape(1, -1)
imgsim_flat = imgsim.reshape(1, -1)
img_flat = torch.from_numpy(img_flat).float().to(device)
imgsim_flat = torch.from_numpy(imgsim_flat).float().to(device)
similarity_png = cosine_similarity(img_flat, imgsim_flat)
print("pixel cosine similarity:", similarity_png.item())
input_tensor = torch.from_numpy(img).to(device)
input_tensorsim = torch.from_numpy(imgsim).to(device)
# print(input_tensor)

# import onnxruntime as ort
# import onnxruntime as ort

# 1️⃣ 加载预训练模型
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.eval()  # 推理（inference）模式，不是训练模式。
features = nn.Sequential(*list(model.children())[:-1])  # nn.Sequential(*) 把这些模块按顺序串成一个新的网络。一个“特征提取器”，不做分类，只负责把图片变成一个 512 维的 embedding。
# list(model.children())[:-1] 就是去掉最后的 fc
# 只保留：
# conv1, bn1, relu, maxpool,
# layer1, layer2, layer3, layer4,
# avgpool

# 2️⃣ 设置运行设备

model.to(device)

# 3️⃣ 定义输入张量
# dummy = torch.randn(1, 3, 480, 640).to(device)
# x = torch.ones(dummy, device=device)  # 放到 device 上

# 4️⃣ PyTorch 前向测试
with torch.no_grad():
    op = features(input_tensor)
    opsim = features(input_tensorsim)
op = op.squeeze(-1).squeeze(-1)
opsim = opsim.squeeze(-1).squeeze(-1)    
# import pdb
# pdb.set_trace()
similarity = cosine_similarity(op, opsim)
print(similarity)
# 5️⃣ 转换为 ONNX
# output_file = '/home/roborock/docker_images_v1.8.x/docker_bushu/awnpu_model_zoo-v0.6.0-20251023-58ae07df/examples/resnet50v2/convert_model/rensnet18.onnx'
# torch.onnx.export(
#     features,
#     dummy,
#     output_file,
#     export_params=True,
#     opset_version=17,       # 推荐最新 opset
#     input_names=["input"],
#     output_names=["output"],
#     dynamic_axes=None
# )

# # 6️⃣ ONNX Runtime 推理
# session = ort.InferenceSession(output_file, providers=["CPUExecutionProvider"])

# # 注意：输入必须是 numpy 且在 CPU 上
# x_np = input_tensor.cpu().numpy().astype('float32')
# input_name = session.get_inputs()[0].name
# output_name = session.get_outputs()[0].name

# features = session.run([output_name], {input_name: img})[0]
# print(features[0][:5])

# ====== 分层看 ResNet18 的相似度 ======
# 把模型拆成几个逻辑 block
blocks = [
    ("conv1",  model.conv1),
    ("bn1",    model.bn1),
    ("relu",   model.relu),
    ("maxpool",model.maxpool),
    ("layer1", model.layer1),
    ("layer2", model.layer2),
    ("layer3", model.layer3),
    ("layer4", model.layer4),
    ("avgpool", model.avgpool),
]


def forward_collect_feats(x, blocks):
    feats = {}
    out = x
    for name, layer in blocks:
        out = layer(out)
        # 展平成 (B, N) 用来算余弦
        feats[name] = out.flatten(1)
    return feats


with torch.no_grad():
    feats1 = forward_collect_feats(input_tensor, blocks)
    feats2 = forward_collect_feats(input_tensorsim, blocks)

print("==== per-layer cosine similarity ====")
for name in feats1.keys():
    sim = cosine_similarity(feats1[name], feats2[name])
    print(f"{name:7s}: {sim.item():.4f}")
