
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch
from torchvision import models
from torchvision.models import ResNet18_Weights
import numpy as np
import cv2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_image(img_path, input_size=(224, 224)):
    # 读取图片
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # resize 到模型的输入尺寸
    img = cv2.resize(img, input_size)

    # HWC -> CHW
    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0  # 归一化到 [0,1]

    # ResNet ImageNet 预处理
    mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
    std  = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
    img = (img - mean) / std

    # 增加 batch 维度 (1,3,H,W)
    img = np.expand_dims(img, axis=0).astype(np.float32)

    return img
img = load_image("/home/roborock/docker_images_v1.8.x/docker_bushu/1.png", input_size=(224, 224))
input_tensor = torch.from_numpy(img).to(device)


import onnxruntime as ort
import cv2
import numpy as np


import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import onnxruntime as ort

# 1️⃣ 加载预训练模型
import pdb
pdb.set_trace()
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.eval()

# 2️⃣ 设置运行设备

model.to(device)

# 3️⃣ 定义输入张量
input_shape = (1, 3, 224, 224)
x = torch.ones(input_shape, device=device)  # 放到 device 上

# 4️⃣ PyTorch 前向测试
with torch.no_grad():
    op = model(input_tensor)
    print(op[0][:5])

# 5️⃣ 转换为 ONNX
output_file = "resnet18.onnx"
torch.onnx.export(
    model,
    x,
    output_file,
    export_params=True,
    opset_version=18,       # 推荐最新 opset
    input_names=["input"],
    output_names=["output"],
    dynamic_axes=None
)

# 6️⃣ ONNX Runtime 推理
session = ort.InferenceSession(output_file, providers=["CPUExecutionProvider"])

# 注意：输入必须是 numpy 且在 CPU 上
x_np = x.cpu().numpy().astype('float32')
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

features = session.run([output_name], {input_name: img})[0]
print(features[0][:5])
