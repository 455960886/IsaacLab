import sys
sys.path.append("/home/roborock/IsaacLab/source/isaaclab")
from isaaclab.pointnet.log.classification.pointnet2_ssg_wo_normals.pointnet2_cls_ssg import get_model as PointNet2ClsMsg
import torch
import open3d as o3d
import torch
import numpy as np
import torch, random, numpy as np
from typing import Optional
import torch.nn as nn
import cv2
seed = 50
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _prepare_pointnet_model() :
    import torch.nn as nn
    # experiment_dir = '/home/roborock/IsaacLab' 
    # classifier = MODEL.get_model(13).cuda()
    # checkpoint = torch.load(str(experiment_dir) + '/best_model.pth')
    # classifier.load_state_dict(checkpoint['model_state_dict'])
    # classifier = classifier.eval()

    # self._point_encoder = classifier.feat
    # self._point_encoder.eval()
    # self._point_encoder.cuda()

    experiment_dir = '/home/roborock/IsaacLab'
    ckpt_path = f"{experiment_dir}/best_model.pth"

    # ✅ 模型输入通道：原模型是 normal_channel=True（6 通道）
    classifier = PointNet2ClsMsg(num_class=40, normal_channel=False).cuda()  

    # ✅ 加载 checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cuda', weights_only=False)

    # 拿出权重字典
    state_dict = checkpoint['model_state_dict']

    # ✅ 加载修正后的权重
    classifier.load_state_dict(state_dict, strict=False)


    classifier.eval()

    # ✅ 仅保留特征提取部分（encoder）
    class PointNet2Encoder(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.normal_channel = True  # 我们只输入 XYZ
            self.sa1 = base_model.sa1
            self.sa2 = base_model.sa2
            self.sa3 = base_model.sa3

        def forward(self, xyz: torch.Tensor):

            B, N, C = xyz.shape
            empty_points = torch.empty(B, 0, N, device=xyz.device, dtype=xyz.dtype)
            l1_xyz, l1_points = self.sa1(xyz, empty_points)
            l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
            l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
            features = l3_points.view(B, 1024)
            return features

    return PointNet2Encoder(classifier).cuda().eval()

def prepare_input(path):
# 读取 ply
    pcd = o3d.io.read_point_cloud(path)

    # 转成 numpy
    points = np.asarray(pcd.points)  # shape: [N, 3]

    # 转成 tensor
    points_tensor = torch.from_numpy(points).float().cuda()  # [N, 3]

    # 扩展 batch 维度
    points_tensor = points_tensor.unsqueeze(0)  # [1, N, 3]

    # 如果模型要求 [B, 3, N] 形状
    points_tensor = points_tensor.permute(0, 2, 1)  

    return points_tensor
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
    mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
    std  = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
    img = (img - mean) / std

    # 增加 batch 维度 (1,3,H,W)
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img

def cosine_similarity(x, y, eps=1e-8):
    dot = (x * y).sum(dim=-1)
    norm_x = x.norm(dim=-1)
    norm_y = y.norm(dim=-1)
    return dot / (norm_x * norm_y + eps)

input1 = prepare_input("/home/roborock/IsaacLab/frame_1.ply")

_point_encoder = _prepare_pointnet_model().eval()
with torch.no_grad():
    pc_feature = _point_encoder(input1)
    print(f"pc_feature{pc_feature}")
# similarity = cosine_similarity(features1, features2)
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import onnxruntime as ort

# 1️⃣ 加载预训练模型
img = load_image("/home/roborock/Downloads/real.png")

img_input = torch.from_numpy(img).to(device)
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.eval()
model.to(device)
res_model = nn.Sequential(*list(model.children())[:-1])
with torch.no_grad():
    img_feature = res_model(img_input)
















































# xyz = torch.randn(1, 3, 1024, device = 'cuda')
# jit_model =torch.jit.script(_point_encoder,example_inputs=(xyz,))
# torch.onnx.export(
#     _point_encoder,
#     xyz,  # 用实际点云 shape 作为 dummy input
#     '/home/roborock/IsaacLab/pointnet.onnx',
#     export_params=True,
#     opset_version=18,
#     do_constant_folding=True,
#     input_names=['points'],
#     output_names=['features'],

#     dynamic_axes=None
# )
# import onnx
# model = onnx.load("./pointnet.onnx")
# model = onnx.shape_inference.infer_shapes(model)
# onnx.save(model, "pointnet_inferred.onnx")
