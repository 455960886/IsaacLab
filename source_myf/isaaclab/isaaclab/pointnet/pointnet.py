
from log.classification.pointnet2_ssg_wo_normals.pointnet2_cls_ssg import get_model as PointNet2ClsMsg
import torch
import open3d as o3d
import torch
import numpy as np
import torch, random, numpy as np
from typing import Optional
seed = 50
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


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


# 读取 ply
pcd = o3d.io.read_point_cloud("/home/roborock/IsaacLab/frame_1.ply")

# 转成 numpy
points = np.asarray(pcd.points)  # shape: [N, 3]

# 转成 tensor
points_tensor = torch.from_numpy(points).float().cuda()  # [N, 3]

# 扩展 batch 维度
points_tensor = points_tensor.unsqueeze(0)  # [1, N, 3]

# 如果模型要求 [B, 3, N] 形状
points_tensor = points_tensor.permute(0, 2, 1)  
pcd = o3d.io.read_point_cloud("/home/roborock/IsaacLab/frame_1.ply")

# 转成 numpy
points = np.asarray(pcd.points)  # shape: [N, 3]

# 转成 tensor
points_tensor = torch.from_numpy(points).float().cuda()  # [N, 3]

# 扩展 batch 维度
points_tensor = points_tensor.unsqueeze(0)  # [1, N, 3]

# 如果模型要求 [B, 3, N] 形状
points_tensor = points_tensor.permute(0, 2, 1)  
_point_encoder = _prepare_pointnet_model().eval()
with torch.no_grad():
    features = _point_encoder(points_tensor)  # [1, 1024]
    features2 = _point_encoder(points_tensor2)
    import pdb
    pdb.set_trace()
print(features)
xyz = torch.randn(1, 3, 1024, device = 'cuda')
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
