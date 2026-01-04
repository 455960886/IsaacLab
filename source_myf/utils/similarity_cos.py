import torch
import torch.nn.functional as F
import numpy as np
import cv2
# def load_feature_from_txt(file_path):
#     """
#     从txt文件中读取一行特征向量（20160维），并返回为 PyTorch tensor。
#     """
#     with open(file_path, 'r') as f:
#         line = f.readline()
#         values = list(map(float, line.strip().split(',')))
#         tensor = torch.tensor(values, dtype=torch.float32)
#     return tensor

# def cosine_sim_torch(x, y):
#     """
#     用 PyTorch 计算余弦相似度。
#     """
#     return F.cosine_similarity(x.unsqueeze(0), y.unsqueeze(0)).item()

# # 替换为你自己的特征文件路径
# sim_feature = np.loadtxt("/home/roborock/下载/sim13_81_feature")
# real_feature = np.loadtxt("/home/roborock/下载/real3_feature")
# # 加载特征向量
# feature1 =  torch.from_numpy(sim_feature)
# feature2 = torch.from_numpy(real_feature)

# # 计算余弦相似度
# similarity = cosine_sim_torch(feature1, feature2)

# print(f"余弦相似度：{similarity:.6f}")
# from scipy.spatial.transform import Rotation as R

# # 欧拉角（单位：度）
# roll = 180
# pitch = -87.5
# yaw = 90

# # 创建旋转对象（XYZ 旋转顺序）
# r = R.from_euler('xyz', [roll, pitch, yaw], degrees=True)

# # 转换为四元数（xyzw）
# quat = r.as_quat()
# print(quat)

def cosine_similarity_rgb(img1_path, img2_path):
    img1 = cv2.imread(img1_path).astype(np.float32)
    img2 = cv2.imread(img2_path).astype(np.float32)
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 展开为向量（包含RGB通道）
    vec1 = img1.flatten()
    vec2 = img2.flatten()

    sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return sim

print(cosine_similarity_rgb("/home/roborock/shitou/capture.2025-11-12 10.04.21.png", "/home/roborock/下载/gripper.png"))
# import numpy as np

# # 旋转矩阵 OpenGL convention
# R_gl = np.array([
#     [0.003874  ,-0.004015,    0.999984],
#     [ -0.999947,-0.009508, 0.003835   ],
#     [ 0.00492  ,-0.999947,  -0.004052 ]
# ])

# # 将旋转矩阵转换为四元数 (w, x, y, z)
# def rotmat_to_quat(R):
#     trace = R[0,0] + R[1,1] + R[2,2]
#     if trace > 0:
#         s = 0.5 / np.sqrt(trace + 1.0)
#         w = 0.25 / s
#         x = (R[2,1] - R[1,2]) * s
#         y = (R[0,2] - R[2,0]) * s
#         z = (R[1,0] - R[0,1]) * s
#     else:
#         if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
#             s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
#             w = (R[2,1] - R[1,2]) / s
#             x = 0.25 * s
#             y = (R[0,1] + R[1,0]) / s
#             z = (R[0,2] + R[2,0]) / s
#         elif R[1,1] > R[2,2]:
#             s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
#             w = (R[0,2] - R[2,0]) / s
#             x = (R[0,1] + R[1,0]) / s
#             y = 0.25 * s
#             z = (R[1,2] + R[2,1]) / s
#         else:
#             s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
#             w = (R[1,0] - R[0,1]) / s
#             x = (R[0,2] + R[2,0]) / s
#             y = (R[1,2] + R[2,1]) / s
#             z = 0.25 * s
#     return np.array([w, x, y, z])

# # 转四元数
# q_gl = rotmat_to_quat(R_gl)
# print("Isaac Sim 四元数 (w,x,y,z):", q_gl)
# # OpenGL -> Isaac convention: q_isaac = (w, x, -z, y)
# q_isaac = np.array([q_gl[0], q_gl[1], -q_gl[3], q_gl[2]])

# print("Isaac Sim 四元数 (w,x,y,z):", q_isaac)
