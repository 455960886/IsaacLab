import open3d as o3d
import numpy as np

# -----------------------------
# 用户参数
# -----------------------------
input_file = "/home/roborock/IsaacLab/frame_1.ply"
output_file = "/home/roborock/IsaacLab/frame_11.ply"

roll  = np.radians(90)   # 绕 X   
pitch = np.radians(0)   # 绕 Y   
yaw   = np.radians(90)     # 绕 Z   

# translation = np.array([0.0, 0.0828, 0.183])
translation = np.array([0.183, 0.0, 0.0659])
# -----------------------------
# 读取点云
# -----------------------------
pcd = o3d.io.read_point_cloud(input_file)
points = np.asarray(pcd.points)

# # -----------------------------
# # 旋转矩阵 Rx Ry Rz
# # -----------------------------
# Rx = np.array([[1, 0, 0],
#                [0, np.cos(roll), -np.sin(roll)],
#                [0, np.sin(roll),  np.cos(roll)]])

# Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
#                [0,              1,             0],
#                [-np.sin(pitch), 0, np.cos(pitch)]])

# Rz = np.array([[ np.cos(yaw), -np.sin(yaw), 0],
#                [ np.sin(yaw),  np.cos(yaw), 0],
#                [0,             0,            1]])

# # 总旋转矩阵 (Z * Y * X)


# # -----------------------------
# # 构造 4x4 transformation 矩阵 T
# # -----------------------------
# x1= np.radians(0.011)
# Rx1 = np.array([[1, 0, 0],
#                [0, np.cos(x1), -np.sin(x1)],
#                [0, np.sin(x1),  np.cos(x1)]])

# R = Rx1@ Rz @ Ry @ Rx 
# T = np.eye(4)
# T[:3, :3] = R
# T[:3,  3] = translation

# print("Transformation Matrix T =\n", T)

# -----------------------------
# 点云转换：使用 4×N 齐次矩阵
# -----------------------------
points_h = np.hstack([points, np.ones((points.shape[0], 1))])   # N×4
# points_transformed_h = (T @ points_h.T).T                       # N×4
points_h = points_h[:, :3]                # 取 xyz
# mask1 = points_h[:, 0] >=0.35
mask2 = points_h[:, 0] <=0.42
mask3 = points_h[:, 2] >= 0.000
mask4 = points_h[:, 1] <=0.08
mask5 = points_h[:, 1] >=-0.08
mask =  mask2 & mask3 &mask4 & mask5 
points_h = points_h[mask]
# 更新点云
pcd.points = o3d.utility.Vector3dVector(points_h)

# 保存点云
o3d.io.write_point_cloud(output_file, pcd)
print(f"点云已保存到 {output_file}")



# mask1 = points[:, 0] >=0.35
# mask2 = points[:, 0] <=0.44
# mask3 = points[:, 2] >= 0.00
# mask4 = points[:, 1] <=0.20
# mask5 = points[:, 1] >=-0.20
# mask = mask1 & mask2 & mask3 & mask4 & mask5
# points = points[mask]