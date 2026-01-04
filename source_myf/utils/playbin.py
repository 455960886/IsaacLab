import numpy as np
import open3d as o3d

# 读取点云
pc = np.fromfile("/home/roborock/下载/NAV_binId10.log", dtype=np.float32).reshape(-1, 4)

# 过滤掉 NaN/Inf
mask = np.all(np.isfinite(pc), axis=1)
pc = pc[mask]
print(pc)
# 过滤 intensity 极端值（可选）
intensity = pc[:, 3]
valid_mask = (intensity >= -1e5) & (intensity <= 1e5)  # 根据你的数据调整阈值
pc = pc[valid_mask]
intensity = pc[:, 3]

print(f"Filtered points: {pc.shape[0]}")

# 创建 Open3D 点云
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc[:, :3])

# intensity 归一化
i_min, i_max = intensity.min(), intensity.max()
if np.isfinite(i_min) and np.isfinite(i_max) and i_max != i_min:
    intensity_norm = (intensity - i_min) / (i_max - i_min)
else:
    intensity_norm = np.zeros_like(intensity)

# 上色

# pcd = pcd.voxel_down_sample(voxel_size=1)  # 5cm 网格
pcd.colors = o3d.utility.Vector3dVector(np.stack([intensity_norm]*3, axis=1))
# 可视化
print("X range:", pc[:,0].min(), pc[:,0].max())
print("Y range:", pc[:,1].min(), pc[:,1].max())
print("Z range:", pc[:,2].min(), pc[:,2].max())

# o3d.visualization.draw_geometries([pcd])



