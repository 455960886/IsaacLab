import numpy as np
import open3d as o3d

# def randomize_pointcloud(pts_np, 
#                          dropout_rate=0.02, 
#                          outlier_ratio=0.001, 
#                          outlier_max_offset=0.03, 
#                          surface_jitter=0.001):
#     """
#     对点云进行随机化：
#     - Dropout: 随机丢弃部分点
#     - Outliers: 随机选择点沿 X 轴正方向偏移
#     - Surface jitter: 沿法线微小扰动

#     pts_np: np.ndarray (N,3)
#     dropout_rate: 点丢失比例
#     outlier_ratio: outlier 点比例
#     outlier_max_offset: outlier 最大偏移距离
#     surface_jitter: 沿法线扰动幅度
#     return: np.ndarray (N,3)
#     """
#     pts = pts_np.copy()
#     N = pts.shape[0]

#     if N == 0:
#         return pts

#     # -------------------------
#     # 1) Dropout
#     # -------------------------
#     mask = np.random.rand(N) > dropout_rate
#     pts = pts[mask]
#     N = pts.shape[0]
#     if N == 0:
#         return pts

#     # -------------------------
#     # 2) Outliers (沿 X 轴正方向)
#     # -------------------------
#     num_outliers = max(1, int(outlier_ratio * N))
#     outlier_idx = np.random.choice(N, num_outliers, replace=False)
#     pts[outlier_idx, 0] += np.random.rand(num_outliers) * outlier_max_offset

#     # -------------------------
#     # 3) Surface jitter (沿法线)
#     # -------------------------
#     pcd_tmp = o3d.geometry.PointCloud()
#     pcd_tmp.points = o3d.utility.Vector3dVector(pts)
#     pcd_tmp.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))
#     normals = np.asarray(pcd_tmp.normals)
#     pts += normals * (np.random.randn(N,3) * surface_jitter)

#     return pts
import torch

def randomize_pointcloud_batch(
    pts,                    # (B, N, 3)
    dropout_rate=0.02,
    outlier_ratio=0.2,
    outlier_max_offset=0.08,
    surface_jitter=0.001,
):
    B, N, _ = pts.shape
    device = pts.device

    pts = pts.clone()

    # 1. Dropout
    # dropout_mask = torch.rand(B, N, device=device) > dropout_rate
    # pts = pts * dropout_mask.unsqueeze(-1)

    # 2. Outliers（沿 X 轴正方向）
    num_outliers = max(1, int(outlier_ratio * N))

    for b in range(B):
        idx = torch.randperm(N, device=device)[:num_outliers].long()  # 确保是 long
        offset = torch.rand(num_outliers, device=device, dtype=pts.dtype) * outlier_max_offset
        pts[b].index_add_(0, idx, torch.stack([offset, torch.zeros_like(offset), torch.zeros_like(offset)], dim=1))


    jitter = torch.randn_like(pts) * surface_jitter
    pts = pts + jitter

    return pts


import open3d as o3d
import torch

def load_ply_as_batch(ply_path, device="cuda"):
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device=device)
    pts = pts.unsqueeze(0)  # (1, N, 3)
    return pts

def save_batch_pcd(pts_batch, prefix="output"):
    B = pts_batch.shape[0]
    for b in range(B):
        pts = pts_batch[b].cpu().numpy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        o3d.io.write_point_cloud(f"{prefix}_b{b}.ply", pcd)


# ------------------------------
# 使用示例
# ------------------------------
if __name__ == "__main__":


    # 读取单个 ply
    pts_batch = load_ply_as_batch("/home/roborock/IsaacLab/frame_000006_3_downsampled.ply", device="cuda")  # -> (1, N, 3)

    # 复制成多个 batch 测试（比如 B=4）
    pts_batch = pts_batch.repeat(4, 1, 1)  # (4, N, 3)

    print("Input batch shape:", pts_batch.shape)

    # 进行 batched 随机增广
    aug_batch = randomize_pointcloud_batch(pts_batch)

    print("Output batch shape:", aug_batch.shape)

    # 保存输出
    save_batch_pcd(aug_batch, prefix="augmented")

# def add_outliers(pts, outlier_ratio=0.001, outlier_max_offset=0.03):
#     """
#     pts: (B, N, 3) torch tensor
#     """
#     B, N, _ = pts.shape
#     device = pts.device
#     num_outliers = max(1, int(outlier_ratio * N))

#     for b in range(B):
#         idx = torch.randperm(N, device=device)[:num_outliers].long()  # 确保是 long
#         offset = torch.rand(num_outliers, device=device, dtype=pts.dtype) * outlier_max_offset
#         pts[b].index_add_(0, idx, torch.stack([offset, torch.zeros_like(offset), torch.zeros_like(offset)], dim=1))

#     return pts
