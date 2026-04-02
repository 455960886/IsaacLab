"""
Simple Point Cloud Noise for Sim-to-Real Transfer

Two noise types:
1. Gaussian noise - random jitter on existing points
2. Cluster noise - small spurious point clusters near the point cloud
"""

import torch


def add_gaussian_noise(points: torch.Tensor, std: float = 0.002) -> torch.Tensor:
    """Add Gaussian noise to point cloud.

    Args:
        points: (N, 3) point cloud tensor
        std: noise standard deviation in meters (default 2mm)

    Returns:
        Noisy point cloud, same shape as input
    """
    noise = torch.randn_like(points) * std
    return points + noise


def add_cluster_noise(
    points: torch.Tensor,
    num_clusters: int = 3,
    points_per_cluster: int = 15,
    distance: float = 0.015,
    spread: float = 0.005,
) -> torch.Tensor:
    """Add random point clusters near the point cloud.

    Args:
        points: (N, 3) point cloud tensor
        num_clusters: number of clusters to add
        points_per_cluster: points in each cluster
        distance: how far clusters spawn from surface (meters)
        spread: size of each cluster (meters)

    Returns:
        Point cloud with some points replaced by cluster points
    """
    device = points.device
    dtype = points.dtype
    N = points.shape[0]

    if N == 0:
        return points

    all_cluster_points = []

    for _ in range(num_clusters):
        # Pick random anchor point from existing cloud
        anchor_idx = torch.randint(0, N, (1,), device=device).item()
        anchor = points[anchor_idx]

        # Random direction
        direction = torch.randn(3, device=device, dtype=dtype)
        direction = direction / (torch.norm(direction) + 1e-8)

        # Cluster center = anchor + offset in random direction
        center = anchor + direction * distance

        # Generate cluster points around center
        cluster = center + torch.randn(points_per_cluster, 3, device=device, dtype=dtype) * spread
        all_cluster_points.append(cluster)

    # Combine all clusters
    clusters = torch.cat(all_cluster_points, dim=0)
    total_new = clusters.shape[0]

    # Replace random original points with cluster points (to keep point count same)
    result = points.clone()
    replace_count = min(total_new, N)
    replace_idx = torch.randperm(N, device=device)[:replace_count]
    result[replace_idx] = clusters[:replace_count]

    return result


def add_noise(
    points: torch.Tensor,
    gaussian_std: float = 0.0002,
    num_clusters: int = 3,
    cluster_prob: float = 0.5,
) -> torch.Tensor:
    """Apply both Gaussian and cluster noise.

    Args:
        points: (N, 3) point cloud tensor
        gaussian_std: Gaussian noise std in meters
        num_clusters: number of clusters to add
        cluster_prob: probability of adding clusters (0-1)

    Returns:
        Noisy point cloud
    """
    # Add Gaussian noise
    result = add_gaussian_noise(points, std=gaussian_std)

    # Maybe add clusters
    if torch.rand(1, device=points.device).item() < cluster_prob:
        result = add_cluster_noise(result, num_clusters=num_clusters)

    return result