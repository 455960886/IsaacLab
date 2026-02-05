import sys
sys.path.append("/home/roborock/IsaacLab/source/isaaclab")
from isaaclab.pointnet.log.classification.pointnet2_ssg_wo_normals.pointnet2_cls_ssg import get_model as PointNet2ClsMsg
import torch
import torch.nn as nn
import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA


class PointNet2VisualizationEncoder(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.sa1 = base_model.sa1
        self.sa2 = base_model.sa2
        self.sa3 = base_model.sa3
    
    def forward(self, xyz: torch.Tensor):
        B, N, C = xyz.shape
        empty_points = torch.empty(B, 0, N, device=xyz.device, dtype=xyz.dtype)
        
        l1_xyz, l1_points = self.sa1(xyz, empty_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        
        return {
            'l1': (l1_xyz, l1_points),
            'l2': (l2_xyz, l2_points),
            'l3': (l3_xyz, l3_points)
        }


def visualize_pointcloud_features(xyz, features, window_name="PointNet Features"):
    """
    Visualize point cloud colored by features
    
    Args:
        xyz: [N, 3] numpy array of point coordinates
        features: [N, C] numpy array of feature vectors
        window_name: name for the visualization window
    """
    # Ensure correct shape [N, 3]
    if xyz.shape[1] != 3:
        xyz = xyz.T
    
    # Ensure arrays are contiguous and float64
    xyz = np.ascontiguousarray(xyz, dtype=np.float64)
    features = np.ascontiguousarray(features, dtype=np.float64)
    
    # Use PCA to reduce features to 3D (RGB)
    pca = PCA(n_components=3)
    features_3d = pca.fit_transform(features)
    
    # Normalize to [0, 1] for coloring
    colors = (features_3d - features_3d.min(axis=0)) / \
             (features_3d.max(axis=0) - features_3d.min(axis=0) + 1e-8)
    colors = np.ascontiguousarray(colors, dtype=np.float64)
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualize
    o3d.visualization.draw_geometries([pcd], window_name=window_name)
    
    print(f"\nVisualization Statistics:")
    print(f"  Points: {xyz.shape[0]}")
    print(f"  Feature dimensions: {features.shape[1]}")
    print(f"  PCA explained variance: {pca.explained_variance_ratio_}")


def prepare_pointnet_model():
    """Load pretrained PointNet2 model"""
    experiment_dir = '/home/roborock/IsaacLab'
    ckpt_path = f"{experiment_dir}/best_model.pth"
    
    classifier = PointNet2ClsMsg(num_class=40, normal_channel=False).cuda()
    checkpoint = torch.load(ckpt_path, map_location='cuda', weights_only=False)
    state_dict = checkpoint['model_state_dict']
    classifier.load_state_dict(state_dict, strict=False)
    classifier.eval()
    
    return classifier


def prepare_input(path):
    """Load point cloud from file"""
    pcd = o3d.io.read_point_cloud(path)
    points = np.asarray(pcd.points)
    points_tensor = torch.from_numpy(points).float().cuda()
    points_tensor = points_tensor.unsqueeze(0)  # [1, N, 3]
    points_tensor = points_tensor.permute(0, 2, 1)  # [1, 3, N]
    return points_tensor


if __name__ == "__main__":
    # Load model
    base_model = prepare_pointnet_model()
    encoder = PointNet2VisualizationEncoder(base_model).cuda().eval()
    
    # Load point clouds
    input1 = prepare_input("/home/roborock/R50_pc/pairs/pair_1/frame_000008_3_downsampled.ply")
    input2 = prepare_input("/home/roborock/R50_pc/pairs/pair_1/frame_sampled_lego_T_2_translated.ply")
    
    # Visualize first point cloud
    print("\n" + "="*60)
    print("VISUALIZING FIRST POINT CLOUD")
    print("="*60)
    with torch.no_grad():
        outputs1 = encoder(input1)
        
        # Check each layer and visualize the one with enough points
        for layer_name in ['l1', 'l2']:
            xyz, features = outputs1[layer_name]
            xyz = xyz[0].cpu().numpy()  # [3, N] or [N, 3]
            features = features[0].permute(1, 0).cpu().numpy()  # [N, C]
            
            # Handle shape - xyz might be [3, N] and need transpose
            if xyz.shape[0] == 3 and xyz.shape[1] > 3:
                num_points = xyz.shape[1]
            else:
                num_points = xyz.shape[0]
            
            print(f"\n{layer_name}: {num_points} points, {features.shape[1]} features")
            
            visualize_pointcloud_features(xyz, features, f"Random PC - {layer_name.upper()}")
    
    # Visualize second point cloud
    print("\n" + "="*60)
    print("VISUALIZING SECOND POINT CLOUD")
    print("="*60)
    with torch.no_grad():
        outputs2 = encoder(input2)
        
        for layer_name in ['l1', 'l2']:
            xyz, features = outputs2[layer_name]
            xyz = xyz[0].cpu().numpy()  # [3, N] or [N, 3]
            features = features[0].permute(1, 0).cpu().numpy()  # [N, C]
            
            # Handle shape - xyz might be [3, N] and need transpose
            if xyz.shape[0] == 3 and xyz.shape[1] > 3:
                num_points = xyz.shape[1]
            else:
                num_points = xyz.shape[0]
            
            print(f"\n{layer_name}: {num_points} points, {features.shape[1]} features")
            
            visualize_pointcloud_features(xyz, features, f"Lego PC - {layer_name.upper()}")
    
    # Compute cosine similarity between global features (L3)
    print("\n" + "="*60)
    print("FEATURE SIMILARITY COMPARISON")
    print("="*60)
    with torch.no_grad():
        # Get L3 global features for both point clouds
        l3_features1 = outputs1['l3'][1].view(-1)  # [1024]
        l3_features2 = outputs2['l3'][1].view(-1)  # [1024]
        
        # Compute cosine similarity
        dot_product = (l3_features1 * l3_features2).sum()
        norm1 = l3_features1.norm()
        norm2 = l3_features2.norm()
        cosine_sim = dot_product / (norm1 * norm2 + 1e-8)
        
        print(f"\nGlobal Feature Cosine Similarity (L3): {cosine_sim.item():.4f}")
        print(f"  (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite)")
        
        # Also compute L2 average similarity
        l2_features1_tensor = torch.from_numpy(outputs1['l2'][1][0].permute(1, 0).cpu().numpy()).float()
        l2_features2_tensor = torch.from_numpy(outputs2['l2'][1][0].permute(1, 0).cpu().numpy()).float()
        
        l2_features1_norm = l2_features1_tensor / (l2_features1_tensor.norm(dim=1, keepdim=True) + 1e-8)
        l2_features2_norm = l2_features2_tensor / (l2_features2_tensor.norm(dim=1, keepdim=True) + 1e-8)
        
        l2_mean_sim = (l2_features1_norm.mean(dim=0) * l2_features2_norm.mean(dim=0)).sum()
        print(f"Average Feature Cosine Similarity (L2): {l2_mean_sim.item():.4f}")