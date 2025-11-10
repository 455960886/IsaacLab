import torch
import numpy as np
import isaaclab.utils.math as math_utils

def transform_world_to_camera(points_world, camera_pos_w, camera_quat_w, apply_x_rotation=True):
    """
    Transform points from world frame to camera frame.
    
    Args:
        points_world: Points in world frame (B, 3) or (B, N, 3)
        camera_pos_w: Camera position in world frame (B, 3)
        camera_quat_w: Camera orientation in world frame (B, 4) in IsaacLab convention (w, x, y, z)
        apply_x_rotation: Whether to apply the 0.5 degree X-axis rotation
    """
    # Get camera rotation matrix from world to camera frame
    # The camera rotation matrix transforms from camera to world frame
    # So we need its inverse to transform from world to camera
    camera_rot_w_to_c = math_utils.matrix_from_quat(camera_quat_w).transpose(-2, -1)
    
    if points_world.dim() == 2:
        points_relative = points_world - camera_pos_w
        points_camera = torch.einsum('bij,bj->bi', camera_rot_w_to_c, points_relative)
    else:
        camera_pos_w = camera_pos_w.unsqueeze(1)
        points_relative = points_world - camera_pos_w
        points_camera = torch.einsum('bij,bnj->bni', camera_rot_w_to_c, points_relative)

    # Apply the same 0.5 degree X-axis rotation as point cloud processing
    if apply_x_rotation:
        theta = torch.deg2rad(torch.tensor(0.5, device=points_camera.device))
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        R_x = torch.tensor([
            [1, 0, 0],
            [0, cos_theta, -sin_theta],
            [0, sin_theta, cos_theta]
        ], device=points_camera.device, dtype=points_camera.dtype)

        if points_world.dim() == 2:
            points_camera = torch.matmul(points_camera, R_x.T)
        else:
            points_camera = torch.matmul(points_camera, R_x.T)
        
        # Invert y-axis
        if points_world.dim() == 2:
            points_camera[:, 1] = -points_camera[:, 1]
        else:
            points_camera[:, :, 1] = -points_camera[:, :, 1]

    return points_camera

def debug_gripper_transformation(env, sensor_cfg_name="depth_camera"):
    """
    Debug function to check gripper transformation accuracy.
    """
    # Get gripper positions in world frame
    left_finger_pos_w = env.scene["finger_frame_1"].data.target_pos_w[:, 0, :]
    right_finger_pos_w = env.scene["finger_frame_2"].data.target_pos_w[:, 0, :]
    
    # Get camera pose
    camera = env.scene.sensors[sensor_cfg_name]
    camera_pos_w = camera.data.pos_w
    camera_quat_w = camera.data.quat_w_ros
    
    # Convert ROS quaternion to IsaacLab convention
    camera_quat_w_isaac = torch.cat([camera_quat_w[:, 3:4], camera_quat_w[:, :3]], dim=-1)
    
    # print(f"Camera position: {camera_pos_w[0].cpu().numpy()}")
    # print(f"Camera quaternion (ROS): {camera_quat_w[0].cpu().numpy()}")
    # print(f"Camera quaternion (Isaac): {camera_quat_w_isaac[0].cpu().numpy()}")
    # print(f"Left finger world: {left_finger_pos_w[0].cpu().numpy()}")
    # print(f"Right finger world: {right_finger_pos_w[0].cpu().numpy()}")
    
    # Transform to camera frame
    left_finger_cam = transform_world_to_camera(left_finger_pos_w, camera_pos_w, camera_quat_w_isaac)
    right_finger_cam = transform_world_to_camera(right_finger_pos_w, camera_pos_w, camera_quat_w_isaac)
    
    # print(f"Left finger camera: {left_finger_cam[0].cpu().numpy()}")
    # print(f"Right finger camera: {right_finger_cam[0].cpu().numpy()}")
    
    return left_finger_cam, right_finger_cam

def label_grippers_in_pointcloud(pointcloud, left_finger_pos_cam, right_finger_pos_cam, 
                                  radius=0.02, label_value=1.0):
    """
    Label points near gripper fingers in the point cloud.
    """
    B, N, _ = pointcloud.shape
    
    # Expand finger positions for broadcasting
    left_pos = left_finger_pos_cam.unsqueeze(1)  # (B, 1, 3)
    right_pos = right_finger_pos_cam.unsqueeze(1)  # (B, 1, 3)
    
    # Compute distances to each finger
    dist_to_left = torch.norm(pointcloud - left_pos, dim=-1)  # (B, N)
    dist_to_right = torch.norm(pointcloud - right_pos, dim=-1)  # (B, N)
    
    # Label points within radius of either finger
    labels = ((dist_to_left < radius) | (dist_to_right < radius)).float() * label_value
    
    # Stack distances for reference
    distances = torch.stack([dist_to_left, dist_to_right], dim=-1)  # (B, N, 2)
    
    return labels, distances

def add_gripper_labels_to_observation(env, pointcloud, sensor_cfg_name="depth_camera", radius=0.02):
    """
    Complete pipeline: get gripper positions, transform to camera frame, and label point cloud.
    """
    # Get gripper positions in world frame
    left_finger_pos_w = env.scene["finger_frame_1"].data.target_pos_w[:, 0, :]
    right_finger_pos_w = env.scene["finger_frame_2"].data.target_pos_w[:, 0, :]

    print(f"World frame - Left: {left_finger_pos_w.shape}, Right: {right_finger_pos_w.shape}")
    
    # Get camera pose in world frame
    camera = env.scene.sensors[sensor_cfg_name]
    camera_pos_w = camera.data.pos_w
    camera_quat_w = camera.data.quat_w_ros  # ROS convention (x, y, z, w)
    
    # Convert ROS quaternion (x,y,z,w) to IsaacLab convention (w,x,y,z)
    camera_quat_w_isaac = torch.cat([camera_quat_w[:, 3:4], camera_quat_w[:, :3]], dim=-1)
    
    # # Debug: print transformation info
    # print(f"Camera position: {camera_pos_w[0].cpu().numpy()}")
    # print(f"Left finger world: {left_finger_pos_w[0].cpu().numpy()}")
    
    # Transform gripper positions to camera frame
    left_finger_pos_cam = transform_world_to_camera(left_finger_pos_w, camera_pos_w, camera_quat_w_isaac)
    right_finger_pos_cam = transform_world_to_camera(right_finger_pos_w, camera_pos_w, camera_quat_w_isaac)

    # print(f"Camera frame - Left: {left_finger_pos_cam[0].cpu().numpy()}")
    # print(f"Camera frame - Right: {right_finger_pos_cam[0].cpu().numpy()}")
    # print(f"Point cloud range - X: [{pointcloud[0,:,0].min().item():.3f}, {pointcloud[0,:,0].max().item():.3f}], "
    #       f"Y: [{pointcloud[0,:,1].min().item():.3f}, {pointcloud[0,:,1].max().item():.3f}], "
    #       f"Z: [{pointcloud[0,:,2].min().item():.3f}, {pointcloud[0,:,2].max().item():.3f}]")
    
    # Label points in point cloud
    labels, distances = label_grippers_in_pointcloud(
        pointcloud, left_finger_pos_cam, right_finger_pos_cam, radius
    )
    
    # Count labeled points for debugging
    num_labeled = labels.sum(dim=1)
    # print(f"Number of labeled points per env: {num_labeled.cpu().numpy()}")
    
    # Concatenate labels as 4th channel
    labeled_pointcloud = torch.cat([pointcloud, labels.unsqueeze(-1)], dim=-1)
    
    gripper_info = {
        'left_pos_camera': left_finger_pos_cam,
        'right_pos_camera': right_finger_pos_cam,
        'left_pos_world': left_finger_pos_w,
        'right_pos_world': right_finger_pos_w,
        'distances': distances,
        'labels': labels
    }

    gripper_clouds = create_gripper_pointclouds(left_finger_pos_cam, right_finger_pos_cam)
    gripper_info['gripper_pointclouds'] = gripper_clouds

    return labeled_pointcloud, gripper_info

def create_gripper_pointclouds(left_finger_pos_cam, right_finger_pos_cam, 
                                num_points=100, radius=0.001):
    """
    Generate small point clouds around gripper finger positions and midpoint.
    The spherical space's radius changes with the distance between gripper points.
    """
    B = left_finger_pos_cam.shape[0]
    device = left_finger_pos_cam.device

    def generate_spherical_points(batch_size, num_pts, rad):
        """Generate random points in a sphere"""
        # Generate random points in unit sphere
        theta = torch.rand(batch_size, num_pts, device=device) * 2 * np.pi
        phi = torch.acos(2 * torch.rand(batch_size, num_pts, device=device) - 1)
        r = torch.rand(batch_size, num_pts, device=device).pow(1/3) * rad

        # Convert to Cartesian
        x = r * torch.sin(phi) * torch.cos(theta)
        y = r * torch.sin(phi) * torch.sin(theta)
        z = r * torch.cos(phi)
        return torch.stack([x, y, z], dim=-1)  # (B, num_points, 3)

    # Generate point clouds around each finger
    offset = generate_spherical_points(B, num_points, radius)
    left_cloud = left_finger_pos_cam.unsqueeze(1) + offset
    right_cloud = right_finger_pos_cam.unsqueeze(1) + offset

    # Calculate midpoint between the two fingers
    midpoint = (left_finger_pos_cam + right_finger_pos_cam) / 2.0
    
    # Calculate distance between fingers
    finger_distance = torch.norm(left_finger_pos_cam - right_finger_pos_cam, dim=-1)
    
    # Make midpoint radius proportional to finger distance
    # When gripper is closed, distance is small -> small sphere
    # When gripper is open, distance is large -> large sphere
    # Using 0.6 multiplier so sphere radius is 60% of finger distance
    midpoint_radius = finger_distance * 0.35
    
    # Ensure minimum radius to avoid zero-sized spheres when gripper is fully closed
    min_radius = 0.005  # 5mm minimum radius
    midpoint_radius = torch.clamp(midpoint_radius, min=min_radius)
    
    # Expand midpoint_radius for broadcasting: (B,) -> (B, 1, 1)
    midpoint_radius_expanded = midpoint_radius.unsqueeze(-1).unsqueeze(-1)
    
    # Generate spherical cloud at midpoint with dynamic radius
    # Use more points for the midpoint sphere since it's larger
    midpoint_num_points = 800
    
    # Generate points in unit sphere and scale by the dynamic radius
    theta = torch.rand(B, midpoint_num_points, device=device) * 2 * np.pi
    phi = torch.acos(2 * torch.rand(B, midpoint_num_points, device=device) - 1)
    r = torch.rand(B, midpoint_num_points, device=device).pow(1/3)
    
    # Scale by the dynamic radius for each batch element
    r = r * midpoint_radius_expanded.squeeze(-1)
    
    # Convert to Cartesian
    x = r * torch.sin(phi) * torch.cos(theta)
    y = r * torch.sin(phi) * torch.sin(theta)
    z = r * torch.cos(phi)
    midpoint_offset = torch.stack([x, y, z], dim=-1)
    
    midpoint_cloud = midpoint.unsqueeze(1) + midpoint_offset
    
    # Debug info
    # print(f"Finger distance: {finger_distance[0].item():.4f}, Midpoint radius: {midpoint_radius[0].item():.4f}")
    
    # Merge all clouds together
    gripper_clouds = torch.cat([left_cloud, right_cloud, midpoint_cloud], dim=1)

    return gripper_clouds


def calculate_pointcloud_density_in_sphere(pointcloud, sphere_center, sphere_radius):
    """
    Calculate density of points within spherical regions.

    Args:
        pointcloud: (B, N, 3) point cloud in camera frame
        sphere_center: (B, 3) center of sphere for each env
        sphere_radius: (B,) radius of sphere for each env

    Returns:
        density: (B,) normalized density value [0, 1]
        num_points_in_sphere: (B,) count of points in sphere
    """
    B, N, _ = pointcloud.shape

    # Expand for broadcasting
    sphere_center = sphere_center.unsqueeze(1)  # (B, 1, 3)
    sphere_radius = sphere_radius.unsqueeze(1)  # (B, 1)

    # Calculate distances from each point to sphere center
    distances = torch.norm(pointcloud - sphere_center, dim=-1)  # (B, N)

    # Count points inside sphere
    inside_sphere = distances < sphere_radius  # (B, N) boolean
    num_points_in_sphere = inside_sphere.sum(dim=1).float()  # (B,)

    # Normalize by total points to get density ratio [0, 1]
    density = num_points_in_sphere / N

    return density, num_points_in_sphere