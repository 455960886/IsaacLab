import torch
import numpy as np
import isaaclab.utils.math as math_utils

def get_base_rotation_matrix(base_joint_angle):
    """
    Create rotation matrix for base rotation around Z-axis.
    
    Args:
        base_joint_angle: (B,) tensor of M0 joint angles in radians
        
    Returns:
        R_base: (B, 3, 3) rotation matrices
    """
    cos_theta = torch.cos(base_joint_angle)
    sin_theta = torch.sin(base_joint_angle)
    zeros = torch.zeros_like(cos_theta)
    ones = torch.ones_like(cos_theta)
    
    # Rotation around Z-axis: [cos -sin 0; sin cos 0; 0 0 1]
    R_base = torch.stack([
        torch.stack([cos_theta, -sin_theta, zeros], dim=-1),
        torch.stack([sin_theta, cos_theta, zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1)
    ], dim=-2)  # (B, 3, 3)
    
    return R_base


def transform_world_to_camera(points_world, env, sensor_cfg_name="depth_camera", 
                                         apply_x_rotation=True):
    """
    Transform points from world frame to camera frame, accounting for base rotation.
    
    This version manually computes the camera pose because camera.data.pos_w and 
    camera.data.quat_w_ros do NOT update when the base rotates.
    
    Args:
        points_world: Points in world frame (B, 3) or (B, N, 3)
        env: IsaacLab environment
        sensor_cfg_name: Name of the camera sensor
        apply_x_rotation: Whether to apply the 0.5 degree X-axis rotation
        
    Returns:
        points_camera: Points in camera frame
    """
    robot = env.scene["robot"]
    device = points_world.device
    
    # Get base joint angle (M0 - rotation around Z)
    base_joint_pos = robot.data.joint_pos[:, 0]  # M0 is at index 0
    
    # Get base position in world frame
    base_pos_w = robot.data.root_pos_w  # (B, 3)
    
    # Get base rotation matrix
    R_base = get_base_rotation_matrix(base_joint_pos)  # (B, 3, 3)
    
    # Camera's local position relative to base (extracted from Step 1)
    camera_local_pos = torch.tensor([0.163100, 0.000000, 0.059900], device=device)
    
    # Camera's static local rotation (from config: rot=((0.7071, 0.7071, 0.0, 0.0)) in OpenGL)
    # This quaternion represents a 90° rotation around X-axis
    # In IsaacLab convention (w, x, y, z)
    camera_local_quat = torch.tensor([[-0.50000480, 0.50000480, -0.49999517, 0.49999517]], device=device)
    camera_local_rot = math_utils.matrix_from_quat(camera_local_quat).squeeze(0)  # (3, 3)
    
    # Compute camera position in world frame
    # camera_pos_w = base_pos_w + R_base @ camera_local_pos
    camera_pos_w = base_pos_w + torch.matmul(
        R_base, camera_local_pos.unsqueeze(-1)
    ).squeeze(-1)  # (B, 3)
    
    # Compute camera rotation in world frame
    # R_camera_w = R_base @ R_camera_local
    R_camera_w = torch.matmul(R_base, camera_local_rot)  # (B, 3, 3)
    
    # Transform points to camera frame
    # R_w_to_c = R_camera_w^T (transpose for inverse)
    R_w_to_c = R_camera_w.transpose(-2, -1)  # (B, 3, 3)
    
    # Handle both 2D and 3D point inputs
    if points_world.dim() == 2:
        # (B, 3) points
        points_relative = points_world - camera_pos_w
        points_camera = torch.matmul(R_w_to_c, points_relative.unsqueeze(-1)).squeeze(-1)
    else:
        # (B, N, 3) points
        camera_pos_w_expanded = camera_pos_w.unsqueeze(1)  # (B, 1, 3)
        points_relative = points_world - camera_pos_w_expanded  # (B, N, 3)
        points_camera = torch.matmul(
            points_relative, R_w_to_c.transpose(-2, -1)
        )  # (B, N, 3)
    
    # Apply the 0.5 degree X-axis rotation if needed
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
    NOW USES THE CORRECTED TRANSFORM.
    """
    # Get gripper positions in world frame
    left_finger_pos_w = env.scene["finger_frame_1"].data.target_pos_w[:, 0, :]
    right_finger_pos_w = env.scene["finger_frame_2"].data.target_pos_w[:, 0, :]
    
    # Transform to camera frame using CORRECTED method
    left_finger_cam = transform_world_to_camera(left_finger_pos_w, env, sensor_cfg_name)
    right_finger_cam = transform_world_to_camera(right_finger_pos_w, env, sensor_cfg_name)
    
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
    NOW USES THE CORRECTED TRANSFORM THAT ACCOUNTS FOR BASE ROTATION.
    """
    # Get gripper positions in world frame
    left_finger_pos_w = env.scene["finger_frame_1"].data.target_pos_w[:, 0, :]
    right_finger_pos_w = env.scene["finger_frame_2"].data.target_pos_w[:, 0, :]

    # Transform gripper positions to camera frame using CORRECTED method
    left_finger_pos_cam = transform_world_to_camera(left_finger_pos_w, env, sensor_cfg_name)
    right_finger_pos_cam = transform_world_to_camera(right_finger_pos_w, env, sensor_cfg_name)
    
    # Label points in point cloud
    labels, distances = label_grippers_in_pointcloud(
        pointcloud, left_finger_pos_cam, right_finger_pos_cam, radius
    )
    
    # Count labeled points for debugging
    num_labeled = labels.sum(dim=1)
    
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
    midpoint_radius = finger_distance * 0.35
    
    # Ensure minimum radius
    min_radius = 0.005
    midpoint_radius = torch.clamp(midpoint_radius, min=min_radius)
    
    # Expand midpoint_radius for broadcasting: (B,) -> (B, 1, 1)
    midpoint_radius_expanded = midpoint_radius.unsqueeze(-1).unsqueeze(-1)
    
    # Generate spherical cloud at midpoint with dynamic radius
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
    
    # Merge all clouds together
    gripper_clouds = torch.cat([left_cloud, right_cloud, midpoint_cloud], dim=1)

    return gripper_clouds


def calculate_pointcloud_density_in_sphere(pointcloud, sphere_center, sphere_radius):
    """
    Calculate density of points within spherical regions.
    Density is scaled by sphere size to account for volume differences.

    Args:
        pointcloud: (B, N, 3) point cloud in camera frame
        sphere_center: (B, 3) center of sphere for each env
        sphere_radius: (B,) radius of sphere for each env

    Returns:
        density: (B,) normalized density value [0, 1], scaled by sphere size
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

    reference_radius = 0.0194  # Open gripper state
    sphere_radius_squeezed = sphere_radius.squeeze(1)  # (B,)

    scale_factor = (reference_radius / sphere_radius_squeezed) ** 3

    # Apply scaling
    scaled_density = density * scale_factor

    # Clamp to [0, 1]
    scaled_density = torch.clamp(scaled_density, 0.0, 1.0)

    return scaled_density, num_points_in_sphere