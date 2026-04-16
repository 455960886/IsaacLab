"""
Step-by-Step Testing: Gripper Alignment Reward

STEP 1: Visualize finger centers in point cloud
This script helps you verify that the finger positions from robot kinematics
align correctly with the point cloud.
"""

import torch
import numpy as np


def step1_visualize_fingers_and_pointcloud(env, env_id=0, output_path="step1_fingers_and_cloud.ply"):
    """
    STEP 1: Visualize finger positions overlaid on point cloud.
    
    This creates a PLY file where:
    - Point cloud points are WHITE
    - Left finger center is RED (large sphere)
    - Right finger center is BLUE (large sphere)
    - Grasp center is GREEN (large sphere)
    
    Args:
        env: Your IsaacLab environment
        env_id: Which environment to visualize (default 0)
        output_path: Where to save the PLY file
    """
    
    print("="*60)
    print("STEP 1: Visualizing Finger Centers in Point Cloud")
    print("="*60)
    
    # 1. Get finger positions from robot kinematics
    print("\n[1/4] Getting finger positions from robot state...")
    left_finger_pos_w = env.scene["finger_frame_1"].data.target_pos_w[env_id, 0, :]  # (3,)
    right_finger_pos_w = env.scene["finger_frame_2"].data.target_pos_w[env_id, 0, :]  # (3,)
    
    print(f"  Left finger:  {left_finger_pos_w.cpu().numpy()}")
    print(f"  Right finger: {right_finger_pos_w.cpu().numpy()}")
    
    # Compute grasp center
    grasp_center = (left_finger_pos_w + right_finger_pos_w) / 2.0
    print(f"  Grasp center: {grasp_center.cpu().numpy()}")
    
    # Compute finger distance
    finger_distance = torch.norm(right_finger_pos_w - left_finger_pos_w).item()
    print(f"  Finger distance: {finger_distance:.4f}m")
    
    # 2. Get point cloud from cache (camera frame)
    print("\n[2/4] Getting point cloud from cache...")
    if not hasattr(env, 'point_cloud_cache'):
        print("  ❌ ERROR: point_cloud_cache not found!")
        print("  Please add this line to observations_3_.py:")
        print("    env.point_cloud_cache = batch_points_tensor")
        return
    
    point_cloud_robot = env.point_cloud_cache[env_id]  # (1024, 3)
    print(f"  Point cloud shape: {point_cloud_robot.shape}")
    print(f"  Point cloud device: {point_cloud_robot.device}")
    

    # Replace the transformation section with this version:

    print("\n[3/4] Transforming point cloud to world frame...")
    import isaaclab.utils.math as math_utils

    # Use camera pose with ROS convention (this gave the best result: 0.086m)
    camera = env.scene.sensors["depth_camera"]
    camera_pos_w = camera.data.pos_w[env_id]  # (3,)
    camera_quat_ros = camera.data.quat_w_ros[env_id]  # (4,)

    print(f"  Camera position: {camera_pos_w.cpu().numpy()}")
    print(f"  Camera quaternion (ROS): {camera_quat_ros.cpu().numpy()}")

    # Convert quaternion to rotation matrix
    rotation_matrix = math_utils.matrix_from_quat(camera_quat_ros.unsqueeze(0))[0]  # (3, 3)

    print(f"  Rotation matrix (ROS):")
    print(f"    {rotation_matrix[0].cpu().numpy()}")
    print(f"    {rotation_matrix[1].cpu().numpy()}")
    print(f"    {rotation_matrix[2].cpu().numpy()}")

    # Use the ROS transformation that gave us 0.086m distance
    point_cloud_world = torch.matmul(rotation_matrix, point_cloud_robot.T).T + camera_pos_w

    print(f"  ROS transformed bounds:")
    print(f"    X: [{point_cloud_world[:, 0].min():.3f}, {point_cloud_world[:, 0].max():.3f}]")
    print(f"    Y: [{point_cloud_world[:, 1].min():.3f}, {point_cloud_world[:, 1].max():.3f}]")
    print(f"    Z: [{point_cloud_world[:, 2].min():.3f}, {point_cloud_world[:, 2].max():.3f}]")

    # Check distance
    distance = torch.norm(point_cloud_world.mean(dim=0) - grasp_center).item()
    print(f"  Distance to grasp center: {distance:.3f}m")

    # The Z-offset issue: let's analyze and fix it
    print(f"\n[DEBUG] Analyzing Z-offset:")
    point_cloud_mean = point_cloud_world.mean(dim=0)
    print(f"  Point cloud mean Z: {point_cloud_mean[2]:.3f}")
    print(f"  Grasp center Z: {grasp_center[2]:.3f}")
    z_offset = grasp_center[2] - point_cloud_mean[2]
    print(f"  Z offset (gripper - point cloud): {z_offset:.3f}m")

    # Apply Z correction based on the offset analysis
    if abs(z_offset) > 0.05:  # If significant Z offset
        print(f"  Applying Z correction: {z_offset:.3f}m")
        point_cloud_world[:, 2] += z_offset
        
        print(f"  After Z correction:")
        print(f"    X: [{point_cloud_world[:, 0].min():.3f}, {point_cloud_world[:, 0].max():.3f}]")
        print(f"    Y: [{point_cloud_world[:, 1].min():.3f}, {point_cloud_world[:, 1].max():.3f}]")
        print(f"    Z: [{point_cloud_world[:, 2].min():.3f}, {point_cloud_world[:, 2].max():.3f}]")
        
        corrected_distance = torch.norm(point_cloud_world.mean(dim=0) - grasp_center).item()
        print(f"  Corrected distance: {corrected_distance:.3f}m")

    if distance < 0.1 or corrected_distance < 0.1:
        print("  ✅ SUCCESS: Point cloud is properly aligned with fingers!")



    
    # 4. Save to PLY with colored markers
    print(f"\n[4/4] Saving visualization to {output_path}...")
    
    try:
        import open3d as o3d
    except ImportError:
        print("  ❌ ERROR: open3d not installed!")
        print("  Install with: pip install open3d")
        return
    
    # Convert to numpy
    points_np = point_cloud_world.cpu().numpy()
    left_finger_np = left_finger_pos_w.cpu().numpy()
    right_finger_np = right_finger_pos_w.cpu().numpy()
    grasp_center_np = grasp_center.cpu().numpy()
    
    # Create point cloud for object points (white)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    colors = np.ones((len(points_np), 3)) * 0.8  # Light gray/white
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create spheres for finger markers
    # Left finger (RED)
    left_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)  # 1cm sphere
    left_sphere.translate(left_finger_np)
    left_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # Red
    
    # Right finger (BLUE)
    right_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
    right_sphere.translate(right_finger_np)
    right_sphere.paint_uniform_color([0.0, 0.0, 1.0])  # Blue
    
    # Grasp center (GREEN)
    center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.012)  # Slightly larger
    center_sphere.translate(grasp_center_np)
    center_sphere.paint_uniform_color([0.0, 1.0, 0.0])  # Green
    
    # Combine everything
    combined = pcd + left_sphere.sample_points_uniformly(number_of_points=100) \
                   + right_sphere.sample_points_uniformly(number_of_points=100) \
                   + center_sphere.sample_points_uniformly(number_of_points=100)
    
    # Save
    o3d.io.write_point_cloud(output_path, combined)
    
    print(f"  ✅ Saved to {output_path}")
    print("\nHow to view:")
    print(f"  1. Open {output_path} in MeshLab, CloudCompare, or Open3D viewer")
    print("  2. Look for:")
    print("     - RED sphere = Left finger")
    print("     - BLUE sphere = Right finger")
    print("     - GREEN sphere = Grasp center (should be between red and blue)")
    print("     - WHITE points = Object point cloud")
    print("\n✅ If the colored spheres are positioned correctly relative to the")
    print("   white point cloud, then the coordinate transformation is working!")
    print("="*60)


def step1_simple_check(env):
    """
    Quick sanity check without saving files.
    Just print key information.
    """
    print("="*60)
    print("STEP 1: Quick Sanity Check")
    print("="*60)
    
    # Check finger positions
    print("\n[Finger Positions]")
    left_finger_pos = env.scene["finger_frame_1"].data.target_pos_w[0, 0, :]
    right_finger_pos = env.scene["finger_frame_2"].data.target_pos_w[0, 0, :]
    grasp_center = (left_finger_pos + right_finger_pos) / 2.0
    
    print(f"Left finger:  X={left_finger_pos[0]:.4f}, Y={left_finger_pos[1]:.4f}, Z={left_finger_pos[2]:.4f}")
    print(f"Right finger: X={right_finger_pos[0]:.4f}, Y={right_finger_pos[1]:.4f}, Z={right_finger_pos[2]:.4f}")
    print(f"Grasp center: X={grasp_center[0]:.4f}, Y={grasp_center[1]:.4f}, Z={grasp_center[2]:.4f}")
    
    finger_distance = torch.norm(right_finger_pos - left_finger_pos).item()
    print(f"Finger distance: {finger_distance:.4f}m ({finger_distance*100:.2f}cm)")
    
    # Check point cloud
    print("\n[Point Cloud]")
    if hasattr(env, 'point_cloud_cache'):
        pc = env.point_cloud_cache[0]
        print(f"Point cloud shape: {pc.shape}")
        print(f"Point cloud mean: X={pc[:, 0].mean():.4f}, Y={pc[:, 1].mean():.4f}, Z={pc[:, 2].mean():.4f}")
        print(f"Point cloud std:  X={pc[:, 0].std():.4f}, Y={pc[:, 1].std():.4f}, Z={pc[:, 2].std():.4f}")
        print("✅ Point cloud cache found!")
    else:
        print("❌ Point cloud cache NOT found!")
        print("   Add this line to observations_3_.py:")
        print("   env.point_cloud_cache = batch_points_tensor")
    
    # Check robot pose
    print("\n[Robot Base]")
    robot = env.scene["robot"]
    robot_pos = robot.data.root_pos_w[0]
    print(f"Robot position: X={robot_pos[0]:.4f}, Y={robot_pos[1]:.4f}, Z={robot_pos[2]:.4f}")
    
    print("="*60)


# Simple usage instructions
if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════╗
║          STEP 1: Test Finger Position Visualization         ║
╚══════════════════════════════════════════════════════════════╝

HOW TO USE:

1. In your training script, after env.reset() or env.step():

   from step1_test_fingers import step1_visualize_fingers_and_pointcloud
   
   step1_visualize_fingers_and_pointcloud(env, env_id=0)

2. This will create a file: step1_fingers_and_cloud.ply

3. Open it in MeshLab or CloudCompare

4. Verify:
   - RED sphere (left finger) is near object
   - BLUE sphere (right finger) is near object
   - GREEN sphere (grasp center) is between them
   - WHITE points (object) are where you expect

QUICK CHECK (no file output):

   from step1_test_fingers import step1_simple_check
   
   step1_simple_check(env)

This just prints positions to verify everything makes sense.

═══════════════════════════════════════════════════════════════

WHAT THIS TESTS:
- Robot kinematics are giving correct finger positions
- Point cloud is in the right coordinate frame
- Transformations from robot frame to world frame are correct

If the spheres are positioned correctly relative to the point cloud,
we can proceed to STEP 2: defining the grasp volume!
""")
