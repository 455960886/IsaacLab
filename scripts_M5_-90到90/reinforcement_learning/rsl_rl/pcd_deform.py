import open3d as o3d
import numpy as np

def estimate_local_geometry(pcd, k_neighbors=20):
    """Estimate curvature and planarity at each point using PCA."""
    points = np.asarray(pcd.points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    curvatures = np.zeros(len(points))
    planarity = np.zeros(len(points))
    normals = np.zeros((len(points), 3))
    
    for i in range(len(points)):
        [_, idx, _] = kdtree.search_knn_vector_3d(points[i], k_neighbors)
        neighbors = points[idx]
        
        centered = neighbors - neighbors.mean(axis=0)
        cov = centered.T @ centered / len(neighbors)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        eigenvalues = np.sort(eigenvalues)
        
        curvatures[i] = eigenvalues[0] / (eigenvalues[2] + 1e-8)
        planarity[i] = (eigenvalues[1] - eigenvalues[0]) / (eigenvalues[2] + 1e-8)
        normals[i] = eigenvectors[:, 0]
    
    curvatures = 1 - (curvatures / (curvatures.max() + 1e-8))
    
    return curvatures, planarity, normals

def round_edges(pcd, points, curvatures, curvature_threshold=0.5, smoothing_radius=0.005):
    """Round edges/corners by applying curvature-weighted Gaussian smoothing."""
    edge_mask = curvatures > curvature_threshold
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    smoothed_points = points.copy()
    
    for i in np.where(edge_mask)[0]:
        [_, idx, dists] = kdtree.search_radius_vector_3d(points[i], smoothing_radius)
        
        if len(idx) < 3:
            continue
            
        neighbors = points[idx]
        dists = np.sqrt(dists)
        
        sigma = smoothing_radius / 3
        weights = np.exp(-dists**2 / (2 * sigma**2))
        weights = weights / weights.sum()
        
        smoothed_pos = (neighbors * weights[:, None]).sum(axis=0)
        blend_factor = curvatures[i]
        smoothed_points[i] = (1 - blend_factor) * points[i] + blend_factor * smoothed_pos
    
    return smoothed_points

def generate_wave_distortion(points, wavelength=0.05, amplitude=0.005, num_waves=3):
    """
    Generate smooth wave-like distortion using sinusoidal functions.
    Much more visible and predictable than RBF.
    """
    distortion = np.zeros(len(points))
    
    # Multiple waves at different frequencies and random directions
    for i in range(num_waves):
        # Random wave direction (2D plane)
        angle = np.random.rand() * 2 * np.pi
        wave_dir = np.array([np.cos(angle), np.sin(angle), 0])
        
        # Random frequency variation
        freq = (1.0 + np.random.randn() * 0.3) / wavelength
        phase = np.random.rand() * 2 * np.pi
        
        # Project points onto wave direction and compute wave
        projection = points @ wave_dir
        wave = np.sin(2 * np.pi * freq * projection + phase)
        
        distortion += wave / num_waves
    
    # Normalize and scale
    distortion = distortion * amplitude
    
    return distortion

def distort_flat_surfaces(points, planarity, normals, 
                          planarity_threshold=0.7, 
                          distortion_amplitude=0.005,
                          wavelength=0.05):
    """
    Add smooth, wave-like distortion to flat surfaces.
    """
    flat_mask = planarity > planarity_threshold
    
    if flat_mask.sum() == 0:
        print("  Warning: No flat surfaces found!")
        return points
    
    print(f"  Distorting {flat_mask.sum()} flat surface points...")
    
    # Generate wave distortion for all points (need context for smooth waves)
    distortion_field = generate_wave_distortion(points, wavelength=wavelength, 
                                                amplitude=distortion_amplitude, num_waves=3)
    
    distorted_points = points.copy()
    
    # Apply distortion along normals, scaled by planarity
    for i in range(len(points)):
        if flat_mask[i]:
            # More planar = more distortion
            displacement = distortion_field[i] * planarity[i]
            distorted_points[i] += normals[i] * displacement
    
    # Print stats for debugging
    actual_displacement = np.linalg.norm(distorted_points[flat_mask] - points[flat_mask], axis=1)
    print(f"  Displacement stats: mean={actual_displacement.mean():.6f}, max={actual_displacement.max():.6f}")
    
    return distorted_points

def add_sim2real_augmentation(input_file, output_file,
                               curvature_threshold=0.45,
                               smoothing_radius=0.006,
                               planarity_threshold=0.6,
                               distortion_amplitude=0.01,  # Increased default
                               distortion_wavelength=0.08,
                               k_neighbors=20):
    """
    Apply sim-to-real augmentations: round edges and distort flat surfaces.
    """
    print(f"Reading: {input_file}")
    pcd = o3d.io.read_point_cloud(input_file)
    points = np.asarray(pcd.points)
    print(f"Points: {len(points)}")
    
    # Print scale info
    bbox_size = points.max(axis=0) - points.min(axis=0)
    print(f"Bounding box size: {bbox_size}")
    print(f"Suggested distortion_amplitude: {bbox_size.mean() * 0.01:.4f} (1% of object size)")
    
    print("Estimating local geometry...")
    curvatures, planarity, normals = estimate_local_geometry(pcd, k_neighbors)
    print(f"  Curvature range: [{curvatures.min():.3f}, {curvatures.max():.3f}]")
    print(f"  Planarity range: [{planarity.min():.3f}, {planarity.max():.3f}]")
    
    print("Rounding edges...")
    points = round_edges(pcd, points, curvatures, curvature_threshold, smoothing_radius)
    
    print("Distorting flat surfaces...")
    points = distort_flat_surfaces(points, planarity, normals, 
                                    planarity_threshold, distortion_amplitude,
                                    distortion_wavelength)
    
    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(points)
    if pcd.has_colors():
        pcd_out.colors = pcd.colors
    
    print(f"Saving to: {output_file}")
    o3d.io.write_point_cloud(output_file, pcd_out)
    print("Done!")

if __name__ == "__main__":
    input_file = "/home/roborock/R50_pc/pairs/pair_3/frame_000006_2_filtered.ply"
    output_file = "/home/roborock/R50_pc/pairs/pair_3/frame_000006_2_deformed.ply"
    
    add_sim2real_augmentation(
        input_file, 
        output_file,
        curvature_threshold=0.45,
        smoothing_radius=0.06,
        planarity_threshold=0.6,          # Lowered to catch more surfaces
        distortion_amplitude=0.01,        # 10x larger - adjust based on printed size
        distortion_wavelength=0.08,       # Adjust based on object size
        k_neighbors=20
    )

    # input_file = "/home/roborock/R50_pc/pairs/pair_3/frame_000006_2_filtered.ply"
    # output_file = "/home/roborock/R50_pc/pairs/pair_3/frame_000006_2_deformed.ply"