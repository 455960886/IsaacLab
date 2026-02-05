import numpy as np
import cv2
import torch


class DepthGradientSmoother:
    """
    Applies selective gradient-based smoothing to depth images.
    Smooths low-gradient regions (inside objects) while preserving edges.
    """
    
    def __init__(
        self,
        threshold_percentile: float = 70,
        transition_width: float = 3.0,
        kernel_size: int = 5,
        sigma: float = 1.0,
        verbose: bool = False
    ):
        """
        Initialize the depth smoother.
        
        Args:
            threshold_percentile: Percentile for gradient threshold (0-100)
            transition_width: Controls smoothness of transition (higher = more gradual)
            kernel_size: Gaussian kernel size for smoothing
            sigma: Gaussian kernel sigma
            verbose: Print debug information
        """
        self.threshold_percentile = threshold_percentile
        self.transition_width = transition_width
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.verbose = verbose
    
    def smooth(self, depth_image):
        """
        Apply selective smoothing to depth image.
        
        Args:
            depth_image: Depth image as numpy array or torch tensor
                        Shape: (H, W) or (B, H, W) or (B, H, W, 1)
        
        Returns:
            smoothed_depth: Smoothed depth image in same format as input
        """
        # Handle torch tensors
        is_tensor = isinstance(depth_image, torch.Tensor)
        if is_tensor:
            device = depth_image.device
            original_shape = depth_image.shape
            depth_np = depth_image.detach().cpu().numpy()
        else:
            depth_np = depth_image
            original_shape = depth_np.shape
        
        # Handle batch dimension
        if depth_np.ndim == 4:  # (B, H, W, 1)
            depth_np = depth_np.squeeze(-1)
        
        if depth_np.ndim == 3:  # (B, H, W)
            # Process each image in batch
            smoothed_list = []
            for i in range(depth_np.shape[0]):
                smoothed = self._smooth_single(depth_np[i])
                smoothed_list.append(smoothed)
            result = np.stack(smoothed_list, axis=0)
        else:  # (H, W)
            result = self._smooth_single(depth_np)
        
        # Reshape to original
        if len(original_shape) == 4:
            result = result[..., np.newaxis]
        
        # Convert back to tensor if needed
        if is_tensor:
            result = torch.from_numpy(result).to(device)
        
        return result
    
    def _smooth_single(self, depth_image: np.ndarray) -> np.ndarray:
        """Smooth a single depth image (H, W)."""
        # Compute gradients
        grad_x = np.zeros_like(depth_image, dtype=np.float64)
        grad_y = np.zeros_like(depth_image, dtype=np.float64)
        
        grad_x[:, :-1] = depth_image[:, 1:] - depth_image[:, :-1]
        grad_y[:-1, :] = depth_image[1:, :] - depth_image[:-1, :]
        
        # Compute gradient magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        if self.verbose:
            print(f"Gradient magnitude - min: {magnitude.min():.6f}, max: {magnitude.max():.6f}, mean: {magnitude.mean():.6f}")
            print(f"Non-zero gradients: {np.count_nonzero(magnitude)} / {magnitude.size}")
        
        # Determine threshold
        non_zero_mag = magnitude[magnitude > 0]
        if len(non_zero_mag) > 0:
            threshold = np.percentile(non_zero_mag, self.threshold_percentile)
        else:
            threshold = 0
        
        if self.verbose:
            print(f"Threshold: {threshold:.6f}, Transition width: {self.transition_width}")
        
        # Create smooth weight map using sigmoid
        weight_map = 1.0 / (1.0 + np.exp((magnitude - threshold) / self.transition_width))
        weight_map[magnitude == 0] = 0  # Don't smooth background
        
        if self.verbose:
            print(f"Average smoothing weight (non-background): {weight_map[magnitude > 0].mean():.3f}")
        
        # Apply Gaussian smoothing to depth
        smoothed_depth = cv2.GaussianBlur(
            depth_image.astype(np.float32), 
            (self.kernel_size, self.kernel_size), 
            self.sigma
        )
        
        # Gradual blend based on weight_map
        result = weight_map * smoothed_depth + (1 - weight_map) * depth_image
        
        return result.astype(depth_image.dtype)


# Example standalone usage
if __name__ == "__main__":
    # Create smoother
    smoother = DepthGradientSmoother(
        threshold_percentile=70,
        transition_width=3.0,
        kernel_size=5,
        sigma=1.0,
        verbose=True
    )
    
    # Read and smooth
    depth = cv2.imread("/home/roborock/depth_gradient_output/data/depth.png", cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    depth = depth.astype(np.float32)
    
    smoothed = smoother.smooth(depth)
    
    # Save
    depth_norm = cv2.normalize(smoothed, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite("/home/roborock/depth_gradient_output/data/depth_smoothed.png", depth_norm.astype(np.uint8))
    
    print("Done!")