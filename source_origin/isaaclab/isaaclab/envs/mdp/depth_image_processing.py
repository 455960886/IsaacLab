import numpy as np
import cv2
import torch


class DepthGradientSmoother:
    """Applies strong smoothing in y-direction to low-gradient regions."""
    
    def __init__(
        self,
        threshold_percentile: float = 70,  # Percentile for threshold
        kernel_size: int = 15,  # Large smoothing
        sigma: float = 5.0  # Strong blur
    ):
        self.threshold_percentile = threshold_percentile
        self.kernel_size = kernel_size
        self.sigma = sigma
    
    def smooth(self, depth_image):
        """Smooth depth image in y-direction."""
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
        if depth_np.ndim == 4:
            depth_np = depth_np.squeeze(-1)
        
        if depth_np.ndim == 3:
            smoothed_list = []
            for i in range(depth_np.shape[0]):
                smoothed = self._smooth_single(depth_np[i])
                smoothed_list.append(smoothed)
            result = np.stack(smoothed_list, axis=0)
        else:
            result = self._smooth_single(depth_np)
        
        # Reshape to original
        if len(original_shape) == 4:
            result = result[..., np.newaxis]
        
        if is_tensor:
            result = torch.from_numpy(result).to(device)
        
        return result
    
    def _smooth_single(self, depth_image: np.ndarray) -> np.ndarray:
        """Smooth single image in y-direction."""
        # Compute y-gradient
        grad_y = np.zeros_like(depth_image, dtype=np.float64)
        grad_y[:-1, :] = depth_image[1:, :] - depth_image[:-1, :]
        
        magnitude = np.abs(grad_y)
        
        # Find threshold
        non_zero_mag = magnitude[magnitude > 0]
        if len(non_zero_mag) > 0:
            threshold = np.percentile(non_zero_mag, self.threshold_percentile)
        else:
            threshold = 0
        
        # Mask: True where gradient is SMALLER than threshold
        smooth_mask = (magnitude > 0) & (magnitude < threshold)
        
        # Apply strong vertical smoothing
        smoothed = cv2.GaussianBlur(
            depth_image.astype(np.float32),
            (1, self.kernel_size),  # Only y-direction
            sigmaX=0,
            sigmaY=self.sigma
        )
        
        # Apply smoothing only where mask is True
        result = depth_image.copy()
        result[smooth_mask] = smoothed[smooth_mask]
        
        return result.astype(depth_image.dtype)