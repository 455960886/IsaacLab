import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import models
import os
import argparse
from pathlib import Path


def load_resnet18_with_hooks():
    """Load ResNet18 model and register forward hooks to extract features at each layer."""
    # Load pre-trained ResNet18
    model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1').eval()
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Dictionary to store intermediate features
    features = {}

    # Define hook function
    def get_activation(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook

    # Register hooks for each major layer
    model.conv1.register_forward_hook(get_activation('conv1'))
    model.bn1.register_forward_hook(get_activation('bn1'))
    model.relu.register_forward_hook(get_activation('relu'))
    model.maxpool.register_forward_hook(get_activation('maxpool'))
    model.layer1.register_forward_hook(get_activation('layer1'))
    model.layer2.register_forward_hook(get_activation('layer2'))
    model.layer3.register_forward_hook(get_activation('layer3'))
    model.layer4.register_forward_hook(get_activation('layer4'))
    model.avgpool.register_forward_hook(get_activation('avgpool'))

    return model, features, device


def prepare_image(image_path, target_size=(224, 224), device='cpu'):
    """Load and preprocess image for ResNet.

    Args:
        image_path: Path to the image file
        target_size: Target size for resizing (height, width)
        device: Device to place tensor on

    Returns:
        Preprocessed image tensor of shape [1, 3, H, W]
    """
    # Load image
    img = Image.open(image_path).convert('RGB')

    # Resize
    img = img.resize(target_size)

    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(img).astype(np.float32) / 255.0

    # Apply ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std

    # Convert to tensor: [H, W, C] -> [C, H, W]
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()

    # Add batch dimension: [C, H, W] -> [1, C, H, W]
    img_tensor = img_tensor.unsqueeze(0).to(device)

    return img_tensor, img


def visualize_feature_maps(features, layer_name, output_dir, image_name, max_channels=64):
    """Visualize feature maps from a layer and save to disk.
    
    Args:
        features: Feature tensor of shape [1, C, H, W]
        layer_name: Name of the layer
        output_dir: Directory to save visualizations
        image_name: Name of the input image (for naming files)
        max_channels: Maximum number of channels to visualize
    """
    # Remove batch dimension
    features = features.squeeze(0)  # [C, H, W]
    
    num_channels = features.shape[0]
    num_to_plot = min(num_channels, max_channels)
    
    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(num_to_plot)))
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(20, 20))
    fig.suptitle(f'{layer_name} - {image_name}\nShape: {tuple(features.shape)}, Showing {num_to_plot}/{num_channels} channels', 
                 fontsize=16, y=0.995)
    
    # Flatten axes for easier iteration
    if grid_size == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each channel
    for idx in range(num_to_plot):
        feature_map = features[idx].cpu().numpy()
        
        # Normalize to [0, 1] for better visualization
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        
        axes[idx].imshow(feature_map, cmap='viridis')
        axes[idx].axis('off')
        axes[idx].set_title(f'Ch {idx}', fontsize=8)
    
    # Hide unused subplots
    for idx in range(num_to_plot, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(output_dir, f'{image_name}_{layer_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")


def visualize_average_activation(features, layer_name, output_dir, image_name):
    """Visualize the average activation across all channels.
    
    Args:
        features: Feature tensor of shape [1, C, H, W]
        layer_name: Name of the layer
        output_dir: Directory to save visualizations
        image_name: Name of the input image
    """
    # Remove batch dimension and average across channels
    features = features.squeeze(0)  # [C, H, W]
    avg_activation = features.mean(dim=0).cpu().numpy()  # [H, W]
    
    # Normalize
    avg_activation = (avg_activation - avg_activation.min()) / (avg_activation.max() - avg_activation.min() + 1e-8)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    plt.imshow(avg_activation, cmap='hot')
    plt.colorbar(label='Normalized Activation')
    plt.title(f'{layer_name} - Average Activation\n{image_name}')
    plt.axis('off')
    
    # Save figure
    save_path = os.path.join(output_dir, f'{image_name}_{layer_name}_avg.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")


def visualize_top_activations(features, layer_name, output_dir, image_name, top_k=16):
    """Visualize the top-k most activated channels.
    
    Args:
        features: Feature tensor of shape [1, C, H, W]
        layer_name: Name of the layer
        output_dir: Directory to save visualizations
        image_name: Name of the input image
        top_k: Number of top channels to visualize
    """
    # Remove batch dimension
    features = features.squeeze(0)  # [C, H, W]
    
    # Calculate mean activation for each channel
    channel_means = features.mean(dim=(1, 2))  # [C]
    
    # Get top-k channels
    top_indices = torch.topk(channel_means, min(top_k, features.shape[0])).indices
    
    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(len(top_indices))))
    
    # Create figure
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    fig.suptitle(f'{layer_name} - Top {len(top_indices)} Activated Channels\n{image_name}', 
                 fontsize=14, y=0.995)
    
    # Flatten axes
    if grid_size == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot top channels
    for plot_idx, channel_idx in enumerate(top_indices):
        feature_map = features[channel_idx].cpu().numpy()
        
        # Normalize
        feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)
        
        axes[plot_idx].imshow(feature_map, cmap='viridis')
        axes[plot_idx].axis('off')
        axes[plot_idx].set_title(f'Ch {channel_idx.item()}\nμ={channel_means[channel_idx].item():.3f}', fontsize=8)
    
    # Hide unused subplots
    for idx in range(len(top_indices), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(output_dir, f'{image_name}_{layer_name}_top.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")


def save_original_image(original_img, output_dir, image_name):
    """Save the original preprocessed image.
    
    Args:
        original_img: PIL Image
        output_dir: Directory to save image
        image_name: Name of the image
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(original_img)
    plt.title(f'Original Image: {image_name}')
    plt.axis('off')
    
    save_path = os.path.join(output_dir, f'{image_name}_original.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize ResNet18 feature maps at different layers.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="feature_visualizations", 
                        help="Directory to save visualizations (default: feature_visualizations)")
    parser.add_argument("--size", type=int, default=224, help="Image size (default: 224)")
    parser.add_argument("--max_channels", type=int, default=64, 
                        help="Maximum number of channels to visualize per layer (default: 64)")
    parser.add_argument("--top_k", type=int, default=16, 
                        help="Number of top activated channels to visualize (default: 16)")
    parser.add_argument("--layers", type=str, nargs='+', 
                        default=['conv1', 'layer1', 'layer2', 'layer3', 'layer4'],
                        help="Layers to visualize (default: conv1 layer1 layer2 layer3 layer4)")
    parser.add_argument("--viz_type", type=str, choices=['all', 'grid', 'avg', 'top'], 
                        default='all', help="Type of visualization (default: all)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get image name
    image_name = Path(args.image).stem
    
    print("="*60)
    print("ResNet18 Feature Visualization")
    print("="*60)
    print(f"\nImage: {args.image}")
    print(f"Output directory: {output_dir}")
    print(f"Layers to visualize: {args.layers}")
    print(f"Visualization type: {args.viz_type}")
    
    # Load model
    print("\nLoading ResNet18 model...")
    model, features_dict, device = load_resnet18_with_hooks()
    
    # Load and preprocess image
    print(f"Loading image (target size: {args.size}x{args.size})...")
    img_tensor, original_img = prepare_image(args.image, (args.size, args.size), device)
    
    # Save original image
    save_original_image(original_img, output_dir, image_name)
    
    # Forward pass
    print("\nPerforming forward pass...")
    with torch.no_grad():
        _ = model(img_tensor)
    
    # Visualize features for each layer
    print("\nGenerating visualizations...")
    print("-"*60)
    
    for layer_name in args.layers:
        if layer_name not in features_dict:
            print(f"Warning: Layer '{layer_name}' not found. Skipping...")
            continue
        
        features = features_dict[layer_name]
        print(f"\nProcessing {layer_name} - Shape: {tuple(features.shape)}")
        
        if args.viz_type in ['all', 'grid']:
            visualize_feature_maps(features, layer_name, output_dir, image_name, args.max_channels)
        
        if args.viz_type in ['all', 'avg']:
            visualize_average_activation(features, layer_name, output_dir, image_name)
        
        if args.viz_type in ['all', 'top']:
            visualize_top_activations(features, layer_name, output_dir, image_name, args.top_k)
    
    print("\n" + "="*60)
    print(f"All visualizations saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()