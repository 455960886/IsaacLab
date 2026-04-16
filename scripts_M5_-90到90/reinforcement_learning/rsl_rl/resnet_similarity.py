import sys
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
from torchvision import models
import random

# Set random seeds for reproducibility
seed = 50
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def load_resnet18_with_hooks():
    """Load ResNet18 model and register forward hooks to extract features at each layer."""
    # Load pre-trained ResNet18
    model = models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1').eval().cuda()

    # Dictionary to store intermediate features
    features = {}

    # Define hook function
    def get_activation(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook

    # Register hooks for each major layer
    # ResNet18 structure: conv1 -> bn1 -> relu -> maxpool -> layer1 -> layer2 -> layer3 -> layer4 -> avgpool -> fc
    model.conv1.register_forward_hook(get_activation('conv1'))
    model.bn1.register_forward_hook(get_activation('bn1'))
    model.relu.register_forward_hook(get_activation('relu'))
    model.maxpool.register_forward_hook(get_activation('maxpool'))
    model.layer1.register_forward_hook(get_activation('layer1'))
    model.layer2.register_forward_hook(get_activation('layer2'))
    model.layer3.register_forward_hook(get_activation('layer3'))
    model.layer4.register_forward_hook(get_activation('layer4'))
    model.avgpool.register_forward_hook(get_activation('avgpool'))

    return model, features


def prepare_image(image_path, target_size=(224, 224)):
    """Load and preprocess image for ResNet.

    Args:
        image_path: Path to the image file
        target_size: Target size for resizing (height, width)

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
    img_tensor = img_tensor.unsqueeze(0).cuda()

    return img_tensor


def cosine_similarity(x, y, eps=1e-8):
    """Compute cosine similarity between two tensors.

    Args:
        x: First tensor (any shape)
        y: Second tensor (same shape as x)
        eps: Small constant to avoid division by zero

    Returns:
        Cosine similarity value
    """
    # Flatten tensors (use reshape instead of view to handle non-contiguous tensors)
    x_flat = x.reshape(-1)
    y_flat = y.reshape(-1)

    # Compute cosine similarity
    dot = (x_flat * y_flat).sum()
    norm_x = x_flat.norm()
    norm_y = y_flat.norm()

    return dot / (norm_x * norm_y + eps)


def main():
    """Main function to compare two images."""
    # Paths to the two images to compare
    image1_path = "/path/to/image1.png"  # REPLACE WITH YOUR IMAGE PATH
    image2_path = "/path/to/image2.png"  # REPLACE WITH YOUR IMAGE PATH

    print("="*60)
    print("ResNet18 Layer-by-Layer Cosine Similarity Analysis")
    print("="*60)

    # Load images
    print(f"\nLoading images...")
    print(f"  Image 1: {image1_path}")
    print(f"  Image 2: {image2_path}")

    img1 = prepare_image(image1_path)
    img2 = prepare_image(image2_path)

    # Compute raw image similarity
    raw_similarity = cosine_similarity(img1, img2)
    print(f"\n{'Layer':<15} {'Feature Shape':<25} {'Cosine Similarity':<20}")
    print("-"*60)
    print(f"{'Raw Image':<15} {str(tuple(img1.shape)):<25} {raw_similarity.item():.6f}")

    # Load ResNet18 with hooks
    model, features_dict = load_resnet18_with_hooks()

    # Forward pass for both images
    with torch.no_grad():
        # Clear features dict
        features_dict.clear()
        _ = model(img1)
        features1 = {k: v.clone() for k, v in features_dict.items()}

        # Clear features dict
        features_dict.clear()
        _ = model(img2)
        features2 = {k: v.clone() for k, v in features_dict.items()}

    # Compare features at each layer
    layer_names = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']

    for layer_name in layer_names:
        feat1 = features1[layer_name]
        feat2 = features2[layer_name]

        similarity = cosine_similarity(feat1, feat2)

        print(f"{layer_name:<15} {str(tuple(feat1.shape)):<25} {similarity.item():.6f}")

    print("="*60)

    # Summary statistics
    print("\nSummary:")
    similarities = [cosine_similarity(features1[ln], features2[ln]).item() for ln in layer_names]
    print(f"  Mean similarity across layers: {np.mean(similarities):.6f}")
    print(f"  Std similarity across layers:  {np.std(similarities):.6f}")
    print(f"  Min similarity:                {np.min(similarities):.6f} (layer: {layer_names[np.argmin(similarities)]})")
    print(f"  Max similarity:                {np.max(similarities):.6f} (layer: {layer_names[np.argmax(similarities)]})")
    print("="*60)


if __name__ == "__main__":
    # Example usage with command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Compare two images using ResNet18 layer-by-layer cosine similarity.")
    parser.add_argument("--image1", type=str, required=True, help="Path to first image")
    parser.add_argument("--image2", type=str, required=True, help="Path to second image")
    parser.add_argument("--size", type=int, default=224, help="Image size (default: 224)")

    args = parser.parse_args()

    # Update image paths from arguments
    image1_path = args.image1
    image2_path = args.image2
    target_size = (args.size, args.size)

    print("="*60)
    print("ResNet18 Layer-by-Layer Cosine Similarity Analysis")
    print("="*60)

    # Load images
    print(f"\nLoading images...")
    print(f"  Image 1: {image1_path}")
    print(f"  Image 2: {image2_path}")
    print(f"  Target size: {target_size}")

    img1 = prepare_image(image1_path, target_size)
    img2 = prepare_image(image2_path, target_size)

    # Compute raw image similarity
    raw_similarity = cosine_similarity(img1, img2)
    print(f"\n{'Layer':<15} {'Feature Shape':<25} {'Cosine Similarity':<20}")
    print("-"*60)
    print(f"{'Raw Image':<15} {str(tuple(img1.shape)):<25} {raw_similarity.item():.6f}")

    # Load ResNet18 with hooks
    model, features_dict = load_resnet18_with_hooks()

    # Forward pass for both images
    with torch.no_grad():
        # Clear features dict
        features_dict.clear()
        _ = model(img1)
        features1 = {k: v.clone() for k, v in features_dict.items()}

        # Clear features dict
        features_dict.clear()
        _ = model(img2)
        features2 = {k: v.clone() for k, v in features_dict.items()}

    # Compare features at each layer
    layer_names = ['conv1', 'bn1', 'relu', 'maxpool', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']

    for layer_name in layer_names:
        feat1 = features1[layer_name]
        feat2 = features2[layer_name]

        similarity = cosine_similarity(feat1, feat2)

        print(f"{layer_name:<15} {str(tuple(feat1.shape)):<25} {similarity.item():.6f}")

    print("="*60)

    # Summary statistics
    print("\nSummary:")
    similarities = [cosine_similarity(features1[ln], features2[ln]).item() for ln in layer_names]
    print(f"  Mean similarity across layers: {np.mean(similarities):.6f}")
    print(f"  Std similarity across layers:  {np.std(similarities):.6f}")
    print(f"  Min similarity:                {np.min(similarities):.6f} (layer: {layer_names[np.argmin(similarities)]})")
    print(f"  Max similarity:                {np.max(similarities):.6f} (layer: {layer_names[np.argmax(similarities)]})")
    print("="*60)
