#!/usr/bin/env python3
"""
Simple Sim-to-Real Feature Comparison Script
Outputs raw image cosine similarity and CNN feature cosine similarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import argparse
import os


class LightweightCNN(nn.Module):
    """Lightweight CNN for RGB processing"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.device = device
        
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Move encoder to device first, then calculate flattened size
        self.image_encoder.to(device)
        
        # Calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224, device=device)
            dummy_output = self.image_encoder(dummy_input)
            self.image_flatten_size = dummy_output.shape[1]
        
        self.image_fc = nn.Sequential(
            nn.Linear(self.image_flatten_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )
        
        self.image_fc.to(device)
    
    def forward(self, x):
        features = self.image_encoder(x)
        features = self.image_fc(features)
        return features


def load_image(image_path, target_size=(224, 224), device='cpu'):
    """Load and preprocess image from file path"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0).to(device)


def compute_cosine_similarity(tensor1, tensor2):
    """Compute cosine similarity"""
    tensor1_flat = tensor1.view(tensor1.size(0), -1)
    tensor2_flat = tensor2.view(tensor2.size(0), -1)
    similarity = F.cosine_similarity(tensor1_flat, tensor2_flat, dim=1)
    return similarity.item()


def main():
    parser = argparse.ArgumentParser(description='Compare sim and real images')
    parser.add_argument('sim_image', type=str, help='Path to simulation image')
    parser.add_argument('real_image', type=str, help='Path to real image')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'])
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Load images from paths
    try:
        sim_tensor = load_image(args.sim_image, device=device)
        real_tensor = load_image(args.real_image, device=device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    except Exception as e:
        print(f"Error loading images: {e}")
        return
    
    # Raw image similarity
    raw_similarity = compute_cosine_similarity(sim_tensor, real_tensor)
    
    # CNN feature similarity
    cnn = LightweightCNN(device=device)
    cnn.eval()
    
    with torch.no_grad():
        sim_features = cnn(sim_tensor)
        real_features = cnn(real_tensor)
    
    print(sim_features)
    print(real_features)

    feature_similarity = compute_cosine_similarity(sim_features, real_features)
    
    # Output
    print(f"Raw Image Cosine Similarity: {raw_similarity:.4f}")
    print(f"CNN Feature Cosine Similarity: {feature_similarity:.4f}")


if __name__ == "__main__":
    main()