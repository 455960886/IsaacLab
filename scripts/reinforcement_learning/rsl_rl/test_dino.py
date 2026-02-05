"""Minimal DINOv2-base test script (768-dim output, no gating required)."""
import sys
import torch
from transformers import AutoModel, AutoImageProcessor
from PIL import Image

image_path = "/home/roborock/rgb_sim_2_real/lego_real.png"

# Load model (frozen) - DINOv2 is freely available, 768-dim output like DINOv3-tiny
model = AutoModel.from_pretrained("facebook/dinov2-base").cuda().eval()
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

# Load and process image
image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt").to("cuda")

# Extract features
with torch.no_grad():
    outputs = model(**inputs)
    features = outputs.last_hidden_state[:, 0]  # CLS token

print(f"Feature shape: {features.shape}")  # [1, 768]
print(f"Feature sample: {features[0, :5]}")
