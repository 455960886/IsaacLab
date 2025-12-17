import cv2
import numpy as np


def cosine_similarity_rgb(img1_path, img2_path):
    img1 = cv2.imread(img1_path).astype(np.float32)  # OpenCV 会自动解码 JPG、PNG 等格式，只要能正常 decode 成矩阵（H×W×3），就可以做余弦相似度。
    img2 = cv2.imread(img2_path).astype(np.float32)

    # 有些 PNG 是 RGBA（4 通道），而 JPG 是 3 通道
    print(f"Image 1 shape: {img1.shape}, channels: {img1.shape[2] if len(img1.shape)==3 else 1}")
    print(f"Image 2 shape: {img2.shape}, channels: {img2.shape[2] if len(img2.shape)==3 else 1}")

    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # 展开为向量（包含RGB通道）
    vec1 = img1.flatten()
    vec2 = img2.flatten()

    sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return sim


print(cosine_similarity_rgb("/home/robo/.local/share/ov/data/documents/Kit/shared/screenshots/capture.2025-11-17 21.04.21.png", "/home/robo/下载/gripper_real.jpg"))