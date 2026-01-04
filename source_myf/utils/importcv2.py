import cv2
import numpy as np
import torch

# 1. 读取图像 (BGR)
img = cv2.imread("/home/roborock/下载/camerareal1.png")

# 2. BGR -> RGB

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 计算平均亮度
mean_brightness = np.mean(gray)
contrast = np.std(gray)
print(f"亮度 :{mean_brightness}")
print(f"对比度:{contrast}")
bright = cv2.convertScaleAbs(img, alpha=1.0, beta=-60) 
img = cv2.cvtColor(bright, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 计算平均亮度
mean_brightness = np.mean(gray)
# print(mean_brightness) 
# cv2.imwrite("/home/roborock/下载/sim1.png", img)

# # 3. 转为 float32 并归一化到 [0, 1]
# img = img.astype(np.float32) / 255.0
# cv2.imwrite("/home/roborock/下载/real_21.png",img )
# 4. 转为 torch tensor
# img_tensor = torch.from_numpy(img)  # [H, W, C]

# 5. (可选) 减去每张图的 mean（每通道）
# mean_tensor = torch.mean(img_tensor, dim=(0, 1), keepdim=True)
# img_tensor = img_tensor - mean_tensor

# 6. 变成 [C, H, W]
# img_tensor = img_tensor.permute(2, 0, 1)

# # 7. 加 batch 维度 -> [1, C, H, W]
# img_tensor = img_tensor.unsqueeze(0)
# print(img_tensor)
# # 8. 保存为 .npy
# img_npy = img_tensor.numpy()
# np.save("/home/roborock/下载/9_nchw_mean.npy", img_npy)

# print("保存成功，shape:", img_npy.shape)
