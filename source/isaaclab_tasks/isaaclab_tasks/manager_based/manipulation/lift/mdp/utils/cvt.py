import cv2

# 读取原图
img = cv2.imread('/home/robo/code/下载/9.png')

# 调整大小为 150x200
resized = cv2.resize(img, (128, 128))  # 注意：尺寸是 (宽, 高)

# 保存或显示
cv2.imwrite('/home/robo/code/下载/128*128.png', resized)
