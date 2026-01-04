import os

# 设置你的图片文件夹路径
folder_path = '/home/roborock/桌面/floor/dataset'  # ✅ 例如：'/Users/yourname/Pictures'

# 可识别的图片扩展名（可根据需要添加）
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif', '.webp')

# 输出文件名（可选）
output_file = '/home/roborock/桌面/floor.txt'

# 打开文件准备写入
with open(output_file, 'w') as f:
    # 遍历文件夹中的文件
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(image_extensions):
            full_path = os.path.abspath(os.path.join(folder_path, filename))
            print(full_path)             # 打印到控制台
            f.write(full_path + '\n')    # 写入到文件

print(f"\n✅ 所有图片路径已写入到文件: {output_file}")
