import os
import shutil

def copy_files_with_keyword(src_dir, dst_dir, keyword):
    """
    递归遍历src_dir及其子文件夹，复制文件名包含keyword的文件到dst_dir。
    
    :param src_dir: 源目录路径，如 /home/roborock/桌面/floor
    :param dst_dir: 目标目录路径，复制到这里
    :param keyword: 文件名包含的关键字
    """
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if keyword in file:
                src_file_path = os.path.join(root, file)
                dst_file_path = os.path.join(dst_dir, file)
                shutil.copy2(src_file_path, dst_file_path)
                print(f"复制文件：{src_file_path} 到 {dst_file_path}")

if __name__ == "__main__":
    src_folder = "/home/roborock/桌面/floor"  # 你的源目录
    dst_folder = "/home/roborock/桌面/floor/dataset"  # 你想复制到这里
    keyword = "JPG_Color"  # 替换成你想找的关键词

    copy_files_with_keyword(src_folder, dst_folder, keyword)
