import numpy as np

def txt_to_xyz_flat_npy(
    txt_path: str,
    npy_path: str,
):
    # 1. 读取 txt，取前三列 (N, 3)
    points = np.loadtxt(txt_path, usecols=(0, 1, 2))

    assert points.ndim == 2 and points.shape[1] == 3, \
        f"Expect (N,3), got {points.shape}"

    # 2. 拆分 xyz
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # 3. reorder -> [xxxx yyyy zzzz]
    reordered = np.concatenate([x, y, z], axis=0)

    # 4. 保存
    np.save(npy_path, reordered)

    print(f"Saved: {npy_path}")
    print(f"Points: {points.shape[0]}, Output shape: {reordered.shape}")



txt_to_xyz_flat_npy('/home/roborock/Downloads/frame_sampled(6).txt','/home/roborock/docker_images_v1.8.x/docker_bushu/f6.npy')
# txt_to_xyz_flat_npy('/home/roborock/Downloads/fps2.txt','/home/roborock/docker_images_v1.8.x/docker_bushu/fps2.npy')
# txt_to_xyz_flat_npy('/home/roborock/Downloads/fps3.txt','/home/roborock/docker_images_v1.8.x/docker_bushu/fps3.npy')