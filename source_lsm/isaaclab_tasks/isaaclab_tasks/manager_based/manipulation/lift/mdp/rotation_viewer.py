import numpy as np
import matplotlib.pyplot as plt

# 定义一个3D向量
v = np.array([0, 0, 1])  # 原始向量
# R1 = np.array([
#     [1, 0, 0],
#     [0, -1, 0],
#     [0, 0, -1]
# ])
# 定义旋转矩阵（绕Z轴旋转theta角）
def rotation_matrix_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [  0.0038746, -0.0040156,  0.9999844],
        [ -0.9999474, -0.0095082,  0.0038363],
        [ 0.0094926,  -0.9999467, -0.0040522 ]
    ])
def rotation_matrix_to_xyz(R):
    """
    将旋转矩阵转换为 XYZ 欧拉角（弧度）
    旋转顺序为 X -> Y -> Z（roll -> pitch -> yaw）

    参数:
        R: 3x3旋转矩阵
    返回:
        tuple (rx, ry, rz): 绕 X, Y, Z 的旋转角度（弧度）
    """
    if R.shape != (3,3):
        raise ValueError("R must be a 3x3 matrix")

    # 防止数值误差导致arcsin越界
    sy = -R[2,0]
    sy = np.clip(sy, -1.0, 1.0)
    ry = np.arcsin(sy)  # 绕Y旋转（pitch）

    if np.abs(sy) < 0.99999:
        rx = np.arctan2(R[2,1], R[2,2])  # 绕X旋转（roll）
        rz = np.arctan2(R[1,0], R[0,0])  # 绕Z旋转（yaw）
    else:
        # 奇异情况（gimbal lock）
        rx = np.arctan2(-R[1,2], R[1,1])
        rz = 0.0
    print(np.degrees(rx), np.degrees(ry), np.degrees(rz))
    return rx, ry, rz
# 旋转角度
theta = np.pi / 4  # 45度
R = rotation_matrix_z(theta)
rotation_matrix_to_xyz(R)
v_rot = R @ v  # 旋转后的向量

# 可视化
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 原始向量
ax.quiver(0, 0, 0, v[0], v[1], v[2], color='b', label='Original')

# 旋转后向量
ax.quiver(0, 0, 0, v_rot[0], v_rot[1], v_rot[2], color='r', label='Rotated')

ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
ax.set_zlim([-1,1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()
