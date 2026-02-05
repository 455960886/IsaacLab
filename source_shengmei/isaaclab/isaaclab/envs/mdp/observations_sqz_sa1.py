# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.sensors import Camera, Imu, RayCaster, RayCasterCamera, TiledCamera
import numpy as np
import torchvision
import cv2
import os
import time
from isaaclab.pointnet.models.pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
import importlib
from isaaclab.pointnet.log.classification.pointnet2_ssg_wo_normals.pointnet2_cls_ssg import get_model as PointNet2ClsMsg
# from .gripper_transform import add_gripper_labels_to_observation, debug_gripper_transformation, transform_world_to_camera
import open3d as o3d


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

"""
Root state.
"""

def base_pos_z(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root height in the simulation world frame."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2].unsqueeze(-1)


def base_lin_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_b


def base_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Root angular velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_b


def projected_gravity(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gravity projection on the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.projected_gravity_b


def root_pos_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root position in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w - env.scene.env_origins


def root_quat_w(
    env: ManagerBasedEnv, make_quat_unique: bool = False, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Asset root orientation (w, x, y, z) in the environment frame.

    If :attr:`make_quat_unique` is True, then returned quaternion is made unique by ensuring
    the quaternion has non-negative real component. This is because both ``q`` and ``-q`` represent
    the same orientation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    quat = asset.data.root_quat_w
    # make the quaternion real-part positive if configured
    return math_utils.quat_unique(quat) if make_quat_unique else quat


def root_lin_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root linear velocity in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w


def root_ang_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Asset root angular velocity in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w


"""
Joint state.
"""


def joint_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids]


def joint_pos_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # with open('output_5142.txt', 'a') as f:
    #     f.write(f"obs1 m: {(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]).mean().item()}\n")
    #     f.write(f"obs1 s: {(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]).std().item()}\n")
    # print("obs1 m: ",(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]).mean().item())
    # print("obs1 s: ",(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]).std().item())
    return asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]


def joint_pos_limit_normalized(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """The joint positions of the asset normalized with the asset's joint limits.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their normalized positions returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return math_utils.scale_transform(
        asset.data.joint_pos[:, asset_cfg.joint_ids],
        asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 0],
        asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, 1],
    )


def joint_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]


def joint_vel_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # with open('output_5142.txt', 'a') as f:
    #     f.write(f"obs2 m: {(asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]).mean().item()}\n")
    #     f.write(f"obs2 s: {(asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]).std().item()}\n")
    # print("obs2 m:",(asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]).mean().item())
    # print("obs2 s:",(asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]).std().item())
    return asset.data.joint_vel[:, asset_cfg.joint_ids] - asset.data.default_joint_vel[:, asset_cfg.joint_ids]


"""
Sensors.
"""


def height_scan(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset


def body_incoming_wrench(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Incoming spatial wrench on bodies of an articulation in the simulation world frame.

    This is the 6-D wrench (force and torque) applied to the body link by the incoming joint force.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # obtain the link incoming forces in world frame
    link_incoming_forces = asset.root_physx_view.get_link_incoming_joint_force()[:, asset_cfg.body_ids]
    return link_incoming_forces.view(env.num_envs, -1)


def imu_orientation(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Imu sensor orientation in the simulation world frame.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an IMU sensor. Defaults to SceneEntityCfg("imu").

    Returns:
        Orientation in the world frame in (w, x, y, z) quaternion form. Shape is (num_envs, 4).
    """
    # extract the used quantities (to enable type-hinting)
    asset: Imu = env.scene[asset_cfg.name]
    # return the orientation quaternion
    return asset.data.quat_w


def imu_ang_vel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Imu sensor angular velocity w.r.t. environment origin expressed in the sensor frame.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an IMU sensor. Defaults to SceneEntityCfg("imu").

    Returns:
        The angular velocity (rad/s) in the sensor frame. Shape is (num_envs, 3).
    """
    # extract the used quantities (to enable type-hinting)
    asset: Imu = env.scene[asset_cfg.name]
    # return the angular velocity
    return asset.data.ang_vel_b


def imu_lin_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("imu")) -> torch.Tensor:
    """Imu sensor linear acceleration w.r.t. the environment origin expressed in sensor frame.

    Args:
        env: The environment.
        asset_cfg: The SceneEntity associated with an IMU sensor. Defaults to SceneEntityCfg("imu").

    Returns:
        The linear acceleration (m/s^2) in the sensor frame. Shape is (num_envs, 3).
    """
    asset: Imu = env.scene[asset_cfg.name]
    return asset.data.lin_acc_b


def image(
    env: ManagerBasedEnv,
    # cnt: int = 0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
    depth_cfg : SceneEntityCfg = SceneEntityCfg("tiled_camera2"),
) -> torch.Tensor:
    """Images of a specific datatype from the camera sensor.

    If the flag :attr:`normalize` is True, post-processing of the images are performed based on their
    data-types:

    - "rgb": Scales the image to (0, 1) and subtracts with the mean of the current image batch.
    - "depth" or "distance_to_camera" or "distance_to_plane": Replaces infinity values with zero.

    Args:
        env: The environment the cameras are placed within.
        sensor_cfg: The desired sensor to read from. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The data type to pull from the desired camera. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        normalize: Whether to normalize the images. This depends on the selected data type.
            Defaults to True.

    Returns:
        The images produced at the last time-step
    """
    # extract the used quantities (to enable type-hinting)
    sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

    # obtain the input image
    images = sensor.data.output[data_type]
    # depth image conversion
    # images = images[:, :, images.shape[2] // 2:, :]

    # obs_image = torch.tensor(images).float().squeeze(0).cpu().numpy()  # Convert to tensor and float type
    # obs_bgr = cv2.cvtColor(obs_image, cv2.COLOR_RGB2BGR)
    # os.makedirs("IMAGES2", exist_ok=True)
    # # os.makedirs("IMAGES17", exist_ok=True)
    step = 0
    step +=1
    # # cv2.imwrite(f"./IMAGES16/observation_{time1}.png", obs_image)
    # cv2.imwrite(f"/home/roborock/下载/{step}.png", images)
    # import pdb
    # pdb.set_trace()
    # with open('output_formres9.txt', 'a') as f:
    #     f.write(f"observation_{time1}.png\n")
    # print(f"./IMAGES2/observation_{time1}.png")
    depth = env.scene.sensors[depth_cfg.name].data.output["distance_to_image_plane"]
    # print("depth shape:",depth.shape)
    # depth_np = depth.squeeze(0).squeeze(-1).cpu().numpy()  # shape [H, W]

    # # 归一化到 0~255
    # depth_norm = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
    # depth_uint8 = (depth_norm * 255).astype(np.uint8)

    # os.makedirs("depth_images", exist_ok=True)
    # timestamp = time.time()
    # cv2.imwrite(f"depth_images/depth_{timestamp}.png", depth_uint8)
    if (data_type == "distance_to_camera") and convert_perspective_to_orthogonal:
        images = math_utils.orthogonalize_perspective_depth(images, sensor.data.intrinsic_matrices)
    # obs_np = rgb_image_tensor.squeeze(0).cpu().numpy()
    # # act_np = actions.cpu().numpy()
    # # os.makedirs(act_log_dir, exist_ok=True)
    # # np.save(os.path.join(act_log_dir, f"act_step_{t}.npy"), act_np)
    # if obs_np.dtype == np.float32 or obs_np.max() <= 1.0:
    #     obs_np = (obs_np * 255).astype(np.uint8)

    # # RGB 转 BGR 再保存
    # obs_bgr = cv2.cvtColor(obs_np, cv2.COLOR_RGB2BGR)
    # os.makedirs("IMAGES1", exist_ok=True)
    # cv2.imwrite(f"./IMAGES1/observation_{cnt}.png", obs_bgr)
    # print(f"./IMAGES1/observation_{cnt}.png")
    # rgb/depth image normalization
    if normalize:
        # print(f"Normalizing images of type: {data_type}")
        if data_type == "rgb":
            images = images.float() / 255.0
            mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
            images -= mean_tensor

            # images = images.float()

            # obs_np2 = images.squeeze(0).cpu().numpy()
            # # act_np = actions.cpu().numpy()
            # # os.makedirs(act_log_dir, exist_ok=True)
            # # np.save(os.path.join(act_log_dir, f"act_step_{t}.npy"), act_np)
            # if obs_np2.dtype == np.float32 or obs_np2.max() <= 1.0:
            #     obs_np2 = (obs_np2 * 255).astype(np.uint8)

            # # RGB 转 BGR 再保存
            # obs_bgr2 = cv2.cvtColor(obs_np2, cv2.COLOR_RGB2BGR)
            # os.makedirs("IMAGES13", exist_ok=True)
            # cv2.imwrite(f"./IMAGES13/observation_{time.time()}.png", obs_np2)

            pass
        elif "distance_to" in data_type or "depth" in data_type:
            images[images == float("inf")] = 0
    # print("image shape11:",images.shape)
    #深度图与RGB图拼接
    images = torch.cat((images,depth),dim=-1)
    # print("image shape22:",images.shape)
    return images.clone()


class image_features(ManagerTermBase):
    """Extracted image features from a pre-trained frozen encoder.

    This term uses models from the model zoo in PyTorch and extracts features from the images.

    It calls the :func:`image` function to get the images and then processes them using the model zoo.

    A user can provide their own model zoo configuration to use different models for feature extraction.
    The model zoo configuration should be a dictionary that maps different model names to a dictionary
    that defines the model, preprocess and inference functions. The dictionary should have the following
    entries:

    - "model": A callable that returns the model when invoked without arguments.
    - "reset": A callable that resets the model. This is useful when the model has a state that needs to be reset.
    - "inference": A callable that, when given the model and the images, returns the extracted features.

    If the model zoo configuration is not provided, the default model zoo configurations are used. The default
    model zoo configurations include the models from Theia :cite:`shang2024theia` and SqueezeNet.
    These models are loaded from `Hugging-Face transformers <https://huggingface.co/docs/transformers/index>`_ and
    `PyTorch torchvision <https://pytorch.org/vision/stable/models.html>`_ respectively.

    Args:
        sensor_cfg: The sensor configuration to poll. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The sensor data type. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        model_zoo_cfg: A user-defined dictionary that maps different model names to their respective configurations.
            Defaults to None. If None, the default model zoo configurations are used.
        model_name: The name of the model to use for inference. Defaults to "squeezenet1_1".
        model_device: The device to store and infer the model on. This is useful when offloading the computation
            from the environment simulation device. Defaults to the environment device.
        inference_kwargs: Additional keyword arguments to pass to the inference function. Defaults to None,
            which means no additional arguments are passed.

    Returns:
        The extracted features tensor. Shape is (num_envs, feature_dim).

    Raises:
        ValueError: When the model name is not found in the provided model zoo configuration.
        ValueError: When the model name is not found in the default model zoo configuration.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)

        # extract parameters from the configuration
        self.model_zoo_cfg: dict = cfg.params.get("model_zoo_cfg")  # type: ignore
        self.model_name: str = cfg.params.get("model_name", "squeezenet1_1")  # type: ignore
        self.model_device: str = cfg.params.get("model_device", env.device)  # type: ignore

        # List of Theia models - These are configured through `_prepare_theia_transformer_model` function
        default_theia_models = [
            "theia-tiny-patch16-224-cddsv",
            "theia-tiny-patch16-224-cdiv",
            "theia-small-patch16-224-cdiv",
            "theia-base-patch16-224-cdiv",
            "theia-small-patch16-224-cddsv",
            "theia-base-patch16-224-cddsv",
        ]
        # List of SqueezeNet models - These are configured through `_prepare_squeezenet_model` function
        default_squeezenet_models = ["squeezenet1_0", "squeezenet1_1"]

        # Check if model name is specified in the model zoo configuration
        if self.model_zoo_cfg is not None and self.model_name not in self.model_zoo_cfg:
            raise ValueError(
                f"Model name '{self.model_name}' not found in the provided model zoo configuration."
                " Please add the model to the model zoo configuration or use a different model name."
                f" Available models in the provided list: {list(self.model_zoo_cfg.keys())}."
                "\nHint: If you want to use a default model, consider using one of the following models:"
                f" {default_theia_models + default_squeezenet_models}. In this case, you can remove the"
                " 'model_zoo_cfg' parameter from the observation term configuration."
            )
        if self.model_zoo_cfg is None:
            if self.model_name in default_theia_models:
                model_config = self._prepare_theia_transformer_model(self.model_name, self.model_device)
            elif self.model_name in default_squeezenet_models:
                model_config = self._prepare_squeezenet_model(self.model_name, self.model_device)
            else:
                raise ValueError(
                    f"Model name '{self.model_name}' not found in the default model zoo configuration."
                    f" Available models: {default_theia_models + default_squeezenet_models}."
                )
        else:
            model_config = self.model_zoo_cfg[self.model_name]

        # Retrieve the model, preprocess and inference functions
        self._model = model_config["model"]()
        self._reset_fn = model_config.get("reset")
        self._inference_fn = model_config["inference"]
        self._prepare_pointnet_model()
        # self.fx, self.fy = 525.0, 525.0
        # self.cx, self.cy = 319.5, 239.5

        # self.fx, self.fy = 117.78, 124.95
        # self.cx, self.cy = 200.0, 150.0

        self._frame_counter = 0


    def reset(self, env_ids: torch.Tensor | None = None):
        # reset the model if a reset function is provided
        # this might be useful when the model has a state that needs to be reset
        # for example: video transformers
        if self._reset_fn is not None:
            self._reset_fn(self._model, env_ids)

    def depth_to_pointcloud(self,depth_image, fx, fy, cx, cy, rgb_image=None, output_path="pointcloud.ply"):
        """
        将深度图转换为点云（可选带颜色）

        参数:
            depth_image : np.ndarray
                深度图（H, W），单位为米。
            fx, fy, cx, cy : float
                相机内参。
            rgb_image : np.ndarray, optional
                彩色图（H, W, 3），与深度图对齐。
            output_path : str
                点云保存路径。
        """
        assert len(depth_image.shape) == 2, "深度图必须是单通道 (H, W)"
        height, width = depth_image.shape
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        # 深度图中无效值置0（避免NaN）
        depth = np.nan_to_num(depth_image, nan=0.0)
        # depth = (depth.max() - depth)
        # print(depth_image.dtype)
        # print("min, max, median:", np.nanmin(depth_image), np.nanmax(depth_image), np.nanmedian(depth_image))
        # print("non-zero fraction:", np.count_nonzero(~np.isnan(depth_image) & (depth_image!=0)) / depth_image.size)
        mask = depth > 0  # 有效深度

        # 反投影到3D空间
        Z = depth[mask]
        X = (u[mask] - cx) * Z / fx
        Y = (v[mask] - cy) * Z / fy
        points = np.stack((X, -Y, Z), axis=-1)

        def save_ply(points, colors=None, output_path="pointcloud.ply"):
            """
            保存点云为 PLY 文件
            参数:
                points: (N, 3) numpy 数组
                colors: (N, 3) numpy 数组 (0~255 或 0~1)
                output_path: 输出文件路径
            """
            # 创建 open3d 点云对象
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)

            if colors is not None:
                if colors.max() > 1.0:
                    colors = colors / 255.0  # 归一化到 [0,1]
                pcd.colors = o3d.utility.Vector3dVector(colors)

            # 保存为 PLY 文件
            o3d.io.write_point_cloud(output_path, pcd)
            print(f"✅ 点云已保存到: {output_path}")

        # save_ply(points, colors=None, output_path=output_path.replace(".ply","_0.ply"))
        theta = np.deg2rad(0.5)
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta),  np.cos(theta)]
        ])

        rotated_points = points @ R_x.T
        # save_ply(rotated_points, colors=None, output_path=output_path.replace(".ply","_1.ply"))
        # save_ply(points, colors=None, output_path=output_path)
        # ===== 距离筛选部分 =====
        points = rotated_points[rotated_points[:,2]<0.16]
        points = points[points[:,1]>-0.05]

        def voxel_down_sample_fixed(points, voxel_size=2.0, num_points=1024, seed=None):
            """
            对点云进行体素下采样，并确保输出固定数量的点。

            参数:
                points: np.ndarray, shape [N, 3]
                voxel_size: float, 体素大小
                num_points: int, 输出固定点数
                seed: int or None, 随机种子（可选）

            返回:
                down_points: np.ndarray, shape [num_points, 3]
            """
            if len(points) == 0:
                # 返回一个全零点云（或可选 raise）
                return np.zeros((num_points, 3), dtype=np.float32)

            if seed is not None:
                np.random.seed(seed)

            decay_rate = voxel_size / 2.0
            dist = np.linalg.norm(points, axis=1)
            p = np.exp(-dist / decay_rate)   # 近处概率大
            p /= p.sum()

            # ✅ 修复点：若点数不足，则允许放回采样
            replace_flag = len(points) < num_points
            indices = np.random.choice(len(points), num_points, replace=replace_flag, p=p)
            down_points = points[indices]

            # ✅ 第二步其实可以省略，但如果你想保持逻辑清晰：
            N = down_points.shape[0]
            if N < num_points:
                extra_indices = np.random.choice(N, num_points - N, replace=True)
                down_points = np.concatenate([down_points, down_points[extra_indices]], axis=0)

            return down_points

        points = voxel_down_sample_fixed(points, voxel_size=2.0)
        # save_ply(points, colors=None, output_path=output_path.replace(".ply","_downsampled8.ply"))
        return points


    # GPU-accelerated version for batch processing
    def depth_to_pointcloud_batch_gpu(self, depth_batch, fx, fy, cx, cy, num_points=1024,
                                      save_ply_debug=False, env_id=0, frame_counter=None, save_dir="debug_pointclouds", env=None):
        """GPU-accelerated batch point cloud generation with systematic PLY saving."""
        import os
        B, H, W = depth_batch.shape
        device = depth_batch.device

        # Setup save directory if debugging
        if save_ply_debug:
            env_dir = os.path.join(save_dir, f"env_{env_id}")
            os.makedirs(env_dir, exist_ok=True)

            # Use frame counter or fallback to timestamp
            if frame_counter is not None:
                prefix = f"frame_{frame_counter:06d}"
            else:
                import time
                prefix = f"time_{int(time.time() * 1000)}"

        # Helper function to save PLY files
        def save_ply(points_np, stage_name):
            """Save point cloud as PLY file."""
            if save_ply_debug and env_id < B:
                filepath = os.path.join(env_dir, f"{prefix}_{stage_name}.ply")
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_np)
                o3d.io.write_point_cloud(filepath, pcd)
                print(f"✅ Saved: {filepath}")

        # Create pixel grid on GPU
        v_coords = torch.arange(H, device=device, dtype=torch.float32)
        u_coords = torch.arange(W, device=device, dtype=torch.float32)
        v, u = torch.meshgrid(v_coords, u_coords, indexing='ij')

        # Expand to batch
        u = u.unsqueeze(0).expand(B, -1, -1)
        v = v.unsqueeze(0).expand(B, -1, -1)

        # Back-project to 3D
        Z = depth_batch
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        points = torch.stack([X, -Y, Z], dim=-1)

        # Save Stage 0: Initial projection
        if save_ply_debug:
            points_initial = points[env_id].reshape(-1, 3).cpu().numpy()
            # points_initial[:, 1] += 0.0632
            save_ply(points_initial, "0_initial")

        # Apply rotation (0.5 degrees)
        def deg2rad(v, device):
            return torch.tensor(v, device=device) * torch.pi / 180.0

        roll  = deg2rad(90.0, device)
        pitch = deg2rad(0.0,  device)
        yaw   = deg2rad(90.0, device)
        c1, s1 = torch.cos(roll), torch.sin(roll)
        c2, s2 = torch.cos(pitch), torch.sin(pitch)
        c3, s3 = torch.cos(yaw), torch.sin(yaw)

        Rx = torch.tensor([
            [1, 0, 0],
            [0,  c1, -s1],
            [0,  s1,  c1]
        ], device=device, dtype=torch.float32)

        Ry = torch.tensor([
            [ c2, 0, s2],
            [  0, 1, 0],
            [-s2, 0, c2]
        ], device=device, dtype=torch.float32)

        Rz = torch.tensor([
            [c3, -s3, 0],
            [s3,  c3, 0],
            [ 0,   0, 1]
        ], device=device, dtype=torch.float32)
        # -----------------------------
        # 构造 4x4 transformation 矩阵 T
        # -----------------------------
        x1 = deg2rad(0.011, device)

        c = torch.cos(x1)
        s = torch.sin(x1)

        Rx1 = torch.tensor([
            [1., 0., 0.],
            [0.,     c,    -s],
            [0.,     s,     c],
        ], device=device)

        R = Rx1@ Rz @ Ry @ Rx
        # print(R)
        # import pdb
        # pdb.set_trace()
        points_flat = points.reshape(B, H * W, 3)
        rotated_points = torch.matmul(points_flat, R.T)
        translation = torch.tensor([0.1654, 0.0, 0.0494], device=device)
        trans_points = rotated_points +translation
        # Save Stage 1: After rotation
        if save_ply_debug:
            points_trans = trans_points[env_id].cpu().numpy()
            save_ply(points_trans, "1_rotated")

        # Apply distance filtering
        # mask1 = rotated_points[:, :, 2] < 0.21
        # mask2 = rotated_points[:, :, 1] > -0.0628
        # mask3 = rotated_points[:, :, 1] < 0.0428
        rand_thresh = np.random.uniform(-0.0003, 0.002)
        mask2 = trans_points[:,:, 0] <=0.42
        # mask3 = trans_points[:,:, 2] >= -0.0003
        mask3 = trans_points[:,:, 2] >= rand_thresh

        # mask4 = trans_points[:,:, 1] <=0.20
        # mask5 = trans_points[:,:, 1] >=-0.20
        mask = mask2 & mask3
        # Save Stage 2: After filtering
        if save_ply_debug:
            points_filtered = trans_points[env_id][mask[env_id]].cpu().numpy()
            save_ply(points_filtered, "2_filtered")

        # Sample fixed number of points
        sampled_points = []
        for b in range(B):
            valid_points = trans_points[b][mask[b]]

            if len(valid_points) == 0:
                sampled_points.append(torch.zeros(num_points, 3, device=device))
            elif len(valid_points) >= num_points:
                dist = torch.norm(valid_points, dim=1)
                weights = torch.exp(-dist / 1.0)
                weights = weights / weights.sum()
                indices = torch.multinomial(weights, num_points, replacement=False)
                sampled_points.append(valid_points[indices])
            else:
                indices = torch.randint(0, len(valid_points), (num_points,), device=device)
                sampled_points.append(valid_points[indices])

        result = torch.stack(sampled_points, dim=0)

        # Save Stage 3: Final downsampled
        if save_ply_debug:
            points_final = result[env_id].cpu().numpy()
            save_ply(points_final, "3_downsampled")

        # from .pointcloud_noise import add_noise

        # for b in range(B):
        #     result[b] = add_noise(result[b])
        #         # Save Stage 3: Final downsampled

        # if save_ply_debug:
        #     points_final = result[env_id].cpu().numpy()
        #     save_ply(points_final, "4_noised")
        # result = randomize_pointcloud_batch_torch(result,dropout_rate=0.02,outlier_ratio=0.02,outlier_max_offset=0.08,surface_jitter=0.001)

        # if save_ply_debug:
        #     points_final = result[env_id].cpu().numpy()
        #     save_ply(points_final, "4_random")

        return result


    def _apply_domain_randomization(
        self,
        images: torch.Tensor,
        save_debug: bool = False,
        step_counter: int = 0
    ) -> torch.Tensor:
        """
        Apply domain randomization (Gaussian blur and noise) to images for sim-to-real transfer.

        Args:
            images: Input images tensor of shape (B, C, H, W) or (B, H, W, C)
            save_debug: If True, save before/after images
            step_counter: Current step number for filename

        Returns:
            Augmented images tensor
        """
        # Convert to float if needed and normalize to [0, 1]
        original_dtype = images.dtype
        if images.dtype == torch.uint8:
            images = images.float() / 255.0

        # Ensure tensor is in (B, C, H, W) format for transforms
        if images.ndim == 4 and images.shape[-1] in [1, 3, 4]:  # (B, H, W, C)
            images = images.permute(0, 3, 1, 2)
            was_channels_last = True
        else:
            was_channels_last = False

        if save_debug:
            self._save_images(images, step_counter, prefix="before_aug")

        # Random Gaussian blur
        # if torch.rand(1).item() < 0.5:  # 50% chance to apply blur
        kernel_size = int(torch.randint(3, 12, (1,)).item())
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
        sigma = torch.rand(1).item() * 3.0 + 0.5  # Random sigma between 0.5-2.0
        images = torchvision.transforms.functional.gaussian_blur(
            images, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma]
        )

        # Random Gaussian noise
        noise_std = torch.rand(1).item() * 0.08 + 0.02  # Random std between 0-0.03
        noise = torch.randn_like(images) * noise_std
        images = torch.clamp(images + noise, 0.0, 1.0)

        if save_debug:
            self._save_images(images, step_counter, prefix="after_aug")

        # Convert back to original format
        if was_channels_last:
            images = images.permute(0, 2, 3, 1)

        if original_dtype == torch.uint8:
            images = (images * 255.0).byte()

        return images

    def _save_images(self, images: torch.Tensor, step: int, prefix: str = "img"):
        """Save images to disk for debugging."""
        import os
        import cv2
        import numpy as np

        save_dir = "debug_augmentation"
        os.makedirs(save_dir, exist_ok=True)

        # Save only the first image in the batch
        img_to_save = images[0]  # Take first image from batch

        # Convert from torch tensor to numpy
        img_np = img_to_save.detach().cpu().numpy()

        # Check the shape and handle accordingly
        print(f"Image shape: {img_np.shape}")  # Debug print

        # If shape is (C, H, W), convert to (H, W, C)
        if img_np.ndim == 3 and img_np.shape[0] in [1, 3, 4]:  # Channel first
            img_np = np.transpose(img_np, (1, 2, 0))

        # Ensure values are in [0, 1] range, then convert to [0, 255]
        img_np = np.clip(img_np, 0, 1)
        img_np = (img_np * 255).astype(np.uint8)

        # Handle different channel counts
        if img_np.shape[-1] == 3:  # RGB
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif img_np.shape[-1] == 1:  # Grayscale
            img_bgr = img_np.squeeze(-1)
        elif img_np.shape[-1] == 4:  # RGBA
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
        else:
            print(f"Warning: Unexpected channel count {img_np.shape[-1]}, saving as-is")
            img_bgr = img_np

        save_path = os.path.join(save_dir, f"{prefix}_step_{step:06d}.png")
        cv2.imwrite(save_path, img_bgr)


    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera"),
        depth_cfg: SceneEntityCfg = SceneEntityCfg("tiled_camera2"),
        data_type: str = "rgb",
        convert_perspective_to_orthogonal: bool = False,
        model_zoo_cfg: dict | None = None,
        model_name: str = "squeezenet1_1",
        model_device: str | None = None,
        inference_kwargs: dict | None = None,
        save_augmentation_debug: bool = False,
    ) -> torch.Tensor:
        # obtain the images from the sensor
        # image_data = image(
        #     env=env,
        #     sensor_cfg=sensor_cfg,
        #     data_type=data_type,
        #     convert_perspective_to_orthogonal=convert_perspective_to_orthogonal,
        #     normalize=False,  # we pre-process based on model
        # )
        sensor: TiledCamera | Camera | RayCasterCamera = env.scene.sensors[sensor_cfg.name]

        # obtain the input image
        images = sensor.data.output[data_type]

        images = images[:, 120:, :, :]

        # Apply domain randomization
        images = self._apply_domain_randomization(
            images,
            save_debug=save_augmentation_debug,
            step_counter=self._frame_counter
        )

        # import pdb
        # pdb.set_trace()

        # store the device of the image
        image_device = images.device
        # forward the images through the model
        features = self._inference_fn(self._model, images, **(inference_kwargs or {}))

        depth = env.scene.sensors[depth_cfg.name].data.output["distance_to_image_plane"]

        cam1 = env.scene.sensors["depth_camera"]
        K = cam1._data.intrinsic_matrices[0]
        fx = K[0][0]
        fy = K[1][1]
        cx = K[0][2]
        cy = K[1][2]

        depth_tensor = depth.squeeze(-1)

        self._frame_counter += 1

        # Generate point clouds on GPU in one batch
        batch_points_tensor = self.depth_to_pointcloud_batch_gpu(
            depth_tensor,
            fx,
            fy,
            cx,
            cy,
            num_points=1024,
            save_ply_debug=False,
            env_id=0,
            frame_counter=self._frame_counter,
            save_dir="debug_pointclouds",
            env=env
        )

        # env.point_cloud_cache = batch_points_tensor
        env.point_cloud_cache = batch_points_tensor.detach()
        env._pcd_cache_step = env.common_step_counter

        pts_input = batch_points_tensor.permute(0, 2, 1).contiguous()

        with torch.no_grad():
            depth_features_batch = self._point_encoder(pts_input)

        # # Change grid_size parameter
        # depth_features_batch = self.voxelize_pointcloud_batch(
        #     batch_points_tensor,
        #     grid_size=16,
        #     x_range=(0.15, 0.5),  # Adjust these to fit your workspace
        #     y_range=(-0.04, 0.04),
        #     z_range=(-0.0004, 0.085),
        #     save_debug=False,  # ← EVERY 10 STEPS
        #     # save_debug=(self._frame_counter % 1 == 0),  # ← EVERY 10 STEPS
        #     frame_counter=self._frame_counter
        # )

        # import pdb
        # pdb.set_trace()

        img_feat_norm = torch.nn.functional.normalize(features, p=2, dim=1)
        # import pdb
        # pdb.set_trace()
        pc_feat_norm = torch.nn.functional.normalize(depth_features_batch, p=2, dim=1)

        features = torch.cat((img_feat_norm,pc_feat_norm),dim=-1)

        return features.detach().to(image_device)

    """
    Helper functions.
    """


    def voxelize_pointcloud_batch(
        self,
        points: torch.Tensor,
        grid_size: int = 8,
        x_range: tuple = (-0.5, 0.5),
        y_range: tuple = (-0.5, 0.5),
        z_range: tuple = (0.0, 1.0),
        save_debug: bool = False,
        frame_counter: int = 0
    ) -> torch.Tensor:
        """
        Convert point cloud batch to voxel grid with occupancy.

        Args:
            points: [B, N, 3] point cloud in world coordinates
            grid_size: Resolution of voxel grid (e.g., 8 = 8x8x8 = 512 voxels)
            x_range: (min, max) bounds for X axis in meters
            y_range: (min, max) bounds for Y axis in meters
            z_range: (min, max) bounds for Z axis in meters
            save_debug: If True, save voxel visualization
            frame_counter: Current frame number for debug filenames

        Returns:
            voxels: [B, grid_size^3] flattened voxel grid
        """
        B, N, _ = points.shape
        device = points.device

        # Unpack bounds
        x_min, x_max = x_range
        y_min, y_max = y_range
        z_min, z_max = z_range

        # Normalize points to [0, grid_size] coordinates
        x_normalized = (points[:, :, 0] - x_min) / (x_max - x_min) * grid_size
        y_normalized = (points[:, :, 1] - y_min) / (y_max - y_min) * grid_size
        z_normalized = (points[:, :, 2] - z_min) / (z_max - z_min) * grid_size

        # Convert to integer indices and clamp
        x_idx = torch.clamp(x_normalized.long(), 0, grid_size - 1)
        y_idx = torch.clamp(y_normalized.long(), 0, grid_size - 1)
        z_idx = torch.clamp(z_normalized.long(), 0, grid_size - 1)

        # Create occupancy grid
        voxels = torch.zeros(B, grid_size, grid_size, grid_size, device=device)

        # Fill occupancy (batch-wise)
        for b in range(B):
            voxels[b, x_idx[b], y_idx[b], z_idx[b]] = 1.0

        # Debug visualization
        if save_debug:
            self._save_voxel_visualization(voxels[0], grid_size, frame_counter)
            occupied_count = (voxels[0] > 0).sum().item()
            print(f"Voxel occupancy: {occupied_count}/{grid_size**3} ({occupied_count/(grid_size**3)*100:.1f}%)")

        # Flatten spatial dimensions [B, D, H, W] -> [B, D*H*W]
        return voxels.flatten(start_dim=1)


    def _save_voxel_visualization(self, voxels: torch.Tensor, grid_size: int, frame_counter: int):
        """
        Save voxel grid visualization.

        Args:
            voxels: [D, H, W] single voxel grid
            grid_size: Grid resolution
            frame_counter: Frame number for filename
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np

        save_dir = "voxel_viz"
        os.makedirs(save_dir, exist_ok=True)

        voxels_np = voxels.cpu().numpy()
        occupied = np.argwhere(voxels_np > 0.1)

        if len(occupied) == 0:
            print(f"Warning: No occupied voxels to visualize")
            return

        fig = plt.figure(figsize=(15, 5))

        # 3D scatter plot
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.scatter(occupied[:, 0], occupied[:, 1], occupied[:, 2],
                c=voxels_np[occupied[:, 0], occupied[:, 1], occupied[:, 2]],
                cmap='viridis', marker='s', s=50, alpha=0.6)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title(f'3D Voxel Grid ({grid_size}³)')
        ax1.set_xlim([0, grid_size])
        ax1.set_ylim([0, grid_size])
        ax1.set_zlim([0, grid_size])

        # Top-down view
        ax2 = fig.add_subplot(132)
        z_slice = voxels_np[:, :, grid_size // 2]
        im2 = ax2.imshow(z_slice.T, origin='lower', cmap='viridis',
                        interpolation='nearest', vmin=0, vmax=1)
        ax2.set_title(f'Top View (Z={grid_size//2})')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        plt.colorbar(im2, ax=ax2)

        # Side view
        ax3 = fig.add_subplot(133)
        y_slice = voxels_np[:, grid_size // 2, :]
        im3 = ax3.imshow(y_slice.T, origin='lower', cmap='viridis',
                        interpolation='nearest', vmin=0, vmax=1)
        ax3.set_title(f'Side View (Y={grid_size//2})')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Z')
        plt.colorbar(im3, ax=ax3)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"voxels_{frame_counter:06d}.png")
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        print(f"Saved voxel visualization: {save_path}")




    def _prepare_theia_transformer_model(self, model_name: str, model_device: str) -> dict:
        """Prepare the Theia transformer model for inference.

        Args:
            model_name: The name of the Theia transformer model to prepare.
            model_device: The device to store and infer the model on.

        Returns:
            A dictionary containing the model and inference functions.
        """
        from transformers import AutoModel

        def _load_model() -> torch.nn.Module:
            """Load the Theia transformer model."""
            model = AutoModel.from_pretrained(f"theaiinstitute/{model_name}", trust_remote_code=True).eval()
            return model.to(model_device)

        def _inference(model, images: torch.Tensor) -> torch.Tensor:
            """Inference the Theia transformer model.

            Args:
                model: The Theia transformer model.
                images: The preprocessed image tensor. Shape is (num_envs, height, width, channel).

            Returns:
                The extracted features tensor. Shape is (num_envs, feature_dim).
            """
            # Move the image to the model device
            image_proc = images.to(model_device)
            # permute the image to (num_envs, channel, height, width)
            image_proc = image_proc.permute(0, 3, 1, 2).float() / 255.0
            # Normalize the image
            mean = torch.tensor([0.485, 0.456, 0.406], device=model_device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=model_device).view(1, 3, 1, 1)
            image_proc = (image_proc - mean) / std

            # Taken from Transformers; inference converted to be GPU only
            features = model.backbone.model(pixel_values=image_proc, interpolate_pos_encoding=True)
            return features.last_hidden_state[:, 1:]

        # return the model, preprocess and inference functions
        return {"model": _load_model, "inference": _inference}


    def _prepare_squeezenet_model(self, model_name: str, model_device: str) -> dict:
        """Prepare the SqueezeNet model for inference.

        Args:
            model_name: The name of the SqueezeNet model to prepare.
            model_device: The device to store and infer the model on.

        Returns:
            A dictionary containing the model and inference functions.
        """
        from torchvision import models

        def _load_model() -> torch.nn.Module:
            """Load the SqueezeNet model."""
            # map the model name to the weights
            squeezenet_weights = {
                "squeezenet1_0": "SqueezeNet1_0_Weights.IMAGENET1K_V1",
                "squeezenet1_1": "SqueezeNet1_1_Weights.IMAGENET1K_V1",
            }

            # load the model
            model = getattr(models, model_name)(weights=squeezenet_weights[model_name]).eval()
            # Remove the final classifier to get features
            model = torch.nn.Sequential(*list(model.children())[:-1])
            return model.to(model_device)

        def _inference(model, images: torch.Tensor) -> torch.Tensor:
            """Inference the SqueezeNet model.

            Args:
                model: The SqueezeNet model.
                images: The preprocessed image tensor. Shape is (num_envs, channel, height, width).

            Returns:
                The extracted features tensor. Shape is (num_envs, feature_dim).
            """
            # move the image to the model device
            image_proc = images.to(model_device)
            # permute the image to (num_envs, channel, height, width)
            image_proc = image_proc.permute(0, 3, 1, 2).float() / 255.0
            # normalize the image
            mean = torch.tensor([0.485, 0.456, 0.406], device=model_device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=model_device).view(1, 3, 1, 1)
            image_proc = (image_proc - mean) / std
            # forward the image through the model
            with torch.no_grad():
                feats = model(image_proc)          # [N, 512, H, W] for SqueezeNet
                feats = torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1))  # Global average pooling
                feats = feats.view(feats.size(0), -1)  # [N, 512]
            return feats

        # return the model, preprocess and inference functions
        return {"model": _load_model, "inference": _inference}



    def _prepare_pointnet_model(self) :

        import torch.nn as nn

        experiment_dir = '/home/roborock/IsaacLab'
        ckpt_path = f"{experiment_dir}/best_model.pth"

        # ✅ 模型输入通道：原模型是 normal_channel=True（6 通道）
        classifier = PointNet2ClsMsg(num_class=40, normal_channel=False).cuda()

        # ✅ 加载 checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cuda', weights_only=False)

        # 拿出权重字典
        state_dict = checkpoint['model_state_dict']

        # ✅ 加载修正后的权重
        classifier.load_state_dict(state_dict, strict=False)
        # print("[INFO] Missing keys:", missing)
        # print("[INFO] Unexpected keys:", unexpected)

        classifier.eval()

        # ✅ 仅保留特征提取部分（encoder）- Only SA1 layer
        class PointNet2Encoder_SA1Only(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.normal_channel = False  # 我们只输入 XYZ
                self.sa1 = base_model.sa1

            def forward(self, xyz):
                B, _, _ = xyz.shape
                norm = None
                l1_xyz, l1_points = self.sa1(xyz, norm)
                # Only use SA1 output - l1_points shape is typically [B, C]
                # For PointNet2, after SA1 we need to get global features
                features = torch.max(l1_points, dim=2)[0]  # Global max pooling
                return features

        self._point_encoder = PointNet2Encoder_SA1Only(classifier).cuda().eval()



"""
Actions.
"""


def randomize_pointcloud_batch_torch(
    pts,
    dropout_rate=0.02,
    outlier_ratio=0.015,
    outlier_max_offset=0.04,
    surface_jitter=0.0005
):
    B, N, _ = pts.shape
    device = pts.device

    pts = pts.clone()

    # 1. Dropout
    # dropout_mask = torch.rand(B, N, device=device) > dropout_rate
    # pts = pts * dropout_mask.unsqueeze(-1)

    # 2. Outliers（沿 X 轴正方向）
    num_outliers = max(1, int(outlier_ratio * N))

    for b in range(B):
        idx = torch.randperm(N, device=device)[:num_outliers].long()  # 确保是 long
        offset = torch.rand(num_outliers, device=device, dtype=pts.dtype) * outlier_max_offset
        pts[b].index_add_(0, idx, torch.stack([offset, torch.zeros_like(offset), torch.zeros_like(offset)], dim=1))


    jitter = torch.randn_like(pts) * surface_jitter
    pts = pts + jitter

    return pts


def last_action(env: ManagerBasedEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment.

    The name of the action term for which the action is required. If None, the
    entire action tensor is returned.
    """
    if action_name is None:
        # with open('output_5142.txt', 'a') as f:
        #     f.write(f"obs5 m: {(env.action_manager.action).mean().item()}\n")
        #     f.write(f"obs5 s: {(env.action_manager.action).std().item()}\n")
        # print("obs5 m:",(env.action_manager.action).mean().item())
        # print("obs5 s:",(env.action_manager.action).std().item())
        return env.action_manager.action
    else:
        # with open('output_5142.txt', 'a') as f:
        #     f.write(f"obs5 m: {(env.action_manager.get_term(action_name).raw_actions).mean().item()}\n")
        #     f.write(f"obs5 s: {(env.action_manager.get_term(action_name).raw_actions).std().item()}\n")
        # print("obs5 m:",(env.action_manager.get_term(action_name).raw_actions).mean().item())
        # print("obs5 s:",(env.action_manager.get_term(action_name).raw_actions).std().item())
        return env.action_manager.get_term(action_name).raw_actions


"""
Commands.
"""


def generated_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""

    # with open('output_5142.txt', 'a') as f:
    #     f.write(f"obs4 m: {(env.command_manager.get_command(command_name)).mean().item()}\n")
    #     f.write(f"obs4 s: {(env.command_manager.get_command(command_name)).std().item()}\n")
    # print("obs4 m:",(env.command_manager.get_command(command_name)).mean().item())
    # print("obs4 s:",(env.command_manager.get_command(command_name)).std().item())
    return env.command_manager.get_command(command_name)
