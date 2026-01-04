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
import cv2
import os
import time
from isaaclab.pointnet.models.pointnet_utils import PointNetEncoder, feature_transform_reguliarzer
import importlib
from isaaclab.pointnet.log.classification.pointnet2_ssg_wo_normals.pointnet2_cls_ssg import get_model as PointNet2ClsMsg
import open3d as o3d
import torch.nn.functional as F
import torchvision

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
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("gripper_camera"),
    data_type: str = "rgb",
    convert_perspective_to_orthogonal: bool = False,
    normalize: bool = True,
    depth_cfg:SceneEntityCfg = SceneEntityCfg("depth_camera")
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
    # images = cv2.imread(, cv2.IMREAD_COLOR)
    # image_path = "/home/roborock/下载/2.jpg"
    # img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
  
    # img_tensor = torch.from_numpy(img_rgb).float()  # shape: (H, W, 3), float32
 
    # img_tensor = img_tensor / 255.0
    depth = env.scene.sensors[depth_cfg.name].data.output["distance_to_image_plane"]
    # img_tensor = img_tensor.permute(2, 0, 1)

    # # 6. Add batch dimension -> (1, C, H, W)
    # images = img_tensor.unsqueeze(0)
    
    # depth image conversion
    # images = images[:, :, images.shape[2] // 2:, :]
    
#     obs_image = images.clone().detach().float().squeeze(0).cpu().numpy()
#   # Convert to tensor and float type
#     obs_bgr = cv2.cvtColor(obs_image, cv2.COLOR_RGB2BGR)
#     cv2.imwrite(f"/home/roborock/下载/obs.png", obs_bgr)
#     # with open('output_formres8.txt', 'a') as f:
    #     f.write(f"observation_{time1}.png\n")
    # print(f"./IMAGES9/observation_{time1}.png")

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
            # obs_image = images.clone().detach().float().squeeze(0).cpu().numpy()
            # obs_bgr = cv2.cvtColor(obs_image, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(f"/home/roborock/下载/obs.png", obs_bgr)
            images = images.float()
            # print(images.shape)
            images = images  / 255.0
            images = images.permute(0, 3, 1, 2)
            target_size = (300, 400)  # (height, width)
            images = F.interpolate(images, size=target_size, mode="bilinear", align_corners=False)
        # # resize 到目标分辨率 (150, 200)
        #     

        #     # 如果想保持原来的 (B, H, W, C) 形状
        #     
            mean_tensor = torch.mean(images, dim=(1, 2), keepdim=True)
            images = images -0.5
            images = images.permute(0, 2, 3, 1)
            # print(f"1 {images.shape}")
            
            # print('minus mean')
            # images = images.float()

            # obs_np2 = images.squeeze(0).cpu().numpy() 
            # # act_np = actions.cpu().numpy() 
            # # os.makedirs(act_log_dir, exist_ok=True)
            # # np.save(os.path.join(act_log_dir, f"act_step_{t}.npy"), act_np)
            # if obs_np2.dtype == np.float32 or obs_np2.max() <= 1.0:
            #     obs_np2 = (obs_np2 * 255).astype(np.uint8)

            # # RGB 转 BGR 再保存
            # obs_bgr2 = cv2.cvtColor(obs_np2, cv2.COLOR_RGB2BGR)
            # os.makedirs("IMAGES6", exist_ok=True)
            # cv2.imwrite(f"./IMAGES6/observation_{time.time()}.png", obs_bgr2)
            
            pass
        elif "distance_to" in data_type or "depth" in data_type:
            images[images == float("inf")] = 0
    images = torch.cat((images,depth),dim=-1)
    # print(f"输入图像 {images}")
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
    model zoo configurations include the models from Theia :cite:`shang2024theia` and ResNet :cite:`he2016deep`.
    These models are loaded from `Hugging-Face transformers <https://huggingface.co/docs/transformers/index>`_ and
    `PyTorch torchvision <https://pytorch.org/vision/stable/models.html>`_ respectively.

    Args:
        sensor_cfg: The sensor configuration to poll. Defaults to SceneEntityCfg("tiled_camera").
        data_type: The sensor data type. Defaults to "rgb".
        convert_perspective_to_orthogonal: Whether to orthogonalize perspective depth images.
            This is used only when the data type is "distance_to_camera". Defaults to False.
        model_zoo_cfg: A user-defined dictionary that maps different model names to their respective configurations.
            Defaults to None. If None, the default model zoo configurations are used.
        model_name: The name of the model to use for inference. Defaults to "resnet18".
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
        self.model_name: str = cfg.params.get("model_name", "resnet18")  # type: ignore
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
        # List of ResNet models - These are configured through `_prepare_resnet_model` function
        default_resnet_models = ["resnet18", "resnet34", "resnet50", "resnet101"]

        # Check if model name is specified in the model zoo configuration
        if self.model_zoo_cfg is not None and self.model_name not in self.model_zoo_cfg:
            raise ValueError(
                f"Model name '{self.model_name}' not found in the provided model zoo configuration."
                " Please add the model to the model zoo configuration or use a different model name."
                f" Available models in the provided list: {list(self.model_zoo_cfg.keys())}."
                "\nHint: If you want to use a default model, consider using one of the following models:"
                f" {default_theia_models + default_resnet_models}. In this case, you can remove the"
                " 'model_zoo_cfg' parameter from the observation term configuration."
            )
        if self.model_zoo_cfg is None:
            if self.model_name in default_theia_models:
                model_config = self._prepare_theia_transformer_model(self.model_name, self.model_device)
            elif self.model_name in default_resnet_models:
                model_config = self._prepare_resnet_model(self.model_name, self.model_device)
            else:
                raise ValueError(
                    f"Model name '{self.model_name}' not found in the default model zoo configuration."
                    f" Available models: {default_theia_models + default_resnet_models}."
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
        

        self.fx, self.fy = 117.78, 124.95
        self.cx, self.cy = 200.0, 150.0

        self._frame_counter = 0


    def reset(self, env_ids: torch.Tensor | None = None):
        # reset the model if a reset function is provided
        # this might be useful when the model has a state that needs to be reset
        # for example: video transformers
        if self._reset_fn is not None:
            self._reset_fn(self._model, env_ids)
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
    # GPU-accelerated version for batch processing
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
        kernel_size = int(torch.randint(3, 8, (1,)).item())
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd kernel size
        sigma = torch.rand(1).item() * 1.5 + 0.5  # Random sigma between 0.5-2.0
        images = torchvision.transforms.functional.gaussian_blur(
            images, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma]
        )

        # Random Gaussian noise
        noise_std = torch.rand(1).item() * 0.03  # Random std between 0-0.03
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
    def depth_to_pointcloud_batch_gpu(self, depth_batch, fx, fy, cx, cy, num_points=1024, 
                                      save_ply_debug=False, env_id=0, frame_counter=None, save_dir="debug_pointclouds"):
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
        # import pdb
        # pdb.set_trace()
        points_flat = points.reshape(B, H * W, 3)
        rotated_points = torch.matmul(points_flat, R.T)
        translation = torch.tensor([0.1654, 0.0, 0.054], device=device)
        trans_points = rotated_points +translation
        # Save Stage 1: After rotation
        if save_ply_debug:
            points_trans = trans_points[env_id].cpu().numpy()
            save_ply(points_trans, "1_rotated")

        # Apply distance filtering
        # mask1 = rotated_points[:, :, 2] < 0.21
        # mask2 = rotated_points[:, :, 1] > -0.0628
        # mask3 = rotated_points[:, :, 1] < 0.0428
        rand_thresh = np.random.uniform(0.0, 0.01)

        mask2 = trans_points[:,:, 0] <=0.42
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
        result = randomize_pointcloud_batch_torch(result,dropout_rate=0.02,outlier_ratio=0.02,outlier_max_offset=0.08,surface_jitter=0.001)
        if save_ply_debug:
            points_final = result[env_id].cpu().numpy()
            save_ply(points_final, "4_random")
        return result


    def __call__(
        self,
        env: ManagerBasedEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("gripper_camera"),
        depth_cfg: SceneEntityCfg = SceneEntityCfg("depth_camera"),
        data_type: str = "rgb",
        convert_perspective_to_orthogonal: bool = False,
        model_zoo_cfg: dict | None = None,
        model_name: str = "resnet18",
        model_device: str | None = None,
        inference_kwargs: dict | None = None,
  
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
        # store the device of the image
        images = self._apply_domain_randomization(
            images, 
            save_debug=False,
            step_counter=self._frame_counter
        )
        image_device = images.device

        # forward the images through the model
        img_features = self._inference_fn(self._model, images, **(inference_kwargs or {}))

        depth = env.scene.sensors[depth_cfg.name].data.output["distance_to_image_plane"]
        cam1 = env.scene.sensors["depth_camera"]
        K = cam1._data.intrinsic_matrices[0]
        fx = K[0][0]
        fy = K[1][1]
        cx = K[0][2]
        cy = K[1][2]
        depth_tensor = depth.squeeze(-1)  # Keep on GPU, no .cpu().numpy()

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
            save_dir="debug_pointclouds"
        )
        env.point_cloud_cache = batch_points_tensor.detach()
        env._pcd_cache_step = env.common_step_counter
        
        pts_input = batch_points_tensor.permute(0, 2, 1).contiguous()
    
        with torch.no_grad():
            depth_features_batch = self._point_encoder(pts_input)
   


        img_feat_norm = torch.nn.functional.normalize(img_features, p=2, dim=1)
        
        pc_feat_norm = torch.nn.functional.normalize(depth_features_batch, p=2, dim=1)
        
        # features = torch.cat((img_feat_norm,pc_feat_norm),dim=-1) # 有normnalize
        features = torch.cat((img_features, depth_features_batch), dim=-1) # 无featrue normalize
        
        return features.detach().to(image_device)

    """
    Helper functions.
    """

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

    def _prepare_resnet_model(self, model_name: str, model_device: str) -> dict:
        """Prepare the ResNet model for inference.

        Args:
            model_name: The name of the ResNet model to prepare.
            model_device: The device to store and infer the model on.

        Returns:
            A dictionary containing the model and inference functions.
        """
        from torchvision import models

        def _load_model() -> torch.nn.Module:
            """Load the ResNet model."""
            # map the model name to the weights
            resnet_weights = {
                "resnet18": "ResNet18_Weights.IMAGENET1K_V1",
                "resnet34": "ResNet34_Weights.IMAGENET1K_V1",
                "resnet50": "ResNet50_Weights.IMAGENET1K_V1",
                "resnet101": "ResNet101_Weights.IMAGENET1K_V1",
            }

            # load the model
            model = getattr(models, model_name)(weights=resnet_weights[model_name]).eval()
            model = torch.nn.Sequential(*list(model.children())[:-1])
            return model.to(model_device)

        def _inference(model, images: torch.Tensor) -> torch.Tensor:
            """Inference the ResNet model.

            Args:
                model: The ResNet model.
                images: The preprocessed image tensor. Shape is (num_envs, channel, height, width).

            Returns:
                The extracted features tensor. Shape is (num_envs, feature_dim).
            """
    #         def load_image(img_path, input_size=(224, 224)):
    # # 读取图片
    #             img = cv2.imread(img_path)
    #             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #             # resize 到模型的输入尺寸
    #             img = cv2.resize(img, input_size)

    #             # HWC -> CHW
    #             img = img.transpose(2, 0, 1).astype(np.float32) / 255.0  # 归一化到 [0,1]

    #             # ResNet ImageNet 预处理
    #             mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
    #             std  = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
    #             img = (img - mean) / std

    #             # 增加 batch 维度 (1,3,H,W)
    #             img = np.expand_dims(img, axis=0).astype(np.float32)

    #             return img
            # img = load_image("/home/roborock/docker_images_v1.8.x/docker_bushu/1.png", input_size=(224, 224))
            # images = torch.from_numpy(img)
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
                feats = model(image_proc)          # [N, 512, 1, 1]
                feats = feats.view(feats.size(0), -1)  # [N, 512]
            return feats

        # return the model, preprocess and inference functions
        return {"model": _load_model, "inference": _inference}
    
    def _prepare_pointnet_model(self) :
        import torch.nn as nn
        # experiment_dir = '/home/roborock/IsaacLab' 
        # classifier = MODEL.get_model(13).cuda()
        # checkpoint = torch.load(str(experiment_dir) + '/best_model.pth')
        # classifier.load_state_dict(checkpoint['model_state_dict'])
        # classifier = classifier.eval()

        # self._point_encoder = classifier.feat
        # self._point_encoder.eval()
        # self._point_encoder.cuda()

        experiment_dir = '/home/roborock/data/private/shengmei/IsaacLab'
        ckpt_path = f"{experiment_dir}/best_model.pth"

        # ✅ 模型输入通道：原模型是 normal_channel=True（6 通道）
        classifier = PointNet2ClsMsg(num_class=40, normal_channel=False).cuda()  

        # ✅ 加载 checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cuda', weights_only=False)

        # 拿出权重字典
        state_dict = checkpoint['model_state_dict']

        # # ✅ 动态修正输入通道权重 mismatch（从 6 -> 3）
        # for key in list(state_dict.keys()):
        #     if 'sa1' in key and 'weight' in key and state_dict[key].dim() == 4:
        #         if state_dict[key].shape[1] == 6:
        #             print(f"[INFO] Trimming {key} from 6→3 input channels.")
        #             state_dict[key] = state_dict[key][:, :3, :, :]  # 截取前3个通道 (XYZ)

        # # ✅ 忽略分类头不匹配部分
        # ignore_keys = ['fc3.weight', 'fc3.bias']
        # for k in ignore_keys:
        #     if k in state_dict:
        #         print(f"[INFO] Removing {k} from checkpoint.")
        #         del state_dict[k]

        # ✅ 加载修正后的权重
        classifier.load_state_dict(state_dict, strict=False)
        # print("[INFO] Missing keys:", missing)
        # print("[INFO] Unexpected keys:", unexpected)

        classifier.eval()

        # ✅ 仅保留特征提取部分（encoder）
        class PointNet2Encoder(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.normal_channel = True  # 我们只输入 XYZ
                self.sa1 = base_model.sa1
                self.sa2 = base_model.sa2
                self.sa3 = base_model.sa3

            def forward(self, xyz):
                B, _, _ = xyz.shape
                norm = None
                l1_xyz, l1_points = self.sa1(xyz, norm)
                l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
                l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
                features = l3_points.view(B, 1024)
                return features

        self._point_encoder = PointNet2Encoder(classifier).cuda().eval()
        

"""
Actions.
"""


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


def randomize_pointcloud_batch_torch(
    pts, 
    dropout_rate=0.02, 
    outlier_ratio=0.015, 
    outlier_max_offset=0.05, 
    surface_jitter=0.001
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
   


def generated_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""

    # with open('output_5142.txt', 'a') as f:
    #     f.write(f"obs4 m: {(env.command_manager.get_command(command_name)).mean().item()}\n")
    #     f.write(f"obs4 s: {(env.command_manager.get_command(command_name)).std().item()}\n")
    # print("obs4 m:",(env.command_manager.get_command(command_name)).mean().item())
    # print("obs4 s:",(env.command_manager.get_command(command_name)).std().item())
    return env.command_manager.get_command(command_name)
