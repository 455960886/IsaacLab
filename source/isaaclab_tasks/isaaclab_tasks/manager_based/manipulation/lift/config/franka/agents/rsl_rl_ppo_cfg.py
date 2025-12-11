# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class LiftCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 10
    experiment_name = "coarse_arm_lift"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.2,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",

        # ========= 新增：视觉 encoder 配置 =========
        # 打开我们在 ActorCritic 里写的 ResNet + PointNet2 encoder
        use_visual_encoder=True,

        # 现在 obs 里只有视觉（图像 + 点云），没有关节等低维状态，所以先设 0
        state_dim=0,

        # 和 image_features 打印出来的图像尺寸保持一致：
        img_channels=3,
        img_height=480,
        img_width=640,

        # 和 depth_to_pointcloud_batch_gpu 里 num_points 一致：
        pcd_points=1024,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.000,
        num_learning_epochs=4,
        num_mini_batches=6,
        learning_rate=1.0e-3,
        # learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.98,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
