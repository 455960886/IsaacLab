# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class PositiveM5GatedActorCriticCfg(RslRlPpoActorCriticCfg):
    """PositiveM5ActorCritic 的额外配置。"""

    # concat = 保持原来的方式：ResNet 特征和 PointNet 特征直接拼接。
    # gated = 启用门控融合：网络自动学习每一维更相信图像还是点云。
    fusion_mode: str = "gated"

    # 当前 observation 的 image term 是 1536 维：
    # 前 512 维来自 ResNet，后 1024 维来自 PointNet，最后 5 维是 joint_pos。
    image_feat_dim: int = 512
    pcd_feat_dim: int = 1024
    proprio_dim: int = 5
    fusion_hidden_dim: int = 256


@configclass
class LiftCubePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 50000
    save_interval = 10
    experiment_name = "coarse_arm_lift"
    empirical_normalization = True
    # empirical_normalization = True

    policy = PositiveM5GatedActorCriticCfg(
        class_name="PositiveM5ActorCritic",
        init_noise_std=0.2,
        actor_hidden_dims=[512, 256, 128, 64],
        critic_hidden_dims=[512, 256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.015,
        num_learning_epochs=5,
        num_mini_batches=12,
        learning_rate=3.0e-4,
        # learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.97,
        desired_kl=0.01,
        max_grad_norm=0.5,
        # max_grad_norm=0.5,
    )
