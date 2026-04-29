# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic import ActorCritic


class _PositiveM5ActorWrapper(nn.Module):
    """Wrap the base actor so that the M5 output is always in [0, pi]."""

    def __init__(self, base_actor: nn.Module, m5_action_index: int = 2, m5_max: float = math.pi / 2):
        super().__init__()
        self.base_actor = base_actor
        self.m5_action_index = m5_action_index
        self.m5_max = m5_max

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        actions = self.base_actor(observations)
        actions = actions.clone()
        actions[..., self.m5_action_index] = torch.sigmoid(actions[..., self.m5_action_index]) * self.m5_max
        return actions


class _GatedVisualFusionActorWrapper(nn.Module):
    """Fuse ResNet and PointNet features before sending observations to the actor."""

    def __init__(
        self,
        base_actor: nn.Module,
        image_feat_dim: int = 512,
        pcd_feat_dim: int = 1024,
        proprio_dim: int = 5,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.base_actor = base_actor
        self.image_feat_dim = image_feat_dim
        self.pcd_feat_dim = pcd_feat_dim
        self.proprio_dim = proprio_dim
        self.expected_obs_dim = image_feat_dim + pcd_feat_dim + proprio_dim

        # 中文说明：
        # ResNet 输出是 512 维，PointNet 输出是 1024 维，二者维度不同，不能直接相加。
        # 这里先把图像特征投影到 1024 维，让它能和点云特征做逐维融合。
        self.image_to_pcd = nn.Sequential(
            nn.Linear(image_feat_dim, pcd_feat_dim),
            nn.LayerNorm(pcd_feat_dim),
            nn.Tanh(),
        )

        # 中文说明：
        # gate 的每一维都在 [0, 1]。
        # gate 越大，这一维越相信图像特征；gate 越小，这一维越相信点云特征。
        self.gate_net = nn.Sequential(
            nn.Linear(image_feat_dim + pcd_feat_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, pcd_feat_dim),
        )
        # 初始时让 gate 偏向点云，训练会逐渐学到什么时候该多用图像。
        nn.init.constant_(self.gate_net[-1].bias, -4.0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.shape[-1] != self.expected_obs_dim:
            raise ValueError(
                f"Gated fusion expects obs dim {self.expected_obs_dim}, got {observations.shape[-1]}."
            )

        image_feat = observations[..., : self.image_feat_dim]
        pcd_start = self.image_feat_dim
        pcd_end = self.image_feat_dim + self.pcd_feat_dim
        pcd_feat = observations[..., pcd_start:pcd_end]
        proprio_feat = observations[..., pcd_end:]

        image_as_pcd = self.image_to_pcd(image_feat)
        gate = torch.sigmoid(self.gate_net(torch.cat((image_feat, pcd_feat), dim=-1)))

        # 中文说明：
        # 这里不是简单拼接，而是让网络自己决定每一维更相信图像还是点云。
        # fused_pcd 仍然保持 1024 维，所以后面的 actor 输入维度不变。
        fused_pcd = gate * image_as_pcd + (1.0 - gate) * pcd_feat
        fused_observations = torch.cat((image_feat, fused_pcd, proprio_feat), dim=-1)
        return self.base_actor(fused_observations)


class PositiveM5ActorCritic(ActorCritic):
    """ActorCritic variant whose M5 action is exported and inferred as an absolute angle in [0, pi]."""

    def __init__(
        self,
        *args,
        m5_action_index: int = 2,
        m5_max: float = math.pi,
        fusion_mode: str = "concat",
        image_feat_dim: int = 512,
        pcd_feat_dim: int = 1024,
        proprio_dim: int = 5,
        fusion_hidden_dim: int = 256,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        actor = self.actor

        if fusion_mode == "gated":
            actor = _GatedVisualFusionActorWrapper(
                actor,
                image_feat_dim=image_feat_dim,
                pcd_feat_dim=pcd_feat_dim,
                proprio_dim=proprio_dim,
                hidden_dim=fusion_hidden_dim,
            )
        elif fusion_mode != "concat":
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}. Expected 'concat' or 'gated'.")

        self.actor = _PositiveM5ActorWrapper(actor, m5_action_index=m5_action_index, m5_max=m5_max)
