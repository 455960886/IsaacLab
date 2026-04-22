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


class PositiveM5ActorCritic(ActorCritic):
    """ActorCritic variant whose M5 action is exported and inferred as an absolute angle in [0, pi]."""

    def __init__(self, *args, m5_action_index: int = 2, m5_max: float = math.pi, **kwargs):
        super().__init__(*args, **kwargs)
        self.actor = _PositiveM5ActorWrapper(self.actor, m5_action_index=m5_action_index, m5_max=m5_max)
