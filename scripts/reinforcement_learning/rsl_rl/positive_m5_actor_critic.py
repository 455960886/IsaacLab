# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import torch
import torch.nn as nn

from rsl_rl.modules.actor_critic import ActorCritic


M345_HALF_RANGES = (math.radians(45.0), math.radians(72.5), math.pi / 2.0)


class SymmetricM345ActionNormalizer(nn.Module):
    """Map raw M3/M4/M5 actor outputs to symmetric joint-position offsets."""

    def __init__(self, gain: float = 0.5):
        super().__init__()
        self.gain = float(gain)
        self.register_buffer(
            "_half_ranges",
            torch.tensor(M345_HALF_RANGES, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        normalized_actions = actions.clone()
        half_ranges = self._half_ranges.to(device=actions.device, dtype=actions.dtype)
        normalized_actions[..., :3] = torch.tanh(actions[..., :3] * self.gain) * half_ranges
        # mean_M3 = tanh(z_M3 * 0.5) * 0.78540
        # mean_M4 = tanh(z_M4 * 0.5) * 1.26536
        # mean_M5 = tanh(z_M5 * 0.5) * 1.57080
        return normalized_actions


class NormalizedM345ActorCritic(ActorCritic):
    """ActorCritic with symmetric tanh-normalized M3/M4/M5 mean action offsets."""

    def __init__(self, *args, m345_action_gain: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.actor = nn.Sequential(*self.actor, SymmetricM345ActionNormalizer(gain=m345_action_gain))


class PositiveM5ActorCritic(NormalizedM345ActorCritic):
    """Legacy name kept for old configs; now normalizes M3/M4/M5 symmetrically."""

    pass
