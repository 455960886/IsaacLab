# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom termination terms for the shoe-navigation task."""

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def robot_object_contact(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate when any (sensor body, filter target) contact force magnitude
    exceeds ``threshold``.

    Unlike :func:`isaaclab.envs.mdp.illegal_contact`, this uses
    ``force_matrix_w`` so contacts with the *filtered* prim only (e.g. the shoe)
    are considered — wheel-vs-ground etc. don't trigger termination.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    fmat = sensor.data.force_matrix_w  # (N_envs, N_bodies, N_filters, 3)
    if fmat is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # max contact force magnitude across all (body, filter) pairs per env
    forces = torch.norm(fmat, dim=-1)
    max_force = forces.flatten(start_dim=1).amax(dim=-1)
    return max_force > threshold
