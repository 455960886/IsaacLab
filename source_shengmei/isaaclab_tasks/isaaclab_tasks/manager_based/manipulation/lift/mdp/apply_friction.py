# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Function to apply high-friction physics materials to gripper and objects."""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedEnv

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def apply_high_friction_materials(env: ManagerBasedRLEnv, env_ids: list[int] | None = None):
    """Apply high-friction physics materials to gripper fingers and objects.

    This function creates and binds physics materials with high friction coefficients
    to improve grasping stability. It should be called once at startup.

    Args:
        env: The environment instance.
        env_ids: Not used, but required for event function signature.
    """
    # Only run once (check if material already exists)
    stage = env.sim.stage
    material_path = "/World/Physics/HighFrictionMaterial"

    # Create high-friction material if it doesn't exist
    if not stage.GetPrimAtPath(material_path).IsValid():
        high_friction_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=5.0,
            dynamic_friction=4.0,
            restitution=0.0,
            friction_combine_mode="max",  # Use maximum friction when objects contact
        )
        sim_utils.spawn_rigid_body_material(material_path, high_friction_cfg)
        print(f"[INFO] Created high-friction material at {material_path}")

    # Apply material to gripper fingers (all environments)
    for env_id in range(env.num_envs):
        env_prim_path = f"/World/envs/env_{env_id}"

        # Gripper finger links (adjust these paths to match your robot)
        gripper_paths = [
            f"{env_prim_path}/Robot/M6_1_leftfinger_link",
            f"{env_prim_path}/Robot/M6_2_rightfinger_link",
        ]

        for gripper_path in gripper_paths:
            if stage.GetPrimAtPath(gripper_path).IsValid():
                sim_utils.bind_physics_material(gripper_path, material_path, stage, stronger_than_descendants=True)

        # Apply to objects in the object pool
        # Adjust object names based on your configuration
        object_names = ["slippers"]  # Add other object names as needed

        for obj_name in object_names:
            obj_path = f"{env_prim_path}/{obj_name}"
            if stage.GetPrimAtPath(obj_path).IsValid():
                sim_utils.bind_physics_material(obj_path, material_path, stage, stronger_than_descendants=True)

    print("[INFO] Applied high-friction materials to gripper and objects")
