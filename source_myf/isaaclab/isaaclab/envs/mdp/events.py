# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import math
import torch
import numpy as np
from typing import TYPE_CHECKING, Literal

import carb
import omni.physics.tensors.impl.api as physx
import omni.usd
from isaacsim.core.utils.extensions import enable_extension
from pxr import Gf, Sdf, UsdGeom, Vt, UsdLux, UsdShade

import isaaclab.sim as sim_utils
from pxr import Usd
import isaaclab.utils.math as math_utils
import omni.usd
from isaaclab.utils.math import sample_uniform
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import torch
import random

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def randomize_multiple_sphere_lights(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    num_lights: int,
) -> None:
    """Randomize multiple SphereLight properties (pos, scale, temp, intensity, color)."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    stage = omni.usd.get_context().get_stage()
    for env_id in env_ids.tolist():
        for i in range(num_lights):
            light_path = f"/World/envs/env_{env_id}/SphereLight_{i}"
            light_prim = stage.GetPrimAtPath(light_path)
            if not light_prim.IsValid():
                alt_path = f"/World/envs/env_{env_id:04d}/SphereLight_{i}"
                light_prim = stage.GetPrimAtPath(alt_path)
            if light_prim.IsValid():
                # 随机位置
                pos = (
                    np.random.uniform(-0.5,0.2 ),
                    np.random.uniform(-0.5, 0.5),
                    np.random.uniform(0.5,1.5),
                )
                light_prim.GetAttribute("xformOp:translate").Set(pos)
                # 随机缩放
                scale = np.random.uniform(0.5, 3.2)
                light_prim.GetAttribute("xformOp:scale").Set((scale, scale, scale))
                # 随机色温
                temp = float(np.random.normal(2500, 8000))
                light_prim.GetAttribute("inputs:colorTemperature").Set(temp)
                # 随机强度
                intensity = float(np.random.normal(1500, 8000))
                light_prim.GetAttribute("inputs:intensity").Set(intensity)
                # 随机颜色
                color = (
                    np.random.uniform(0.7, 0.99),
                    np.random.uniform(0.7, 0.99),
                    np.random.uniform(0.7, 0.99),
                )
                light_prim.GetAttribute("inputs:color").Set(color)
            else:
                print(f"[Warning] SphereLight_{i} not found for env {env_id}")
def load_texture_files_from_txt(txt_path: str) -> list[str]:
    texture_files = []
    with open(txt_path, "r", encoding="utf-8") as f:  # 指定utf-8编码
        for line in f:
            line = line.strip()
            if line:
                texture_files.append(line)
    return texture_files

def randomize_floor_texture(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    texture_txt_path:str,
) -> None:
    """Randomize floor texture on reset."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    stage = omni.usd.get_context().get_stage()

    for env_id in env_ids.tolist():
        
        floor_path = f"/World/envs/env_{env_id}/FloorwithPanels/simple_room/Towel_Room01_floor_bottom_218/Towel_Room01_floor_bottom"
        floor_prim = stage.GetPrimAtPath(floor_path)
        if not floor_prim.IsValid():
            print(f"[Warning] floor prim not found at {floor_path}")
            continue
        if texture_txt_path is None:
            print("[Error] texture_txt_path is None")
            return

        texture_files = load_texture_files_from_txt(texture_txt_path)
        # 找到绑定的材质
        material, _ = UsdShade.MaterialBindingAPI(floor_prim).ComputeBoundMaterial()
        if not material:
            print(f"[Warning] no material bound to {floor_path}")
            continue

        # 获取 shader
        shader = UsdShade.Shader(omni.usd.get_shader_from_material(material, get_prim=True))

        # 随机选择一张贴图
        tex = np.random.choice(texture_files)
        shader.GetInput("diffuse_texture").Set(tex)
def randomize_wall_texture(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    texture_txt_path:str,
) -> None:
    """Randomize wall texture on reset."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    stage = omni.usd.get_context().get_stage()

    for env_id in env_ids.tolist():
        
        wall_path = f"/World/envs/env_{env_id}/FloorwithPanels/simple_room/Towel_Room01_wall_side001_286/Towel_Room01_wall_side001"
        
        wall_prim = stage.GetPrimAtPath(wall_path)
        if not wall_prim.IsValid():
            print(f"[Warning] wall prim not found at {wall_path}")
            continue
        if texture_txt_path is None:
            print("[Error] texture_txt_path is None")
            return

        texture_files = load_texture_files_from_txt(texture_txt_path)
        # 找到绑定的材质
        material, _ = UsdShade.MaterialBindingAPI(wall_prim).ComputeBoundMaterial()
        if not material:
            print(f"[Warning] no material bound to {wall_path}")
            continue

        # 获取 shader
        shader = UsdShade.Shader(omni.usd.get_shader_from_material(material, get_prim=True))

        # 随机选择一张贴图
        tex = np.random.choice(texture_files)
        shader.GetInput("diffuse_texture").Set(tex)
        
def gripper_z_force_limit(
    env: ManagerBasedEnv,
    z_threshold: float = 5.0,
    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_left"),
    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_right"),
    check_either: bool = True,
) -> torch.Tensor:
    
    left_sensor: ContactSensor = env.scene.sensors[left_sensor_cfg.name]
    right_sensor: ContactSensor = env.scene.sensors[right_sensor_cfg.name]

    if left_sensor.data.net_forces_w is None or right_sensor.data.net_forces_w is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    # Extract Z-component (index 2) of contact forces
    left_z_force = torch.abs(left_sensor.data.net_forces_w[:, 0, 2])
    right_z_force = torch.abs(right_sensor.data.net_forces_w[:, 0, 2])

    # Check if forces exceed threshold
    left_exceeds = left_z_force > z_threshold
    right_exceeds = right_z_force > z_threshold

    if check_either:
        # Terminate if either finger exceeds threshold
        terminate = left_exceeds | right_exceeds
    else:
        # Terminate only if both fingers exceed threshold
        terminate = left_exceeds & right_exceeds

    return terminate
def randomize_rigid_body_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    scale_range: tuple[float, float] | dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
    relative_child_path: str | None = None,
):
    """Randomize the scale of a rigid body asset in the USD stage.

    This function modifies the "xformOp:scale" property of all the prims corresponding to the asset.

    It takes a tuple or dictionary for the scale ranges. If it is a tuple, then the scaling along
    individual axis is performed equally. If it is a dictionary, the scaling is independent across each dimension.
    The keys of the dictionary are ``x``, ``y``, and ``z``. The values are tuples of the form ``(min, max)``.

    If the dictionary does not contain a key, the range is set to one for that axis.

    Relative child path can be used to randomize the scale of a specific child prim of the asset.
    For example, if the asset at prim path expression "/World/envs/env_.*/Object" has a child
    with the path "/World/envs/env_.*/Object/mesh", then the relative child path should be "mesh" or
    "/mesh".

    .. attention::
        Since this function modifies USD properties that are parsed by the physics engine once the simulation
        starts, the term should only be used before the simulation starts playing. This corresponds to the
        event mode named "usd". Using it at simulation time, may lead to unpredictable behaviors.

    .. note::
        When randomizing the scale of individual assets, please make sure to set
        :attr:`isaaclab.scene.InteractiveSceneCfg.replicate_physics` to False. This ensures that physics
        parser will parse the individual asset properties separately.
    """
    # check if sim is running
    if env.sim.is_playing():
        raise RuntimeError(
            "Randomizing scale while simulation is running leads to unpredictable behaviors."
            " Please ensure that the event term is called before the simulation starts by using the 'usd' mode."
        )

    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    if isinstance(asset, Articulation):
        raise ValueError(
            "Scaling an articulation randomly is not supported, as it affects joint attributes and can cause"
            " unexpected behavior. To achieve different scales, we recommend generating separate USD files for"
            " each version of the articulation and using multi-asset spawning. For more details, refer to:"
            " https://isaac-sim.github.io/IsaacLab/main/source/how-to/multi_asset_spawning.html"
        )

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # acquire stage
    stage = omni.usd.get_context().get_stage()
    # resolve prim paths for spawning and cloning
    prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)

    # sample scale values
    if isinstance(scale_range, dict):
        range_list = [scale_range.get(key, (1.0, 1.0)) for key in ["x", "y", "z"]]
        ranges = torch.tensor(range_list, device="cpu")
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu")
    else:
        rand_samples = math_utils.sample_uniform(*scale_range, (len(env_ids), 1), device="cpu")
        rand_samples = rand_samples.repeat(1, 3)
    # convert to list for the for loop
    rand_samples = rand_samples.tolist()

    # apply the randomization to the parent if no relative child path is provided
    # this might be useful if user wants to randomize a particular mesh in the prim hierarchy
    if relative_child_path is None:
        relative_child_path = ""
    elif not relative_child_path.startswith("/"):
        relative_child_path = "/" + relative_child_path

    # use sdf changeblock for faster processing of USD properties
    with Sdf.ChangeBlock():
        for i, env_id in enumerate(env_ids):
            # path to prim to randomize
            prim_path = prim_paths[env_id] + relative_child_path
            # spawn single instance
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

            # get the attribute to randomize
            scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
            # if the scale attribute does not exist, create it
            has_scale_attr = scale_spec is not None
            if not has_scale_attr:
                scale_spec = Sdf.AttributeSpec(prim_spec, prim_path + ".xformOp:scale", Sdf.ValueTypeNames.Double3)

            # set the new scale
            scale_spec.default = Gf.Vec3f(*rand_samples[i])

            # ensure the operation is done in the right ordering if we created the scale attribute.
            # otherwise, we assume the scale attribute is already in the right order.
            # note: by default isaac sim follows this ordering for the transform stack so any asset
            #   created through it will have the correct ordering
            if not has_scale_attr:
                op_order_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
                if op_order_spec is None:
                    op_order_spec = Sdf.AttributeSpec(
                        prim_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                    )
                op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])


class randomize_rigid_body_material(ManagerTermBase):
    """Randomize the physics materials on all geometries of the asset.

    This function creates a set of physics materials with random static friction, dynamic friction, and restitution
    values. The number of materials is specified by ``num_buckets``. The materials are generated by sampling
    uniform random values from the given ranges.

    The material properties are then assigned to the geometries of the asset. The assignment is done by
    creating a random integer tensor of shape  (num_instances, max_num_shapes) where ``num_instances``
    is the number of assets spawned and ``max_num_shapes`` is the maximum number of shapes in the asset (over
    all bodies). The integer values are used as indices to select the material properties from the
    material buckets.

    If the flag ``make_consistent`` is set to ``True``, the dynamic friction is set to be less than or equal to
    the static friction. This obeys the physics constraint on friction values. However, it may not always be
    essential for the application. Thus, the flag is set to ``False`` by default.

    .. attention::
        This function uses CPU tensors to assign the material properties. It is recommended to use this function
        only during the initialization of the environment. Otherwise, it may lead to a significant performance
        overhead.

    .. note::
        PhysX only allows 64000 unique physics materials in the scene. If the number of materials exceeds this
        limit, the simulation will crash. Due to this reason, we sample the materials only once during initialization.
        Afterwards, these materials are randomly assigned to the geometries of the asset.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.

        Raises:
            ValueError: If the asset is not a RigidObject or an Articulation.
        """
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if not isinstance(self.asset, (RigidObject, Articulation)):
            raise ValueError(
                f"Randomization term 'randomize_rigid_body_material' not supported for asset: '{self.asset_cfg.name}'"
                f" with type: '{type(self.asset)}'."
            )

        # obtain number of shapes per body (needed for indexing the material properties correctly)
        # note: this is a workaround since the Articulation does not provide a direct way to obtain the number of shapes
        #  per body. We use the physics simulation view to obtain the number of shapes per body.
        if isinstance(self.asset, Articulation) and self.asset_cfg.body_ids != slice(None):
            self.num_shapes_per_body = []
            for link_path in self.asset.root_physx_view.link_paths[0]:
                link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
                self.num_shapes_per_body.append(link_physx_view.max_shapes)
            # ensure the parsing is correct
            num_shapes = sum(self.num_shapes_per_body)
            expected_shapes = self.asset.root_physx_view.max_shapes
            if num_shapes != expected_shapes:
                raise ValueError(
                    "Randomization term 'randomize_rigid_body_material' failed to parse the number of shapes per body."
                    f" Expected total shapes: {expected_shapes}, but got: {num_shapes}."
                )
        else:
            # in this case, we don't need to do special indexing
            self.num_shapes_per_body = None

        # obtain parameters for sampling friction and restitution values
        static_friction_range = cfg.params.get("static_friction_range", (1.0, 1.0))
        dynamic_friction_range = cfg.params.get("dynamic_friction_range", (1.0, 1.0))
        restitution_range = cfg.params.get("restitution_range", (0.0, 0.0))
        num_buckets = int(cfg.params.get("num_buckets", 1))

        # sample material properties from the given ranges
        # note: we only sample the materials once during initialization
        #   afterwards these are randomly assigned to the geometries of the asset
        range_list = [static_friction_range, dynamic_friction_range, restitution_range]
        ranges = torch.tensor(range_list, device="cpu")
        self.material_buckets = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (num_buckets, 3), device="cpu")

        # ensure dynamic friction is always less than static friction
        make_consistent = cfg.params.get("make_consistent", False)
        if make_consistent:
            self.material_buckets[:, 1] = torch.min(self.material_buckets[:, 0], self.material_buckets[:, 1])

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        num_buckets: int,
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = False,
    ):
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        # randomly assign material IDs to the geometries
        total_num_shapes = self.asset.root_physx_view.max_shapes
        bucket_ids = torch.randint(0, num_buckets, (len(env_ids), total_num_shapes), device="cpu")
        material_samples = self.material_buckets[bucket_ids]

        # retrieve material buffer from the physics simulation
        materials = self.asset.root_physx_view.get_material_properties()

        # update material buffer with new samples
        if self.num_shapes_per_body is not None:
            # sample material properties from the given ranges
            for body_id in self.asset_cfg.body_ids:
                # obtain indices of shapes for the body
                start_idx = sum(self.num_shapes_per_body[:body_id])
                end_idx = start_idx + self.num_shapes_per_body[body_id]
                # assign the new materials
                # material samples are of shape: num_env_ids x total_num_shapes x 3
                materials[env_ids, start_idx:end_idx] = material_samples[:, start_idx:end_idx]
        else:
            # assign all the materials
            materials[env_ids] = material_samples[:]

        # apply to simulation
        self.asset.root_physx_view.set_material_properties(materials, env_ids)


def randomize_rigid_body_mass(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    recompute_inertia: bool = True,
):
    """Randomize the mass of the bodies by adding, scaling, or setting random values.

    This function allows randomizing the mass of the bodies of the asset. The function samples random values from the
    given distribution parameters and adds, scales, or sets the values into the physics simulation based on the operation.

    If the ``recompute_inertia`` flag is set to ``True``, the function recomputes the inertia tensor of the bodies
    after setting the mass. This is useful when the mass is changed significantly, as the inertia tensor depends
    on the mass. It assumes the body is a uniform density object. If the body is not a uniform density object,
    the inertia tensor may not be accurate.

    .. tip::
        This function uses CPU tensors to assign the body masses. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current masses of the bodies (num_assets, num_bodies)
    masses = asset.root_physx_view.get_masses()

    # apply randomization on default values
    # this is to make sure when calling the function multiple times, the randomization is applied on the
    # default values and not the previously randomized values
    masses[env_ids[:, None], body_ids] = asset.data.default_mass[env_ids[:, None], body_ids].clone()

    # sample from the given range
    # note: we modify the masses in-place for all environments
    #   however, the setter takes care that only the masses of the specified environments are modified
    masses = _randomize_prop_by_op(
        masses, mass_distribution_params, env_ids, body_ids, operation=operation, distribution=distribution
    )

    # set the mass into the physics simulation
    asset.root_physx_view.set_masses(masses, env_ids)

    # recompute inertia tensors if needed
    if recompute_inertia:
        # compute the ratios of the new masses to the initial masses
        ratios = masses[env_ids[:, None], body_ids] / asset.data.default_mass[env_ids[:, None], body_ids]
        # scale the inertia tensors by the the ratios
        # since mass randomization is done on default values, we can use the default inertia tensors
        inertias = asset.root_physx_view.get_inertias()
        if isinstance(asset, Articulation):
            # inertia has shape: (num_envs, num_bodies, 9) for articulation
            inertias[env_ids[:, None], body_ids] = (
                asset.data.default_inertia[env_ids[:, None], body_ids] * ratios[..., None]
            )
        else:
            # inertia has shape: (num_envs, 9) for rigid object
            inertias[env_ids] = asset.data.default_inertia[env_ids] * ratios
        # set the inertia tensors into the physics simulation
        asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_rigid_body_collider_offsets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    rest_offset_distribution_params: tuple[float, float] | None = None,
    contact_offset_distribution_params: tuple[float, float] | None = None,
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the collider parameters of rigid bodies in an asset by adding, scaling, or setting random values.

    This function allows randomizing the collider parameters of the asset, such as rest and contact offsets.
    These correspond to the physics engine collider properties that affect the collision checking.

    The function samples random values from the given distribution parameters and applies the operation to
    the collider properties. It then sets the values into the physics simulation. If the distribution parameters
    are not provided for a particular property, the function does not modify the property.

    Currently, the distribution parameters are applied as absolute values.

    .. tip::
        This function uses CPU tensors to assign the collision properties. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")

    # sample collider properties from the given ranges and set into the physics simulation
    # -- rest offsets
    if rest_offset_distribution_params is not None:
        rest_offset = asset.root_physx_view.get_rest_offsets().clone()
        rest_offset = _randomize_prop_by_op(
            rest_offset,
            rest_offset_distribution_params,
            None,
            slice(None),
            operation="abs",
            distribution=distribution,
        )
        asset.root_physx_view.set_rest_offsets(rest_offset, env_ids.cpu())
    # -- contact offsets
    if contact_offset_distribution_params is not None:
        contact_offset = asset.root_physx_view.get_contact_offsets().clone()
        contact_offset = _randomize_prop_by_op(
            contact_offset,
            contact_offset_distribution_params,
            None,
            slice(None),
            operation="abs",
            distribution=distribution,
        )
        asset.root_physx_view.set_contact_offsets(contact_offset, env_ids.cpu())


def randomize_physics_scene_gravity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    gravity_distribution_params: tuple[list[float], list[float]],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize gravity by adding, scaling, or setting random values.

    This function allows randomizing gravity of the physics scene. The function samples random values from the
    given distribution parameters and adds, scales, or sets the values into the physics simulation based on the
    operation.

    The distribution parameters are lists of two elements each, representing the lower and upper bounds of the
    distribution for the x, y, and z components of the gravity vector. The function samples random values for each
    component independently.

    .. attention::
        This function applied the same gravity for all the environments.

    .. tip::
        This function uses CPU tensors to assign gravity.
    """
    # get the current gravity
    gravity = torch.tensor(env.sim.cfg.gravity, device="cpu").unsqueeze(0)
    dist_param_0 = torch.tensor(gravity_distribution_params[0], device="cpu")
    dist_param_1 = torch.tensor(gravity_distribution_params[1], device="cpu")
    gravity = _randomize_prop_by_op(
        gravity,
        (dist_param_0, dist_param_1),
        None,
        slice(None),
        operation=operation,
        distribution=distribution,
    )
    # unbatch the gravity tensor into a list
    gravity = gravity[0].tolist()

    # set the gravity into the physics simulation
    physics_sim_view: physx.SimulationView = sim_utils.SimulationContext.instance().physics_sim_view
    physics_sim_view.set_gravity(carb.Float3(*gravity))


def randomize_actuator_gains(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    stiffness_distribution_params: tuple[float, float] | None = None,
    damping_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the actuator gains in an articulation by adding, scaling, or setting random values.

    This function allows randomizing the actuator stiffness and damping gains.

    The function samples random values from the given distribution parameters and applies the operation to the joint properties.
    It then sets the values into the actuator models. If the distribution parameters are not provided for a particular property,
    the function does not modify the property.

    .. tip::
        For implicit actuators, this function uses CPU tensors to assign the actuator gains into the simulation.
        In such cases, it is recommended to use this function only during the initialization of the environment.
    """
    # Extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    def randomize(data: torch.Tensor, params: tuple[float, float]) -> torch.Tensor:
        return _randomize_prop_by_op(
            data, params, dim_0_ids=None, dim_1_ids=actuator_indices, operation=operation, distribution=distribution
        )

    # Loop through actuators and randomize gains
    for actuator in asset.actuators.values():
        if isinstance(asset_cfg.joint_ids, slice):
            # we take all the joints of the actuator
            actuator_indices = slice(None)
            if isinstance(actuator.joint_indices, slice):
                global_indices = slice(None)
            else:
                global_indices = torch.tensor(actuator.joint_indices, device=asset.device)
        elif isinstance(actuator.joint_indices, slice):
            # we take the joints defined in the asset config
            global_indices = actuator_indices = torch.tensor(asset_cfg.joint_ids, device=asset.device)
        else:
            # we take the intersection of the actuator joints and the asset config joints
            actuator_joint_indices = torch.tensor(actuator.joint_indices, device=asset.device)
            asset_joint_ids = torch.tensor(asset_cfg.joint_ids, device=asset.device)
            # the indices of the joints in the actuator that have to be randomized
            actuator_indices = torch.nonzero(torch.isin(actuator_joint_indices, asset_joint_ids)).view(-1)
            if len(actuator_indices) == 0:
                continue
            # maps actuator indices that have to be randomized to global joint indices
            global_indices = actuator_joint_indices[actuator_indices]
        # Randomize stiffness
        if stiffness_distribution_params is not None:
            stiffness = actuator.stiffness[env_ids].clone()
            stiffness[:, actuator_indices] = asset.data.default_joint_stiffness[env_ids][:, global_indices].clone()
            randomize(stiffness, stiffness_distribution_params)
            actuator.stiffness[env_ids] = stiffness
            if isinstance(actuator, ImplicitActuator):
                asset.write_joint_stiffness_to_sim(stiffness, joint_ids=actuator.joint_indices, env_ids=env_ids)
        # Randomize damping
        if damping_distribution_params is not None:
            damping = actuator.damping[env_ids].clone()
            damping[:, actuator_indices] = asset.data.default_joint_damping[env_ids][:, global_indices].clone()
            randomize(damping, damping_distribution_params)
            actuator.damping[env_ids] = damping
            if isinstance(actuator, ImplicitActuator):
                asset.write_joint_damping_to_sim(damping, joint_ids=actuator.joint_indices, env_ids=env_ids)


def randomize_joint_parameters(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    friction_distribution_params: tuple[float, float] | None = None,
    armature_distribution_params: tuple[float, float] | None = None,
    lower_limit_distribution_params: tuple[float, float] | None = None,
    upper_limit_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the simulated joint parameters of an articulation by adding, scaling, or setting random values.

    This function allows randomizing the joint parameters of the asset. These correspond to the physics engine
    joint properties that affect the joint behavior. The properties include the joint friction coefficient, armature,
    and joint position limits.

    The function samples random values from the given distribution parameters and applies the operation to the
    joint properties. It then sets the values into the physics simulation. If the distribution parameters are
    not provided for a particular property, the function does not modify the property.

    .. tip::
        This function uses CPU tensors to assign the joint properties. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    # sample joint properties from the given ranges and set into the physics simulation
    # joint friction coefficient
    if friction_distribution_params is not None:
        friction_coeff = _randomize_prop_by_op(
            asset.data.default_joint_friction_coeff.clone(),
            friction_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )
        asset.write_joint_friction_coefficient_to_sim(
            friction_coeff[env_ids[:, None], joint_ids], joint_ids=joint_ids, env_ids=env_ids
        )

    # joint armature
    if armature_distribution_params is not None:
        armature = _randomize_prop_by_op(
            asset.data.default_joint_armature.clone(),
            armature_distribution_params,
            env_ids,
            joint_ids,
            operation=operation,
            distribution=distribution,
        )
        asset.write_joint_armature_to_sim(armature[env_ids[:, None], joint_ids], joint_ids=joint_ids, env_ids=env_ids)

    # joint position limits
    if lower_limit_distribution_params is not None or upper_limit_distribution_params is not None:
        joint_pos_limits = asset.data.default_joint_pos_limits.clone()
        # -- randomize the lower limits
        if lower_limit_distribution_params is not None:
            joint_pos_limits[..., 0] = _randomize_prop_by_op(
                joint_pos_limits[..., 0],
                lower_limit_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )
        # -- randomize the upper limits
        if upper_limit_distribution_params is not None:
            joint_pos_limits[..., 1] = _randomize_prop_by_op(
                joint_pos_limits[..., 1],
                upper_limit_distribution_params,
                env_ids,
                joint_ids,
                operation=operation,
                distribution=distribution,
            )

        # extract the position limits for the concerned joints
        joint_pos_limits = joint_pos_limits[env_ids[:, None], joint_ids]
        if (joint_pos_limits[..., 0] > joint_pos_limits[..., 1]).any():
            raise ValueError(
                "Randomization term 'randomize_joint_parameters' is setting lower joint limits that are greater than"
                " upper joint limits. Please check the distribution parameters for the joint position limits."
            )
        # set the position limits into the physics simulation
        asset.write_joint_position_limit_to_sim(
            joint_pos_limits, joint_ids=joint_ids, env_ids=env_ids, warn_limit_violation=False
        )


def randomize_fixed_tendon_parameters(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    stiffness_distribution_params: tuple[float, float] | None = None,
    damping_distribution_params: tuple[float, float] | None = None,
    limit_stiffness_distribution_params: tuple[float, float] | None = None,
    lower_limit_distribution_params: tuple[float, float] | None = None,
    upper_limit_distribution_params: tuple[float, float] | None = None,
    rest_length_distribution_params: tuple[float, float] | None = None,
    offset_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the simulated fixed tendon parameters of an articulation by adding, scaling, or setting random values.

    This function allows randomizing the fixed tendon parameters of the asset.
    These correspond to the physics engine tendon properties that affect the joint behavior.

    The function samples random values from the given distribution parameters and applies the operation to the tendon properties.
    It then sets the values into the physics simulation. If the distribution parameters are not provided for a
    particular property, the function does not modify the property.

    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.fixed_tendon_ids == slice(None):
        tendon_ids = slice(None)  # for optimization purposes
    else:
        tendon_ids = torch.tensor(asset_cfg.fixed_tendon_ids, dtype=torch.int, device=asset.device)

    # sample tendon properties from the given ranges and set into the physics simulation
    # stiffness
    if stiffness_distribution_params is not None:
        stiffness = _randomize_prop_by_op(
            asset.data.default_fixed_tendon_stiffness.clone(),
            stiffness_distribution_params,
            env_ids,
            tendon_ids,
            operation=operation,
            distribution=distribution,
        )
        asset.set_fixed_tendon_stiffness(stiffness[env_ids[:, None], tendon_ids], tendon_ids, env_ids)

    # damping
    if damping_distribution_params is not None:
        damping = _randomize_prop_by_op(
            asset.data.default_fixed_tendon_damping.clone(),
            damping_distribution_params,
            env_ids,
            tendon_ids,
            operation=operation,
            distribution=distribution,
        )
        asset.set_fixed_tendon_damping(damping[env_ids[:, None], tendon_ids], tendon_ids, env_ids)

    # limit stiffness
    if limit_stiffness_distribution_params is not None:
        limit_stiffness = _randomize_prop_by_op(
            asset.data.default_fixed_tendon_limit_stiffness.clone(),
            limit_stiffness_distribution_params,
            env_ids,
            tendon_ids,
            operation=operation,
            distribution=distribution,
        )
        asset.set_fixed_tendon_limit_stiffness(limit_stiffness[env_ids[:, None], tendon_ids], tendon_ids, env_ids)

    # position limits
    if lower_limit_distribution_params is not None or upper_limit_distribution_params is not None:
        limit = asset.data.default_fixed_tendon_pos_limits.clone()
        # -- lower limit
        if lower_limit_distribution_params is not None:
            limit[..., 0] = _randomize_prop_by_op(
                limit[..., 0],
                lower_limit_distribution_params,
                env_ids,
                tendon_ids,
                operation=operation,
                distribution=distribution,
            )
        # -- upper limit
        if upper_limit_distribution_params is not None:
            limit[..., 1] = _randomize_prop_by_op(
                limit[..., 1],
                upper_limit_distribution_params,
                env_ids,
                tendon_ids,
                operation=operation,
                distribution=distribution,
            )

        # check if the limits are valid
        tendon_limits = limit[env_ids[:, None], tendon_ids]
        if (tendon_limits[..., 0] > tendon_limits[..., 1]).any():
            raise ValueError(
                "Randomization term 'randomize_fixed_tendon_parameters' is setting lower tendon limits that are greater"
                " than upper tendon limits."
            )
        asset.set_fixed_tendon_position_limit(tendon_limits, tendon_ids, env_ids)

    # rest length
    if rest_length_distribution_params is not None:
        rest_length = _randomize_prop_by_op(
            asset.data.default_fixed_tendon_rest_length.clone(),
            rest_length_distribution_params,
            env_ids,
            tendon_ids,
            operation=operation,
            distribution=distribution,
        )
        asset.set_fixed_tendon_rest_length(rest_length[env_ids[:, None], tendon_ids], tendon_ids, env_ids)

    # offset
    if offset_distribution_params is not None:
        offset = _randomize_prop_by_op(
            asset.data.default_fixed_tendon_offset.clone(),
            offset_distribution_params,
            env_ids,
            tendon_ids,
            operation=operation,
            distribution=distribution,
        )
        asset.set_fixed_tendon_offset(offset[env_ids[:, None], tendon_ids], tendon_ids, env_ids)

    # write the fixed tendon properties into the simulation
    asset.write_fixed_tendon_properties_to_sim(tendon_ids, env_ids)


def apply_external_force_torque(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize the external forces and torques applied to the bodies.

    This function creates a set of random forces and torques sampled from the given ranges. The number of forces
    and torques is equal to the number of bodies times the number of environments. The forces and torques are
    applied to the bodies by calling ``asset.set_external_force_and_torque``. The forces and torques are only
    applied when ``asset.write_data_to_sim()`` is called in the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    # resolve number of bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    # sample random forces and torques
    size = (len(env_ids), num_bodies, 3)
    forces = math_utils.sample_uniform(*force_range, size, asset.device)
    torques = math_utils.sample_uniform(*torque_range, size, asset.device)
    # set the forces and torques into the buffers
    # note: these are only applied when you call: `asset.write_data_to_sim()`
    asset.set_external_force_and_torque(forces, torques, env_ids=env_ids, body_ids=asset_cfg.body_ids)


def push_by_setting_velocity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Push the asset by setting the root velocity to a random value within the given ranges.

    This creates an effect similar to pushing the asset with a random impulse that changes the asset's velocity.
    It samples the root velocity from the given ranges and sets the velocity into the physics simulation.

    The function takes a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
    are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form ``(min, max)``.
    If the dictionary does not contain a key, the velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # velocities
    vel_w = asset.data.root_vel_w[env_ids]
    # sample random velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    vel_w += math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
    # set the velocities into the physics simulation
    asset.write_root_velocity_to_sim(vel_w, env_ids=env_ids)


def reset_root_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)
    

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples[:, 0:3]
    orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # cube_sizes = math_utils.sample_uniform(size_range[0], size_range[1], (len(env_ids),), device=asset.device)
    # cube_scales = torch.stack([cube_sizes] * 3, dim=-1)  # 均匀缩放

    # 设置 cube 的局部缩放 (需要 asset 是 RigidObject)
    # set_cube_local_scale(env, asset, env_ids, cube_scales)
    # print(positions[0])

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
from pxr import Gf, UsdGeom

def set_cube_local_scale(env, asset, env_ids: torch.Tensor, scales: torch.Tensor):
    """
    通过 env 获取 stage，设置对应物体的局部缩放。
    
    env: ManagerBasedEnv
    asset: RigidObject
    env_ids: 环境索引列表
    scales: 尺寸tensor，shape (N,3)
    """
    stage = env.stage  # 从 env 拿 USD stage
    for i, scale in zip(env_ids.cpu().numpy(), scales.cpu().numpy()):
        prim_path = f"/World/envs/env_{i}/Object"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            print(f"警告：没有找到 prim {prim_path}")
            continue
        from pxr import Gf, UsdGeom
        xformable = UsdGeom.Xformable(prim)
        xformable.AddScaleOp().Set(Gf.Vec3f(*scale))


def reset_root_state_with_random_orientation(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root position and velocities sampled randomly within the given ranges
    and the asset root orientation sampled randomly from the SO(3).

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation uniformly from the SO(3) and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of position and velocity ranges for each axis and rotation:

    * :attr:`pose_range` - a dictionary of position ranges for each axis. The keys of the dictionary are ``x``,
      ``y``, and ``z``. The orientation is sampled uniformly from the SO(3).
    * :attr:`velocity_range` - a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
      are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``.

    The values are tuples of the form ``(min, max)``. If the dictionary does not contain a particular key,
    the position is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # get default root state
    root_states = asset.data.default_root_state[env_ids].clone()

    # poses
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)

    positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_samples
    orientations = math_utils.random_orientation(len(env_ids), device=asset.device)

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = root_states[:, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


def reset_root_state_from_terrain(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state by sampling a random valid pose from the terrain.

    This function samples a random valid pose(based on flat patches) from the terrain and sets the root state
    of the asset to this position. The function also samples random velocities from the given ranges and sets them
    into the physics simulation.

    The function takes a dictionary of position and velocity ranges for each axis and rotation:

    * :attr:`pose_range` - a dictionary of pose ranges for each axis. The keys of the dictionary are ``roll``,
      ``pitch``, and ``yaw``. The position is sampled from the flat patches of the terrain.
    * :attr:`velocity_range` - a dictionary of velocity ranges for each axis and rotation. The keys of the dictionary
      are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``.

    The values are tuples of the form ``(min, max)``. If the dictionary does not contain a particular key,
    the position is set to zero for that axis.

    Note:
        The function expects the terrain to have valid flat patches under the key "init_pos". The flat patches
        are used to sample the random pose for the robot.

    Raises:
        ValueError: If the terrain does not have valid flat patches under the key "init_pos".
    """
    # access the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain

    # obtain all flat patches corresponding to the valid poses
    valid_positions: torch.Tensor = terrain.flat_patches.get("init_pos")
    if valid_positions is None:
        raise ValueError(
            "The event term 'reset_root_state_from_terrain' requires valid flat patches under 'init_pos'."
            f" Found: {list(terrain.flat_patches.keys())}"
        )

    # sample random valid poses
    ids = torch.randint(0, valid_positions.shape[2], size=(len(env_ids),), device=env.device)
    positions = valid_positions[terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids], ids]
    positions += asset.data.default_root_state[env_ids, :3]

    # sample random orientations
    range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device=asset.device)

    # convert to quaternions
    orientations = math_utils.quat_from_euler_xyz(rand_samples[:, 0], rand_samples[:, 1], rand_samples[:, 2])

    # sample random velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=asset.device)

    velocities = asset.data.default_root_state[env_ids, 7:13] + rand_samples

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)

import random
import torch
import numpy as np


def reset_camera_focal_length_per_env(
    env,
    env_ids: torch.Tensor,
    focal_length_range: tuple[float, float] = (10.0, 50.0),
    camera_cfg=None,
):
    
    
        if camera_cfg is None:
            camera_cfg = env.cfg.scene.tiled_camera

        # 给每个环境采样一个焦距，单位 mm
        sampled_focal_lengths = [
            random.uniform(focal_length_range[0], focal_length_range[1]) for _ in env_ids
        ]

        # 获取 camera_sensor
        camera_sensor = env.scene.sensors["tiled_camera"]

        for i, focal_length_mm in zip(env_ids.tolist(), sampled_focal_lengths):
            focal_length_cm = focal_length_mm   # 转成 cm

            # 取对应环境的USD相机Prim
            sensor_prim = camera_sensor._sensor_prims[i]

            # 直接修改焦距属性
            param_attr = sensor_prim.GetFocalLengthAttr()
            omni.usd.set_prop_val(param_attr, focal_length_cm)

            # print(f"[Env {i}] Set focal_length to {focal_length_mm:.2f} cm ")
    


  

def reset_joints_by_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints by scaling the default position and velocity by the given ranges.

    This function samples random values from the given ranges and scales the default joint positions and velocities
    by these values. The scaled values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # get default joint state
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    # scale these values randomly
    joint_pos *= math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel *= math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

# def reset_cube_size_by_range(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor,
#     size_range: tuple[float, float],
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
# ):


#     # Get the cube entity
#     cube: RigidObject = env.scene[asset_cfg.name]

#     # Sample random sizes
#     cube_sizes = math_utils.sample_uniform(*size_range, (len(env_ids),), device=env.device)

#     # Apply scaling to each cube prim
#     for i, env_id in enumerate(env_ids):
#         prim_path = cube._physics_view.get_prim_path(env_id)
#         size = cube_sizes[i].item()
#         set_local_scale(prim_path, (size, size, size))



#     return cube_sizes  # 如果你要用于 logging 或训练



def reset_joints_by_offset(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the robot joints with offsets around the default position and velocity by the given ranges.

    This function samples random values from the given ranges and biases the default joint positions and velocities
    by these values. The biased values are then set into the physics simulation.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # get default joint state
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    # bias these values randomly
    joint_pos += math_utils.sample_uniform(*position_range, joint_pos.shape, joint_pos.device)
    joint_vel += math_utils.sample_uniform(*velocity_range, joint_vel.shape, joint_vel.device)

    # clamp joint pos to limits
    joint_pos_limits = asset.data.soft_joint_pos_limits[env_ids]
    joint_pos = joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    # clamp joint vel to limits
    joint_vel_limits = asset.data.soft_joint_vel_limits[env_ids]
    joint_vel = joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)

    # set into the physics simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def reset_nodal_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset nodal state to a random position and velocity uniformly within the given ranges.

    This function randomizes the nodal position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default nodal position, before setting
      them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of position and velocity ranges for each axis. The keys of the
    dictionary are ``x``, ``y``, ``z``. The values are tuples of the form ``(min, max)``.
    If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: DeformableObject = env.scene[asset_cfg.name]
    # get default root state
    nodal_state = asset.data.default_nodal_state_w[env_ids].clone()

    # position
    range_list = [position_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 1, 3), device=asset.device)

    nodal_state[..., :3] += rand_samples

    # velocities
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device=asset.device)
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 1, 3), device=asset.device)

    nodal_state[..., 3:] += rand_samples

    # set into the physics simulation
    asset.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)


def reset_scene_to_default(env: ManagerBasedEnv, env_ids: torch.Tensor):
    """Reset the scene to the default state specified in the scene configuration."""
    # rigid bodies
    for rigid_object in env.scene.rigid_objects.values():
        # obtain default and deal with the offset for env origins
        default_root_state = rigid_object.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        rigid_object.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        rigid_object.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
    # articulations
    for articulation_asset in env.scene.articulations.values():
        # obtain default and deal with the offset for env origins
        default_root_state = articulation_asset.data.default_root_state[env_ids].clone()
        default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
        # set into the physics simulation
        articulation_asset.write_root_pose_to_sim(default_root_state[:, :7], env_ids=env_ids)
        articulation_asset.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids=env_ids)
        # obtain default joint positions
        default_joint_pos = articulation_asset.data.default_joint_pos[env_ids].clone()
        default_joint_vel = articulation_asset.data.default_joint_vel[env_ids].clone()
        # set into the physics simulation
        articulation_asset.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)
    # deformable objects
    for deformable_object in env.scene.deformable_objects.values():
        # obtain default and set into the physics simulation
        nodal_state = deformable_object.data.default_nodal_state_w[env_ids].clone()
        deformable_object.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)


class randomize_visual_texture_material(ManagerTermBase):
    """Randomize the visual texture of bodies on an asset using Replicator API.

    This function randomizes the visual texture of the bodies of the asset using the Replicator API.
    The function samples random textures from the given texture paths and applies them to the bodies
    of the asset. The textures are projected onto the bodies and rotated by the given angles.

    .. note::
        The function assumes that the asset follows the prim naming convention as:
        "{asset_prim_path}/{body_name}/visuals" where the body name is the name of the body to
        which the texture is applied. This is the default prim ordering when importing assets
        from the asset converters in Isaac Lab.

    .. note::
        When randomizing the texture of individual assets, please make sure to set
        :attr:`isaaclab.scene.InteractiveSceneCfg.replicate_physics` to False. This ensures that physics
        parser will parse the individual asset properties separately.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the event term.
            env: The environment instance.
        """
        super().__init__(cfg, env)

        # enable replicator extension if not already enabled
        enable_extension("omni.replicator.core")
        # we import the module here since we may not always need the replicator
        import omni.replicator.core as rep

        # read parameters from the configuration
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg")
        texture_paths = cfg.params.get("texture_paths")
        event_name = cfg.params.get("event_name")
        texture_rotation = cfg.params.get("texture_rotation", (0.0, 0.0))

        # check to make sure replicate_physics is set to False, else raise error
        # note: We add an explicit check here since texture randomization can happen outside of 'prestartup' mode
        #   and the event manager doesn't check in that case.
        if env.cfg.scene.replicate_physics:
            raise RuntimeError(
                "Unable to randomize visual texture material with scene replication enabled."
                " For stable USD-level randomization, please disable scene replication"
                " by setting 'replicate_physics' to False in 'InteractiveSceneCfg'."
            )

        # convert from radians to degrees
        texture_rotation = tuple(math.degrees(angle) for angle in texture_rotation)

        # obtain the asset entity
        asset = env.scene[asset_cfg.name]

        # join all bodies in the asset
        body_names = asset_cfg.body_names
        if isinstance(body_names, str):
            body_names_regex = body_names
        elif isinstance(body_names, list):
            body_names_regex = "|".join(body_names)
        else:
            body_names_regex = ".*"

        # create the affected prim path
        # TODO: Remove the hard-coded "/visuals" part.
        prim_path = f"{asset.cfg.prim_path}/{body_names_regex}/visuals"

        # Create the omni-graph node for the randomization term
        def rep_texture_randomization():
            prims_group = rep.get.prims(path_pattern=prim_path)

            with prims_group:
                rep.randomizer.texture(
                    textures=texture_paths, project_uvw=True, texture_rotate=rep.distribution.uniform(*texture_rotation)
                )

            return prims_group.node

        # Register the event to the replicator
        with rep.trigger.on_custom_event(event_name=event_name):
            rep_texture_randomization()

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        event_name: str,
        asset_cfg: SceneEntityCfg,
        texture_paths: list[str],
        texture_rotation: tuple[float, float] = (0.0, 0.0),
    ):
        # import replicator
        import omni.replicator.core as rep

        # only send the event to the replicator
        # note: This triggers the nodes for all the environments.
        #   We need to investigate how to make it happen only for a subset based on env_ids.
        rep.utils.send_og_event(event_name)


class randomize_visual_color(ManagerTermBase):
    """Randomize the visual color of bodies on an asset using Replicator API.

    This function randomizes the visual color of the bodies of the asset using the Replicator API.
    The function samples random colors from the given colors and applies them to the bodies
    of the asset.

    The function assumes that the asset follows the prim naming convention as:
    "{asset_prim_path}/{mesh_name}" where the mesh name is the name of the mesh to
    which the color is applied. For instance, if the asset has a prim path "/World/asset"
    and a mesh named "body_0/mesh", the prim path for the mesh would be
    "/World/asset/body_0/mesh".

    The colors can be specified as a list of tuples of the form ``(r, g, b)`` or as a dictionary
    with the keys ``r``, ``g``, ``b`` and values as tuples of the form ``(low, high)``.
    If a dictionary is used, the function will sample random colors from the given ranges.

    .. note::
        When randomizing the color of individual assets, please make sure to set
        :attr:`isaaclab.scene.InteractiveSceneCfg.replicate_physics` to False. This ensures that physics
        parser will parse the individual asset properties separately.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        """Initialize the randomization term."""
        super().__init__(cfg, env)

        # enable replicator extension if not already enabled
        enable_extension("omni.replicator.core")
        # we import the module here since we may not always need the replicator
        import omni.replicator.core as rep

        # read parameters from the configuration
        asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg")
        colors = cfg.params.get("colors")
        event_name = cfg.params.get("event_name")
        mesh_name: str = cfg.params.get("mesh_name", "")  # type: ignore

        # check to make sure replicate_physics is set to False, else raise error
        # note: We add an explicit check here since texture randomization can happen outside of 'prestartup' mode
        #   and the event manager doesn't check in that case.
        if env.cfg.scene.replicate_physics:
            raise RuntimeError(
                "Unable to randomize visual color with scene replication enabled."
                " For stable USD-level randomization, please disable scene replication"
                " by setting 'replicate_physics' to False in 'InteractiveSceneCfg'."
            )

        # obtain the asset entity
        asset = env.scene[asset_cfg.name]

        # create the affected prim path
        if not mesh_name.startswith("/"):
            mesh_name = "/" + mesh_name
        mesh_prim_path = f"{asset.cfg.prim_path}{mesh_name}"
        # TODO: Need to make it work for multiple meshes.

        # parse the colors into replicator format
        if isinstance(colors, dict):
            # (r, g, b) - low, high --> (low_r, low_g, low_b) and (high_r, high_g, high_b)
            color_low = [colors[key][0] for key in ["r", "g", "b"]]
            color_high = [colors[key][1] for key in ["r", "g", "b"]]
            colors = rep.distribution.uniform(color_low, color_high)
        else:
            colors = list(colors)

        # Create the omni-graph node for the randomization term
        def rep_texture_randomization():
            prims_group = rep.get.prims(path_pattern=mesh_prim_path)

            with prims_group:
                rep.randomizer.color(colors=colors)

            return prims_group.node

        # Register the event to the replicator
        with rep.trigger.on_custom_event(event_name=event_name):
            rep_texture_randomization()

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        event_name: str,
        asset_cfg: SceneEntityCfg,
        colors: list[tuple[float, float, float]] | dict[str, tuple[float, float]],
        mesh_name: str = "",
    ):
        # import replicator
        import omni.replicator.core as rep

        # only send the event to the replicator
        rep.utils.send_og_event(event_name)


"""
Internal helper functions.
"""


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data

def randomize_object_pool_selection(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
):
    """Randomize which object from pool is active per environment."""
    # Handle None env_ids (startup mode - apply to all environments)
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    object_collection = env.scene[asset_cfg.name]
    num_objects = len(object_collection.object_names)
    # Create buffer to store active object indices if not exists
    if not hasattr(env, 'active_object_indices'):
        env.active_object_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    for env_idx in env_ids:
        active_idx = torch.randint(0, num_objects, (1,), device=env.device).item()
        env.active_object_indices[env_idx] = active_idx
        for obj_idx in range(num_objects):
            pos = torch.tensor([0.28, 0.0, 0.0] if obj_idx == active_idx else [100.0, 100.0, -10.0], device=env.device)
            root_state = object_collection.data.default_object_state[env_idx, obj_idx].clone()
            root_state[:3] = pos + env.scene.env_origins[env_idx]
            root_state[3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
            root_state[7:] = 0.0
            object_collection.write_object_state_to_sim(
                root_state.unsqueeze(0),
                env_ids=torch.tensor([env_idx], device=env.device),
                object_ids=torch.tensor([obj_idx], device=env.device)
            )

# def randomize_object_and_position(
#     env: ManagerBasedEnv,
#     env_ids: torch.Tensor,
#     pose_range: dict[str, tuple[float, float]],
#     rigid_asset_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
# ):
#     """
#     自动处理 object_pool 只有 1 个 or 多个物体的情况。
#     单物体：直接随机位置，不选择 active index
#     多物体：保留你原来的逻辑（随机选择 + 其它物体丢到 10000m）
#     """
#     device = env.device
#     if not hasattr(env, "active_object_indices"):
#         env.active_object_indices = torch.zeros(env.scene.num_envs, dtype=torch.long, device=env.device)
#     if env_ids is None:
#         env_ids = torch.arange(env.scene.num_envs, device=device)
#     try:
#         obj_col = env.scene[rigid_asset_cfg.name]
#     except Exception:
#         return
#     num_envs = len(env_ids)
#     env_origins = env.scene.env_origins[env_ids]
#     # [N, num_objs, 13]
#     default_states = obj_col.data.default_object_state[env_ids].clone()
#     state_w = obj_col.data.object_state_w[env_ids].clone()
#     num_objs = len(obj_col.cfg.rigid_objects)
#     env_idx = torch.arange(num_envs, device=device)
#     # =============================================================================
#     # Case 1: 单物体 → 不需要随机选物体，不需要 active_object_indices
#     # =============================================================================
#     if num_objs == 1:
#         env.active_object_indices[env_ids] = 0
#         base_pos = default_states[:, 0, 0:3]
#         base_quat = default_states[:, 0, 3:7]
#         # pose range
#         range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll","pitch","yaw"]]
#         ranges = torch.tensor(range_list, device=device)
#         rand_pose = torch.empty((num_envs, 6), device=device)
#         for k in range(6):
#             rand_pose[:, k] = torch.empty(num_envs, device=device).uniform_(ranges[k, 0], ranges[k, 1])
#         delta_pos = rand_pose[:, 0:3]
#         world_pos = env_origins + base_pos + delta_pos
#         # random rotation
#         delta_quat = math_utils.quat_from_euler_xyz(rand_pose[:, 3], rand_pose[:, 4], rand_pose[:, 5])
#         world_quat = math_utils.quat_mul(base_quat, delta_quat)
#         state_w[:, 0, 0:3] = world_pos
#         state_w[:, 0, 3:7] = world_quat
#         state_w[:, 0, 7:13] = 0.0  # clear velocities
#         obj_col.write_object_link_pose_to_sim(state_w[..., :7], env_ids=env_ids)
#         obj_col.write_object_com_velocity_to_sim(state_w[..., 7:], env_ids=env_ids)
#         return  # 单物体逻辑结束
#     # =============================================================================
#     # Case 2: 多物体 → 保留原本逻辑
#     # =============================================================================
#     # 随机选择 active object
#     active_obj_ids = torch.randint(0, num_objs, (num_envs,), device=device)
#     # 1) 确保 active_object_indices 是 (env.scene.num_envs,) 的全局buffer
#     if (not hasattr(env, "active_object_indices")) or (env.active_object_indices.numel() != env.scene.num_envs):
#         env.active_object_indices = torch.zeros(env.scene.num_envs, dtype=torch.long, device=env.device)
#     # 2) env_ids 可能只是子集：只更新这些 env
#     # num_envs = len(env_ids)  # 你原来就是这么算的:contentReference[oaicite:3]{index=3}
#     active_obj_ids = torch.randint(0, num_objs, (num_envs,), device=env.device)
#     env.active_object_indices[env_ids] = active_obj_ids  # ✅关键：别覆盖整个tensor
#     # hide 未选中的物体
#     dummy_z = 10000.0
#     for i in range(num_envs):
#         for j in range(num_objs):
#             if j != int(active_obj_ids[i]):
#                 state_w[i, j, 0:3] = torch.tensor([0.0, 0.0, dummy_z], device=device)
#                 state_w[i, j, 7:13] = 0.0
#     # pose range
#     range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
#     ranges = torch.tensor(range_list, device=device)
#     rand_pose = torch.empty((num_envs, 6), device=device)
#     for k in range(6):
#         rand_pose[:, k] = torch.empty(num_envs, device=device).uniform_(ranges[k, 0], ranges[k, 1])
#     # 对每个 env 的 active object 单独设置随机 pose
#     base_pos = default_states[env_idx, active_obj_ids, 0:3]
#     base_quat = default_states[env_idx, active_obj_ids, 3:7]
#     delta_pos = rand_pose[:, 0:3]
#     world_pos = env_origins + base_pos + delta_pos
#     delta_quat = math_utils.quat_from_euler_xyz(rand_pose[:, 3], rand_pose[:, 4], rand_pose[:, 5])
#     world_quat = math_utils.quat_mul(base_quat, delta_quat)
#     # 写回 active object
#     state_w[env_idx, active_obj_ids, 0:3] = world_pos
#     state_w[env_idx, active_obj_ids, 3:7] = world_quat
#     state_w[env_idx, active_obj_ids, 7:13] = 0.0
#     obj_col.write_object_link_pose_to_sim(state_w[..., :7], env_ids=env_ids)
#     obj_col.write_object_com_velocity_to_sim(state_w[..., 7:], env_ids=env_ids)

def reset_object_pool_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
):
    """Reset active objects in object pool to random position and velocity uniformly within given ranges."""
    from isaaclab.assets import RigidObjectCollection
    import isaaclab.utils.math as math_utils

    # Extract object collection
    object_collection: RigidObjectCollection = env.scene[asset_cfg.name]

    # Get active object indices
    if not hasattr(env, 'active_object_indices'):
        raise RuntimeError("active_object_indices not found")

    # Prepare pose ranges (same as original)
    pose_range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    pose_ranges = torch.tensor(pose_range_list, device=object_collection.device)
    pose_rand_samples = math_utils.sample_uniform(
        pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), 6), device=object_collection.device
    )

    # Prepare velocity ranges (same as original)
    vel_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    vel_ranges = torch.tensor(vel_range_list, device=object_collection.device)
    vel_rand_samples = math_utils.sample_uniform(
        vel_ranges[:, 0], vel_ranges[:, 1], (len(env_ids), 6), device=object_collection.device
    )

    # Reset each environment's active object
    for idx, env_idx in enumerate(env_ids):
        active_obj_idx = env.active_object_indices[env_idx].item()

        # Get default state for active object
        root_state = object_collection.data.default_object_state[env_idx, active_obj_idx].clone()

        # Apply pose randomization (same logic as original)
        position = root_state[0:3] + env.scene.env_origins[env_idx] + pose_rand_samples[idx, 0:3]
        orientation_delta = math_utils.quat_from_euler_xyz(
            pose_rand_samples[idx, 3], pose_rand_samples[idx, 4], pose_rand_samples[idx, 5]
        )
        orientation = math_utils.quat_mul(root_state[3:7], orientation_delta)

        # Apply velocity randomization (same logic as original)
        velocity = root_state[7:13] + vel_rand_samples[idx]

        # Combine and write (adapted for object collection)
        new_state = torch.cat([position, orientation, velocity], dim=-1)
        object_collection.write_object_state_to_sim(
            new_state.unsqueeze(0),
            env_ids=torch.tensor([env_idx], device=object_collection.device),
            object_ids=torch.tensor([active_obj_idx], device=object_collection.device)
        )