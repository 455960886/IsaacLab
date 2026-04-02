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
from typing import TYPE_CHECKING, Literal
import numpy as np
import carb
import omni.physics.tensors.impl.api as physx
import omni.usd
from isaacsim.core.utils.extensions import enable_extension
from pxr import Gf, Sdf, UsdGeom, Vt, UsdShade, Usd

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import sample_uniform
from isaaclab.actuators import ImplicitActuator
# import omni.isaac.core.utils.stage as stage_utils
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


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



def randomize_sphere_light_intensity(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float] = (800.0, 2500.0),
) -> None:
    """Randomize only the intensity of sphere lights for specific environments.

    Args:
        env: Environment instance
        env_ids: Environment indices to randomize
        intensity_range: Range for light intensity values
    """

    try:
        import omni.usd
        from pxr import UsdLux

        stage = omni.usd.get_context().get_stage()
        num_envs = len(env_ids)

        # Sample random intensities for all environments at once
        intensities = sample_uniform(intensity_range[0], intensity_range[1], (num_envs,), device=env.device)

        # Apply to each environment's sphere light
        for i, env_id in enumerate(env_ids):
            # Path to your sphere light in each environment
            sphere_light_path = f"/World/envs/env_{env_id}/SphereLight"
            light_prim = stage.GetPrimAtPath(sphere_light_path)

            if light_prim.IsValid():
                sphere_light = UsdLux.SphereLight(light_prim)
                sphere_light.GetIntensityAttr().Set(float(intensities[i].item()))
            else:
                # Try alternative path if the first doesn't work
                alt_path = f"/World/envs/env_{env_id:04d}/SphereLight"
                alt_prim = stage.GetPrimAtPath(alt_path)
                if alt_prim.IsValid():
                    sphere_light = UsdLux.SphereLight(alt_prim)
                    sphere_light.GetIntensityAttr().Set(float(intensities[i].item()))
                else:
                    print(f"Warning: Could not find sphere light at env {env_id}")

    except Exception as e:
        print(f"Warning: Sphere light intensity randomization failed: {e}")


def randomize_light_color_temperature(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    temperature_range: tuple[float, float] = (3000.0, 6500.0),
) -> None:
    """Randomize light color temperature.

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize.
        temperature_range: Range for color temperature in Kelvin.
    """
    num_envs = len(env_ids)
    # Sample random temperature
    temperature = sample_uniform(temperature_range[0], temperature_range[1], (num_envs,), device=env.device)

    try:
        import omni.usd
        from pxr import UsdLux

        stage = omni.usd.get_context().get_stage()
        # dome_light_prim = stage.GetPrimAtPath("/World/light")  # Adjust path

        # Apply to each environment's sphere light
        for i, env_id in enumerate(env_ids):
            # Path to your sphere light in each environment
            sphere_light_path = f"/World/envs/env_{env_id}/SphereLight"
            light_prim = stage.GetPrimAtPath(sphere_light_path)

            sphere_light = UsdLux.SphereLight(light_prim)
            sphere_light.GetColorTemperatureAttr().Set(float(temperature[i].item()))
            # print(f"sphere light {i} temperature set to {temperature[i]}")


    except Exception as e:
        print(f"Warning: Sphere light intensity randomization failed: {e}")


def randomize_lighting_combined(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float] = (500.0, 2000.0),
    temperature_range: tuple[float, float] = (3000.0, 6500.0),
) -> None:
    """Combined lighting randomization function.

    Args:
        env: The environment instance.
        env_ids: Environment indices to randomize.
        intensity_range: Range for light intensity.
        temperature_range: Range for color temperature.
    """
    randomize_dome_light_intensity(env, env_ids, intensity_range)
    randomize_light_color_temperature(env, env_ids, temperature_range)



def randomize_object_size(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    size_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    # Check if sim is running
    if env.sim.is_playing():
        raise RuntimeError(
            "Randomizing scale while simulation is running leads to unpredictable behaviors."
            " Please ensure that the event term is called before the simulation starts by using the 'usd' mode."
        )

    # Extract the asset
    asset: RigidObject = env.scene[asset_cfg.name]

    # Resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # Acquire stage
    stage = omni.usd.get_context().get_stage()
    # Resolve prim paths
    prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)

    # Sample scale values for each dimension
    range_list = [size_range.get(key, (1.0, 1.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu")
    rand_samples = rand_samples.tolist()

    # Use sdf changeblock for faster processing
    with Sdf.ChangeBlock():
        for i, env_id in enumerate(env_ids):
            prim_path = prim_paths[env_id]
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

            # Get or create the scale attribute
            scale_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOp:scale")
            has_scale_attr = scale_spec is not None
            if not has_scale_attr:
                scale_spec = Sdf.AttributeSpec(prim_spec, prim_path + ".xformOp:scale", Sdf.ValueTypeNames.Double3)

            # Set the new scale
            scale_spec.default = Gf.Vec3f(*rand_samples[i])

            # Ensure correct transform ordering if we created the scale attribute
            if not has_scale_attr:
                op_order_spec = prim_spec.GetAttributeAtPath(prim_path + ".xformOpOrder")
                if op_order_spec is None:
                    op_order_spec = Sdf.AttributeSpec(
                        prim_spec, UsdGeom.Tokens.xformOpOrder, Sdf.ValueTypeNames.TokenArray
                    )
                op_order_spec.default = Vt.TokenArray(["xformOp:translate", "xformOp:orient", "xformOp:scale"])


def set_camera_rt_subframes(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    subframes: int = 4,
    camera_name: str = "gripper_camera",
):
    """Set RT subframes for camera to reduce ghosting."""
    try:
        import omni.usd
        from pxr import Sdf, UsdGeom
        
        stage = omni.usd.get_context().get_stage()
        
        # Resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=env.device)
            
        for env_id in env_ids:
            # Construct camera path based on your camera configuration
            camera_prim_path = f"/World/envs/env_{env_id}/Robot/M5_wrist_link/gripper_camera"
            camera_prim = stage.GetPrimAtPath(camera_prim_path)
            
            if camera_prim.IsValid():
                # First, check if the attribute already exists, if not create it
                attr = camera_prim.GetAttribute("rtx:subframes")
                if not attr:
                    # Create the attribute with integer type
                    attr = camera_prim.CreateAttribute("rtx:subframes", Sdf.ValueTypeNames.Int)
                
                # Now set the value
                attr.Set(subframes)
                print(f"Set RT subframes to {subframes} for camera in env {env_id}")
            else:
                print(f"Warning: Camera prim not found at {camera_prim_path}")
                
    except Exception as e:
        print(f"Error setting RT subframes: {e}")


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

    # set into the physics simulation
    asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


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


def reset_joints_by_offset_selective(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_range: tuple[float, float],
    velocity_range: tuple[float, float],
    joint_names: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset specific robot joints with offsets, leaving other joints at default values.

    Similar to reset_joints_by_offset but only randomizes specified joints.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # get default joint state
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = asset.data.default_joint_vel[env_ids].clone()

    # find indices of specified joints
    joint_indices = [asset.joint_names.index(name) for name in joint_names if name in asset.joint_names]

    # generate offsets only for specified joints
    num_envs = len(env_ids)
    num_joints = len(joint_indices)
    pos_offsets = math_utils.sample_uniform(*position_range, (num_envs, num_joints), device=joint_pos.device)
    vel_offsets = math_utils.sample_uniform(*velocity_range, (num_envs, num_joints), device=joint_vel.device)

    # apply offsets only to specified joints
    for i, joint_idx in enumerate(joint_indices):
        joint_pos[:, joint_idx] += pos_offsets[:, i]
        joint_vel[:, joint_idx] += vel_offsets[:, i]

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

        # ===============================                                                                                                                               
        # Reset joint position targets to default (prevents moving to old targets)                                                                                                    
        articulation_asset.data.joint_pos_target[env_ids] = default_joint_pos.clone()                                                                                                 
        articulation_asset._joint_pos_target_sim[env_ids] = default_joint_pos.clone()                                                                                                 
        # Reset velocity targets to zero                                                                                                                                              
        articulation_asset.data.joint_vel_target[env_ids] = 0.0                                                                                                                       
        articulation_asset._joint_vel_target_sim[env_ids] = 0.0                                                                                                                       
        # =============================================          
    
    # deformable objects
    for deformable_object in env.scene.deformable_objects.values():
        # obtain default and set into the physics simulation
        nodal_state = deformable_object.data.default_nodal_state_w[env_ids].clone()
        deformable_object.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)


def force_clear_object_visuals(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> None:
    """Force clear object visuals and reset to avoid ghost images.

    This function uses USD visibility controls to force refresh the object visuals.
    """
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    try:
        import omni.usd
        from pxr import UsdGeom

        asset = env.scene[asset_cfg.name]
        stage = omni.usd.get_context().get_stage()

        # First make objects invisible, then visible again to clear ghost images
        for env_id in env_ids:
            object_prim_path = asset.cfg.prim_path.replace("{ENV_REGEX_NS}", f"/World/envs/env_{env_id}")
            object_prim = stage.GetPrimAtPath(object_prim_path)

            if object_prim.IsValid():
                # Method 1: Use visibility attribute
                imageable = UsdGeom.Imageable(object_prim)
                if imageable:
                    # Make invisible
                    imageable.MakeInvisible()
                    # Immediately make visible again
                    imageable.MakeVisible()

                # Method 2: Force geometry refresh by toggling purpose
                geom_prim = UsdGeom.Gprim(object_prim)
                if geom_prim:
                    current_purpose = geom_prim.GetPurposeAttr().Get()
                    # Toggle purpose to force refresh
                    geom_prim.GetPurposeAttr().Set("proxy")
                    geom_prim.GetPurposeAttr().Set(current_purpose or "default")

        # Now do normal position reset
        reset_root_state_uniform(env, env_ids, pose_range, velocity_range, asset_cfg)

    except Exception as e:
        print(f"Warning: Visual clearing failed, using normal reset: {e}")
        reset_root_state_uniform(env, env_ids, pose_range, velocity_range, asset_cfg)


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


# def randomize_object_pool_selection(
#     env: ManagerBasedRLEnv,
#     env_ids: torch.Tensor | None,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
# ):
#     """Randomize which object from pool is active per environment."""
#     # Handle None env_ids (startup mode - apply to all environments)
#     if env_ids is None:
#         env_ids = torch.arange(env.num_envs, device=env.device)

#     object_collection = env.scene[asset_cfg.name]
#     num_objects = len(object_collection.object_names)

#     # Create buffer to store active object indices if not exists
#     if not hasattr(env, 'active_object_indices'):
#         env.active_object_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

#     for env_idx in env_ids:
#         active_idx = torch.randint(0, num_objects, (1,), device=env.device).item()
#         env.active_object_indices[env_idx] = active_idx

#         for obj_idx in range(num_objects):
#             pos = torch.tensor([0.28, 0.0, 0.0] if obj_idx == active_idx else [100.0, 100.0, -10.0], device=env.device)

#             root_state = object_collection.data.default_object_state[env_idx, obj_idx].clone()
#             root_state[:3] = pos + env.scene.env_origins[env_idx]
#             root_state[3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
#             root_state[7:] = 0.0

#             object_collection.write_object_state_to_sim(
#                 root_state.unsqueeze(0),
#                 env_ids=torch.tensor([env_idx], device=env.device),
#                 object_ids=torch.tensor([obj_idx], device=env.device)
#             )


def randomize_object_pool_selection(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
    balanced: bool = True,
):

    import omni.usd
    from pxr import UsdGeom, UsdPhysics, PhysxSchema

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    object_collection = env.scene[asset_cfg.name]
    num_objects = len(object_collection.object_names)
    stage = omni.usd.get_context().get_stage()

    if not hasattr(env, 'active_object_indices'):
        env.active_object_indices = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    num_env_ids = len(env_ids)

    if balanced and num_env_ids > 1:
        # Balanced sampling: ensure approximately equal distribution
        base_count = num_env_ids // num_objects
        remainder = num_env_ids % num_objects
        assignments = []
        for obj_idx in range(num_objects):
            count = base_count + (1 if obj_idx < remainder else 0)
            assignments.extend([obj_idx] * count)

        assignments = torch.tensor(assignments, device=env.device, dtype=torch.long)
        perm = torch.randperm(len(assignments), device=env.device)
        assignments = assignments[perm]
    else:
        # Pure random selection
        assignments = torch.randint(0, num_objects, (num_env_ids,), device=env.device)

    # Apply assignments to each environment
    for i, env_idx in enumerate(env_ids):
        active_idx = assignments[i].item()
        env.active_object_indices[env_idx] = active_idx

        for obj_idx in range(num_objects):
            obj_name = object_collection.object_names[obj_idx]
            obj_cfg = list(env.cfg.scene.object_pool.rigid_objects.values())[obj_idx]
            obj_prim_path = obj_cfg.prim_path.replace("{ENV_REGEX_NS}", f"/World/envs/env_{env_idx}")

            obj_prim = stage.GetPrimAtPath(obj_prim_path)

            if obj_idx == active_idx:
                # Enable active object
                if obj_prim.IsValid():
                    obj_prim.SetActive(True)
                    UsdGeom.Imageable(obj_prim).MakeVisible()

                    rigid_body_api = UsdPhysics.RigidBodyAPI(obj_prim)
                    if rigid_body_api:
                        rigid_body_api.GetRigidBodyEnabledAttr().Set(True)

                    collision_api = UsdPhysics.CollisionAPI(obj_prim)
                    if collision_api:
                        collision_api.GetCollisionEnabledAttr().Set(True)

                pos = torch.tensor([0.28, 0.0, 0.0], device=env.device)
            else:
                # Disable inactive object
                if obj_prim.IsValid():
                    obj_prim.SetActive(False)
                    UsdGeom.Imageable(obj_prim).MakeInvisible()
                    
                    rigid_body_api = UsdPhysics.RigidBodyAPI(obj_prim)
                    if rigid_body_api:
                        rigid_body_api.GetRigidBodyEnabledAttr().Set(False)
                    
                    collision_api = UsdPhysics.CollisionAPI(obj_prim)
                    if collision_api:
                        collision_api.GetCollisionEnabledAttr().Set(False)

                pos = torch.tensor([100.0, 100.0, -10.0], device=env.device)

            root_state = object_collection.data.default_object_state[env_idx, obj_idx].clone()
            root_state[:3] = pos + env.scene.env_origins[env_idx]
            root_state[3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
            root_state[7:] = 0.0

            object_collection.write_object_state_to_sim(
                root_state.unsqueeze(0),
                env_ids=torch.tensor([env_idx], device=env.device),
                object_ids=torch.tensor([obj_idx], device=env.device)
            )


def reset_object_pool_state_uniform_for_object(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_names: str | list[str],
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    yaw_ranges: list[tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
):
    """Reset active pool objects only for environments where one of ``object_names`` is active.

    This allows per-object pose randomization ranges on top of the default
    ``reset_object_pool_state_uniform`` event. Pass either a single object name
    or a list of object names that should share the same pose randomization range.
    When ``yaw_ranges`` is provided, yaw is sampled from one of those disjoint
    intervals instead of the continuous ``pose_range["yaw"]`` interval.
    """
    from isaaclab.assets import RigidObjectCollection
    import isaaclab.utils.math as math_utils

    if not hasattr(env, "active_object_indices"):
        raise RuntimeError("active_object_indices not found")

    object_collection: RigidObjectCollection = env.scene[asset_cfg.name]

    if isinstance(object_names, str):
        object_names = [object_names]

    invalid_object_names = [name for name in object_names if name not in object_collection.object_names]
    if invalid_object_names:
        raise ValueError(
            f"Object(s) {invalid_object_names} not found in pool. Available: {object_collection.object_names}"
        )

    target_obj_ids = torch.tensor(
        [object_collection.object_names.index(name) for name in object_names],
        device=object_collection.device,
        dtype=torch.long,
    )

    # Filter env_ids to only those where one of the target objects is active.
    active_for_env = env.active_object_indices[env_ids]
    mask = (active_for_env[:, None] == target_obj_ids[None, :]).any(dim=1)
    filtered_env_ids = env_ids[mask]
    filtered_obj_ids = active_for_env[mask]
    if len(filtered_env_ids) == 0:
        return

    # Sample pose randomization for the filtered envs.
    pose_range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    pose_ranges = torch.tensor(pose_range_list, device=object_collection.device)
    pose_rand_samples = math_utils.sample_uniform(
        pose_ranges[:, 0], pose_ranges[:, 1], (len(filtered_env_ids), 6), device=object_collection.device
    )

    if yaw_ranges is not None:
        yaw_ranges_tensor = torch.tensor(yaw_ranges, device=object_collection.device, dtype=pose_rand_samples.dtype)
        yaw_range_indices = torch.randint(
            0, yaw_ranges_tensor.shape[0], (len(filtered_env_ids),), device=object_collection.device
        )
        selected_yaw_ranges = yaw_ranges_tensor[yaw_range_indices]
        pose_rand_samples[:, 5] = torch.rand(
            len(filtered_env_ids), device=object_collection.device, dtype=pose_rand_samples.dtype
        ) * (selected_yaw_ranges[:, 1] - selected_yaw_ranges[:, 0]) + selected_yaw_ranges[:, 0]
    vel_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    vel_ranges = torch.tensor(vel_range_list, device=object_collection.device)
    vel_rand_samples = math_utils.sample_uniform(
        vel_ranges[:, 0], vel_ranges[:, 1], (len(filtered_env_ids), 6), device=object_collection.device
    )

    for idx, (env_idx, target_obj_idx) in enumerate(zip(filtered_env_ids, filtered_obj_ids)):
        root_state = object_collection.data.default_object_state[env_idx, target_obj_idx].clone()
        position = root_state[0:3] + env.scene.env_origins[env_idx] + pose_rand_samples[idx, 0:3]
        orientation_delta = math_utils.quat_from_euler_xyz(
            pose_rand_samples[idx, 3], pose_rand_samples[idx, 4], pose_rand_samples[idx, 5]
        )
        orientation = math_utils.quat_mul(root_state[3:7], orientation_delta)
        velocity = root_state[7:13] + vel_rand_samples[idx]
        new_state = torch.cat([position, orientation, velocity], dim=-1)
        object_collection.write_object_state_to_sim(
            new_state.unsqueeze(0),
            env_ids=env_idx.unsqueeze(0),
            object_ids=target_obj_idx.unsqueeze(0),
        )


def reset_object_pool_state_uniform_for_object1(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_names: str | list[str],
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    yaw_ranges: list[tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
):
    """Reset active pool objects only for environments where one of ``object_names`` is active.

    This allows per-object pose randomization ranges on top of the default
    ``reset_object_pool_state_uniform`` event. Pass either a single object name
    or a list of object names that should share the same pose randomization range.
    When ``yaw_ranges`` is provided, yaw is sampled from one of those disjoint
    intervals instead of the continuous ``pose_range["yaw"]`` interval.
    """
    from isaaclab.assets import RigidObjectCollection
    import isaaclab.utils.math as math_utils

    if not hasattr(env, "active_object_indices"):
        raise RuntimeError("active_object_indices not found")

    object_collection: RigidObjectCollection = env.scene[asset_cfg.name]

    if isinstance(object_names, str):
        object_names = [object_names]

    invalid_object_names = [name for name in object_names if name not in object_collection.object_names]
    if invalid_object_names:
        raise ValueError(
            f"Object(s) {invalid_object_names} not found in pool. Available: {object_collection.object_names}"
        )

    target_obj_ids = torch.tensor(
        [object_collection.object_names.index(name) for name in object_names],
        device=object_collection.device,
        dtype=torch.long,
    )

    # Filter env_ids to only those where one of the target objects is active.
    active_for_env = env.active_object_indices[env_ids]
    mask = (active_for_env[:, None] == target_obj_ids[None, :]).any(dim=1)
    filtered_env_ids = env_ids[mask]
    filtered_obj_ids = active_for_env[mask]
    if len(filtered_env_ids) == 0:
        return

    # Sample pose randomization for the filtered envs.
    pose_range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    pose_ranges = torch.tensor(pose_range_list, device=object_collection.device)
    pose_rand_samples = math_utils.sample_uniform(
        pose_ranges[:, 0], pose_ranges[:, 1], (len(filtered_env_ids), 6), device=object_collection.device
    )

    if yaw_ranges is not None:
        yaw_ranges_tensor = torch.tensor(yaw_ranges, device=object_collection.device, dtype=pose_rand_samples.dtype)
        yaw_range_indices = torch.randint(
            0, yaw_ranges_tensor.shape[0], (len(filtered_env_ids),), device=object_collection.device
        )
        selected_yaw_ranges = yaw_ranges_tensor[yaw_range_indices]
        pose_rand_samples[:, 5] = torch.rand(
            len(filtered_env_ids), device=object_collection.device, dtype=pose_rand_samples.dtype
        ) * (selected_yaw_ranges[:, 1] - selected_yaw_ranges[:, 0]) + selected_yaw_ranges[:, 0]
    vel_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    vel_ranges = torch.tensor(vel_range_list, device=object_collection.device)
    vel_rand_samples = math_utils.sample_uniform(
        vel_ranges[:, 0], vel_ranges[:, 1], (len(filtered_env_ids), 6), device=object_collection.device
    )

    for idx, (env_idx, target_obj_idx) in enumerate(zip(filtered_env_ids, filtered_obj_ids)):
        root_state = object_collection.data.default_object_state[env_idx, target_obj_idx].clone()
        position = root_state[0:3] + env.scene.env_origins[env_idx] + pose_rand_samples[idx, 0:3]
        orientation_delta = math_utils.quat_from_euler_xyz(
            pose_rand_samples[idx, 3], pose_rand_samples[idx, 4], pose_rand_samples[idx, 5]
        )
        orientation = math_utils.quat_mul(root_state[3:7], orientation_delta)
        velocity = root_state[7:13] + vel_rand_samples[idx]
        new_state = torch.cat([position, orientation, velocity], dim=-1)
        object_collection.write_object_state_to_sim(
            new_state.unsqueeze(0),
            env_ids=env_idx.unsqueeze(0),
            object_ids=target_obj_idx.unsqueeze(0),
        )


def reset_object_pool_state_uniform_for_object2(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_names: str | list[str],
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    yaw_ranges: list[tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
):
    """Reset active pool objects only for environments where one of ``object_names`` is active.

    This allows per-object pose randomization ranges on top of the default
    ``reset_object_pool_state_uniform`` event. Pass either a single object name
    or a list of object names that should share the same pose randomization range.
    When ``yaw_ranges`` is provided, yaw is sampled from one of those disjoint
    intervals instead of the continuous ``pose_range["yaw"]`` interval.
    """
    from isaaclab.assets import RigidObjectCollection
    import isaaclab.utils.math as math_utils

    if not hasattr(env, "active_object_indices"):
        raise RuntimeError("active_object_indices not found")

    object_collection: RigidObjectCollection = env.scene[asset_cfg.name]

    if isinstance(object_names, str):
        object_names = [object_names]

    invalid_object_names = [name for name in object_names if name not in object_collection.object_names]
    if invalid_object_names:
        raise ValueError(
            f"Object(s) {invalid_object_names} not found in pool. Available: {object_collection.object_names}"
        )

    target_obj_ids = torch.tensor(
        [object_collection.object_names.index(name) for name in object_names],
        device=object_collection.device,
        dtype=torch.long,
    )

    # Filter env_ids to only those where one of the target objects is active.
    active_for_env = env.active_object_indices[env_ids]
    mask = (active_for_env[:, None] == target_obj_ids[None, :]).any(dim=1)
    filtered_env_ids = env_ids[mask]
    filtered_obj_ids = active_for_env[mask]
    if len(filtered_env_ids) == 0:
        return

    # Sample pose randomization for the filtered envs.
    pose_range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    pose_ranges = torch.tensor(pose_range_list, device=object_collection.device)
    pose_rand_samples = math_utils.sample_uniform(
        pose_ranges[:, 0], pose_ranges[:, 1], (len(filtered_env_ids), 6), device=object_collection.device
    )

    if yaw_ranges is not None:
        yaw_ranges_tensor = torch.tensor(yaw_ranges, device=object_collection.device, dtype=pose_rand_samples.dtype)
        yaw_range_indices = torch.randint(
            0, yaw_ranges_tensor.shape[0], (len(filtered_env_ids),), device=object_collection.device
        )
        selected_yaw_ranges = yaw_ranges_tensor[yaw_range_indices]
        pose_rand_samples[:, 5] = torch.rand(
            len(filtered_env_ids), device=object_collection.device, dtype=pose_rand_samples.dtype
        ) * (selected_yaw_ranges[:, 1] - selected_yaw_ranges[:, 0]) + selected_yaw_ranges[:, 0]
    vel_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    vel_ranges = torch.tensor(vel_range_list, device=object_collection.device)
    vel_rand_samples = math_utils.sample_uniform(
        vel_ranges[:, 0], vel_ranges[:, 1], (len(filtered_env_ids), 6), device=object_collection.device
    )

    for idx, (env_idx, target_obj_idx) in enumerate(zip(filtered_env_ids, filtered_obj_ids)):
        root_state = object_collection.data.default_object_state[env_idx, target_obj_idx].clone()
        position = root_state[0:3] + env.scene.env_origins[env_idx] + pose_rand_samples[idx, 0:3]
        orientation_delta = math_utils.quat_from_euler_xyz(
            pose_rand_samples[idx, 3], pose_rand_samples[idx, 4], pose_rand_samples[idx, 5]
        )
        orientation = math_utils.quat_mul(root_state[3:7], orientation_delta)
        velocity = root_state[7:13] + vel_rand_samples[idx]
        new_state = torch.cat([position, orientation, velocity], dim=-1)
        object_collection.write_object_state_to_sim(
            new_state.unsqueeze(0),
            env_ids=env_idx.unsqueeze(0),
            object_ids=target_obj_idx.unsqueeze(0),
        )


def reset_object_pool_state_uniform_for_object3(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_names: str | list[str],
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    yaw_ranges: list[tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
):
    """Reset active pool objects only for environments where one of ``object_names`` is active.

    This allows per-object pose randomization ranges on top of the default
    ``reset_object_pool_state_uniform`` event. Pass either a single object name
    or a list of object names that should share the same pose randomization range.
    When ``yaw_ranges`` is provided, yaw is sampled from one of those disjoint
    intervals instead of the continuous ``pose_range["yaw"]`` interval.
    """
    from isaaclab.assets import RigidObjectCollection
    import isaaclab.utils.math as math_utils

    if not hasattr(env, "active_object_indices"):
        raise RuntimeError("active_object_indices not found")

    object_collection: RigidObjectCollection = env.scene[asset_cfg.name]

    if isinstance(object_names, str):
        object_names = [object_names]

    invalid_object_names = [name for name in object_names if name not in object_collection.object_names]
    if invalid_object_names:
        raise ValueError(
            f"Object(s) {invalid_object_names} not found in pool. Available: {object_collection.object_names}"
        )

    target_obj_ids = torch.tensor(
        [object_collection.object_names.index(name) for name in object_names],
        device=object_collection.device,
        dtype=torch.long,
    )

    # Filter env_ids to only those where one of the target objects is active.
    active_for_env = env.active_object_indices[env_ids]
    mask = (active_for_env[:, None] == target_obj_ids[None, :]).any(dim=1)
    filtered_env_ids = env_ids[mask]
    filtered_obj_ids = active_for_env[mask]
    if len(filtered_env_ids) == 0:
        return

    # Sample pose randomization for the filtered envs.
    pose_range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    pose_ranges = torch.tensor(pose_range_list, device=object_collection.device)
    pose_rand_samples = math_utils.sample_uniform(
        pose_ranges[:, 0], pose_ranges[:, 1], (len(filtered_env_ids), 6), device=object_collection.device
    )

    if yaw_ranges is not None:
        yaw_ranges_tensor = torch.tensor(yaw_ranges, device=object_collection.device, dtype=pose_rand_samples.dtype)
        yaw_range_indices = torch.randint(
            0, yaw_ranges_tensor.shape[0], (len(filtered_env_ids),), device=object_collection.device
        )
        selected_yaw_ranges = yaw_ranges_tensor[yaw_range_indices]
        pose_rand_samples[:, 5] = torch.rand(
            len(filtered_env_ids), device=object_collection.device, dtype=pose_rand_samples.dtype
        ) * (selected_yaw_ranges[:, 1] - selected_yaw_ranges[:, 0]) + selected_yaw_ranges[:, 0]
    vel_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    vel_ranges = torch.tensor(vel_range_list, device=object_collection.device)
    vel_rand_samples = math_utils.sample_uniform(
        vel_ranges[:, 0], vel_ranges[:, 1], (len(filtered_env_ids), 6), device=object_collection.device
    )

    for idx, (env_idx, target_obj_idx) in enumerate(zip(filtered_env_ids, filtered_obj_ids)):
        root_state = object_collection.data.default_object_state[env_idx, target_obj_idx].clone()
        position = root_state[0:3] + env.scene.env_origins[env_idx] + pose_rand_samples[idx, 0:3]
        orientation_delta = math_utils.quat_from_euler_xyz(
            pose_rand_samples[idx, 3], pose_rand_samples[idx, 4], pose_rand_samples[idx, 5]
        )
        orientation = math_utils.quat_mul(root_state[3:7], orientation_delta)
        velocity = root_state[7:13] + vel_rand_samples[idx]
        new_state = torch.cat([position, orientation, velocity], dim=-1)
        object_collection.write_object_state_to_sim(
            new_state.unsqueeze(0),
            env_ids=env_idx.unsqueeze(0),
            object_ids=target_obj_idx.unsqueeze(0),
        )


def reset_object_pool_state_uniform_for_object4(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_names: str | list[str],
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    yaw_ranges: list[tuple[float, float]] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
):
    """Reset active pool objects only for environments where one of ``object_names`` is active.

    This allows per-object pose randomization ranges on top of the default
    ``reset_object_pool_state_uniform`` event. Pass either a single object name
    or a list of object names that should share the same pose randomization range.
    When ``yaw_ranges`` is provided, yaw is sampled from one of those disjoint
    intervals instead of the continuous ``pose_range["yaw"]`` interval.
    """
    from isaaclab.assets import RigidObjectCollection
    import isaaclab.utils.math as math_utils

    if not hasattr(env, "active_object_indices"):
        raise RuntimeError("active_object_indices not found")

    object_collection: RigidObjectCollection = env.scene[asset_cfg.name]

    if isinstance(object_names, str):
        object_names = [object_names]

    invalid_object_names = [name for name in object_names if name not in object_collection.object_names]
    if invalid_object_names:
        raise ValueError(
            f"Object(s) {invalid_object_names} not found in pool. Available: {object_collection.object_names}"
        )

    target_obj_ids = torch.tensor(
        [object_collection.object_names.index(name) for name in object_names],
        device=object_collection.device,
        dtype=torch.long,
    )

    # Filter env_ids to only those where one of the target objects is active.
    active_for_env = env.active_object_indices[env_ids]
    mask = (active_for_env[:, None] == target_obj_ids[None, :]).any(dim=1)
    filtered_env_ids = env_ids[mask]
    filtered_obj_ids = active_for_env[mask]
    if len(filtered_env_ids) == 0:
        return

    # Sample pose randomization for the filtered envs.
    pose_range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    pose_ranges = torch.tensor(pose_range_list, device=object_collection.device)
    pose_rand_samples = math_utils.sample_uniform(
        pose_ranges[:, 0], pose_ranges[:, 1], (len(filtered_env_ids), 6), device=object_collection.device
    )

    if yaw_ranges is not None:
        yaw_ranges_tensor = torch.tensor(yaw_ranges, device=object_collection.device, dtype=pose_rand_samples.dtype)
        yaw_range_indices = torch.randint(
            0, yaw_ranges_tensor.shape[0], (len(filtered_env_ids),), device=object_collection.device
        )
        selected_yaw_ranges = yaw_ranges_tensor[yaw_range_indices]
        pose_rand_samples[:, 5] = torch.rand(
            len(filtered_env_ids), device=object_collection.device, dtype=pose_rand_samples.dtype
        ) * (selected_yaw_ranges[:, 1] - selected_yaw_ranges[:, 0]) + selected_yaw_ranges[:, 0]
    vel_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    vel_ranges = torch.tensor(vel_range_list, device=object_collection.device)
    vel_rand_samples = math_utils.sample_uniform(
        vel_ranges[:, 0], vel_ranges[:, 1], (len(filtered_env_ids), 6), device=object_collection.device
    )

    for idx, (env_idx, target_obj_idx) in enumerate(zip(filtered_env_ids, filtered_obj_ids)):
        root_state = object_collection.data.default_object_state[env_idx, target_obj_idx].clone()
        position = root_state[0:3] + env.scene.env_origins[env_idx] + pose_rand_samples[idx, 0:3]
        orientation_delta = math_utils.quat_from_euler_xyz(
            pose_rand_samples[idx, 3], pose_rand_samples[idx, 4], pose_rand_samples[idx, 5]
        )
        orientation = math_utils.quat_mul(root_state[3:7], orientation_delta)
        velocity = root_state[7:13] + vel_rand_samples[idx]
        new_state = torch.cat([position, orientation, velocity], dim=-1)
        object_collection.write_object_state_to_sim(
            new_state.unsqueeze(0),
            env_ids=env_idx.unsqueeze(0),
            object_ids=target_obj_idx.unsqueeze(0),
        )


def initialize_point_cloud_cache(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
):
    """
    Initialize point cloud cache storage.
    Called once at startup (env_ids will be None).
    """
    # Initialize cache attributes
    env.point_cloud_cache = None
    env._pcd_cache_step = -1
    print("[INFO] Point cloud cache initialized")


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


def load_texture_files_from_txt(txt_path: str) -> list[str]:
    texture_files = []
    with open(txt_path, "r", encoding="utf-8") as f:  # 指定utf-8编码
        for line in f:
            line = line.strip()
            if line:
                texture_files.append(line)
    return texture_files


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
                scale = np.random.uniform(0.5, 2.0)
                light_prim.GetAttribute("xformOp:scale").Set((scale, scale, scale))
                # 随机色温
                temp = float(np.random.uniform(2000, 10000))
                light_prim.GetAttribute("inputs:colorTemperature").Set(temp)
                # 随机强度
                intensity = float(np.random.uniform(1000, 8000))
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


def randomize_global_sphere_lights(
    env: "ManagerBasedEnv",
    env_ids: torch.Tensor,
    light_paths: list[str],
    intensity_range: tuple[float, float] = (50000.0, 1000000.0),
    temperature_range: tuple[float, float] = (2500.0, 11000.0),
    color_variation: float = 0.3,
    position_variation: tuple[float, float, float] = (5.0, 5.0, 3.0),
) -> None:
    """Randomize global sphere light properties (intensity, color temperature, color, position).

    This function randomizes global lights that are shared across all environments,
    which is more memory efficient than per-environment lights.

    Best used with mode="interval" and is_global_time=True for time-based randomization.

    Args:
        env: Environment instance.
        env_ids: Environment indices.
        light_paths: List of USD paths to the global sphere lights.
        intensity_range: Min and max intensity values for randomization.
        temperature_range: Min and max color temperature values (Kelvin).
            - 2500K: warm candlelight
            - 5500K: daylight
            - 11000K: cool blue sky
        color_variation: Maximum deviation from white (1.0) for RGB color channels.
        position_variation: Max offset (x, y, z) from original position for each light.
    """
    try:
        import omni.usd
        from pxr import Gf

        stage = omni.usd.get_context().get_stage()

        # Base positions for each light (matching the config)
        base_positions = {
            "/World/GlobalLight_0": (0.0, 0.0, 25.0),
            "/World/GlobalLight_1": (-20.0, -20.0, 20.0),
            "/World/GlobalLight_2": (20.0, 20.0, 20.0),
        }

        for light_path in light_paths:
            light_prim = stage.GetPrimAtPath(light_path)

            if not light_prim.IsValid():
                print(f"[Warning] Global light not found at {light_path}")
                continue

            # Randomize intensity (dramatic range: dim to very bright)
            intensity = float(np.random.uniform(intensity_range[0], intensity_range[1]))
            intensity_attr = light_prim.GetAttribute("inputs:intensity")
            if intensity_attr:
                intensity_attr.Set(intensity)

            # Ensure color temperature is enabled
            enable_temp_attr = light_prim.GetAttribute("inputs:enableColorTemperature")
            if enable_temp_attr:
                enable_temp_attr.Set(True)

            # Randomize color temperature (warm orange to cool blue)
            temp = float(np.random.uniform(temperature_range[0], temperature_range[1]))
            temp_attr = light_prim.GetAttribute("inputs:colorTemperature")
            if temp_attr:
                temp_attr.Set(temp)

            # Randomize color with more dramatic variation
            color = (
                float(np.clip(np.random.uniform(1.0 - color_variation, 1.0 + color_variation * 0.5), 0.5, 1.0)),
                float(np.clip(np.random.uniform(1.0 - color_variation, 1.0 + color_variation * 0.5), 0.5, 1.0)),
                float(np.clip(np.random.uniform(1.0 - color_variation, 1.0 + color_variation * 0.5), 0.5, 1.0)),
            )
            color_attr = light_prim.GetAttribute("inputs:color")
            if color_attr:
                color_attr.Set(Gf.Vec3f(*color))

            # Randomize position around base position
            if light_path in base_positions:
                base_pos = base_positions[light_path]
                new_pos = Gf.Vec3d(
                    base_pos[0] + np.random.uniform(-position_variation[0], position_variation[0]),
                    base_pos[1] + np.random.uniform(-position_variation[1], position_variation[1]),
                    base_pos[2] + np.random.uniform(-position_variation[2], position_variation[2]),
                )
                translate_attr = light_prim.GetAttribute("xformOp:translate")
                if translate_attr:
                    translate_attr.Set(new_pos)

    except Exception as e:
        print(f"[Warning] Global sphere light randomization failed: {e}")
        import traceback
        traceback.print_exc()


def log_object_distribution(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
):
    """Log the distribution of objects across environments at startup."""
    if not hasattr(env, 'active_object_indices'):
        print("⚠️  Warning: active_object_indices not found!")
        return

    object_collection = env.scene[asset_cfg.name]
    object_names = object_collection.object_names
    num_objects = len(object_names)

    print("\n" + "="*60)
    print("🎯 Object Distribution Across Environments:")
    print("="*60)

    for obj_idx in range(num_objects):
        count = (env.active_object_indices == obj_idx).sum().item()
        percentage = (count / env.num_envs) * 100
        obj_name = object_names[obj_idx]
        print(f"  {obj_name:20s}: {count:3d} envs ({percentage:5.1f}%)")

    print("="*60 + "\n")