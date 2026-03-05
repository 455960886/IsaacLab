import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from .rewards import get_active_object_states


def active_object_quat_w(env, object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool")):
    _, quat_w = get_active_object_states(env, object_cfg)
    return quat_w  # (N,4) wxyz


def obj_pos_rel_ee(env,
                   object_cfg: SceneEntityCfg = SceneEntityCfg("object_pool"),
                   ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame")):
    obj_pos_w, _ = get_active_object_states(env, object_cfg)
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]  # (N,3)
    return obj_pos_w - ee_pos_w


def contact_flags_y(env,
                    left_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_left"),
                    right_sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces_right"),
                    thr: float = 1.5):
    left = env.scene.sensors[left_sensor_cfg.name]
    right = env.scene.sensors[right_sensor_cfg.name]
    if left.data.net_forces_w is None or right.data.net_forces_w is None:
        return torch.zeros((env.num_envs, 2), device=env.device)
    ly = torch.abs(left.data.net_forces_w[:, 0, 1])
    ry = torch.abs(right.data.net_forces_w[:, 0, 1])
    return torch.stack([(ly > thr).float(), (ry > thr).float()], dim=-1)


def joint_pos(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return env.scene[asset_cfg.name].data.joint_pos


def joint_vel(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return env.scene[asset_cfg.name].data.joint_vel
