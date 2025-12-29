# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def robot_position_2d(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robot position in 2D (x, y) in world frame.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot asset
        
    Returns:
        Robot position tensor of shape [num_envs, 2] (x, y)
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, :2]


def robot_heading(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Robot heading (yaw) angle in world frame.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot asset
        
    Returns:
        Robot yaw tensor of shape [num_envs, 1] (radians)
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    # Extract yaw from quaternion (w, x, y, z order in Isaac Lab)
    # Using formula: yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
    yaw = torch.atan2(
        2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
        1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2)
    )
    return yaw.unsqueeze(-1)


def goal_position_2d(
    env: ManagerBasedRLEnv,
    command_name: str = "goal_command",
) -> torch.Tensor:
    """Goal position in 2D (x, y) in world frame.
    
    Args:
        env: The environment instance
        command_name: Name of the goal command
        
    Returns:
        Goal position tensor of shape [num_envs, 2] (x, y)
    """
    goal_cmd = env.command_manager.get_command(command_name)
    return goal_cmd[:, :2]


def distance_to_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "goal_command",
) -> torch.Tensor:
    """Distance from robot to goal.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot asset
        command_name: Name of the goal command
        
    Returns:
        Distance tensor of shape [num_envs, 1] (meters)
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    robot_pos = asset.data.root_pos_w[:, :2]
    goal_pos = env.command_manager.get_command(command_name)[:, :2]
    distance = torch.norm(robot_pos - goal_pos, dim=1, keepdim=True)
    return distance


def direction_to_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "goal_command",
) -> torch.Tensor:
    """Normalized direction vector from robot to goal.
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot asset
        command_name: Name of the goal command
        
    Returns:
        Direction tensor of shape [num_envs, 2] (normalized vector)
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    robot_pos = asset.data.root_pos_w[:, :2]
    goal_pos = env.command_manager.get_command(command_name)[:, :2]
    direction = goal_pos - robot_pos
    distance = torch.norm(direction, dim=1, keepdim=True)
    # Normalize direction, avoid division by zero
    direction_normalized = direction / torch.clamp(distance, min=1e-8)
    return direction_normalized
