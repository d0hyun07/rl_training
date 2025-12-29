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
    robot_pos = asset.data.root_pos_w[:, :2]
    # Check for nan/inf values and replace with zeros
    robot_pos = torch.where(torch.isfinite(robot_pos), robot_pos, torch.zeros_like(robot_pos))
    return robot_pos


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
    
    # Check for nan/inf values in quaternion
    quat = torch.where(torch.isfinite(quat), quat, torch.tensor([1.0, 0.0, 0.0, 0.0], device=quat.device).expand_as(quat))
    
    # Extract yaw from quaternion (w, x, y, z order in Isaac Lab)
    # Using formula: yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
    yaw = torch.atan2(
        2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
        1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2)
    )
    
    # Clamp yaw to valid range and check for nan/inf
    yaw = torch.clamp(yaw, min=-3.14159, max=3.14159)
    yaw = torch.where(torch.isfinite(yaw), yaw, torch.zeros_like(yaw))
    
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
    goal_pos = goal_cmd[:, :2]
    # Check for nan/inf values and replace with zeros
    goal_pos = torch.where(torch.isfinite(goal_pos), goal_pos, torch.zeros_like(goal_pos))
    return goal_pos


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
    
    # Check for nan/inf values and replace with zeros
    robot_pos = torch.where(torch.isfinite(robot_pos), robot_pos, torch.zeros_like(robot_pos))
    goal_pos = torch.where(torch.isfinite(goal_pos), goal_pos, torch.zeros_like(goal_pos))
    
    distance = torch.norm(robot_pos - goal_pos, dim=1, keepdim=True)
    # Clamp distance to avoid inf/nan
    distance = torch.clamp(distance, min=0.0, max=1e6)
    
    # Final check: replace any nan/inf with zero
    distance = torch.where(torch.isfinite(distance), distance, torch.zeros_like(distance))
    
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
    
    # Check for nan/inf values and replace with zeros
    robot_pos = torch.where(torch.isfinite(robot_pos), robot_pos, torch.zeros_like(robot_pos))
    goal_pos = torch.where(torch.isfinite(goal_pos), goal_pos, torch.zeros_like(goal_pos))
    
    direction = goal_pos - robot_pos
    distance = torch.norm(direction, dim=1, keepdim=True)
    # Clamp distance to avoid inf/nan
    distance = torch.clamp(distance, min=1e-8, max=1e6)
    # Normalize direction, avoid division by zero
    direction_normalized = direction / distance
    
    # Final check: replace any nan/inf with zero
    direction_normalized = torch.where(
        torch.isfinite(direction_normalized), 
        direction_normalized, 
        torch.zeros_like(direction_normalized)
    )
    
    return direction_normalized
