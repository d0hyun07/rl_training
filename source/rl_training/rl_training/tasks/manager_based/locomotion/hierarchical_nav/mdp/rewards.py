# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def progress_to_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "goal_command",
) -> torch.Tensor:
    """Reward for making progress towards the goal.
    
    This reward encourages the robot to reduce the distance to the goal.
    It is computed as the negative change in distance (positive when getting closer).
    
    Args:
        env: The environment instance
        asset_cfg: Configuration for the robot asset
        command_name: Name of the goal command
        
    Returns:
        Progress reward tensor of shape [num_envs]
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    robot_pos = asset.data.root_pos_w[:, :2]
    goal_pos = env.command_manager.get_command(command_name)[:, :2]
    current_distance = torch.norm(robot_pos - goal_pos, dim=1)
    
    # Get previous distance from environment (stored by environment)
    if hasattr(env, "_prev_distance_to_goal") and env._prev_distance_to_goal is not None:
        prev_distance = env._prev_distance_to_goal
        progress = prev_distance - current_distance
    else:
        # First step, no progress yet
        progress = torch.zeros_like(current_distance)
    
    # Store current distance for next step
    env._prev_distance_to_goal = current_distance.clone()
    
    return progress


def goal_reached_bonus(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "goal_command",
) -> torch.Tensor:
    """Bonus reward for reaching the goal.
    
    Args:
        env: The environment instance
        threshold: Distance threshold for goal reached (meters)
        asset_cfg: Configuration for the robot asset
        command_name: Name of the goal command
        
    Returns:
        Bonus reward tensor of shape [num_envs] (1.0 if reached, 0.0 otherwise)
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    robot_pos = asset.data.root_pos_w[:, :2]
    goal_pos = env.command_manager.get_command(command_name)[:, :2]
    distance = torch.norm(robot_pos - goal_pos, dim=1)
    reached = (distance < threshold).float()
    return reached


def timeout_penalty(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    """Penalty for episode timeout.
    
    This penalty is applied when the episode terminates due to timeout
    (i.e., robot didn't reach the goal in time).
    
    Args:
        env: The environment instance
        
    Returns:
        Penalty tensor of shape [num_envs] (-1.0 if timeout, 0.0 otherwise)
    """
    # Check if episode is near timeout (90% of max episode length)
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        return torch.zeros(env.num_envs, device=env.device)
    
    if not hasattr(env, "max_episode_length"):
        return torch.zeros(env.num_envs, device=env.device)
    
    # Check if we're in the last 10% of the episode
    timeout_threshold = env.max_episode_length * 0.9
    near_timeout = (env.episode_length_buf.float() >= timeout_threshold).float()
    
    # Only apply penalty if goal is not reached
    asset: RigidObject = env.scene["robot"]
    robot_pos = asset.data.root_pos_w[:, :2]
    goal_pos = env.command_manager.get_command("goal_command")[:, :2]
    distance = torch.norm(robot_pos - goal_pos, dim=1)
    goal_not_reached = (distance > 1.0).float()
    
    penalty = -near_timeout * goal_not_reached
    return penalty
