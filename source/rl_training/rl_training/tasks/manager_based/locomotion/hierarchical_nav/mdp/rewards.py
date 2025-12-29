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
    
    # Check for nan/inf values and replace with zeros
    robot_pos = torch.where(torch.isfinite(robot_pos), robot_pos, torch.zeros_like(robot_pos))
    goal_pos = torch.where(torch.isfinite(goal_pos), goal_pos, torch.zeros_like(goal_pos))
    
    current_distance = torch.norm(robot_pos - goal_pos, dim=1)
    # Clamp distance to avoid inf/nan
    current_distance = torch.clamp(current_distance, min=0.0, max=1e6)
    
    # Get previous distance from environment (stored by environment)
    if hasattr(env, "_prev_distance_to_goal") and env._prev_distance_to_goal is not None:
        prev_distance = env._prev_distance_to_goal
        # Ensure prev_distance is also finite
        prev_distance = torch.where(torch.isfinite(prev_distance), prev_distance, current_distance)
        progress = prev_distance - current_distance
        # Clamp progress to avoid extreme values
        progress = torch.clamp(progress, min=-10.0, max=10.0)
    else:
        # First step, no progress yet
        progress = torch.zeros_like(current_distance)
    
    # Store current distance for next step (ensure it's finite)
    env._prev_distance_to_goal = torch.where(
        torch.isfinite(current_distance), 
        current_distance.clone(), 
        torch.zeros_like(current_distance)
    )
    
    # Final check: replace any nan/inf with zero
    progress = torch.where(torch.isfinite(progress), progress, torch.zeros_like(progress))
    
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
    
    # Check for nan/inf values and replace with zeros
    robot_pos = torch.where(torch.isfinite(robot_pos), robot_pos, torch.zeros_like(robot_pos))
    goal_pos = torch.where(torch.isfinite(goal_pos), goal_pos, torch.zeros_like(goal_pos))
    
    distance = torch.norm(robot_pos - goal_pos, dim=1)
    # Clamp distance to avoid inf/nan
    distance = torch.clamp(distance, min=0.0, max=1e6)
    reached = (distance < threshold).float()
    
    # Final check: replace any nan/inf with zero
    reached = torch.where(torch.isfinite(reached), reached, torch.zeros_like(reached))
    
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
    
    # Check for nan/inf values and replace with zeros
    robot_pos = torch.where(torch.isfinite(robot_pos), robot_pos, torch.zeros_like(robot_pos))
    goal_pos = torch.where(torch.isfinite(goal_pos), goal_pos, torch.zeros_like(goal_pos))
    
    distance = torch.norm(robot_pos - goal_pos, dim=1)
    # Clamp distance to avoid inf/nan
    distance = torch.clamp(distance, min=0.0, max=1e6)
    goal_not_reached = (distance > 1.0).float()
    
    penalty = -near_timeout * goal_not_reached
    
    # Final check: replace any nan/inf with zero
    penalty = torch.where(torch.isfinite(penalty), penalty, torch.zeros_like(penalty))
    
    return penalty
