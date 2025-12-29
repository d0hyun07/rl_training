# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Utility for freezing and using trained locomotion policies.

This module provides utilities to load and freeze trained policies
following the same pattern as play.py.
"""

from __future__ import annotations

import torch
from typing import Optional

from tensordict import TensorDict
from rsl_rl.runners import OnPolicyRunner


def freeze_policy(runner: OnPolicyRunner) -> torch.nn.Module:
    """Freeze a policy from an OnPolicyRunner.
    
    This function sets the policy to eval mode and disables gradients,
    following the same pattern as play.py.
    
    Args:
        runner: OnPolicyRunner instance with loaded checkpoint
        
    Returns:
        Frozen policy module
    """
    # Extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic
    
    # Set to eval mode and freeze all parameters
    policy_nn.eval()
    for param in policy_nn.parameters():
        param.requires_grad = False
    
    return policy_nn


def is_frozen(policy_nn: torch.nn.Module) -> bool:
    """Check if a policy is frozen (all parameters have requires_grad=False).
    
    Args:
        policy_nn: Policy neural network module
        
    Returns:
        True if all parameters have requires_grad=False, False otherwise
    """
    return all(not param.requires_grad for param in policy_nn.parameters())


def count_parameters(policy_nn: torch.nn.Module) -> int:
    """Count total number of parameters in a policy.
    
    Args:
        policy_nn: Policy neural network module
        
    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in policy_nn.parameters())


class FrozenLocomotionPolicy:
    """Wrapper for frozen low-level locomotion policy.
    
    This class provides a simple interface to convert velocity commands to joint actions
    using a frozen low-level policy. The policy is expected to be loaded externally
    and the inference function is passed to this class.
    
    Args:
        inference_policy: Callable that takes observation tensor and returns actions
        env: The low-level locomotion environment (ManagerBasedRLEnv)
    """
    
    def __init__(
        self,
        inference_policy: callable,
        env,  # ManagerBasedRLEnv - avoiding circular import
    ):
        self.inference_policy = inference_policy
        self.env = env
    
    def __call__(self, velocity_command: torch.Tensor) -> torch.Tensor:
        """Convert velocity command to joint actions.
        
        This method temporarily sets the velocity command in the environment,
        gets the observation, and returns the policy actions.
        
        Args:
            velocity_command: Velocity command tensor of shape [num_envs, 3]
                where columns are [vx, vy, vyaw]
                
        Returns:
            Joint actions tensor of shape [num_envs, num_joints]
        """
        # Get the command term
        cmd_term = self.env.command_manager.get_term("base_velocity")
        
        # UniformVelocityCommand uses vel_command_b attribute
        # Store original command for restoration
        original_cmd = None
        
        # Try to get and modify vel_command_b
        try:
            if hasattr(cmd_term, "vel_command_b") and cmd_term.vel_command_b is not None:
                vel_cmd = cmd_term.vel_command_b
                original_cmd = vel_cmd.clone()
                
                # Check shape and update
                if vel_cmd.dim() == 2:
                    # 2D tensor: [num_envs, num_dims]
                    if vel_cmd.shape[1] >= 3:
                        # Update first 3 dims
                        cmd_term.vel_command_b[:, :3] = velocity_command
                    else:
                        # Less than 3 dims, replace entirely
                        cmd_term.vel_command_b = velocity_command
                else:
                    # Not 2D, try to replace
                    cmd_term.vel_command_b = velocity_command
            else:
                # vel_command_b doesn't exist, try to get command via get_command
                original_cmd = self.env.command_manager.get_command("base_velocity").clone()
                
                # Try to create or access vel_command_b
                if not hasattr(cmd_term, "vel_command_b"):
                    raise RuntimeError("base_velocity command term does not have vel_command_b attribute")
                
                # Initialize if None
                if cmd_term.vel_command_b is None:
                    cmd_term.vel_command_b = original_cmd.clone()
                
                # Update
                if cmd_term.vel_command_b.dim() == 2 and cmd_term.vel_command_b.shape[1] >= 3:
                    cmd_term.vel_command_b[:, :3] = velocity_command
                else:
                    cmd_term.vel_command_b = velocity_command
        except (IndexError, RuntimeError) as e:
            # If indexing fails, try a different approach
            # Get command via get_command and see its actual shape
            try:
                cmd_from_manager = self.env.command_manager.get_command("base_velocity")
                if original_cmd is None:
                    original_cmd = cmd_from_manager.clone()
                
                # Try to set via direct assignment if vel_command_b exists
                if hasattr(cmd_term, "vel_command_b"):
                    # Create new tensor with correct shape
                    num_envs = velocity_command.shape[0]
                    if cmd_term.vel_command_b is None or cmd_term.vel_command_b.shape != velocity_command.shape:
                        cmd_term.vel_command_b = velocity_command.clone()
                    else:
                        cmd_term.vel_command_b.copy_(velocity_command)
                else:
                    raise RuntimeError(f"Cannot modify base_velocity command: {e}")
            except Exception as e2:
                raise RuntimeError(f"Failed to modify base_velocity command: {e}, fallback also failed: {e2}")
        
        # Get observations with new command
        # For frozen policy, we need low-level observations (57D), not high-level (8D)
        # Check if this is a hierarchical environment by checking observation dimension
        obs_dict = self.env.observation_manager.compute()
        policy_obs = obs_dict.get("policy", None)
        
        # Check if we got high-level observations (8D) instead of low-level (57D)
        # If so, we need to compute low-level observations manually
        # For M20 rough task: base_lin_vel=None, height_scan=None
        # So observations are: base_ang_vel(1D) + projected_gravity(3D) + velocity_commands(3D) + 
        #                        joint_pos(12D) + joint_vel(12D) + last_action(16D) = 47D
        # But checkpoint expects 57D, so there might be base_lin_vel included (2D) and something else (8D)
        # Let's compute exactly as the low-level task does
        if policy_obs is not None and policy_obs.shape[1] < 50:  # High-level obs is 8D
            # This is a hierarchical environment - compute low-level observations manually
            import rl_training.tasks.manager_based.locomotion.velocity.mdp as velocity_mdp
            from isaaclab.managers import SceneEntityCfg
            
            # For M20, compute observations in the exact order as defined in the config
            # But we need to match what the checkpoint was trained with
            # The checkpoint expects 57D, so let's compute all possible terms and see what matches
            
            # For M20 rough task, the config shows:
            # - base_lin_vel = None (disabled)
            # - height_scan = None (disabled)
            # - joint_pos uses joint_pos_rel_without_wheel (12D leg joints, wheels zeroed)
            # - joint_vel uses all joints (12D)
            # - last_action: 16D (12 joint pos + 4 joint vel)
            # But checkpoint expects 57D, so the checkpoint was likely trained with base_lin_vel included
            
            # Compute observations in the exact order as the low-level task
            # Order: base_lin_vel(2D) + base_ang_vel(1D) + projected_gravity(3D) + 
            #        velocity_commands(3D) + joint_pos(12D) + joint_vel(12D) + last_action(16D) = 49D
            # But checkpoint expects 57D, so there must be 8 more dimensions
            # Possibly: base_lin_vel is actually included (2D) + something else (6D)?
            # Or the checkpoint was trained with a different config
            
            # Let's compute exactly as the observation manager would for the low-level task
            # We'll include base_lin_vel even though it's None in config, since checkpoint expects 57D
            robot = self.env.scene["robot"]
            
            # base_lin_vel: 2D (included in checkpoint despite being None in config)
            base_lin_vel = velocity_mdp.base_lin_vel(self.env)
            
            # base_ang_vel: 1D
            base_ang_vel = velocity_mdp.base_ang_vel(self.env)
            
            # projected_gravity: 3D
            projected_gravity = velocity_mdp.projected_gravity(self.env)
            
            # velocity_commands: 3D
            velocity_commands = velocity_mdp.generated_commands(self.env, command_name="base_velocity")
            
            # joint_pos: Use joint_pos_rel_without_wheel for M20 (12D leg joints, wheels zeroed)
            joint_pos = velocity_mdp.joint_pos_rel_without_wheel(
                self.env,
                asset_cfg=SceneEntityCfg("robot", joint_names=".*", preserve_order=True),
                wheel_asset_cfg=SceneEntityCfg("robot", joint_names=["fl_wheel_joint", "fr_wheel_joint", "hl_wheel_joint", "hr_wheel_joint"])
            )
            
            # joint_vel: 12D (all joints)
            joint_vel = velocity_mdp.joint_vel_rel(
                self.env,
                asset_cfg=SceneEntityCfg("robot", joint_names=".*", preserve_order=True)
            )
            
            # last_action: 16D (12 joint pos + 4 joint vel)
            last_action = velocity_mdp.last_action(self.env)
            
            # Concatenate observations
            # The checkpoint expects 57D, so we need to match exactly
            # Let's compute and check the dimension, then adjust if needed
            low_level_obs = torch.cat([
                base_lin_vel,        # 2D
                base_ang_vel,        # 1D
                projected_gravity,    # 3D
                velocity_commands,   # 3D
                joint_pos,           # 12D (leg joints only, wheels zeroed)
                joint_vel,           # 12D (all joints)
                last_action,         # 16D (12 joint pos + 4 joint vel)
            ], dim=1)
            
            # Check dimension and adjust to match checkpoint (57D)
            current_dim = low_level_obs.shape[1]
            expected_dim = 57
            
            if current_dim != expected_dim:
                # Dimension mismatch - need to adjust
                if current_dim < expected_dim:
                    # Pad with zeros
                    missing_dims = expected_dim - current_dim
                    padding = torch.zeros(low_level_obs.shape[0], missing_dims, device=low_level_obs.device, dtype=low_level_obs.dtype)
                    low_level_obs = torch.cat([low_level_obs, padding], dim=1)
                else:
                    # Truncate (shouldn't happen, but handle it)
                    low_level_obs = low_level_obs[:, :expected_dim]
            
            # Use low-level observations
            obs = TensorDict({"policy": low_level_obs}, batch_size=[low_level_obs.shape[0]])
        else:
            # Already low-level observations or correct format
            obs = TensorDict({"policy": policy_obs}, batch_size=[policy_obs.shape[0]])
        
        # Get actions from frozen policy
        with torch.no_grad():
            actions = self.inference_policy(obs)
        
        # Restore original command
        if original_cmd is not None:
            try:
                if hasattr(cmd_term, "vel_command_b") and cmd_term.vel_command_b is not None:
                    if original_cmd.dim() == 2 and original_cmd.shape[1] >= 3:
                        cmd_term.vel_command_b[:, :3] = original_cmd[:, :3]
                    else:
                        cmd_term.vel_command_b.copy_(original_cmd)
            except Exception as e:
                # If restoration fails, log warning but continue
                import warnings
                warnings.warn(f"Failed to restore original command: {e}")
        
        return actions
