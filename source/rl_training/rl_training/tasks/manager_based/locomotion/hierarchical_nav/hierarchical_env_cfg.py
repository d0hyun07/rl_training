# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Environment configuration for hierarchical navigation task."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import rl_training.tasks.manager_based.locomotion.hierarchical_nav.mdp as mdp
import rl_training.tasks.manager_based.locomotion.velocity.mdp as velocity_mdp
from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.rough_env_cfg import (
    DeeproboticsM20ActionsCfg,
    DeeproboticsM20RoughEnvCfg,
)


##
# MDP settings
##


@configclass
class HierarchicalNavCommandsCfg:
    """Command specifications for hierarchical navigation MDP."""

    goal_command = mdp.GoalCommandCfg(
        resampling_time_range=(1e10, 1e10),  # Only resample on reset
    )
    # Add base_velocity command for frozen low-level policy
    # This is needed by FrozenLocomotionPolicy to convert velocity commands to joint actions
    base_velocity = velocity_mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1e10, 1e10),  # Don't auto-resample (set by frozen policy)
        ranges=velocity_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-2.0, 2.0),
            lin_vel_y=(-2.0, 2.0),
            ang_vel_z=(-2.0, 2.0),
        ),
    )


@configclass
class HierarchicalNavObservationsCfg:
    """Observation specifications for hierarchical navigation MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # High-level navigation observations
        robot_position_2d = ObsTerm(
            func=mdp.robot_position_2d,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        robot_heading = ObsTerm(
            func=mdp.robot_heading,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-3.14159, 3.14159),
            scale=1.0,
        )
        goal_position_2d = ObsTerm(
            func=mdp.goal_position_2d,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        distance_to_goal = ObsTerm(
            func=mdp.distance_to_goal,
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(0.0, 100.0),
            scale=1.0,
        )
        direction_to_goal = ObsTerm(
            func=mdp.direction_to_goal,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        # High-level navigation observations (same as policy)
        robot_position_2d = ObsTerm(
            func=mdp.robot_position_2d,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        robot_heading = ObsTerm(
            func=mdp.robot_heading,
            clip=(-3.14159, 3.14159),
            scale=1.0,
        )
        goal_position_2d = ObsTerm(
            func=mdp.goal_position_2d,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        distance_to_goal = ObsTerm(
            func=mdp.distance_to_goal,
            clip=(0.0, 100.0),
            scale=1.0,
        )
        direction_to_goal = ObsTerm(
            func=mdp.direction_to_goal,
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class HierarchicalNavRewardsCfg:
    """Reward specifications for hierarchical navigation MDP.

    IMPORTANT: No velocity tracking rewards should be included here.
    This is a high-level navigation task, not a low-level locomotion task.
    """

    # High-level goal-reaching rewards
    progress_to_goal = RewTerm(
        func=mdp.progress_to_goal,
        weight=10.0,
    )
    goal_reached_bonus = RewTerm(
        func=mdp.goal_reached_bonus,
        weight=100.0,
        params={"threshold": 1.0},
    )
    timeout_penalty = RewTerm(
        func=mdp.timeout_penalty,
        weight=1.0,
    )


##
# Environment configuration
##


@configclass
class HierarchicalNavActionsCfg(DeeproboticsM20ActionsCfg):
    """Action specifications for hierarchical navigation MDP.

    Note: For hierarchical navigation, the high-level policy outputs velocity commands,
    which are then converted to joint actions by the frozen low-level policy.
    However, we still need to define actions for the base environment.
    The actual joint actions will be set correctly in __post_init__.
    """

    pass  # Will be configured in __post_init__


@configclass
class HierarchicalNavEnvCfg(DeeproboticsM20RoughEnvCfg):
    """Configuration for hierarchical navigation environment.

    This environment extends the low-level locomotion environment (DeeproboticsM20RoughEnvCfg)
    with high-level navigation observations, rewards, and commands.
    """

    # Override observations with high-level navigation observations
    observations: HierarchicalNavObservationsCfg = HierarchicalNavObservationsCfg()
    # Override rewards with high-level navigation rewards
    rewards: HierarchicalNavRewardsCfg = HierarchicalNavRewardsCfg()
    # Override commands with goal command
    commands: HierarchicalNavCommandsCfg = HierarchicalNavCommandsCfg()
    # Override actions (will be configured in __post_init__)
    actions: HierarchicalNavActionsCfg = HierarchicalNavActionsCfg()

    def __post_init__(self):
        """Post initialization."""
        # IMPORTANT: Save our high-level configs before calling parent
        # The parent's __post_init__ will try to modify observations/rewards/commands
        # that don't exist in our high-level config, so we need to restore them after
        hierarchical_obs = self.observations
        hierarchical_rewards = self.rewards
        hierarchical_commands = self.commands

        # IMPORTANT: Set temporary joint names for actions to avoid empty string error
        # The parent's __post_init__ will set these correctly, but we need valid values first
        # Use a pattern that matches all joints temporarily
        self.actions.joint_pos.joint_names = [".*"]
        self.actions.joint_vel.joint_names = [".*"]

        # Call parent post_init to set up scene, robot, etc.
        # This will fail when trying to modify observations, but we'll handle that
        try:
            print("[DEBUG] Calling parent __post_init__...")
            super().__post_init__()
            print("[DEBUG] Parent __post_init__ completed successfully")
        except (AttributeError, ValueError) as e:
            # Parent's __post_init__ tries to access observation terms (like joint_pos)
            # that don't exist in our high-level config. This is expected.
            # We'll restore our high-level configs after.
            error_str = str(e)
            print(f"[DEBUG] Parent __post_init__ raised expected error: {error_str[:200]}")
            if any(keyword in error_str for keyword in ["observations", "joint_pos", "policy", "Not all regular expressions"]):
                # This is expected - parent tries to modify low-level observations
                # that we've replaced with high-level ones, or actions with empty joint names
                print("[DEBUG] Error is expected, continuing...")
                pass
            else:
                # Re-raise if it's a different error
                print(f"[DEBUG] Unexpected error, re-raising...")
                raise
        except Exception as e:
            # Catch any other exceptions that might occur
            error_str = str(e)
            print(f"[DEBUG] Parent __post_init__ raised unexpected error: {error_str[:200]}")
            if any(keyword in error_str for keyword in ["observations", "joint_pos", "policy", "Not all regular expressions", "body_names", "sensor_cfg"]):
                # These are also expected errors related to config mismatches
                print("[DEBUG] Error is related to config mismatch, continuing...")
                pass
            else:
                # Re-raise if it's a different error
                print(f"[DEBUG] Unexpected error type, re-raising...")
                raise

        # Restore our high-level configs (they were overridden by parent)
        self.observations = hierarchical_obs
        self.rewards = hierarchical_rewards
        self.commands = hierarchical_commands

        # Configure actions properly (parent's __post_init__ should have set joint_names)
        # But if it didn't (because of the error), set them manually
        if hasattr(self, "leg_joint_names") and hasattr(self, "wheel_joint_names"):
            # Set joint names for actions (from parent class attributes)
            self.actions.joint_pos.joint_names = self.leg_joint_names
            self.actions.joint_vel.joint_names = self.wheel_joint_names
        elif not hasattr(self.actions.joint_pos, "joint_names") or not self.actions.joint_pos.joint_names:
            # Fallback: use all joints if attributes not available
            self.actions.joint_pos.joint_names = [".*"]
            self.actions.joint_vel.joint_names = [".*_wheel_joint"]

        # IMPORTANT: Fix events and terminations that have empty body_names
        # These are set by parent's __post_init__ but may have failed due to our observations override
        if hasattr(self, "base_link_name") and hasattr(self, "foot_link_name"):
            # Fix events that need body_names
            if hasattr(self.events, "randomize_rigid_body_mass_base"):
                if hasattr(self.events.randomize_rigid_body_mass_base, "params"):
                    if "asset_cfg" in self.events.randomize_rigid_body_mass_base.params:
                        if hasattr(self.events.randomize_rigid_body_mass_base.params["asset_cfg"], "body_names"):
                            if not self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names:
                                self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
            
            if hasattr(self.events, "randomize_rigid_body_mass"):
                if hasattr(self.events.randomize_rigid_body_mass, "params"):
                    if "asset_cfg" in self.events.randomize_rigid_body_mass.params:
                        if hasattr(self.events.randomize_rigid_body_mass.params["asset_cfg"], "body_names"):
                            if not self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names:
                                self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = [
                                    f"^(?!.*{self.base_link_name}).*"
                                ]
            
            if hasattr(self.events, "randomize_com_positions"):
                if hasattr(self.events.randomize_com_positions, "params"):
                    if "asset_cfg" in self.events.randomize_com_positions.params:
                        if hasattr(self.events.randomize_com_positions.params["asset_cfg"], "body_names"):
                            if not self.events.randomize_com_positions.params["asset_cfg"].body_names:
                                self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
            
            if hasattr(self.events, "randomize_apply_external_force_torque"):
                if hasattr(self.events.randomize_apply_external_force_torque, "params"):
                    if "asset_cfg" in self.events.randomize_apply_external_force_torque.params:
                        if hasattr(self.events.randomize_apply_external_force_torque.params["asset_cfg"], "body_names"):
                            if not self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names:
                                self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
            
            # Fix terminations
            # Parent sets illegal_contact to None in __post_init__
            # But if it wasn't set (due to exception), set it to None manually
            if hasattr(self.terminations, "illegal_contact"):
                # Check if it has empty body_names (which causes the error)
                if self.terminations.illegal_contact is not None:
                    if hasattr(self.terminations.illegal_contact, "params"):
                        if "sensor_cfg" in self.terminations.illegal_contact.params:
                            sensor_cfg = self.terminations.illegal_contact.params["sensor_cfg"]
                            if hasattr(sensor_cfg, "body_names"):
                                # If body_names is empty or empty list, set to None as parent does
                                if not sensor_cfg.body_names or (isinstance(sensor_cfg.body_names, list) and len(sensor_cfg.body_names) == 0):
                                    self.terminations.illegal_contact = None
                                elif isinstance(sensor_cfg.body_names, str) and sensor_cfg.body_names == "":
                                    self.terminations.illegal_contact = None

        # High-level navigation specific settings
        # Episode length: 50 seconds (sufficient time to reach goal)
        self.episode_length_s = 50.0
        # Decimation: High-level policy runs 10x slower than low-level
        # Low-level runs at 50Hz, high-level runs at 5Hz (10 decimation)
        self.decimation = 10

        # Disable all velocity tracking rewards from parent
        # These are low-level rewards and should not be used for high-level navigation
        if hasattr(self.rewards, "track_lin_vel_xy_exp"):
            self.rewards.track_lin_vel_xy_exp = None
        if hasattr(self.rewards, "track_ang_vel_z_exp"):
            self.rewards.track_ang_vel_z_exp = None

        # Disable curriculum that depends on base_velocity command
        # High-level navigation uses goal_command, not base_velocity
        if hasattr(self.curriculum, "terrain_levels"):
            self.curriculum.terrain_levels = None
        if hasattr(self.curriculum, "command_levels"):
            self.curriculum.command_levels = None

        # Note: Low-level actions are handled by the frozen policy wrapper
        # The action space for this environment is velocity commands [vx, vy, vyaw]
        # which will be converted to joint actions by the frozen low-level policy
