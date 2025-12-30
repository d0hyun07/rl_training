# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING, Sequence

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class GoalCommand(CommandTerm):
    """Command generator that generates goal positions with curriculum learning.

    This command generator creates random goal positions around the robot.
    The distance range changes based on the curriculum phase.
    """

    cfg: "GoalCommandCfg"
    """The configuration of the command generator."""

    def __init__(self, cfg: "GoalCommandCfg", env: ManagerBasedEnv):
        """Initialize the command generator.
        
        Args:
            cfg: The configuration of the command generator.
            env: The environment object.
        """
        super().__init__(cfg, env)
        # Command is [goal_x, goal_y] in world frame
        self._command = torch.zeros(self.num_envs, 2, device=self.device)
        # Curriculum phase: 0=Phase 1, 1=Phase 2, 2=Phase 3
        self.curriculum_phase = 0
        # Track goal reached success rate for curriculum learning
        self._goal_reached_count = 0
        self._total_episodes = 0
        self._curriculum_update_interval = 500  # Update curriculum every 500 episodes (increased for stability)
        # Success rate thresholds for advancing curriculum: 0.5 (50%), 0.7 (70%)
        # Increased thresholds to prevent premature curriculum advancement
        self._curriculum_thresholds = [0.5, 0.7]

    @property
    def command(self) -> torch.Tensor:
        """Return the current command. Shape is (num_envs, 2)."""
        return self._command

    def _update_command(self):
        """Update and store the current commands."""
        # Commands are already stored in self._command, nothing to do here
        pass

    def _update_metrics(self):
        """Update metrics for the command generator and update curriculum."""
        # Update curriculum based on goal reached success rate
        # Only update if we have enough episodes tracked
        if self._total_episodes >= self._curriculum_update_interval:
            if self._total_episodes > 0:
                success_rate = self._goal_reached_count / self._total_episodes
                
                # Clamp success rate to [0, 1] to prevent >100% values
                success_rate = min(1.0, max(0.0, success_rate))
                
                # Advance curriculum phase if success rate is high enough
                if self.curriculum_phase < len(self._curriculum_thresholds):
                    if success_rate >= self._curriculum_thresholds[self.curriculum_phase]:
                        old_phase = self.curriculum_phase
                        self.curriculum_phase += 1
                        print(f"[CURRICULUM] Advancing from phase {old_phase + 1} to phase {self.curriculum_phase + 1} "
                              f"(success rate: {success_rate:.2%}, episodes: {self._total_episodes})")
                    else:
                        print(f"[CURRICULUM] Phase {self.curriculum_phase + 1} progress: "
                              f"{success_rate:.2%} / {self._curriculum_thresholds[self.curriculum_phase]:.2%} "
                              f"(episodes: {self._total_episodes})")
                
                # Reset counters
                self._goal_reached_count = 0
                self._total_episodes = 0

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample goal positions for specified environments.
        
        Args:
            env_ids: Environment IDs to resample commands for.
        """
        # Track goal reached status when episodes reset
        # _resample_command is called when episodes reset, so we check if goal was reached
        # before resampling by checking the current distance to goal
        try:
            if hasattr(self._env, "scene") and "robot" in self._env.scene:
                robot = self._env.scene["robot"]
                robot_pos = robot.data.root_pos_w[env_ids, :2]
                goal_pos = self._command[env_ids, :2]
                distances = torch.norm(robot_pos - goal_pos, dim=1)
                goal_reached = (distances < 0.5).sum().item()  # 0.5m threshold
                self._goal_reached_count += goal_reached
                self._total_episodes += len(env_ids)
        except (AttributeError, KeyError, TypeError):
            # If scene or robot doesn't exist, skip tracking (e.g., during initialization)
            # Still count episodes to avoid undercounting
            self._total_episodes += len(env_ids)
        
        # Get distance range based on curriculum phase
        # Start with closer goals for easier learning
        if self.curriculum_phase == 0:
            # Phase 1: 0.5-2m (reduced from 1-3m for easier initial learning)
            min_dist, max_dist = 0.5, 2.0
        elif self.curriculum_phase == 1:
            # Phase 2: 1.5-4m (reduced from 3-7m)
            min_dist, max_dist = 1.5, 4.0
        else:
            # Phase 3: 3-8m (reduced from 5-10m)
            min_dist, max_dist = 3.0, 8.0

        # Sample distance and angle
        num_envs = len(env_ids)
        distances = torch.rand(num_envs, device=self.device) * (max_dist - min_dist) + min_dist
        angles = torch.rand(num_envs, device=self.device) * 2 * math.pi

        # Get robot position
        # CommandTerm stores environment as self._env
        robot_pos = self._env.scene["robot"].data.root_pos_w[env_ids, :2]

        # Compute goal position
        goal_x = robot_pos[:, 0] + distances * torch.cos(angles)
        goal_y = robot_pos[:, 1] + distances * torch.sin(angles)

        # Store goal positions
        self._command[env_ids, 0] = goal_x
        self._command[env_ids, 1] = goal_y


@configclass
class GoalCommandCfg(CommandTermCfg):
    """Configuration for goal position command generator."""

    class_type: type = GoalCommand
    resampling_time_range: tuple[float, float] = (1e10, 1e10)
    """The range of time to resample the command in seconds. Set to large values to prevent automatic resampling."""
