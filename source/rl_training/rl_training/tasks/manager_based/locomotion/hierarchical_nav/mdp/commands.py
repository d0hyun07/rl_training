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

    @property
    def command(self) -> torch.Tensor:
        """Return the current command. Shape is (num_envs, 2)."""
        return self._command

    def _update_command(self):
        """Update and store the current commands."""
        # Commands are already stored in self._command, nothing to do here
        pass

    def _update_metrics(self):
        """Update metrics for the command generator."""
        # No metrics to update for goal position commands
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample goal positions for specified environments.
        
        Args:
            env_ids: Environment IDs to resample commands for.
        """
        # Get distance range based on curriculum phase
        if self.curriculum_phase == 0:
            # Phase 1: 1-3m
            min_dist, max_dist = 1.0, 3.0
        elif self.curriculum_phase == 1:
            # Phase 2: 3-7m
            min_dist, max_dist = 3.0, 7.0
        else:
            # Phase 3: 5-10m
            min_dist, max_dist = 5.0, 10.0

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
