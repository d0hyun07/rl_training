# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Hierarchical navigation task for goal-reaching with frozen low-level policy."""

import gymnasium as gym

from . import agents
from .hierarchical_nav_env import HierarchicalNavEnv

__all__ = ["HierarchicalNavEnv"]

##
# Register Gym environments.
##

gym.register(
    id="Hierarchical-Nav-Deeprobotics-M20-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hierarchical_env_cfg:HierarchicalNavEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:HierarchicalNavPPORunnerCfg",
    },
)

