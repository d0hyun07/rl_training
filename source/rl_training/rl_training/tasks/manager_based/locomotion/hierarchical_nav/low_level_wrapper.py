# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

"""Low-level policy wrapper for hierarchical navigation.

This wrapper integrates a frozen low-level locomotion policy into the
high-level navigation environment. It converts high-level velocity commands
to low-level joint actions using the frozen policy.
"""

from __future__ import annotations

import torch
import numpy as np
import gymnasium as gym
from gymnasium.core import ActType, ObsType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from rl_training.utils.frozen_policy import FrozenLocomotionPolicy
import rl_training.tasks.manager_based.locomotion.hierarchical_nav.mdp as mdp


class LowLevelPolicyWrapper(gym.Wrapper):
    """Wrapper that integrates frozen low-level policy for hierarchical navigation.
    
    This wrapper:
    1. Takes high-level velocity commands [vx, vy, vyaw] as actions
    2. Converts them to joint actions using frozen low-level policy
    3. Executes decimation steps in the low-level environment
    4. Computes high-level observations and rewards
    
    Args:
        env: Hierarchical navigation environment (ManagerBasedRLEnv with goal_command)
        frozen_policy_path: Path to frozen low-level policy checkpoint
        decimation: Number of low-level steps per high-level step (default: 10)
    """
    
    def __init__(
        self,
        env: ManagerBasedRLEnv,
        frozen_policy_path: str,
        decimation: int = 10,
    ):
        """Initialize the wrapper.
        
        Args:
            env: Hierarchical navigation environment (unwrapped ManagerBasedRLEnv)
            frozen_policy_path: Path to frozen policy checkpoint
            decimation: Number of low-level steps per high-level step
        """
        # Initialize gym.Wrapper
        super().__init__(env)
        
        # Store decimation
        self.decimation = decimation
        self._frozen_policy_path = frozen_policy_path
        
        # Override action space: high-level velocity commands [vx, vy, vyaw]
        # Store as instance variable and also as property to ensure it's always accessible
        self._action_space = gym.spaces.Box(
            low=np.array([-2.0, -2.0, -2.0], dtype=np.float32),
            high=np.array([2.0, 2.0, 2.0], dtype=np.float32),
            dtype=np.float32,
        )
        
        # Override observation space: high-level navigation observations (8D)
        obs_dim = 8  # robot_pos(2) + heading(1) + goal_pos(2) + distance(1) + direction(2)
        self._observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        
        # IMPORTANT: Store reference to unwrapped environment
        # We'll override its single_action_space in the unwrapped property
        unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env
        self._unwrapped_env = unwrapped_env
        if hasattr(unwrapped_env, 'single_action_space'):
            # Store original for restoration if needed
            self._original_single_action_space = unwrapped_env.single_action_space
            # Override with our 3D action space
            unwrapped_env.single_action_space = self._action_space
            print(f"[INFO] Overrode unwrapped.single_action_space: {unwrapped_env.single_action_space}")
        
        # Load frozen policy
        print(f"[INFO] Loading frozen policy from: {frozen_policy_path}")
        self.frozen_policy_wrapper = self._load_frozen_policy(frozen_policy_path, env)
        
        if self.frozen_policy_wrapper is None:
            raise RuntimeError(f"Failed to load frozen policy from {frozen_policy_path}")
        
        print(f"[INFO] Frozen policy loaded successfully")
        print(f"[INFO] Low-level policy wrapper created (decimation={decimation})")
        print(f"[INFO] Action space: {self._action_space}")
        print(f"[INFO] Observation space: {self._observation_space}")
        
        # Store previous distance for progress reward
        self._prev_distance_to_goal = None
    
    @property
    def action_space(self):
        """Return high-level action space (3D velocity commands)."""
        return self._action_space
    
    @property
    def observation_space(self):
        """Return high-level observation space (8D navigation observations)."""
        return self._observation_space
    
    @property
    def single_action_space(self):
        """Return single action space (required by RslRlVecEnvWrapper).
        
        RslRlVecEnvWrapper reads self.unwrapped.single_action_space to determine num_actions.
        """
        return self._action_space
    
    @property
    def single_observation_space(self):
        """Return single observation space (required by RslRlVecEnvWrapper)."""
        return self._observation_space
    
    @property
    def unwrapped(self):
        """Return unwrapped environment.
        
        We need to return the actual unwrapped environment, but ensure
        RslRlVecEnvWrapper reads our single_action_space.
        """
        # Return the actual unwrapped environment
        unwrapped = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        # Temporarily override its single_action_space to our 3D space
        # This ensures RslRlVecEnvWrapper reads the correct action space
        if hasattr(unwrapped, 'single_action_space'):
            # Store original if not already stored
            if not hasattr(self, '_original_unwrapped_single_action_space'):
                self._original_unwrapped_single_action_space = unwrapped.single_action_space
            # Override with our 3D action space
            unwrapped.single_action_space = self._action_space
        return unwrapped
    
    def _load_frozen_policy(self, frozen_policy_path: str, env: ManagerBasedRLEnv) -> FrozenLocomotionPolicy:
        """Load frozen policy from checkpoint.
        
        Args:
            frozen_policy_path: Path to frozen policy checkpoint
            env: The hierarchical navigation environment
            
        Returns:
            FrozenLocomotionPolicy instance
        """
        import os
        import torch
        from omegaconf import OmegaConf
        from rsl_rl.runners import OnPolicyRunner
        from rsl_rl.modules import ActorCritic
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rl_training.utils.frozen_policy import freeze_policy
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
        
        # Check if base_velocity command exists (should be added in hierarchical_env_cfg.py)
        # Use unwrapped to access command_manager (env might be wrapped)
        unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env
        has_base_velocity = "base_velocity" in unwrapped_env.command_manager._terms
        
        if not has_base_velocity:
            raise RuntimeError(
                "base_velocity command not found in hierarchical environment. "
                "Please ensure HierarchicalNavCommandsCfg includes base_velocity command."
            )
        
        print("[INFO] base_velocity command found in hierarchical environment")
        
        # Get checkpoint directory (parent of checkpoint file)
        checkpoint_dir = os.path.dirname(frozen_policy_path)
        agent_config_path = os.path.join(checkpoint_dir, "params", "agent.yaml")
        
        # Try to load agent config from checkpoint directory
        if os.path.exists(agent_config_path):
            print(f"[INFO] Loading agent config from checkpoint: {agent_config_path}")
            agent_cfg_dict = OmegaConf.load(agent_config_path)
            # Convert OmegaConf to dict
            agent_cfg_dict = OmegaConf.to_container(agent_cfg_dict, resolve=True)
        else:
            print(f"[WARNING] Agent config not found at {agent_config_path}, using default config")
            # Load agent config from registry
            low_level_task = "Rough-Deeprobotics-M20-v0"
            agent_cfg = load_cfg_from_registry(low_level_task, "rsl_rl_cfg_entry_point")
            agent_cfg_dict = OmegaConf.to_container(OmegaConf.structured(agent_cfg), resolve=True)
        
        # Set device (use unwrapped_env)
        agent_cfg_dict["device"] = unwrapped_env.device
        
        # Load checkpoint to get model state dict
        checkpoint = torch.load(frozen_policy_path, map_location=unwrapped_env.device)
        model_state_dict = checkpoint.get("model_state_dict", checkpoint)
        
        # Infer observation and action dimensions from checkpoint
        actor_first_layer_key = None
        for key in model_state_dict.keys():
            if "actor.0.weight" in key or ("actor" in key and "0.weight" in key):
                actor_first_layer_key = key
                break
        
        if not actor_first_layer_key:
            raise RuntimeError("Could not find actor first layer in checkpoint")
        
        actor_obs_dim = model_state_dict[actor_first_layer_key].shape[1]
        
        # Find critic observation dimension
        critic_first_layer_key = None
        for key in model_state_dict.keys():
            if "critic.0.weight" in key or ("critic" in key and "0.weight" in key):
                critic_first_layer_key = key
                break
        
        if not critic_first_layer_key:
            # Fallback: assume same as actor
            critic_obs_dim = actor_obs_dim
            print(f"[WARNING] Could not find critic first layer, assuming critic_obs_dim = actor_obs_dim = {actor_obs_dim}")
        else:
            critic_obs_dim = model_state_dict[critic_first_layer_key].shape[1]
        
        # Find action dimension from log_std
        action_dim = None
        for key in model_state_dict.keys():
            if "log_std" in key:
                action_dim = model_state_dict[key].shape[0]
                break
        
        if action_dim is None:
            # Fallback: find last actor layer
            actor_layers = []
            for key in model_state_dict.keys():
                if "actor" in key and "weight" in key:
                    parts = key.split(".")
                    try:
                        idx = parts.index("actor")
                        if idx + 1 < len(parts):
                            layer_num = int(parts[idx + 1])
                            actor_layers.append((layer_num, key))
                    except (ValueError, IndexError):
                        continue
            
            if actor_layers:
                actor_layers.sort(key=lambda x: x[0], reverse=True)
                action_dim = model_state_dict[actor_layers[0][1]].shape[0]
            else:
                raise RuntimeError("Could not infer action dimension from checkpoint")
        
        print(f"[INFO] Inferred dimensions from checkpoint: actor_obs={actor_obs_dim}, critic_obs={critic_obs_dim}, action={action_dim}")
        
        # Infer network architecture from state dict
        # Extract actor hidden dims
        actor_hidden_dims = []
        actor_layer_keys = []
        for key in model_state_dict.keys():
            if "actor" in key and "weight" in key and "log_std" not in key:
                parts = key.split(".")
                try:
                    idx = parts.index("actor")
                    if idx + 1 < len(parts):
                        layer_num = int(parts[idx + 1])
                        actor_layer_keys.append((layer_num, key))
                except (ValueError, IndexError):
                    continue
        
        actor_layer_keys.sort(key=lambda x: x[0])
        for layer_num, key in actor_layer_keys:
            if layer_num == 0:
                continue  # Skip input layer
            weight_shape = model_state_dict[key].shape
            if len(weight_shape) == 2:
                actor_hidden_dims.append(weight_shape[1])
        
        # Extract critic hidden dims
        critic_hidden_dims = []
        critic_layer_keys = []
        for key in model_state_dict.keys():
            if "critic" in key and "weight" in key:
                parts = key.split(".")
                try:
                    idx = parts.index("critic")
                    if idx + 1 < len(parts):
                        layer_num = int(parts[idx + 1])
                        critic_layer_keys.append((layer_num, key))
                except (ValueError, IndexError):
                    continue
        
        critic_layer_keys.sort(key=lambda x: x[0])
        for layer_num, key in critic_layer_keys:
            if layer_num == 0:
                continue  # Skip input layer
            weight_shape = model_state_dict[key].shape
            if len(weight_shape) == 2:
                critic_hidden_dims.append(weight_shape[1])
        
        # Fallback to agent config if inference failed
        if not actor_hidden_dims:
            actor_hidden_dims = agent_cfg_dict.get("policy", {}).get("actor_hidden_dims", [512, 256, 128])
        if not critic_hidden_dims:
            critic_hidden_dims = agent_cfg_dict.get("policy", {}).get("critic_hidden_dims", [512, 256, 128])
        
        print(f"[INFO] Inferred network architecture: actor={actor_hidden_dims}, critic={critic_hidden_dims}")
        
        # Create a minimal dummy environment wrapper that provides the correct observation structure
        # This is needed because ActorCritic is created by OnPolicyRunner using env.get_observations()
        from tensordict import TensorDict
        
        # Create a simple dummy environment class that will pass RslRlVecEnvWrapper checks
        from isaaclab.envs import ManagerBasedRLEnv
        
        class DummyLowLevelEnvUnwrapped(ManagerBasedRLEnv):
            """Dummy unwrapped environment that passes isinstance checks."""
            
            def __init__(self, num_envs, actor_obs_dim, critic_obs_dim, device, action_dim):
                # Don't call super().__init__() - it would try to create a real environment
                # Use object.__setattr__ to bypass property setters
                import gymnasium as gym
                import numpy as np
                
                # Set attributes using object.__setattr__ to bypass any property setters
                object.__setattr__(self, '_num_envs', num_envs)
                object.__setattr__(self, '_device', device)
                object.__setattr__(self, '_obs_dim', actor_obs_dim)  # For policy
                object.__setattr__(self, '_critic_obs_dim', critic_obs_dim)  # For critic
                object.__setattr__(self, '_max_episode_length', 1000)
                object.__setattr__(self, '_is_closed', False)  # For __del__ compatibility
                # Add command_manager, reward_manager, termination_manager, and curriculum_manager to prevent AttributeError in __del__
                object.__setattr__(self, 'command_manager', None)
                object.__setattr__(self, 'reward_manager', None)
                object.__setattr__(self, 'termination_manager', None)
                object.__setattr__(self, 'curriculum_manager', None)
                
                # Create dummy action space (required by RslRlVecEnvWrapper)
                object.__setattr__(self, 'single_action_space', gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(action_dim,),
                    dtype=np.float32,
                ))
            
            @property
            def unwrapped(self):
                """Return self as unwrapped (for compatibility)."""
                return self
            
            @property
            def num_envs(self):
                return self._num_envs
            
            @property
            def device(self):
                return self._device
            
            @property
            def max_episode_length(self):
                return self._max_episode_length
            
            def _get_observations(self):
                """Internal method to get observations (called by RslRlVecEnvWrapper)."""
                from tensordict import TensorDict
                # Return both policy and critic observations
                # Use actor_obs_dim for policy, critic_obs_dim for critic
                # We'll store these in the class
                return {
                    "policy": torch.zeros(self._num_envs, self._obs_dim, device=self._device),
                    "critic": torch.zeros(self._num_envs, getattr(self, '_critic_obs_dim', self._obs_dim), device=self._device),
                }
        
        class DummyLowLevelEnv:
            """Dummy environment wrapper that provides correct observation structure for policy loading."""
            
            def __init__(self, num_envs, actor_obs_dim, critic_obs_dim, device, action_dim):
                self._num_envs = num_envs
                self._device = device
                self.obs_dim = actor_obs_dim  # For policy
                # Create unwrapped that will pass isinstance check
                self.unwrapped = DummyLowLevelEnvUnwrapped(num_envs, actor_obs_dim, critic_obs_dim, device, action_dim)
            
            @property
            def num_envs(self):
                return self._num_envs
            
            @property
            def device(self):
                return self._device
            
            @property
            def max_episode_length(self):
                return self.unwrapped._max_episode_length
            
            def get_observations(self):
                """Return dummy observations with correct structure."""
                from tensordict import TensorDict
                return TensorDict(
                    {
                        "policy": torch.zeros(self.num_envs, self.obs_dim, device=self.device),
                        "critic": torch.zeros(self.num_envs, self.unwrapped._critic_obs_dim, device=self.device),
                    },
                    batch_size=[self.num_envs]
                )
            
            def reset(self, **kwargs):
                """Dummy reset."""
                return self.get_observations(), {}
            
            def step(self, actions):
                """Dummy step."""
                return self.get_observations(), torch.zeros(self.num_envs, device=self.device), \
                       torch.zeros(self.num_envs, dtype=torch.bool, device=self.device), \
                       torch.zeros(self.num_envs, dtype=torch.bool, device=self.device), {}
            
            def close(self):
                """Dummy close."""
                pass
        
        # Create dummy environment (use both actor and critic observation dimensions)
        dummy_env = DummyLowLevelEnv(unwrapped_env.num_envs, actor_obs_dim, critic_obs_dim, unwrapped_env.device, action_dim)
        
        # Wrap dummy environment for RSL-RL
        wrapped_dummy_env = RslRlVecEnvWrapper(dummy_env, clip_actions=agent_cfg_dict.get("clip_actions", True))
        
        # Reset to initialize
        wrapped_dummy_env.reset()
        
        # Create runner with dummy environment (this will create ActorCritic with correct structure)
        runner = OnPolicyRunner(
            wrapped_dummy_env,
            agent_cfg_dict,
            device=unwrapped_env.device,
        )
        
        # Load checkpoint weights
        runner.load(frozen_policy_path)
        
        # Extract and freeze the policy network
        try:
            policy_nn = runner.alg.policy
        except AttributeError:
            policy_nn = runner.alg.actor_critic
        
        # Freeze the policy
        policy_nn.eval()
        for param in policy_nn.parameters():
            param.requires_grad = False
        
        print("[INFO] Frozen policy network created and loaded from checkpoint")
        
        # Create inference function
        def inference_policy(obs):
            """Inference function for frozen policy."""
            with torch.no_grad():
                # obs should be a TensorDict with "policy" key
                # policy_nn.act_inference expects a TensorDict, not a raw tensor
                if isinstance(obs, TensorDict):
                    # Already a TensorDict, use as-is
                    return policy_nn.act_inference(obs)
                elif isinstance(obs, dict):
                    # Convert dict to TensorDict
                    obs_td = TensorDict(obs, batch_size=[obs.get("policy", obs.get(list(obs.keys())[0])).shape[0]])
                    return policy_nn.act_inference(obs_td)
                else:
                    # Raw tensor - wrap in TensorDict
                    obs_td = TensorDict({"policy": obs}, batch_size=[obs.shape[0]])
                    return policy_nn.act_inference(obs_td)
        
        # Create FrozenLocomotionPolicy wrapper
        # Use unwrapped env for command_manager access
        frozen_policy_wrapper = FrozenLocomotionPolicy(
            inference_policy=inference_policy,
            env=unwrapped_env,  # Use unwrapped hierarchical env for command_manager access
        )
        
        return frozen_policy_wrapper
    
    def reset(self, **kwargs) -> tuple[ObsType, dict[str, Any]]:
        """Reset the environment and return high-level observations.
        
        Returns:
            High-level observations and info dict
        """
        # Reset high-level environment
        obs, info = self.env.reset(**kwargs)
        
        # Get high-level observations
        high_level_obs = self._get_high_level_obs()
        
        # RslRlVecEnvWrapper expects TensorDict for observations
        from tensordict import TensorDict
        
        # Ensure observation is torch tensor
        if not isinstance(high_level_obs, torch.Tensor):
            high_level_obs = torch.tensor(high_level_obs, device=self.device, dtype=torch.float32)
        
        # Create TensorDict with "policy" and "critic" keys (RslRlVecEnvWrapper expects both)
        # For hierarchical navigation, policy and critic use the same observations
        obs_dict = TensorDict({
            "policy": high_level_obs,
            "critic": high_level_obs.clone(),  # Same observations for critic
        }, batch_size=[high_level_obs.shape[0]])
        
        return obs_dict, info
    
    def step(self, action: ActType) -> tuple[ObsType, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Step the environment with high-level action.
        
        Args:
            action: High-level velocity command [vx, vy, vyaw] of shape [num_envs, 3]
        
        Returns:
            High-level observations, rewards, terminated, truncated, info
        """
        # Convert action to tensor if needed
        if not isinstance(action, torch.Tensor):
            action = torch.tensor(action, device=self.device, dtype=torch.float32)
        
        # Ensure correct shape [num_envs, 3]
        if action.dim() == 1:
            action = action.unsqueeze(0)
        
        # Get num_envs from unwrapped environment
        unwrapped_env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        num_envs = unwrapped_env.num_envs
        
        # Handle action shape issues
        if action.shape[0] != num_envs:
            if action.shape[0] == 1 and num_envs > 1:
                action = action.expand(num_envs, -1)
            else:
                raise ValueError(f"Action batch size {action.shape[0]} doesn't match num_envs {num_envs}")
        
        # Handle case where action might be 16D (from base environment) instead of 3D
        # This can happen if OnPolicyRunner was initialized with wrong action space
        if action.shape[1] == 16:
            # Take first 3 dimensions (assuming they correspond to velocity command)
            # Only warn once to avoid spam
            if not hasattr(self, '_warned_16d_action'):
                print(f"[WARNING] Received 16D action instead of 3D. Using first 3 dimensions. "
                      f"This suggests OnPolicyRunner was initialized with wrong action space. "
                      f"Action space should be 3D but got {action.shape[1]}D.")
                self._warned_16d_action = True
            action = action[:, :3]
        elif action.shape[1] != 3:
            raise ValueError(
                f"Expected action shape [num_envs, 3], got {action.shape}. "
                f"This suggests the action space was not properly overridden. "
                f"Current action_space: {self.action_space}, shape: {self.action_space.shape if hasattr(self.action_space, 'shape') else 'N/A'}"
            )
        
        # Step high-level environment decimation times
        # Each step: convert velocity command to joint actions using frozen policy
        last_info = None
        for step_idx in range(self.decimation):
            # Convert high-level velocity command to low-level joint actions
            low_level_actions = self.frozen_policy_wrapper(action)
            
            # Step high-level environment with joint actions
            # The hierarchical environment expects joint actions (from base class)
            step_result = self.env.step(low_level_actions)
            
            # Handle different return formats (gymnasium vs custom)
            if len(step_result) == 4:
                obs, reward, terminated, truncated = step_result
                info = {}
            elif len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
            else:
                raise ValueError(f"Unexpected step return format: {len(step_result)} values")
            
            # Store info from last step
            last_info = info
            
            # Early termination check: if all environments are done, break early
            if isinstance(terminated, torch.Tensor):
                if terminated.all():
                    break
            elif isinstance(terminated, (bool, np.ndarray)):
                if np.all(terminated):
                    break
        
        # Get high-level observations and rewards
        high_level_obs = self._get_high_level_obs()
        high_level_reward = self._compute_high_level_reward()
        high_level_terminated, high_level_truncated = self._compute_high_level_terminations(
            terminated, truncated
        )
        
        # Merge info dicts
        if isinstance(last_info, dict):
            info = last_info.copy()
        else:
            info = {}
        
        # Add high-level metrics
        info["hierarchical/step_count"] = self.decimation
        
        # RslRlVecEnvWrapper expects TensorDict for observations
        # Convert observation to TensorDict format
        from tensordict import TensorDict
        
        # Ensure observation is torch tensor
        if not isinstance(high_level_obs, torch.Tensor):
            high_level_obs = torch.tensor(high_level_obs, device=self.device, dtype=torch.float32)
        
        # Create TensorDict with "policy" and "critic" keys (RslRlVecEnvWrapper expects both)
        # For hierarchical navigation, policy and critic use the same observations
        obs_dict = TensorDict({
            "policy": high_level_obs,
            "critic": high_level_obs.clone(),  # Same observations for critic
        }, batch_size=[high_level_obs.shape[0]])
        
        # Ensure reward, terminated, truncated are torch tensors
        if not isinstance(high_level_reward, torch.Tensor):
            high_level_reward = torch.tensor(high_level_reward, device=self.device, dtype=torch.float32)
        if not isinstance(high_level_terminated, torch.Tensor):
            high_level_terminated = torch.tensor(high_level_terminated, device=self.device, dtype=torch.bool)
        if not isinstance(high_level_truncated, torch.Tensor):
            high_level_truncated = torch.tensor(high_level_truncated, device=self.device, dtype=torch.bool)
        
        return obs_dict, high_level_reward, high_level_terminated, high_level_truncated, info
    
    def _get_high_level_obs(self) -> torch.Tensor:
        """Get high-level observations from current state.
        
        Returns:
            High-level observation tensor of shape [num_envs, 8]
        """
        # Use unwrapped env to access scene, command_manager, etc.
        unwrapped_env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        
        # Use MDP observation functions
        robot_pos_2d = mdp.robot_position_2d(unwrapped_env)
        robot_heading_val = mdp.robot_heading(unwrapped_env)
        goal_pos_2d = mdp.goal_position_2d(unwrapped_env)
        distance = mdp.distance_to_goal(unwrapped_env)
        direction = mdp.direction_to_goal(unwrapped_env)
        
        # Concatenate observations
        obs = torch.cat([
            robot_pos_2d,      # [num_envs, 2]
            robot_heading_val, # [num_envs, 1]
            goal_pos_2d,       # [num_envs, 2]
            distance,          # [num_envs, 1]
            direction,         # [num_envs, 2]
        ], dim=1)  # [num_envs, 8]
        
        return obs
    
    def _compute_high_level_reward(self) -> torch.Tensor:
        """Compute high-level rewards.
        
        Returns:
            High-level reward tensor of shape [num_envs]
        """
        # Use unwrapped env to access reward_manager, step_dt, etc.
        unwrapped_env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        
        # Use reward manager to compute rewards (it already has the weights configured)
        # The reward manager computes all active reward terms and sums them
        # RewardManager.compute() requires dt argument
        dt = unwrapped_env.step_dt
        reward_dict = unwrapped_env.reward_manager.compute(dt=dt)
        
        # Sum all rewards
        # Handle different return types: Tensor, TensorDict, or regular dict
        from tensordict import TensorDict
        
        if isinstance(reward_dict, torch.Tensor):
            # Already a tensor, return as-is (handle sparse if needed)
            if reward_dict.is_sparse:
                reward_dict = reward_dict.to_dense()
            return reward_dict
        elif isinstance(reward_dict, TensorDict):
            # TensorDict - iterate over keys to avoid sparse tensor issues
            total_reward = torch.zeros(unwrapped_env.num_envs, device=unwrapped_env.device)
            for key in reward_dict.keys():
                reward = reward_dict[key]
                if isinstance(reward, torch.Tensor):
                    if reward.is_sparse:
                        reward = reward.to_dense()
                    total_reward = total_reward + reward
                else:
                    total_reward = total_reward + reward
        else:
            # Regular dict - iterate over items
            total_reward = torch.zeros(unwrapped_env.num_envs, device=unwrapped_env.device)
            for key, reward in reward_dict.items():
                if isinstance(reward, torch.Tensor):
                    if reward.is_sparse:
                        reward = reward.to_dense()
                    total_reward = total_reward + reward
                else:
                    total_reward = total_reward + reward
        
        return total_reward
    
    def _compute_high_level_terminations(
        self,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute high-level terminations.
        
        Args:
            terminated: Low-level terminated flags
            truncated: Low-level truncated flags
        
        Returns:
            High-level terminated and truncated flags
        """
        # Check if goal is reached
        unwrapped_env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        distance = mdp.distance_to_goal(unwrapped_env).squeeze(-1)  # [num_envs]
        goal_reached = (distance < 1.0)  # threshold from goal_reached_bonus
        
        # High-level termination: goal reached or low-level terminated
        high_level_terminated = goal_reached | terminated.bool()
        
        # High-level truncation: same as low-level (timeout)
        high_level_truncated = truncated.bool()
        
        return high_level_terminated, high_level_truncated
    
    @property
    def device(self):
        """Get device from wrapped environment."""
        unwrapped_env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        return unwrapped_env.device
    
    @property
    def num_envs(self):
        """Get number of environments."""
        unwrapped_env = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        return unwrapped_env.num_envs
    
    def get_observations(self):
        """Get current high-level observations.
        
        This method is used by RslRlVecEnvWrapper to get observations.
        
        Returns:
            TensorDict with "policy" and "critic" keys containing high-level observations
        """
        from tensordict import TensorDict
        
        high_level_obs = self._get_high_level_obs()
        
        # Return as TensorDict with both "policy" and "critic" keys
        # RslRlVecEnvWrapper expects both observation groups
        return TensorDict({
            "policy": high_level_obs,
            "critic": high_level_obs.clone(),  # Same observations for critic
        }, batch_size=[high_level_obs.shape[0]])

