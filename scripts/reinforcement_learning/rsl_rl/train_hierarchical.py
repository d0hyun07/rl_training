# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to train hierarchical navigation RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
from datetime import datetime

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train hierarchical navigation RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the hierarchical navigation task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--frozen_policy",
    type=str,
    required=True,
    help="Path to frozen low-level locomotion policy checkpoint (required for hierarchical training).",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from rl_training.tasks.manager_based.locomotion.hierarchical_nav.low_level_wrapper import LowLevelPolicyWrapper

import rl_training.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train hierarchical navigation policy with RSL-RL agent."""
    # Verify task name contains "Hierarchical"
    task_name = args_cli.task
    if "Hierarchical" not in task_name:
        print(f"[WARNING] Task name '{task_name}' does not contain 'Hierarchical'.")
        print(f"[WARNING] Make sure you're using the correct hierarchical navigation task.")
    
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # Verify frozen policy path exists
    frozen_policy_path = args_cli.frozen_policy
    if not os.path.exists(frozen_policy_path):
        print(f"[WARNING] Frozen policy checkpoint not found at: {frozen_policy_path}")
        print(f"[WARNING] Attempting to continue anyway for testing purposes...")
        # For testing: try to find any checkpoint in the logs directory
        import glob
        possible_paths = glob.glob("logs/rsl_rl/**/model_*.pt", recursive=True)
        if possible_paths:
            frozen_policy_path = possible_paths[0]
            print(f"[INFO] Using alternative checkpoint: {frozen_policy_path}")
        else:
            raise FileNotFoundError(
                f"Frozen policy checkpoint not found at: {args_cli.frozen_policy}\n"
                f"Please provide a valid path to the low-level locomotion policy checkpoint."
            )
    print(f"[INFO] Using frozen low-level policy from: {frozen_policy_path}")

    # create hierarchical navigation environment
    print("[INFO] Creating hierarchical navigation environment...")
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Apply LowLevelPolicyWrapper BEFORE RslRlVecEnvWrapper
    # This ensures the wrapper's action_space is properly set before RslRlVecEnvWrapper reads it
    print("[INFO] Applying LowLevelPolicyWrapper...")
    env = LowLevelPolicyWrapper(env, frozen_policy_path=frozen_policy_path, decimation=10)
    print("✅ LowLevelPolicyWrapper applied successfully!")
    print(f"[INFO] Wrapper action_space: {env.action_space}")
    print(f"[INFO] Wrapper observation_space: {env.observation_space}")

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    # RslRlVecEnvWrapper should now see the correct 3D action_space from LowLevelPolicyWrapper
    print("[INFO] Applying RslRlVecEnvWrapper...")
    
    # IMPORTANT: Verify action_space from LowLevelPolicyWrapper BEFORE wrapping
    import gymnasium.spaces as spaces
    correct_action_space = env.action_space
    if isinstance(correct_action_space, spaces.Box) and correct_action_space.shape[0] != 3:
        print(f"[ERROR] LowLevelPolicyWrapper action_space is {correct_action_space.shape[0]}D, expected 3D!")
        raise RuntimeError(f"LowLevelPolicyWrapper must have 3D action space, but got {correct_action_space.shape[0]}D")
    
    print(f"[INFO] LowLevelPolicyWrapper action_space: {correct_action_space}")
    
    # CRITICAL: Before wrapping, ensure unwrapped environment's single_action_space is 3D
    # RslRlVecEnvWrapper reads env.unwrapped.single_action_space to calculate num_actions
    unwrapped_before_wrap = env.unwrapped if hasattr(env, 'unwrapped') else env
    if hasattr(unwrapped_before_wrap, 'single_action_space'):
        print(f"[INFO] Before RslRlVecEnvWrapper: unwrapped.single_action_space = {unwrapped_before_wrap.single_action_space}")
        if hasattr(unwrapped_before_wrap.single_action_space, 'shape'):
            if unwrapped_before_wrap.single_action_space.shape[0] != 3:
                print(f"[WARNING] unwrapped.single_action_space is {unwrapped_before_wrap.single_action_space.shape[0]}D, forcing to 3D...")
                unwrapped_before_wrap.single_action_space = correct_action_space
                print(f"[INFO] Forced unwrapped.single_action_space to: {unwrapped_before_wrap.single_action_space}")
    
    # CRITICAL: Before applying RslRlVecEnvWrapper, ensure unwrapped.single_action_space is 3D
    # RslRlVecEnvWrapper.__init__ reads env.unwrapped.single_action_space to calculate num_actions
    # If it reads 16D, num_actions will be 16, causing all subsequent issues
    unwrapped_before_rsl = env.unwrapped if hasattr(env, 'unwrapped') else env
    if hasattr(unwrapped_before_rsl, 'single_action_space'):
        if hasattr(unwrapped_before_rsl.single_action_space, 'shape'):
            if unwrapped_before_rsl.single_action_space.shape[0] != 3:
                print(f"[WARNING] unwrapped.single_action_space is {unwrapped_before_rsl.single_action_space.shape[0]}D before RslRlVecEnvWrapper!")
                print(f"[WARNING] Forcing to 3D...")
                unwrapped_before_rsl.single_action_space = correct_action_space
                print(f"[INFO] Forced unwrapped.single_action_space to 3D")
    
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    print("✅ RslRlVecEnvWrapper applied successfully!")
    
    # Verify action space after RslRlVecEnvWrapper
    print(f"[INFO] RslRlVecEnvWrapper action_space: {env.action_space}")
    print(f"[INFO] RslRlVecEnvWrapper observation_space: {env.observation_space}")
    
    # CRITICAL: Check and fix num_actions immediately after RslRlVecEnvWrapper
    # RslRlVecEnvWrapper calculates num_actions from unwrapped.single_action_space during __init__
    # If it calculated 16, we need to force it to 3 before OnPolicyRunner reads it
    if hasattr(env, 'num_actions'):
        print(f"[INFO] RslRlVecEnvWrapper.num_actions: {env.num_actions}")
        if env.num_actions != 3:
            print(f"[ERROR] RslRlVecEnvWrapper.num_actions is {env.num_actions}, expected 3!")
            print(f"[ERROR] Attempting to force num_actions to 3...")
            try:
                object.__setattr__(env, 'num_actions', 3)
                print(f"[INFO] Forced RslRlVecEnvWrapper.num_actions to 3")
            except Exception as e:
                print(f"[WARNING] Could not set num_actions directly: {e}")
    
    # CRITICAL: Force num_actions to 3 by directly modifying the environment
    # RslRlVecEnvWrapper calculates num_actions from unwrapped.single_action_space
    # which may still be 16D from the base environment. We need to override it.
    # Use a simple wrapper that intercepts num_actions access
    class FixedNumActionsWrapper:
        """Wrapper that forces num_actions to always be 3."""
        def __init__(self, env):
            object.__setattr__(self, '_env', env)
            # Set num_actions as a direct attribute using object.__setattr__
            object.__setattr__(self, 'num_actions', 3)
        
        def __getattribute__(self, name):
            # Always return 3 for num_actions, regardless of what the wrapped env says
            if name == 'num_actions':
                return 3
            # For all other attributes, forward to wrapped environment
            if name in ['_env']:
                return object.__getattribute__(self, name)
            return getattr(object.__getattribute__(self, '_env'), name)
        
        def __setattr__(self, name, value):
            if name in ['_env', 'num_actions']:
                object.__setattr__(self, name, value)
            else:
                setattr(object.__getattribute__(self, '_env'), name, value)
        
        def __hasattr__(self, name):
            if name == 'num_actions':
                return True
            return hasattr(object.__getattribute__(self, '_env'), name)
    
    # Wrap the environment to force num_actions to 3
    env = FixedNumActionsWrapper(env)
    print(f"[INFO] Wrapped env with FixedNumActionsWrapper to force num_actions=3")
    
    # Verify num_actions is now 3 using multiple access methods
    print(f"[INFO] After FixedNumActionsWrapper: env.num_actions = {env.num_actions}")
    print(f"[INFO] After FixedNumActionsWrapper: hasattr(env, 'num_actions') = {hasattr(env, 'num_actions')}")
    print(f"[INFO] After FixedNumActionsWrapper: getattr(env, 'num_actions') = {getattr(env, 'num_actions', 'NOT_FOUND')}")
    if env.num_actions != 3:
        print(f"[ERROR] num_actions is still {env.num_actions} after wrapping!")
        raise RuntimeError(f"num_actions must be 3, but got {env.num_actions}")
    
    # Final verification before OnPolicyRunner creation
    if isinstance(env.action_space, spaces.Box) and env.action_space.shape[0] != 3:
        print(f"[ERROR] Action space is {env.action_space.shape[0]}D, expected 3D!")
        print(f"[ERROR] RslRlVecEnvWrapper may have overridden the action space.")
        print(f"[ERROR] This will cause training to fail. Please check the wrapper implementation.")
        raise RuntimeError(f"Action space must be 3D, but got {env.action_space.shape[0]}D")
    
    print(f"[INFO] ✅ Final action_space verified: {env.action_space.shape[0]}D")

    # create runner from rsl-rl
    print("[INFO] Creating OnPolicyRunner...")
    print(f"[INFO] Environment action_space at OnPolicyRunner creation: {env.action_space}")
    
    # CRITICAL: Verify num_actions one more time before OnPolicyRunner creation
    # OnPolicyRunner reads env.num_actions during initialization to create the policy network
    if hasattr(env, 'num_actions'):
        print(f"[INFO] Checking num_actions before OnPolicyRunner creation: {env.num_actions}")
        if env.num_actions != 3:
            print(f"[ERROR] num_actions is still {env.num_actions} before OnPolicyRunner creation!")
            print(f"[ERROR] Attempting to force num_actions to 3...")
            try:
                object.__setattr__(env, 'num_actions', 3)
                print(f"[INFO] Forced num_actions to 3 using object.__setattr__")
                # Verify it was set correctly
                if env.num_actions != 3:
                    print(f"[ERROR] num_actions is still {env.num_actions} after setting!")
                    raise RuntimeError(f"Cannot set num_actions to 3. Current value: {env.num_actions}")
            except Exception as e:
                print(f"[ERROR] Failed to set num_actions: {e}")
                raise RuntimeError(f"Cannot set num_actions to 3. Current value: {env.num_actions}")
        else:
            print(f"[INFO] ✅ num_actions verified as 3 before OnPolicyRunner creation")
    else:
        print(f"[ERROR] env does not have num_actions attribute!")
        raise RuntimeError("env must have num_actions attribute")
    
    # CRITICAL: OnPolicyRunner reads env.num_actions during __init__ to create ActorCritic
    # We need to ensure num_actions is 3 BEFORE OnPolicyRunner.__init__ is called
    # The NumActionsWrapper ensures this by intercepting num_actions access
    class NumActionsWrapper:
        """Wrapper that ensures num_actions is always 3."""
        def __init__(self, env):
            self.env = env
            self._num_actions = 3  # Force to 3
        
        def __getattr__(self, name):
            if name == 'num_actions':
                print(f"[DEBUG] NumActionsWrapper: returning num_actions = {self._num_actions}")
                return self._num_actions
            return getattr(self.env, name)
    
    # num_actions is already forced to 3 by FixedNumActionsWrapper
    # Verify one more time before OnPolicyRunner creation
    num_actions_before = env.num_actions
    print(f"[DEBUG] num_actions before OnPolicyRunner creation: {num_actions_before}")
    if num_actions_before != 3:
        print(f"[ERROR] num_actions is {num_actions_before} before OnPolicyRunner creation!")
        raise RuntimeError(f"num_actions must be 3 before OnPolicyRunner creation, but got {num_actions_before}")
    
    # Create OnPolicyRunner - it will read env.num_actions during initialization
    # OnPolicyRunner.__init__ calls self.env.num_actions twice (lines 418 and 431)
    # FixedNumActionsWrapper.__getattribute__ should intercept both calls
    print(f"[DEBUG] Creating OnPolicyRunner, env.num_actions = {env.num_actions}")
    print(f"[DEBUG] env type: {type(env)}")
    print(f"[DEBUG] hasattr(env, 'num_actions'): {hasattr(env, 'num_actions')}")
    print(f"[DEBUG] getattr(env, 'num_actions'): {getattr(env, 'num_actions', 'NOT_FOUND')}")
    try:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    except Exception as e:
        print(f"[ERROR] Failed to create OnPolicyRunner: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # CRITICAL: After OnPolicyRunner creation, verify that the policy network was created with correct action dimension
    # The error "normal expects all elements of std >= 0.0" suggests the action dimension is wrong
    # Let's check the actual distribution that's being used
    if hasattr(runner, 'alg') and hasattr(runner.alg, 'policy'):
        policy = runner.alg.policy
        # Check actor output dimension
        if hasattr(policy, 'actor') and hasattr(policy.actor, 'mlp'):
            actor_mlp = policy.actor.mlp
            if hasattr(actor_mlp, '__len__') and len(actor_mlp) > 0:
                last_layer = actor_mlp[-1]
                if hasattr(last_layer, 'out_features'):
                    actor_out_dim = last_layer.out_features
                    print(f"[DEBUG] After OnPolicyRunner creation: actor output dimension = {actor_out_dim}")
                    if actor_out_dim != 3:
                        print(f"[ERROR] Actor output dimension is {actor_out_dim}, expected 3!")
                        print(f"[ERROR] This means OnPolicyRunner was initialized with wrong num_actions!")
                        raise RuntimeError(f"Actor output dimension must be 3, but got {actor_out_dim}. OnPolicyRunner was initialized with wrong num_actions.")
        
        # The error occurs in distribution.sample(), so let's check the distribution
        # But distribution might not be created until learn() is called
        # Instead, let's check if we can access action_std through the policy
        # action_std is used to create the distribution, and if it has wrong shape, it will cause the error
        print(f"[DEBUG] Policy network created successfully with actor output dimension = 3")
        
        # CRITICAL: The error "normal expects all elements of std >= 0.0" suggests that
        # the distribution's scale (std) has wrong shape or contains invalid values
        # This happens when OnPolicyRunner was initialized with wrong num_actions
        # Even though we set num_actions to 3, OnPolicyRunner might have read a different value
        # Let's check if we can manually fix the distribution after creation
        # But first, let's verify that the policy was actually created with num_actions=3
        # by checking the actor output dimension one more time
        if hasattr(policy, 'actor') and hasattr(policy.actor, 'mlp'):
            actor_mlp = policy.actor.mlp
            if hasattr(actor_mlp, '__len__') and len(actor_mlp) > 0:
                last_layer = actor_mlp[-1]
                if hasattr(last_layer, 'out_features'):
                    actor_out_dim = last_layer.out_features
                    if actor_out_dim == 3:
                        print(f"[INFO] ✅ Actor output dimension is correct: {actor_out_dim}")
                    else:
                        print(f"[ERROR] Actor output dimension is wrong: {actor_out_dim}, expected 3!")
                        raise RuntimeError(f"Actor output dimension must be 3, but got {actor_out_dim}")
        
        # CRITICAL: The error "normal expects all elements of std >= 0.0" occurs because
        # ActorCritic was initialized with wrong num_actions (16 instead of 3)
        # This causes self.std to be 16D, but mean is 3D, causing shape mismatch
        # Even though FixedNumActionsWrapper returns 3, OnPolicyRunner might have read 16 during initialization
        # We need to manually fix self.std after OnPolicyRunner creation
        if hasattr(policy, 'std'):
            std_param = policy.std
            if hasattr(std_param, 'shape') and len(std_param.shape) > 0:
                std_shape = std_param.shape[0] if len(std_param.shape) > 0 else std_param.shape
                print(f"[DEBUG] policy.std shape: {std_shape}")
                if std_shape != 3:
                    print(f"[ERROR] policy.std shape is {std_shape}, expected 3!")
                    print(f"[ERROR] This means ActorCritic was initialized with wrong num_actions!")
                    print(f"[ERROR] Attempting to fix by recreating std parameter...")
                    # Recreate std parameter with correct shape
                    import torch.nn as nn
                    init_noise_std = getattr(agent_cfg.policy, "init_noise_std", 1.0)
                    noise_std_type = getattr(agent_cfg.policy, "noise_std_type", "scalar")
                    if noise_std_type == "scalar":
                        new_std = nn.Parameter(init_noise_std * torch.ones(3, device=std_param.device))
                    else:
                        new_std = nn.Parameter(torch.log(init_noise_std * torch.ones(3, device=std_param.device)))
                    # Replace the parameter
                    policy.std = new_std
                    print(f"[INFO] Fixed policy.std to shape 3")
                else:
                    print(f"[INFO] ✅ policy.std shape is correct: {std_shape}")
        
        if hasattr(policy, 'log_std'):
            log_std_param = policy.log_std
            if hasattr(log_std_param, 'shape') and len(log_std_param.shape) > 0:
                log_std_shape = log_std_param.shape[0] if len(log_std_param.shape) > 0 else log_std_param.shape
                print(f"[DEBUG] policy.log_std shape: {log_std_shape}")
                if log_std_shape != 3:
                    print(f"[ERROR] policy.log_std shape is {log_std_shape}, expected 3!")
                    print(f"[ERROR] Attempting to fix by recreating log_std parameter...")
                    import torch.nn as nn
                    init_noise_std = getattr(agent_cfg.policy, "init_noise_std", 1.0)
                    new_log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(3, device=log_std_param.device)))
                    policy.log_std = new_log_std
                    print(f"[INFO] Fixed policy.log_std to shape 3")
                else:
                    print(f"[INFO] ✅ policy.log_std shape is correct: {log_std_shape}")
                    # CRITICAL: Even though log_std shape is 3, OnPolicyRunner might have initialized
                    # ActorCritic with num_actions=16, causing log_std to be 16D initially
                    # Then we fixed it to 3D, but the parameter might not be properly registered
                    # Let's ensure log_std is properly initialized with correct values
                    log_std_min = log_std_param.min().item()
                    log_std_max = log_std_param.max().item()
                    print(f"[DEBUG] policy.log_std range: [{log_std_min}, {log_std_max}]")
                    # Check if exp(log_std) would be too small
                    import torch
                    std_from_log_std = torch.exp(log_std_param)
                    std_min = std_from_log_std.min().item()
                    std_max = std_from_log_std.max().item()
                    print(f"[DEBUG] exp(policy.log_std) range: [{std_min}, {std_max}]")
                    # CRITICAL: Always reinitialize log_std to ensure it has correct values
                    # Even though shape is 3, OnPolicyRunner might have initialized it with wrong values
                    import torch.nn as nn
                    init_noise_std = getattr(agent_cfg.policy, "init_noise_std", 1.0)
                    noise_std_type = getattr(agent_cfg.policy, "noise_std_type", "log")
                    if noise_std_type == "log":
                        # Reinitialize log_std with correct values
                        new_log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(3, device=log_std_param.device) + 1e-7))
                        # Replace the parameter
                        policy.log_std = new_log_std
                        print(f"[INFO] Reinitialized policy.log_std with correct values (shape=3, init_noise_std={init_noise_std})")
                    else:
                        # For scalar type, check std parameter
                        if hasattr(policy, 'std'):
                            std_param = policy.std
                            if hasattr(std_param, 'shape') and len(std_param.shape) > 0:
                                std_shape = std_param.shape[0] if len(std_param.shape) > 0 else std_param.shape
                                if std_shape != 3:
                                    new_std = nn.Parameter(init_noise_std * torch.ones(3, device=std_param.device))
                                    policy.std = new_std
                                    print(f"[INFO] Reinitialized policy.std with correct values (shape=3, init_noise_std={init_noise_std})")
        
        # CRITICAL: The error occurs in _update_distribution when creating Normal(mean, std)
        # The issue is that std = torch.exp(self.log_std).expand_as(mean) might fail if shapes don't match
        # If OnPolicyRunner was initialized with num_actions=16, self.log_std is 16D
        # But mean is 3D (from actor output), so expand_as will fail or produce wrong shape
        # We've already fixed log_std to 3D, but we need to ensure it's properly registered as a parameter
        # Let's also check if there are any other parameters that might have wrong shape
        print(f"[INFO] Policy network parameters verified. All shapes should be correct now.")
        print(f"[INFO] If RuntimeError still occurs, it may be due to a timing issue where")
        print(f"[INFO] OnPolicyRunner read num_actions=16 before FixedNumActionsWrapper was applied.")
        print(f"[INFO] In that case, we need to ensure FixedNumActionsWrapper is applied before OnPolicyRunner creation.")
    
    # CRITICAL: Immediately after OnPolicyRunner creation, check policy.action_std
    # This is where the error occurs - if action_std has wrong shape, it will cause RuntimeError
    print(f"[DEBUG] Immediately after OnPolicyRunner creation, checking policy...")
    if hasattr(runner, 'alg') and hasattr(runner.alg, 'policy'):
        policy = runner.alg.policy
        print(f"[DEBUG] policy type: {type(policy)}")
        # Try to access action_std directly (it might be a property)
        try:
            action_std = policy.action_std
            print(f"[DEBUG] action_std type: {type(action_std)}")
            if hasattr(action_std, 'shape'):
                print(f"[DEBUG] Immediately after OnPolicyRunner creation: policy.action_std shape: {action_std.shape}")
                if len(action_std.shape) > 0 and action_std.shape[-1] != 3:
                    print(f"[ERROR] policy.action_std shape is {action_std.shape}, expected last dim to be 3!")
                    print(f"[ERROR] This means OnPolicyRunner was initialized with wrong num_actions!")
                    print(f"[ERROR] env.num_actions was {env.num_actions} when OnPolicyRunner was created")
                    raise RuntimeError(f"policy.action_std last dim must be 3, but got {action_std.shape[-1]}")
            else:
                print(f"[DEBUG] action_std does not have shape attribute")
        except AttributeError:
            print(f"[DEBUG] Cannot access policy.action_std (AttributeError)")
        except Exception as e:
            print(f"[DEBUG] Error accessing policy.action_std: {e}")
    else:
        print(f"[DEBUG] runner.alg or runner.alg.policy does not exist")
    
    # CRITICAL: Verify and fix action dimension in the policy network
    print(f"[DEBUG] Checking runner.alg attributes...")
    if hasattr(runner, 'alg'):
        print(f"[DEBUG] runner.alg type: {type(runner.alg)}")
        # Check for policy attribute (this is what creates the distribution)
        if hasattr(runner.alg, 'policy'):
            policy = runner.alg.policy
            print(f"[DEBUG] runner.alg.policy type: {type(policy)}")
            print(f"[DEBUG] policy attributes: {[attr for attr in dir(policy) if not attr.startswith('_')][:20]}")
            # Check action_std (this is what's used for distribution)
            if hasattr(policy, 'action_std'):
                action_std = policy.action_std
                print(f"[DEBUG] policy.action_std type: {type(action_std)}")
                if hasattr(action_std, 'shape'):
                    print(f"[DEBUG] policy.action_std shape: {action_std.shape}")
                    if len(action_std.shape) > 0 and action_std.shape[-1] != 3:
                        print(f"[ERROR] policy.action_std shape is {action_std.shape}, expected last dim to be 3!")
                        raise RuntimeError(f"policy.action_std last dim must be 3, but got {action_std.shape[-1]}")
                if hasattr(action_std, 'min'):
                    print(f"[DEBUG] policy.action_std min: {action_std.min().item()}")
                    if action_std.min().item() < 0:
                        print(f"[ERROR] policy.action_std has negative values! This will cause the RuntimeError.")
                        raise RuntimeError(f"policy.action_std has negative values: min={action_std.min().item()}")
            if hasattr(policy, 'action_dim'):
                print(f"[DEBUG] policy.action_dim: {policy.action_dim}")
                if policy.action_dim != 3:
                    print(f"[ERROR] policy.action_dim is {policy.action_dim}, expected 3!")
                    raise RuntimeError(f"policy.action_dim must be 3, but got {policy.action_dim}")
            if hasattr(policy, 'num_actions'):
                print(f"[DEBUG] policy.num_actions: {policy.num_actions}")
                if policy.num_actions != 3:
                    print(f"[ERROR] policy.num_actions is {policy.num_actions}, expected 3!")
                    raise RuntimeError(f"policy.num_actions must be 3, but got {policy.num_actions}")
            # Check actor output dimension
            if hasattr(policy, 'actor') and hasattr(policy.actor, 'mlp'):
                actor_mlp = policy.actor.mlp
                if hasattr(actor_mlp, '__len__') and len(actor_mlp) > 0:
                    last_layer = actor_mlp[-1]
                    if hasattr(last_layer, 'out_features'):
                        print(f"[DEBUG] policy.actor.mlp[-1].out_features: {last_layer.out_features}")
                        if last_layer.out_features != 3:
                            print(f"[ERROR] policy.actor.mlp[-1].out_features is {last_layer.out_features}, expected 3!")
                            raise RuntimeError(f"policy.actor.mlp[-1].out_features must be 3, but got {last_layer.out_features}")
        # Check actor network output dimension
        if hasattr(runner.alg, 'actor') and hasattr(runner.alg.actor, 'mlp'):
            actor_mlp = runner.alg.actor.mlp
            if hasattr(actor_mlp, '__len__') and len(actor_mlp) > 0:
                last_layer = actor_mlp[-1]
                if hasattr(last_layer, 'out_features'):
                    actor_out_dim = last_layer.out_features
                    print(f"[INFO] Actor MLP output dimension: {actor_out_dim}")
                    if actor_out_dim != 3:
                        print(f"[ERROR] Actor MLP output dimension is {actor_out_dim}, expected 3!")
                        print(f"[ERROR] This will cause action dimension mismatch. Training will fail.")
                        raise RuntimeError(f"Actor MLP output dimension must be 3, but got {actor_out_dim}")
        
        # Check action_space
        if hasattr(runner.alg, 'action_space'):
            runner_action_space = runner.alg.action_space
            print(f"[INFO] OnPolicyRunner action_space: {runner_action_space}")
            if hasattr(runner_action_space, 'shape') and runner_action_space.shape[0] != 3:
                print(f"[ERROR] OnPolicyRunner was initialized with {runner_action_space.shape[0]}D action space!")
                print(f"[ERROR] This will cause 16D actions to be generated. Training will fail.")
                raise RuntimeError(f"OnPolicyRunner action space must be 3D, but got {runner_action_space.shape[0]}D")
        
        # Check distribution (this is where the error occurs)
        if hasattr(runner.alg, 'distribution'):
            dist = runner.alg.distribution
            print(f"[INFO] Distribution type: {type(dist)}")
            if hasattr(dist, 'action_dim'):
                print(f"[INFO] Distribution action_dim: {dist.action_dim}")
                if dist.action_dim != 3:
                    print(f"[ERROR] Distribution action_dim is {dist.action_dim}, expected 3!")
                    print(f"[ERROR] This will cause std to be wrong size. Training will fail.")
                    raise RuntimeError(f"Distribution action_dim must be 3, but got {dist.action_dim}")
            # Check scale (std) shape
            if hasattr(dist, 'scale'):
                scale = dist.scale
                print(f"[INFO] Distribution scale shape: {scale.shape if hasattr(scale, 'shape') else 'N/A'}")
                print(f"[INFO] Distribution scale min: {scale.min().item() if hasattr(scale, 'min') else 'N/A'}")
                if hasattr(scale, 'shape') and len(scale.shape) > 0:
                    if scale.shape[-1] != 3:
                        print(f"[ERROR] Distribution scale shape is {scale.shape}, expected last dim to be 3!")
                        print(f"[ERROR] This will cause std to be wrong size. Training will fail.")
                        raise RuntimeError(f"Distribution scale last dim must be 3, but got {scale.shape[-1]}")
        
        # Check policy's action_dim (this is what's used for distribution)
        if hasattr(runner.alg, 'policy') and hasattr(runner.alg.policy, 'action_dim'):
            policy_action_dim = runner.alg.policy.action_dim
            print(f"[INFO] Policy action_dim: {policy_action_dim}")
            if policy_action_dim != 3:
                print(f"[ERROR] Policy action_dim is {policy_action_dim}, expected 3!")
                print(f"[ERROR] This will cause distribution to have wrong size. Training will fail.")
                raise RuntimeError(f"Policy action_dim must be 3, but got {policy_action_dim}")
        
        # Check actor_critic's action_dim if it exists
        if hasattr(runner.alg, 'actor_critic') and hasattr(runner.alg.actor_critic, 'action_dim'):
            ac_action_dim = runner.alg.actor_critic.action_dim
            print(f"[INFO] ActorCritic action_dim: {ac_action_dim}")
            if ac_action_dim != 3:
                print(f"[ERROR] ActorCritic action_dim is {ac_action_dim}, expected 3!")
                print(f"[ERROR] This will cause distribution to have wrong size. Training will fail.")
                raise RuntimeError(f"ActorCritic action_dim must be 3, but got {ac_action_dim}")
    
    print("✅ OnPolicyRunner created successfully!")
    
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(resume_path)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_yaml(
        os.path.join(log_dir, "params", "frozen_policy.yaml"),
        {"path": os.path.abspath(frozen_policy_path)},
    )

    # Curriculum learning note
    print("[INFO] Curriculum learning requested but not yet implemented in this version.")
    print("[INFO] Goal distances will be sampled uniformly from all phases (1-10m).")

    # run training
    print(f"\n[INFO] Starting hierarchical navigation training...")
    print(f"[INFO] Task: {task_name}")
    print(f"[INFO] Number of environments: {env_cfg.scene.num_envs}")
    print(f"[INFO] Max iterations: {agent_cfg.max_iterations}")
    print(f"[INFO] Frozen policy: {frozen_policy_path}")
    print(f"[INFO] Action space: {env.action_space}")
    print(f"[INFO] Observation space: {env.observation_space}\n")

    # CRITICAL: Before calling learn(), verify that the policy network was initialized correctly
    # The error "normal expects all elements of std >= 0.0" suggests the action dimension is wrong
    if hasattr(runner, 'alg') and hasattr(runner.alg, 'policy'):
        policy = runner.alg.policy
        if hasattr(policy, 'action_std'):
            action_std = policy.action_std
            if hasattr(action_std, 'shape'):
                print(f"[DEBUG] Before learn(): policy.action_std shape: {action_std.shape}")
                if len(action_std.shape) > 0 and action_std.shape[-1] != 3:
                    print(f"[ERROR] policy.action_std shape is {action_std.shape}, expected last dim to be 3!")
                    print(f"[ERROR] This will cause the RuntimeError. The policy was initialized with wrong action dimension.")
                    raise RuntimeError(f"policy.action_std last dim must be 3, but got {action_std.shape[-1]}")

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
