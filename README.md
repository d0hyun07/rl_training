rl_training

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-2.3.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Discord](https://img.shields.io/badge/-Discord-5865F2?style=flat&logo=Discord&logoColor=white)](https://discord.gg/gdM9mQutC8)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--64-orange.svg)](https://releases.ubuntu.com/22.04/)
[![License](https://img.shields.io/badge/license-BSD%203--Clause-blue.svg)](https://opensource.org/license/bsd-3-clause)

## Tutorial Videos
We've released the following tutorials for training and deploying a reinforcement learning policy. Please check it out on [Bilibili](https://b23.tv/UoIqsFn) or [YouTube](https://youtube.com/playlist?list=PLy9YHJvMnjO0X4tx_NTWugTUMJXUrOgFH&si=pjUGF5PbFf3tGLFz)! 

## Overview

**rl_training** is a RL training library for deeprobotics robots, based on IsaacLab. The table below lists all available environments:

| Robot Model         | Environment Name (ID)                                      | Screenshot |
|---------------------|------------------------------------------------------------|------------|
| [Deeprobotics Lite3](https://www.deeprobotics.cn/robot/index/product1.html) | Rough-Deeprobotics-Lite3-v0 | <img src="./docs/imgs/deeprobotics_lite3.png" alt="Lite3" width="300">
| [Deeprobotics M20](https://www.deeprobotics.cn/robot/index/lynx.html) | Rough-Deeprobotics-M20-v0 | <img src="./docs/imgs/deeprobotics_m20.png" alt="deeprobotics_m20" width="300">

> [!NOTE]
> If you want to deploy policies in mujoco or real robots, please use the corresponding deploy repo in [Deep Robotics Github Center](https://github.com/DeepRoboticsLab).
## Contribution 

Everyone is welcome to contribute to this repo. If you discover a bug or optimize our training config, just submit a pull request and we will look into it.
## Installation

- Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html). We recommend using the conda installation as it simplifies calling Python scripts from the terminal.

- Clone this repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

  ```bash
  git clone --recurse-submodules https://github.com/DeepRoboticsLab/rl_training.git
  ```

- Using a python interpreter that has Isaac Lab installed, install the library

  ```bash
  python -m pip install -e source/rl_training
  ```

- Verify that the extension is correctly installed by running the following command to print all the available environments in the extension:

  ```bash
  python scripts/tools/list_envs.py
  ```

<details>

<summary>Setup as Omniverse Extension (Optional, click to expand)</summary>

We provide an example UI extension that will load upon enabling your extension defined in `source/rl_training/rl_training/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of your repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon** (☰), then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to `rl_trainingb/source`
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon** (☰), then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

</details>

## Try examples

Deeprobotics Lite3:

```bash
# Train
python scripts/reinforcement_learning/rsl_rl/train.py --task=Rough-Deeprobotics-Lite3-v0 --headless

# Play
python scripts/reinforcement_learning/rsl_rl/play.py --task=Rough-Deeprobotics-Lite3-v0 --num_envs=10
```

Deeprobotics M20:

```bash
# Train
python scripts/reinforcement_learning/rsl_rl/train.py --task=Rough-Deeprobotics-M20-v0 --headless

# Play
python scripts/reinforcement_learning/rsl_rl/play.py --task=Rough-Deeprobotics-M20-v0 --num_envs=10
```

> [!NOTE]
> If you want to control a **SINGLE ROBOT** with the keyboard during playback, add `--keyboard` at the end of the play script.
>
> ```
> Key bindings:
> ====================== ========================= ========================
> Command                Key (+ve axis)            Key (-ve axis)
> ====================== ========================= ========================
> Move along x-axis      Numpad 8 / Arrow Up       Numpad 2 / Arrow Down
> Move along y-axis      Numpad 4 / Arrow Right    Numpad 6 / Arrow Left
> Rotate along z-axis    Numpad 7 / Z              Numpad 9 / X
> ====================== ========================= ========================
> ```
* Record video of a trained agent (requires installing `ffmpeg`), add `--video --video_length 200`
* Play/Train with 32 environments, add `--num_envs 32`
* Play on specific folder or checkpoint, add `--load_run run_folder_name --checkpoint model.pt`
* Resume training from folder or checkpoint, add `--resume --load_run run_folder_name --checkpoint model.pt`

## Trained Results
You can download our trained examples for reference of the training process. This should be expected if you run our training process properly. [M20](https://drive.google.com/file/d/1PAGIHOI6Lsge_JA9kU3w7RU-GyqXybob/view?usp=drive_link) and [Lite3](https://drive.google.com/file/d/1EgCVQA552x3YiaRhPRv675TSd01yqF0c/view?usp=drive_link)



## Multi-gpu acceleration
* To train with multiple GPUs, use the following command, where --nproc_per_node represents the number of available GPUs:
    ```bash
    python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 scripts/reinforcement_learning/rsl_rl/train.py --task=<ENV_NAME> --headless 
    python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 scripts/reinforcement_learning/rsl_rl/train.py --task=Rough-Deeprobotics-Lite3-v0 --headless --distributed --num_envs=2048
    ```
* Note: each gpu will have the same number of envs specified in the config, to use the previous total number of envs, devide it by the number of gpus.
* To scale up training beyond multiple GPUs on a single machine, it is also possible to train across multiple nodes. To train across multiple nodes/machines, it is required to launch an individual process on each node.

    For the master node, use the following command, where --nproc_per_node represents the number of available GPUs, and --nnodes represents the number of nodes:
    ```bash
    python -m torch.distributed.run --nproc_per_node=2 --nnodes=2 --node_rank=0 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=localhost:5555 scripts/reinforcement_learning/rsl_rl/train.py --task=<ENV_NAME> --headless --distributed
    ```
    Note that the port (`5555`) can be replaced with any other available port.
    For non-master nodes, use the following command, replacing `--node_rank` with the index of each machine:
    ```bash
    python -m torch.distributed.run --nproc_per_node=2 --nnodes=2 --node_rank=1 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=ip_of_master_machine:5555 scripts/reinforcement_learning/rsl_rl/train.py --task=<ENV_NAME> --headless --distributed
    ```

## Tensorboard

To view tensorboard, run:

```bash
tensorboard --logdir=logs
```

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing. In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

**Note: Replace `<path-to-isaac-lab>` with your own IsaacLab path.**

```json
{
    "python.languageServer": "Pylance",
    "python.analysis.extraPaths": [
        "${workspaceFolder}/source/rl_training",
        "/<path-to-isaac-lab>/source/isaaclab",
        "/<path-to-isaac-lab>/source/isaaclab_assets",
        "/<path-to-isaac-lab>/source/isaaclab_mimic",
        "/<path-to-isaac-lab>/source/isaaclab_rl",
        "/<path-to-isaac-lab>/source/isaaclab_tasks",
    ]
}
```

### Clean USD Caches

Temporary USD files are generated in `/tmp/IsaacLab/usd_{date}_{time}_{random}` during simulation runs. These files can consume significant disk space and can be cleaned by:

```bash
rm -rf /tmp/IsaacLab/usd_*
```

## Hierarchical Navigation Training

### Overview

Hierarchical navigation training uses a two-level architecture:
- **High-level policy**: Learns to navigate to goals using velocity commands [vx, vy, vyaw]
- **Low-level policy**: Pre-trained locomotion policy (frozen) that converts velocity commands to joint actions

### Prerequisites

Before training the hierarchical navigation policy, you need a pre-trained low-level locomotion policy checkpoint. Train a low-level policy first:

```bash
# Train low-level locomotion policy for Deeprobotics M20
python scripts/reinforcement_learning/rsl_rl/train.py --task=Rough-Deeprobotics-M20-v0 --headless --num_envs 4096 --max_iterations 20000
```

The checkpoint will be saved in `logs/rsl_rl/deeprobotics_m20_rough/<run_name>/model_<iteration>.pt`

### Testing the Hierarchical Navigation Environment

Before training, you can test the hierarchical navigation environment setup:

```bash
# Test hierarchical navigation environment
# Requires: --checkpoint pointing to a low-level policy checkpoint
python scripts/reinforcement_learning/rsl_rl/test_hierarchical_nav.py \
    --task=Rough-Deeprobotics-M20-v0 \
    --headless \
    --num_envs 4 \
    --checkpoint logs/rsl_rl/deeprobotics_m20_rough/<run_name>/model_19999.pt
```

This will:
- Load the frozen low-level policy
- Create the hierarchical navigation environment
- Run a few test steps with random actions
- Verify that observations, rewards, and actions are working correctly

### Training the Hierarchical Navigation Policy

To train the hierarchical navigation policy:

```bash
# Train hierarchical navigation policy
# Requires: --frozen_policy pointing to a low-level policy checkpoint
python scripts/reinforcement_learning/rsl_rl/train_hierarchical.py \
    --task=Hierarchical-Nav-Deeprobotics-M20-v0 \
    --headless \
    --num_envs 4096 \
    --max_iterations 10000 \
    --frozen_policy logs/rsl_rl/deeprobotics_m20_rough/<run_name>/model_19999.pt
```

**Arguments:**
- `--task`: Must be `Hierarchical-Nav-Deeprobotics-M20-v0` for hierarchical navigation
- `--frozen_policy`: **Required**. Path to the pre-trained low-level locomotion policy checkpoint
- `--num_envs`: Number of parallel environments (default: 4096)
- `--max_iterations`: Number of training iterations (default: 10000)
- `--headless`: Run without GUI (required for headless servers)

**Training outputs:**
- Checkpoints: `logs/rsl_rl/hierarchical_nav/<run_name>/model_<iteration>.pt`
- TensorBoard logs: `logs/rsl_rl/hierarchical_nav/<run_name>/events.out.tfevents.*`
- Configuration files: `logs/rsl_rl/hierarchical_nav/<run_name>/params/`

### Monitoring Training Progress

**Option 1: TensorBoard (Recommended)**

```bash
# Start TensorBoard
tensorboard --logdir=logs/rsl_rl/hierarchical_nav --port=6006

# Open browser to http://localhost:6006
```

**Option 2: View Logs Script**

```bash
# View latest training logs
python scripts/view_training_logs.py
```

**Key metrics to monitor:**
- `Episode_Reward/progress_to_goal`: Should increase over time
- `Episode_Reward/goal_reached_bonus`: Should increase as agent learns to reach goals
- `Mean episode length`: Should increase as agent learns to navigate longer distances
- `Loss/value_function`: Should decrease over time

### Resuming Training

To resume training from a checkpoint:

```bash
python scripts/reinforcement_learning/rsl_rl/train_hierarchical.py \
    --task=Hierarchical-Nav-Deeprobotics-M20-v0 \
    --headless \
    --num_envs 4096 \
    --frozen_policy logs/rsl_rl/deeprobotics_m20_rough/<run_name>/model_19999.pt \
    --resume \
    --load_run <hierarchical_run_name> \
    --checkpoint model_<iteration>.pt
```

## Acknowledgements

The project uses some code from the following open-source code repositories:

- [fan-ziqi/robot_lab](https://github.com/fan-ziqi/robot_lab)
