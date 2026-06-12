# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GR00T-WholeBodyControl is NVIDIA's codebase for humanoid whole-body controllers. It contains three main subpackages:

- **`gear_sonic/`** — The primary SONIC training and deployment stack (PPO-based whole-body motion tracking)
- **`decoupled_wbc/`** — Legacy decoupled controller (RL for lower body + IK for upper body), used in GR00T N1.5/N1.6
- **`motionbricks/`** — Real-time latent generative model for interactive motion control (subproject)

**Target robot**: Unitree G1 (29 DOF body + hands). Two hand variants are supported:
- `dex3` — default Unitree DEX3 hands
- `brainco` — BrainCo hands (use `instantiate_g1_robot_model(hand_type="brainco")`)

## Environments & Installation

Each use case has its own isolated venv. Install scripts use `uv` and live in `install_scripts/`:

| Use case | Venv | Install command |
|---|---|---|
| MuJoCo simulation | `.venv_sim` | `bash install_scripts/install_mujoco_sim.sh` |
| VR teleoperation | `.venv_teleop` | `bash install_scripts/install_pico.sh` |
| Data collection | `.venv_data_collection` | `bash install_scripts/install_data_collection.sh` |
| Training (requires Isaac Lab separately) | Isaac Lab env | `pip install -e "gear_sonic/[training]"` |

Package-level extras to install manually:
```bash
pip install -e "gear_sonic/"                    # base deps only
pip install -e "gear_sonic/[sim]"               # adds mujoco, tyro, pyzmq, pin, etc.
pip install -e "gear_sonic/[teleop]"            # adds pyzmq, msgpack, pin, pyvista
pip install -e "gear_sonic/[training]"          # adds hydra, wandb, trl, accelerate
pip install -e "gear_sonic/[data_collection]"   # adds lerobot, av, pyzmq
```

**Git LFS**: Required for meshes and ONNX models. Run `git lfs pull` after cloning.

## Common Commands

### Linting & Formatting

```bash
# Check only (mirrors CI)
bash lint.sh
# or via make:
make run-checks        # isort + black + ruff check

# Auto-fix
bash lint.sh --fix
make format            # isort + black (no ruff)
```

Linters: `ruff` (line length 115, Python 3.10 target), `black` (line length 100), `isort` (black profile). The `external_dependencies/` directory is excluded from all linters.

### Environment Check

```bash
python check_environment.py            # full check
python check_environment.py --training # training-specific
python check_environment.py --deploy   # deployment-specific
```

### MuJoCo Simulation

```bash
# Open any MJCF scene in viewer (no-gravity by default)
python open_mj_scene.py [path/to/scene.xml]
# Default: gear_sonic/data/robot_model/model_data/g1/with_brainco/scene_41dof.xml

# Run the full sim loop (requires .venv_sim or gear_sonic[sim])
python gear_sonic/scripts/run_sim_loop.py --wbc_version sonic_model12
python gear_sonic/scripts/run_sim_loop.py --wbc_version sonic_model12_brainco  # brainco hands
```

### Training (requires Isaac Lab)

```bash
# Finetune from checkpoint (64+ GPUs recommended for full scale)
accelerate launch --num_processes=8 gear_sonic/train_agent_trl.py \
    +exp=manager/universal_token/all_modes/sonic_release \
    +checkpoint=sonic_release/last.pt \
    num_envs=4096 headless=True

# Evaluate
python gear_sonic/eval_agent_trl.py +exp=...
```

### Data Processing

```bash
# Convert Bones-SEED CSV data to motion library format
python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py \
    --input /path/to/bones_seed/g1/csv/ --output data/motion_lib_bones_seed/robot \
    --fps 30 --fps_source 120 --individual --num_workers 16

# Filter and copy converted data
python gear_sonic/data_process/filter_and_copy_bones_data.py \
    --source data/motion_lib_bones_seed/robot --dest data/motion_lib_bones_seed/robot_filtered
```

### Download Checkpoints

```bash
python download_from_hf.py           # inference checkpoint
python download_from_hf.py --training # training checkpoint + SMPL data
```

## Architecture

### `gear_sonic/` Package Structure

```
gear_sonic/
├── train_agent_trl.py      # PPO training entry point (Hydra + accelerate)
├── eval_agent_trl.py       # Evaluation entry point
├── trl/                    # Training library
│   ├── trainer/            # PPO trainer (ppo_trainer.py, ppo_trainer_aux_loss.py)
│   ├── modules/            # Actor-critic, UniversalTokenModule (FSQ bottleneck)
│   ├── losses/             # Auxiliary loss implementations
│   └── utils/              # Rotation conversion, torch transforms
├── envs/                   # Isaac Lab RL environments
│   ├── manager_env/        # Main modular tracking environment
│   └── wrapper/            # Environment wrappers
├── config/                 # Hydra configuration tree
│   ├── base.yaml           # Root config
│   ├── exp/                # Experiment presets (manager/, universal_token/)
│   ├── actor_critic/       # Network architecture configs
│   └── trainer/            # PPO hyperparameter configs
├── data/                   # Robot models and assets
│   ├── robot_model/        # RobotModel class + G1 instantiation
│   │   ├── instantiation/g1.py   # instantiate_g1_robot_model() factory
│   │   └── model_data/g1/        # with_dex3/ and with_brainco/ variants
│   └── assets/robot_description/ # MJCF, URDF, meshes
├── utils/
│   ├── mujoco_sim/         # MuJoCo simulation stack
│   │   ├── base_sim.py     # DefaultEnv + BaseSimulator (physics loop, viewers)
│   │   ├── configs.py      # BaseConfig / SimLoopConfig (dataclass-based)
│   │   ├── simulator_factory.py  # SimulatorFactory + init_channel
│   │   ├── robot.py        # Legacy Robot config container
│   │   └── wbc_configs/    # Per-model YAML configs (sonic_model12*.yaml)
│   └── teleop/             # ZMQ-based teleop utilities
└── scripts/                # Runnable entry points
    ├── run_sim_loop.py          # MuJoCo sim entry point
    ├── pico_manager_thread_server.py  # PICO VR stream server
    ├── launch_data_collection.py
    ├── run_data_exporter.py
    └── run_vla_inference.py
```

### Key Architectural Concepts

**SONIC Policy (UniversalTokenModule)**: The policy uses an encoder → FSQ quantizer → decoder pipeline. Multiple named encoders (`g1`, `smpl`, `teleop`) convert different observation types to a shared latent space, then a Finite Scalar Quantizer discretizes it. Configs live in `gear_sonic/config/actor_critic/`.

**Configuration system**: Training uses Hydra. The `SimLoopConfig` / `BaseConfig` dataclasses (in `configs.py`) load from YAML files in `utils/mujoco_sim/wbc_configs/`. `WBC_VERSIONS` constant (`sonic_model12`, `sonic_model12_brainco`) controls which YAML is loaded.

**MuJoCo sim loop**: `DefaultEnv` owns the MuJoCo model/data and computes PD torques from Unitree SDK commands. `BaseSimulator` wraps it with rate-limiting and viewer threads. Communication with the WBC policy happens via `UnitreeSdk2Bridge` (ZMQ-based Unitree SDK2 channel protocol).

**Robot model variants**: `instantiate_g1_robot_model(hand_type="dex3"|"brainco")` picks between `model_data/g1/with_dex3/` and `model_data/g1/with_brainco/`. The `robot.py` `Robot` class is legacy — prefer the `robot_model` package.

**Inter-process communication**: ZMQ is used throughout (sim ↔ policy ↔ teleop ↔ data collection). ZMQ header size is 1280 bytes (changed 2026-03-24). The `unitree_sdk2py` bridge implements the Unitree SDK2 channel protocol.


## Custom information & instructions

This repo is a fork from the NVIDIA repo.
We are working and focusing exclusively on gear_sonic and gear_sonic_deploy, do not look at all at motionbricks or decoupleb_wbc.
Originally, only the dex3 hands were supported. The brainco hands were added in a PR, as well as teleoperation using the Meta Quest.  
Now, we are focusing exclusively on the brainco hands version with the Meta quest teleoperation.  
The end goal of the repo is to teleoperate the G1, equipped with Brainco hands, to manipulate an object and record the trajectories (robot data + object state).  
The focus is now on testing the pose estimation pipeline in simulation.