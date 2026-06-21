# Zhenglu IRIS H100 Reproduction Notes

These helper scripts are intended for the server copy of this repository at:

```bash
/data/share/hxd/zhenglu/iris
```

They keep the official IRIS source and config files unchanged.

## Environment

Use the conda environment `zhenglu_iris` with Python 3.8. The setup script creates it if it does not already exist.

Official IRIS was developed with `torch==1.11.0` and `torchvision==0.12.0`, but these helpers do not use that stack by default on H100 because it predates Hopper/H100 support. The default setup installs:

```bash
torch==2.4.1 torchvision==0.19.1
```

from the official PyTorch CUDA 12.1 wheel index, while keeping the non-torch IRIS dependencies close to the official pins.

## Setup

From the server repo root:

```bash
cd /data/share/hxd/zhenglu/iris
bash scripts/setup_zhenglu_iris_env.sh
```

If the server is missing system packages such as `ffmpeg`, `xvfb`, OpenGL, SDL2, or X11 libraries, rerun with:

```bash
INSTALL_SYSTEM_DEPS=1 bash scripts/setup_zhenglu_iris_env.sh
```

The script writes a timestamped log under `logs/`.

## Smoke Test

Run a short debugging-only sanity check on one game and one visible GPU:

```bash
cd /data/share/hxd/zhenglu/iris
bash scripts/smoke_one_game.sh BreakoutNoFrameskip-v4
```

To select a different visible CUDA device:

```bash
CUDA_DEVICE=1 bash scripts/smoke_one_game.sh BoxingNoFrameskip-v4
```

This smoke run intentionally uses tiny overrides and should not be treated as a reproduction run.

## Four-Game H100 Training

Run one official-style training job per visible CUDA device:

```bash
cd /data/share/hxd/zhenglu/iris
bash scripts/run_4games_h100.sh
```

The launcher starts:

- `BreakoutNoFrameskip-v4`
- `BoxingNoFrameskip-v4`
- `SeaquestNoFrameskip-v4`
- `RoadRunnerNoFrameskip-v4`

It uses `wandb.mode=disabled`, sets each process to `common.device=cuda:0` inside its own `CUDA_VISIBLE_DEVICES=<id>` scope, and writes logs under `logs/` plus Hydra outputs under `outputs/h100_4games_<timestamp>/`.

The current report showed only one visible H100 MIG 40GB instance. Four-game training needs four visible CUDA devices. If fewer than four are visible, the launcher exits by default. For debugging only, run as many jobs as visible with:

```bash
ALLOW_FEWER_GPUS=1 bash scripts/run_4games_h100.sh
```
