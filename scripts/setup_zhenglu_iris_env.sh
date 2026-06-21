#!/usr/bin/env bash
set -Eeuo pipefail

EXPECTED_ROOT="/data/share/hxd/zhenglu/iris"
ENV_NAME="zhenglu_iris"
PYPI_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
PYPI_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-0}"

timestamp="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs
log_file="logs/setup_zhenglu_iris_env_${timestamp}.log"
exec > >(tee -a "${log_file}") 2>&1

on_error() {
  local status=$?
  echo
  echo "[ERROR] setup failed at line ${BASH_LINENO[0]} with exit code ${status}"
  echo "[ERROR] full log: ${log_file}"
  exit "${status}"
}
trap on_error ERR

echo "IRIS server environment setup"
echo "Timestamp: ${timestamp}"
echo "Repo root: $(pwd)"
echo "Expected server root: ${EXPECTED_ROOT}"
echo "Log file: ${log_file}"
if [[ "$(pwd)" != "${EXPECTED_ROOT}" ]]; then
  echo "[WARN] current directory is not ${EXPECTED_ROOT}; continuing because the script may be inspected or staged elsewhere."
fi

apt_packages=(
  build-essential gcc g++ make cmake pkg-config
  ffmpeg xvfb
  libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libfontconfig1 libsdl2-2.0-0
  zlib1g-dev
)

if [[ "${INSTALL_SYSTEM_DEPS}" == "1" ]]; then
  if [[ "$(id -u)" == "0" ]]; then
    echo "Installing Ubuntu system dependencies with apt as root."
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y "${apt_packages[@]}"
  elif command -v sudo >/dev/null 2>&1; then
    echo "Installing Ubuntu system dependencies with sudo apt."
    sudo apt-get update
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "${apt_packages[@]}"
  else
    echo "[WARN] INSTALL_SYSTEM_DEPS=1 but neither root nor sudo is available."
    echo "Suggested command:"
    echo "  sudo apt-get update && sudo apt-get install -y ${apt_packages[*]}"
  fi
else
  echo "Skipping system package installation. To enable it, rerun with INSTALL_SYSTEM_DEPS=1."
  echo "Suggested command:"
  echo "  sudo apt-get update && sudo apt-get install -y ${apt_packages[*]}"
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda command not found on PATH."
  exit 1
fi

conda_base="$(conda info --base)"
# shellcheck source=/dev/null
source "${conda_base}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  echo "Creating conda environment ${ENV_NAME} with Python 3.8."
  conda create -n "${ENV_NAME}" -y python=3.8
else
  echo "Conda environment ${ENV_NAME} already exists."
fi

conda activate "${ENV_NAME}"
echo "Active Python: $(command -v python)"
python --version

echo "Pinning pip build frontend for gym==0.21 compatibility."
python -m pip install --upgrade \
  -i "${PYPI_MIRROR}" --trusted-host "${PYPI_TRUSTED_HOST}" \
  pip==23.0.1 setuptools==65.5.0 wheel==0.38.4

echo "Installing H100-capable PyTorch stack from the official CUDA 12.1 PyTorch wheel index."
python -m pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu121

echo "Installing IRIS Python dependencies with Tsinghua PyPI mirror."
python -m pip install \
  -i "${PYPI_MIRROR}" --trusted-host "${PYPI_TRUSTED_HOST}" \
  numpy==1.23.5 \
  ale-py==0.7.4 \
  einops==0.3.2 \
  "gym[accept-rom-license]==0.21.0" \
  hydra-core==1.1.1 \
  opencv-python==4.8.1.78 \
  Pillow==9.5.0 \
  "protobuf==3.20.*" \
  psutil==5.8.0 \
  pygame==2.1.2 \
  requests \
  tqdm==4.66.4 \
  wandb==0.12.7 \
  "AutoROM[accept-rom-license]==0.4.2"

echo "Attempting Atari ROM installation. Missing command/module variants are non-fatal."
set +e
AutoROM --accept-license
autorom_status_cli=$?
python -m autorom --accept-license
autorom_status_module=$?
set -e
echo "AutoROM command status: ${autorom_status_cli}"
echo "python -m autorom status: ${autorom_status_module}"
if [[ "${autorom_status_cli}" -ne 0 && "${autorom_status_module}" -ne 0 ]]; then
  echo "[WARN] both AutoROM invocation forms failed; check the log before running Atari training."
fi

run_diag() {
  local name="$1"
  shift
  echo
  echo "===== ${name} ====="
  set +e
  "$@"
  local status=$?
  set -e
  if [[ "${status}" -eq 0 ]]; then
    echo "[OK] ${name}"
  else
    echo "[FAIL] ${name} exited with status ${status}"
  fi
  return 0
}

run_diag "pip check" python -m pip check

run_diag "package imports and versions" python - <<'PY'
import importlib

modules = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("gym", "gym"),
    ("ale_py", "ale_py"),
    ("cv2", "cv2"),
    ("pygame", "pygame"),
    ("PIL", "PIL"),
    ("hydra", "hydra"),
    ("omegaconf", "omegaconf"),
    ("wandb", "wandb"),
    ("google.protobuf", "google.protobuf"),
]

for label, module_name in modules:
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", "unknown")
    print(f"{label}: {version}")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  run_diag "nvidia-smi" nvidia-smi
else
  echo
  echo "===== nvidia-smi ====="
  echo "[FAIL] nvidia-smi not found"
fi

run_diag "torch CUDA smoke test" python - <<'PY'
import torch
import torch.nn.functional as F

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
for idx in range(torch.cuda.device_count()):
    print(f"device {idx}:", torch.cuda.get_device_name(idx))

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")

device = torch.device("cuda:0")
a = torch.randn(1024, 1024, device=device)
b = torch.randn(1024, 1024, device=device)
c = a @ b
print("matmul mean:", float(c.mean().detach().cpu()))

x = torch.randn(2, 3, 64, 64, device=device)
w = torch.randn(8, 3, 3, 3, device=device)
y = F.conv2d(x, w, padding=1)
torch.cuda.synchronize()
print("conv2d shape:", tuple(y.shape))
PY

run_diag "Atari gym smoke test" python - <<'PY'
import gym

games = [
    "BreakoutNoFrameskip-v4",
    "BoxingNoFrameskip-v4",
    "SeaquestNoFrameskip-v4",
    "RoadRunnerNoFrameskip-v4",
]

failures = []
for game in games:
    try:
        env = gym.make(game)
        obs = env.reset()
        action = env.action_space.sample()
        step_result = env.step(action)
        env.close()
        print(f"{game}: reset obs={getattr(obs, 'shape', type(obs))}, step items={len(step_result)}")
    except Exception as exc:
        failures.append((game, repr(exc)))
        print(f"{game}: FAIL {exc!r}")

if failures:
    raise SystemExit(f"Atari failures: {failures}")
PY

run_diag "cv2 video writer smoke test" python - <<'PY'
import os
import tempfile

import cv2
import numpy as np

path = os.path.join(tempfile.gettempdir(), "iris_cv2_smoke.mp4")
writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
if not writer.isOpened():
    raise SystemExit("cv2.VideoWriter did not open")
for frame_idx in range(5):
    frame = np.full((64, 64, 3), frame_idx * 40, dtype=np.uint8)
    writer.write(frame)
writer.release()
print(path, os.path.getsize(path))
PY

run_diag "pygame dummy display smoke test" env SDL_VIDEODRIVER=dummy python - <<'PY'
import pygame

pygame.display.init()
screen = pygame.display.set_mode((64, 64))
screen.fill((20, 40, 60))
pygame.display.flip()
pygame.display.quit()
print("pygame dummy display ok")
PY

echo
echo "Setup finished. Review failures above if any diagnostics failed."
echo "Full log: ${log_file}"
