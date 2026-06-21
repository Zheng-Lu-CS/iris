#!/usr/bin/env bash
set -Eeuo pipefail

EXPECTED_ROOT="/data/share/hxd/zhenglu/iris"
ENV_NAME="zhenglu_iris"
PYPI_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
PYPI_TRUSTED_HOST="pypi.tuna.tsinghua.edu.cn"
INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-0}"
LOG_DIR="${LOG_DIR:-logs/env_setup}"
TORCH_VERSION="${TORCH_VERSION:-2.4.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.19.1}"
TORCH_CUDA_TAG="${TORCH_CUDA_TAG:-cu121}"
PYTORCH_WHEEL_INDEX="${PYTORCH_WHEEL_INDEX:-https://mirrors.aliyun.com/pytorch-wheels/${TORCH_CUDA_TAG}}"
PYTORCH_OFFICIAL_INDEX="https://download.pytorch.org/whl/${TORCH_CUDA_TAG}"
PYTORCH_INSTALL_SOURCE="${PYTORCH_INSTALL_SOURCE:-mirror}"
PYTORCH_WHEEL_DIR="${PYTORCH_WHEEL_DIR:-}"
ALLOW_OPENCV_HEADLESS_FALLBACK="${ALLOW_OPENCV_HEADLESS_FALLBACK:-1}"
OPENCV_PACKAGE="${OPENCV_PACKAGE:-auto}"
PIP_TIMEOUT="${PIP_TIMEOUT:-120}"
PIP_RETRIES="${PIP_RETRIES:-10}"

timestamp="$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"
log_file="${LOG_DIR}/setup_zhenglu_iris_env_${timestamp}.log"
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
echo "PyTorch install source: ${PYTORCH_INSTALL_SOURCE}"
echo "PyTorch wheel index: ${PYTORCH_WHEEL_INDEX}"
echo "OpenCV headless fallback: ${ALLOW_OPENCV_HEADLESS_FALLBACK}"
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

has_runtime_library() {
  local lib="$1"
  command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -Fq "${lib}"
}

if [[ "${OPENCV_PACKAGE}" == "auto" ]]; then
  if [[ "${ALLOW_OPENCV_HEADLESS_FALLBACK}" == "1" ]] && ! has_runtime_library "libGL.so.1"; then
    OPENCV_PACKAGE="opencv-python-headless"
    echo "libGL.so.1 is not visible via ldconfig; using ${OPENCV_PACKAGE}==4.8.1.78 for headless training startup."
  else
    OPENCV_PACKAGE="opencv-python"
    echo "Using ${OPENCV_PACKAGE}==4.8.1.78."
  fi
else
  echo "Using requested OpenCV package: ${OPENCV_PACKAGE}==4.8.1.78."
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
  --timeout "${PIP_TIMEOUT}" --retries "${PIP_RETRIES}" \
  -i "${PYPI_MIRROR}" --trusted-host "${PYPI_TRUSTED_HOST}" \
  pip==23.0.1 setuptools==65.5.0 wheel==0.38.4

if python - <<PY
import sys
try:
    import torch
    import torchvision
except Exception as exc:
    print(f"PyTorch import check failed: {exc!r}")
    sys.exit(1)
torch_ok = torch.__version__.startswith("${TORCH_VERSION}+${TORCH_CUDA_TAG}")
vision_ok = torchvision.__version__.startswith("${TORCHVISION_VERSION}+${TORCH_CUDA_TAG}") or torchvision.__version__.startswith("${TORCHVISION_VERSION}")
cuda_ok = getattr(torch.version, "cuda", None) == "12.1"
print("existing torch:", torch.__version__, "torch cuda:", getattr(torch.version, "cuda", None))
print("existing torchvision:", torchvision.__version__)
sys.exit(0 if (torch_ok and vision_ok and cuda_ok) else 1)
PY
then
  torch_ready=0
else
  torch_ready=1
fi

if [[ "${torch_ready}" -eq 0 ]]; then
  echo "Requested H100-capable PyTorch stack is already installed; skipping PyTorch download."
else
  echo "Installing H100-capable PyTorch stack."
  echo "Default source is a domestic PyTorch wheel mirror to avoid very slow download.pytorch.org transfers."
  case "${PYTORCH_INSTALL_SOURCE}" in
    mirror)
      python -m pip install \
        --timeout "${PIP_TIMEOUT}" --retries "${PIP_RETRIES}" \
        --no-cache-dir \
        -i "${PYPI_MIRROR}" --trusted-host "${PYPI_TRUSTED_HOST}" \
        "torch==${TORCH_VERSION}+${TORCH_CUDA_TAG}" \
        "torchvision==${TORCHVISION_VERSION}+${TORCH_CUDA_TAG}" \
        -f "${PYTORCH_WHEEL_INDEX}/"
      ;;
    official)
      python -m pip install \
        --timeout "${PIP_TIMEOUT}" --retries "${PIP_RETRIES}" \
        --no-cache-dir \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" \
        --index-url "${PYTORCH_OFFICIAL_INDEX}"
      ;;
    conda)
      conda install -y \
        "pytorch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" pytorch-cuda=12.1 \
        -c pytorch -c nvidia
      ;;
    local)
      if [[ -z "${PYTORCH_WHEEL_DIR}" || ! -d "${PYTORCH_WHEEL_DIR}" ]]; then
        echo "[ERROR] PYTORCH_INSTALL_SOURCE=local requires PYTORCH_WHEEL_DIR to point to a directory containing torch/torchvision wheels."
        exit 1
      fi
      python -m pip install \
        --timeout "${PIP_TIMEOUT}" --retries "${PIP_RETRIES}" \
        --no-index --find-links "${PYTORCH_WHEEL_DIR}" \
        "torch==${TORCH_VERSION}+${TORCH_CUDA_TAG}" \
        "torchvision==${TORCHVISION_VERSION}+${TORCH_CUDA_TAG}"
      ;;
    skip)
      echo "PYTORCH_INSTALL_SOURCE=skip set; skipping PyTorch installation."
      ;;
    *)
      echo "[ERROR] unknown PYTORCH_INSTALL_SOURCE=${PYTORCH_INSTALL_SOURCE}; use mirror, official, conda, local, or skip."
      exit 1
      ;;
  esac
fi

if [[ "${OPENCV_PACKAGE}" == "opencv-python-headless" || "${OPENCV_PACKAGE}" == "opencv-python" ]]; then
  echo "Resetting OpenCV wheels before installing ${OPENCV_PACKAGE}==4.8.1.78."
  python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless || true
  python -m pip install \
    --timeout "${PIP_TIMEOUT}" --retries "${PIP_RETRIES}" \
    -i "${PYPI_MIRROR}" --trusted-host "${PYPI_TRUSTED_HOST}" \
    --no-deps "${OPENCV_PACKAGE}==4.8.1.78"
else
  echo "[ERROR] OPENCV_PACKAGE must be auto, opencv-python, or opencv-python-headless; got ${OPENCV_PACKAGE}."
  exit 1
fi

echo "Installing IRIS Python dependencies with Tsinghua PyPI mirror."
python -m pip install \
  --timeout "${PIP_TIMEOUT}" --retries "${PIP_RETRIES}" \
  -i "${PYPI_MIRROR}" --trusted-host "${PYPI_TRUSTED_HOST}" \
  numpy==1.23.5 \
  ale-py==0.7.4 \
  einops==0.3.2 \
  "gym[accept-rom-license]==0.21.0" \
  hydra-core==1.1.1 \
  "${OPENCV_PACKAGE}==4.8.1.78" \
  Pillow==9.5.0 \
  "protobuf==3.20.*" \
  psutil==5.8.0 \
  pygame==2.1.2 \
  requests \
  tqdm==4.66.4 \
  wandb==0.12.7 \
  "AutoROM[accept-rom-license]==0.4.2"

echo "Attempting Atari ROM installation. Missing command/module variants are non-fatal."
if AutoROM --accept-license; then
  autorom_status_cli=0
else
  autorom_status_cli=$?
fi
if python -m autorom --accept-license; then
  autorom_status_module=0
else
  autorom_status_module=$?
fi
echo "AutoROM command status: ${autorom_status_cli}"
echo "python -m autorom status: ${autorom_status_module}"
if [[ "${autorom_status_cli}" -ne 0 && "${autorom_status_module}" -ne 0 ]]; then
  echo "[WARN] both AutoROM invocation forms failed; check the log before running Atari training."
fi

echo "Checking OpenCV runtime import."
if python - <<'PY'
import cv2
print("cv2:", cv2.__version__)
PY
then
  echo "[OK] cv2 imports with ${OPENCV_PACKAGE}."
else
  echo "[WARN] cv2 import failed. On headless servers this is commonly caused by missing libGL.so.1."
  if [[ "${ALLOW_OPENCV_HEADLESS_FALLBACK}" == "1" ]]; then
    echo "Replacing opencv-python with opencv-python-headless==4.8.1.78 so IRIS can import cv2 without system OpenGL libraries."
    python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless || true
    python -m pip install \
      --timeout "${PIP_TIMEOUT}" --retries "${PIP_RETRIES}" \
      -i "${PYPI_MIRROR}" --trusted-host "${PYPI_TRUSTED_HOST}" \
      --no-deps opencv-python-headless==4.8.1.78
    python - <<'PY'
import cv2
print("cv2 headless:", cv2.__version__)
PY
  else
    echo "[WARN] OpenCV headless fallback disabled. Install system package libgl1, or rerun with ALLOW_OPENCV_HEADLESS_FALLBACK=1."
  fi
fi

check_runtime_libraries() {
  echo
  echo "===== runtime library precheck ====="
  local missing=0
  local libs=(
    "libGL.so.1"
    "libSM.so.6"
    "libXrender.so.1"
    "libfontconfig.so.1"
    "libSDL2-2.0.so.0"
  )
  if ! command -v ldconfig >/dev/null 2>&1; then
    echo "[WARN] ldconfig not found; skipping system shared-library precheck."
    return 0
  fi
  for lib in "${libs[@]}"; do
    if ldconfig -p 2>/dev/null | grep -Fq "${lib}"; then
      echo "[OK] ${lib}"
    else
      echo "[MISS] ${lib}"
      missing=1
    fi
  done
  if [[ "${missing}" -ne 0 ]]; then
    echo "[WARN] Some system runtime libraries are missing. If you want full non-headless OpenCV/pygame/video support, install:"
    echo "  sudo apt-get update && sudo apt-get install -y ${apt_packages[*]}"
    echo "[WARN] The script will continue and report smoke-test status instead of aborting."
  fi
  return 0
}

check_runtime_libraries

run_diag() {
  local name="$1"
  shift
  echo
  echo "===== ${name} ====="
  trap - ERR
  set +e
  "$@"
  local status=$?
  set -e
  trap on_error ERR
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
import traceback

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

failures = []
for label, module_name in modules:
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"{label}: {version}")
    except Exception as exc:
        failures.append((label, module_name, repr(exc)))
        print(f"{label}: FAIL {exc!r}")
        traceback.print_exc()

if failures:
    raise SystemExit(f"import failures: {failures}")
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
