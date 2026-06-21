#!/usr/bin/env bash
set -Eeuo pipefail

ENV_NAME="zhenglu_iris"
GAME="${1:-BreakoutNoFrameskip-v4}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

timestamp="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs outputs
safe_game="$(printf '%s\n' "${GAME}" | sed -E 's/NoFrameskip-v4$//; s/[^A-Za-z0-9_.-]+/_/g')"
log_file="logs/smoke_${safe_game}_${timestamp}.log"
run_dir="outputs/smoke_${safe_game}_${timestamp}"

exec > >(tee -a "${log_file}") 2>&1

echo "IRIS one-game smoke run"
echo "Game: ${GAME}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_DEVICE}"
echo "Run dir: ${run_dir}"
echo "Log: ${log_file}"

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda command not found on PATH."
  exit 1
fi

conda_base="$(conda info --base)"
# shellcheck source=/dev/null
source "${conda_base}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS
export MKL_NUM_THREADS

python src/main.py \
  "env.train.id=${GAME}" \
  common.device=cuda:0 \
  common.epochs=1 \
  collection.train.config.num_steps=20 \
  training.tokenizer.steps_per_epoch=1 \
  training.world_model.steps_per_epoch=1 \
  training.actor_critic.steps_per_epoch=1 \
  evaluation.should=False \
  wandb.mode=disabled \
  "hydra.run.dir=${run_dir}"

echo "Smoke run finished successfully."
