#!/usr/bin/env bash
set -Eeuo pipefail

EXPECTED_ROOT="/data/share/hxd/zhenglu/iris"
ENV_NAME="zhenglu_iris"
ALLOW_FEWER_GPUS="${ALLOW_FEWER_GPUS:-0}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

timestamp="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs outputs
run_root="outputs/h100_4games_${timestamp}"
mkdir -p "${run_root}"
summary_file="logs/train_4games_${timestamp}.summary"

exec > >(tee -a "${summary_file}") 2>&1

echo "IRIS H100 four-game launcher"
echo "Timestamp: ${timestamp}"
echo "Repo root: $(pwd)"
echo "Expected server root: ${EXPECTED_ROOT}"
echo "Run root: ${run_root}"
echo "Summary: ${summary_file}"
if [[ "$(pwd)" != "${EXPECTED_ROOT}" ]]; then
  echo "[WARN] current directory is not ${EXPECTED_ROOT}; continuing."
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda command not found on PATH."
  exit 1
fi

conda_base="$(conda info --base)"
# shellcheck source=/dev/null
source "${conda_base}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

gpu_count="$(python - <<'PY'
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    print(0)
PY
)"
if [[ "${gpu_count}" -eq 0 ]] && command -v nvidia-smi >/dev/null 2>&1; then
  gpu_count="$(nvidia-smi -L | grep -c '^GPU ' || true)"
fi

echo "Visible CUDA devices: ${gpu_count}"
if [[ "${gpu_count}" -lt 4 ]]; then
  echo "[WARN] fewer than 4 CUDA devices are visible. Current preflight previously showed one H100 MIG 40GB instance."
  if [[ "${ALLOW_FEWER_GPUS}" != "1" ]]; then
    echo "[ERROR] refusing to start 4-game training. Set ALLOW_FEWER_GPUS=1 to run as many games as visible for debugging."
    exit 1
  fi
  echo "ALLOW_FEWER_GPUS=1 set; launching ${gpu_count} job(s) for debugging."
fi

games=(
  BreakoutNoFrameskip-v4
  BoxingNoFrameskip-v4
  SeaquestNoFrameskip-v4
  RoadRunnerNoFrameskip-v4
)

num_jobs=4
if [[ "${gpu_count}" -lt "${num_jobs}" ]]; then
  num_jobs="${gpu_count}"
fi

if [[ "${num_jobs}" -eq 0 ]]; then
  echo "[ERROR] no CUDA devices visible; cannot launch training."
  exit 1
fi

declare -a pids=()
declare -a job_names=()
declare -a job_logs=()

sanitize_game() {
  local game="$1"
  printf '%s\n' "${game}" | sed -E 's/NoFrameskip-v4$//; s/[^A-Za-z0-9_.-]+/_/g'
}

for idx in $(seq 0 $((num_jobs - 1))); do
  game="${games[$idx]}"
  short_game="$(sanitize_game "${game}")"
  job_log="logs/train_${short_game}_${timestamp}.log"
  job_run_dir="${run_root}/${short_game}"
  mkdir -p "${job_run_dir}"

  echo "Launching ${game} on visible CUDA device ${idx}"
  (
    export CUDA_VISIBLE_DEVICES="${idx}"
    export WANDB_MODE=disabled
    export HYDRA_FULL_ERROR=1
    export OMP_NUM_THREADS
    export MKL_NUM_THREADS
    python src/main.py \
      "env.train.id=${game}" \
      common.device=cuda:0 \
      wandb.mode=disabled \
      evaluation.tokenizer.save_reconstructions=False \
      "hydra.run.dir=${job_run_dir}"
  ) >"${job_log}" 2>&1 &

  pid=$!
  pids+=("${pid}")
  job_names+=("${game}")
  job_logs+=("${job_log}")
  echo "PID ${pid}: ${game} -> CUDA_VISIBLE_DEVICES=${idx}, log=${job_log}, run_dir=${job_run_dir}"
done

echo
echo "Waiting for ${#pids[@]} job(s)."
failed=0
for idx in "${!pids[@]}"; do
  pid="${pids[$idx]}"
  game="${job_names[$idx]}"
  job_log="${job_logs[$idx]}"
  if wait "${pid}"; then
    echo "[OK] ${game} finished successfully. Log: ${job_log}"
  else
    status=$?
    echo "[FAIL] ${game} failed with exit code ${status}. Log: ${job_log}"
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "[ERROR] one or more training jobs failed."
  exit 1
fi

echo "All training jobs completed successfully."
