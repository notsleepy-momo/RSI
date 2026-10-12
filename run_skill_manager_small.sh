#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
DEVICE_MAP="${DEVICE_MAP:-auto}"
MODEL_DTYPE="${MODEL_DTYPE:-float16}"

export ALFWORLD_DATA="${ALFWORLD_DATA:-${PROJECT_DIR}/data}"

mkdir -p "${ALFWORLD_DATA}" "${PROJECT_DIR}/results"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi

echo "[1/2] Ensuring official ALFWorld data are present..."
"${PYTHON_BIN}" "${PROJECT_DIR}/download_alfworld_data.py" --data-dir "${ALFWORLD_DATA}"

echo "[2/2] Running the multi-round Skill Manager experiment..."
"${PYTHON_BIN}" "${PROJECT_DIR}/ollama_adapter/skill_manager_experiment.py" \
  --model "${MODEL}" \
  --device-map "${DEVICE_MAP}" \
  --dtype "${MODEL_DTYPE}" \
  --split-manifest "${PROJECT_DIR}/configs/splits_small_5.json" \
  --output-dir "${PROJECT_DIR}/results/skill_manager_small" \
  "$@"

echo "Results: ${PROJECT_DIR}/results/skill_manager_small"
