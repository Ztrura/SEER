#!/usr/bin/env bash
set -euo pipefail

# 用法：GPUS="0 1 2 3 4 5 6 7" bash run_infer_v1.sh
GPUS=(${GPUS:-3})
NUM_SHARDS=${#GPUS[@]}

VERSION=v1

INPUT_FILE=/home/dhm_41310/ssd/trzhang/SEER/infer/all_prompt_dataset.jsonl
MODEL=/home/dhm_41310/hdd/trzhang/models/GRPO_VLM_FULL_ROUND2/v1-20260224-084717/checkpoint-215
VOXTELL_MODEL_DIR=${VOXTELL_MODEL_DIR:-/home/dhm_41310/hdd/trzhang/models/VoxTell/voxtell_v1.1}
OUTPUT_ROOT=/home/dhm_41310/ssd/trzhang/SEER/test_final

OUT_DIR=${OUTPUT_ROOT}/${VERSION}
PAR_DIR=${OUTPUT_ROOT}/${VERSION}/parallel

PY=infer_v1.py
MERGE_PY=merge_v1.py

mkdir -p "${OUT_DIR}" "${PAR_DIR}"

ROUND_DIR="${PAR_DIR}/inference"
rm -rf "${ROUND_DIR}"
mkdir -p "${ROUND_DIR}"

echo "===== inference: ${NUM_SHARDS} shards on GPUs: ${GPUS[*]} ====="

pids=()
for shard_id in "${!GPUS[@]}"; do
  gpu="${GPUS[$shard_id]}"
  shard_dir="${ROUND_DIR}/shard_${shard_id}"
  mkdir -p "${shard_dir}/log"

  (
    set -euo pipefail
    echo "[START] $(date '+%F %T') shard=${shard_id}/${NUM_SHARDS} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" python "${PY}" \
      --input "${INPUT_FILE}" \
      --model "${MODEL}" \
      --voxtell_model_dir "${VOXTELL_MODEL_DIR}" \
      --detail_out "${shard_dir}/detail.jsonl" \
      --report_out "${shard_dir}/report.csv" \
      --num_shards "${NUM_SHARDS}" \
      --shard_id "${shard_id}" \
      > "${shard_dir}/log/stdout.log" 2> "${shard_dir}/log/stderr.log"
    echo "[DONE ] $(date '+%F %T') shard=${shard_id}/${NUM_SHARDS} gpu=${gpu}"
  ) &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "ERROR: inference has failed shard(s). Check ${ROUND_DIR}/shard_*/log/" >&2
  exit 1
fi

echo "===== deterministic merge ====="
python "${MERGE_PY}" \
  --detail_glob "${ROUND_DIR}/shard_*/detail.jsonl" \
  --merged_detail "${OUT_DIR}/detail_v1.jsonl" \
  --merged_report "${OUT_DIR}/report_v1.csv"

echo "===== Inference finished successfully ====="
