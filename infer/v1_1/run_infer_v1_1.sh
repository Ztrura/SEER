#!/usr/bin/env bash
set -euo pipefail

# GPUS="0 1 2 3 4 5 6 7" bash run_infer_v1_1.sh
GPUS=(${GPUS:-0 1 2 3})
NUM_SHARDS=${#GPUS[@]}

VERSION=v1_1
ADAPTERS=/home/dhm_41310/hdd/trzhang/models/SEER/GRPO_VLM/v9-20260418-135553/checkpoint-4099
PENALTY=1.0

SLEEP_SEC=30
INPUT_FILE=/home/dhm_41310/ssd/trzhang/SEER/SEER-Trace/seer_trace_test.jsonl
MODEL=/home/dhm_41310/hdd/trzhang/models/Qwen3-VL-4B-Instruct
VOXTELL_MODEL_DIR=${VOXTELL_MODEL_DIR:-/home/dhm_41310/hdd/trzhang/models/VoxTell/voxtell_v1.1}
OUTPUT_ROOT=/home/dhm_41310/ssd/trzhang/SEER/test_final

OUT_DIR=${OUTPUT_ROOT}/${VERSION}
SKILL_BANK_DIR=${OUTPUT_ROOT}/${VERSION}/skill_bank
PAR_DIR=${OUTPUT_ROOT}/${VERSION}/parallel
CACHE_DIR=${OUTPUT_ROOT}/${VERSION}/cache

PY=infer_v1_1.py
MERGE_PY=merge_v1_1.py

mkdir -p "${OUT_DIR}" "${SKILL_BANK_DIR}" "${PAR_DIR}" "${CACHE_DIR}"

run_one_round() {
  local round_name="$1"     # round0 / round1 / round2
  local mode="$2"           # round0_no_skill / roundN_with_skill
  local skill_in="$3"       # empty for round0, merged latest for roundN

  local use_min_score_args=()
  if [[ "${mode}" == "roundN_with_skill" ]]; then
    use_min_score_args=(--skill_use_min_score 18.0)
  fi

  local skill_in_args=()
  local merge_skill_in_args=()
  if [[ -n "${skill_in}" ]]; then
    skill_in_args=(--skill_bank_in "${skill_in}")
    merge_skill_in_args=(--skill_bank_in "${skill_in}")
  fi

  local round_dir="${PAR_DIR}/${round_name}"
  rm -rf "${round_dir}"
  mkdir -p "${round_dir}"

  echo "===== ${round_name}: ${NUM_SHARDS} shards on GPUs: ${GPUS[*]} ====="

  local pids=()
  for shard_id in "${!GPUS[@]}"; do
    local gpu="${GPUS[$shard_id]}"
    local shard_dir="${round_dir}/shard_${shard_id}"
    mkdir -p "${shard_dir}/skill_bank" "${shard_dir}/log"

    (
      set -euo pipefail
      echo "[START] $(date '+%F %T') ${round_name} shard=${shard_id}/${NUM_SHARDS} gpu=${gpu}"
      CUDA_VISIBLE_DEVICES="${gpu}" python "${PY}" \
        --mode "${mode}" \
        --round_name "${round_name}" \
        --input "${INPUT_FILE}" \
        --model "${MODEL}" \
        --adapters "${ADAPTERS}" \
        --voxtell_model_dir "${VOXTELL_MODEL_DIR}" \
        --detail_out "${shard_dir}/detail.jsonl" \
        --report_out "${shard_dir}/report.csv" \
        --skill_bank_dir "${shard_dir}/skill_bank" \
        "${skill_in_args[@]}" \
        --raw_dice_cache "${CACHE_DIR}/raw_shard_${shard_id}.json" \
        --audit_risk_penalty "${PENALTY}" \
        --skill_min_gain 0.3 \
        --skill_require_format \
        --freeze_skill_bank_during_round \
        --num_shards "${NUM_SHARDS}" \
        --shard_id "${shard_id}" \
        --new_skill_out "${shard_dir}/new_skills.jsonl" \
        --audit_out "${shard_dir}/audit_records.jsonl" \
        "${use_min_score_args[@]}" \
        > "${shard_dir}/log/stdout.log" 2> "${shard_dir}/log/stderr.log"
      echo "[DONE ] $(date '+%F %T') ${round_name} shard=${shard_id}/${NUM_SHARDS} gpu=${gpu}"
    ) &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "ERROR: ${round_name} has failed shard(s). Check ${round_dir}/shard_*/log/" >&2
    exit 1
  fi

  echo "===== ${round_name}: deterministic merge/replay ====="
  python "${MERGE_PY}" \
    --input "${INPUT_FILE}" \
    --mode "${mode}" \
    --round_name "${round_name}" \
    --skill_bank_dir "${SKILL_BANK_DIR}" \
    "${merge_skill_in_args[@]}" \
    --detail_glob "${round_dir}/shard_*/detail.jsonl" \
    --new_skill_glob "${round_dir}/shard_*/new_skills.jsonl" \
    --audit_glob "${round_dir}/shard_*/audit_records.jsonl" \
    --merged_detail "${OUT_DIR}/detail_r${round_name#round}.jsonl" \
    --merged_report "${OUT_DIR}/report_r${round_name#round}.csv" \
    --latency_summary_out "${OUT_DIR}/latency_r${round_name#round}.csv" \
    --merged_skill_latest "${SKILL_BANK_DIR}/latest.jsonl" \
    --merged_skill_round "${SKILL_BANK_DIR}/${round_name}.jsonl" \
    --audit_risk_penalty "${PENALTY}" \
    --skill_bank_max_ratio 0.05 \
    --skill_max_per_group 10

  echo "sleep ${SLEEP_SEC}s before next round..."
  sleep "${SLEEP_SEC}"
}

run_one_round "round0" "round0_no_skill" ""
run_one_round "round1" "roundN_with_skill" "${SKILL_BANK_DIR}/latest.jsonl"
run_one_round "round2" "roundN_with_skill" "${SKILL_BANK_DIR}/latest.jsonl"

echo "===== All rounds finished successfully ====="
