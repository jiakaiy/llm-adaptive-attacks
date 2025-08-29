#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="/home/hubing/llm-adaptive-attacks/qwq32b-sft-output"
SUBSET="/home/hubing/llm-adaptive-attacks/sft_subset_inputs.jsonl"
PRED="/home/hubing/llm-adaptive-attacks/sft_subset_predictedy.jsonl"
DEEP="/home/hubing/llm-adaptive-attacks/sft_subset_deepseek_results.jsonl"

# API keys (no spaces after commas)
export DEEPSEEK_API_KEYS="sk-4d502fe569664b97a40e89d1c544f290,sk-638a094c184f4624b2ac8a49afb4eb1a,sk-76aaab152a43454dac800f14be12fb26"

echo "[step] check consolidated weights in: $OUT_DIR"

# If ZeRO-3 saved partitioned checkpoints, consolidate once (robust: root + optional tag)
if [ ! -f "$OUT_DIR/pytorch_model.bin" ] && [ ! -f "$OUT_DIR/model.safetensors" ]; then
  echo "[info] no pytorch_model.bin/model.safetensors found; attempting zero_to_fp32"

  if [ -f "$OUT_DIR/latest" ]; then
    echo "[info] using 'latest' pointer in $OUT_DIR"
    set +e
    python -m deepspeed.utils.zero_to_fp32 "$OUT_DIR" "$OUT_DIR/pytorch_model.bin"
    rc=$?
    set -e
  else
    latest_tag="$(ls -1 "$OUT_DIR"/global_step* 2>/dev/null | sed 's#.*/##' | sort | tail -n1 || true)"
    if [ -n "${latest_tag:-}" ]; then
      echo "[info] found checkpoint tag: $latest_tag"
      set +e
      python -m deepspeed.utils.zero_to_fp32 "$OUT_DIR" "$OUT_DIR/pytorch_model.bin" --tag "$latest_tag"
      rc=$?
      set -e
    else
      echo "[warn] no 'latest' file or global_step* dirs in $OUT_DIR"
      rc=0
    fi
  fi

  if [ "${rc:-0}" -ne 0 ]; then
    echo "[warn] zero_to_fp32 failed with code $rc (continuing; generation may fail if no weights are present)"
  fi
fi

# Sanity: require config + some weight file before generation
if [ ! -f "$OUT_DIR/config.json" ]; then
  echo "[error] $OUT_DIR/config.json not found. Did training save the model folder here?"
  exit 1
fi
if [ ! -f "$OUT_DIR/pytorch_model.bin" ] && [ ! -f "$OUT_DIR/model.safetensors" ]; then
  echo "[error] No consolidated weights in $OUT_DIR (pytorch_model.bin/model.safetensors missing)."
  echo "        Re-run zero_to_fp32 manually with the correct --tag, then retry."
  exit 1
fi

# Make sure no deepspeed/accelerate flags leak into generation
unset ACCELERATE_USE_DEEPSPEED ACCELERATE_MIXED_PRECISION ACCELERATE_CPU_AFFINITY ACCELERATE_DEEPSPEED_CONFIG_FILE

mkdir -p logs

echo "[step] running generation + deepseek on subset indices"
python sft_Qwen_train.py \
  --skip_train \
  --model_id "$OUT_DIR" \
  --subset_from_indexed filtered_severe_prompts_indexed.jsonl \
  --subset_indices "201,209,729,839,901,1839,1945,2147,2238,2270,4318" \
  --subset_out "$SUBSET" \
  --predict_on_test \
  --test_pred_out "$PRED" \
  --run_deepseek_on_test \
  --deepseek_out "$DEEP" \
  --deepseek_model deepseek-reasoner \
  --deepseek_temperature 0.5 --deepseek_top_p 1.0 \
  --deepseek_max_tokens 2000 --deepseek_concurrency 12 --deepseek_timeout 60 \
  --bf16 \
  --gen_temperature 0.1 --gen_top_p 0.9 \
  --gen_max_new_tokens 856 --gen_batch_size 1 \
  2>&1 | tee "logs/test_subset_$(date +'%Y%m%d_%H%M%S').log"

echo "[done] outputs (if successful):"
echo "  $SUBSET"
echo "  $PRED"
echo "  $DEEP"
