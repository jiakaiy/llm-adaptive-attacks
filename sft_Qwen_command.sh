#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="/home/hubing/llm-adaptive-attacks/qwq32b-sft-output"
SUBSET="/home/hubing/llm-adaptive-attacks/sft_subset_inputs.jsonl"
PRED="/home/hubing/llm-adaptive-attacks/sft_subset_predictedy.jsonl"
DEEP="/home/hubing/llm-adaptive-attacks/sft_subset_deepseek_results.jsonl"

# API keys (no spaces after commas)
export DEEPSEEK_API_KEYS="sk-4d502fe569664b97a40e89d1c544f290,sk-638a094c184f4624b2ac8a49afb4eb1a,sk-76aaab152a43454dac800f14be12fb26"

# If ZeRO-3 saved partitioned checkpoints, consolidate once:
if [ ! -f "$OUT_DIR/pytorch_model.bin" ] && [ ! -f "$OUT_DIR/model.safetensors" ]; then
  latest_ckpt=$(ls -dt "$OUT_DIR"/global_step* 2>/dev/null | head -n1 || true)
  if [ -n "${latest_ckpt:-}" ]; then
    echo "[info] consolidating ZeRO-3 shards from: $latest_ckpt"
    python -m deepspeed.utils.zero_to_fp32 "$latest_ckpt" "$OUT_DIR/pytorch_model.bin"
  else
    echo "[warn] no consolidated weights and no global_step* found in $OUT_DIR"
  fi
fi

# Make sure no deepspeed/accelerate flags leak into generation
unset ACCELERATE_USE_DEEPSPEED ACCELERATE_MIXED_PRECISION ACCELERATE_CPU_AFFINITY ACCELERATE_DEEPSPEED_CONFIG_FILE

mkdir -p logs

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
