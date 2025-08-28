#!/usr/bin/env bash
set -euo pipefail

# DeepSeek API keys (as provided)
export DEEPSEEK_API_KEYS="sk-4d502fe569664b97a40e89d1c544f290, sk-638a094c184f4624b2ac8a49afb4eb1a, sk-76aaab152a43454dac800f14be12fb26"

python sft_Qwen_train.py \
  --skip_train \
  --model_id /home/hubing/llm-adaptive-attacks/qwq32b-sft-output \
  --subset_from_indexed filtered_severe_prompts_indexed.jsonl \
  --subset_indices "201,209,729,839,901,1839,1945,2147,2238,2270,4318" \
  --subset_out /home/hubing/llm-adaptive-attacks/sft_subset_inputs.jsonl \
  --predict_on_test \
  --test_pred_out /home/hubing/llm-adaptive-attacks/sft_subset_predictedy.jsonl \
  --run_deepseek_on_test \
  --deepseek_out /home/hubing/llm-adaptive-attacks/sft_subset_deepseek_results.jsonl \
  --deepseek_model deepseek-reasoner \
  --deepseek_temperature 0.5 --deepseek_top_p 1.0 \
  --deepseek_max_tokens 2000 --deepseek_concurrency 12 --deepseek_timeout 60 \
  --bf16 \
  --gen_temperature 0.1 --gen_top_p 0.9 \
  --gen_max_new_tokens 856 --gen_batch_size 1
