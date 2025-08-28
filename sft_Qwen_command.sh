cat > sft_Qwen_command.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail

pip3 install -U pip
pip3 install -r requirements.txt

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_PROJECT=qwq32b-sft
mkdir -p logs

export DEEPSEEK_API_KEYS="sk-4d502fe569664b97a40e89d1c544f290,sk-638a094c184f4624b2ac8a49afb4eb1a,sk-76aaab152a43454dac800f14be12fb26"

DS_CFG='{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,
    "overlap_comm": true,
    "contiguous_gradients": true,
    "reduce_bucket_size": 5e8,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6
  },
  "gradient_accumulation_steps": 1,
  "train_micro_batch_size_per_gpu": 1
}'

DEBUG_SYS=1 PYTHONUNBUFFERED=1 accelerate launch --num_processes 8 sft_Qwen_train.py \
  --model_id Qwen/QwQ-32B \
  --output_dir /home/hubing/llm-adaptive-attacks/qwq32b-sft-output \
  --bf16 --gradient_checkpointing \
  --deepspeed_config "$DS_CFG" \
  --batch_size 1 --grad_accum 1 \
  --max_seq_len 4096 \
  --report_to wandb \
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
  --gen_temperature 0.1 --gen_top_p 0.9 \
  --gen_max_new_tokens 856 --gen_batch_size 1 \
  2>&1 | tee "logs/train_subset_$(date +'%Y%m%d_%H%M%S').log"
SH
chmod +x train_qwq32b_subset_acc.sh
