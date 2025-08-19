#!/usr/bin/env bash
set -euo pipefail

# --- Make sure common python locations are on PATH (macOS/Homebrew/pyenv) ---
export PATH="/opt/homebrew/bin:/usr/local/bin:$HOME/.pyenv/shims:$PATH"

# --- point the OpenAI client to DeepSeek (openai-sdk compatible) ---
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.deepseek.com}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-https://api.deepseek.com}"

# --- disable W&B everywhere (no API key needed) ---
export WANDB_DISABLED="true"

# --- pick a Python executable ---
PY="$(command -v python3 || true)"
if [[ -z "${PY}" ]]; then
  PY="$(command -v python || true)"
fi
if [[ -z "${PY}" ]]; then
  echo "❌ Python not found. Install with Homebrew (brew install python) or activate your venv."
  exit 1
fi
echo "Using Python at: ${PY}"
"${PY}" --version

# === Your two DeepSeek keys (HARD-CODED) ===
KEY_A="sk-058ddf786d4a4a8c93699b9f52fc5ed4"
KEY_B="sk-ac93d577a9aa45feb5c0826dbc76b8a0"

mkdir -p logs

# Which rounds to run: all | round1 | round2
ROUND="${ROUND:-round2}"

# Overwrite policy: set to 1 to wipe Round-2 JSONLs and re-run from scratch
OVERWRITE_ROUND2="${OVERWRITE_ROUND2:-1}"

echo "🚀 Starting shards with round: $ROUND"
echo "   OVERWRITE_ROUND2=${OVERWRITE_ROUND2}"
echo

# --- helper: run a shard (optionally overwrite) ---
run_shard () {
  local KEY="$1"
  local START="$2"
  local COUNT="$3"
  local OUT="$4"
  local OVERWRITE="${5:-0}"   # 1 = pass --overwrite and (optionally) delete file first
  local FOREGROUND="${6:-0}"  # 1 = stream to console too
  local LOG="logs/${OUT%.jsonl}.log"

  # If overwriting, remove existing shard file first so we truly start fresh
  if [[ "$OVERWRITE" == "1" && -f "$OUT" ]]; then
    echo "🧹 Removing existing $OUT for a clean run..."
    rm -f "$OUT"
  fi

  local cli_overwrite=()
  if [[ "$OVERWRITE" == "1" ]]; then
    cli_overwrite=(--overwrite)
  fi

  echo "[$(date '+%F %T')] START ${OUT} (start=${START}, count=${COUNT}, overwrite=${OVERWRITE})" | tee -a "$LOG"

  if [[ "$FOREGROUND" == "1" ]]; then
    OPENAI_API_KEY="$KEY" "${PY}" phaseA_chunk.py \
      --start "${START}" --count "${COUNT}" --out "${OUT}" "${cli_overwrite[@]}" \
      --target-model deepseek-reasoner \
      --judge-model no-judge \
      --debug \
      2>&1 | tee -a "$LOG"
  else
    OPENAI_API_KEY="$KEY" "${PY}" phaseA_chunk.py \
      --start "${START}" --count "${COUNT}" --out "${OUT}" "${cli_overwrite[@]}" \
      --target-model deepseek-reasoner \
      --judge-model no-judge \
      --debug \
      >> "$LOG" 2>&1 &
  fi
}

# ---------------- Round 1 (shards 1–4) ----------------
if [[ "$ROUND" == "all" || "$ROUND" == "round1" ]]; then
  # Round 1 defaults to append (no overwrite)
  run_shard "$KEY_A"  560  500  run_A_0560_1059.jsonl 0 1
  run_shard "$KEY_B" 1060  500  run_A_1060_1559.jsonl 0 0
  run_shard "$KEY_A" 1560  500  run_A_1560_2059.jsonl 0 0
  run_shard "$KEY_B" 2060  500  run_A_2060_2559.jsonl 0 0
  wait
  echo "Round 1 complete."
fi

# ---------------- Round 2 (shards 5–8) ----------------
if [[ "$ROUND" == "all" || "$ROUND" == "round2" ]]; then
  # If OVERWRITE_ROUND2=1, wipe files so we truly start fresh
  if [[ "$OVERWRITE_ROUND2" == "1" ]]; then
    echo "🧨 Overwriting Round 2 outputs: removing existing JSONLs..."
    rm -f run_A_2560_3059.jsonl run_A_3060_3559.jsonl run_A_3560_4059.jsonl run_A_4060_4392.jsonl || true
  fi

  # Run Round 2; pass overwrite flag per OVERWRITE_ROUND2
  run_shard "$KEY_A" 2560  500  run_A_2560_3059.jsonl "$OVERWRITE_ROUND2" 1
  run_shard "$KEY_B" 3060  500  run_A_3060_3559.jsonl "$OVERWRITE_ROUND2" 0
  run_shard "$KEY_A" 3560  500  run_A_3560_4059.jsonl "$OVERWRITE_ROUND2" 0
  run_shard "$KEY_B" 4060  333  run_A_4060_4392.jsonl "$OVERWRITE_ROUND2" 0   # 4392 inclusive
  wait
  echo "All shards done."
fi

echo
echo "Tip: follow logs:"
echo "  tail -f logs/run_A_2560_3059.log logs/run_A_3060_3559.log logs/run_A_3560_4059.log logs/run_A_4060_4392.log"
