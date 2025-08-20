#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sft_Qwen_train.py

What this script does
---------------------
1) Read X from sft_Qwen_traindata.jsonl
   - Each line: {"text": "... original prompt: <PROMPT>\\n(output prompt:|Please put your changed prompt here:) ..."}
   - X IS NOT MODIFIED.
2) Read y from two files (STRICT keys for both):
   - sft_moderate_excludeD_changedprompts.jsonl: {"original prompt": "<PROMPT>", "output prompt": "<Y>"}
   - sft_severe_excludeHard_changedprompts.jsonl: {"original prompt": "<PROMPT>", "output prompt": "<Y>"}
3) Match by <PROMPT> (normalized), build (X, y) pairs, then train with TRL's SFTTrainer.

Example (8×H100):
  accelerate launch --num_processes 8 sft_Qwen_train.py \
    --model_id Qwen/Qwen1.5-1.8B-Chat --epochs 1 --batch_size 2 --grad_accum 4 \
    --bf16 --gradient_checkpointing --report_to none
"""

import argparse
import json
import os
import re
import unicodedata
from typing import List, Dict, Iterable, Tuple, Optional

# ---------------- I/O helpers ----------------
def load_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: line {i} invalid JSON: {e}")
    return out

def write_jsonl(path: str, records: Iterable[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")

# ---------------- text utils ----------------
def norm_text(s: str) -> str:
    """NFKC normalize, trim, collapse whitespace, lowercase."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()

# Accept both labels after the original prompt
EXTRACT_PATTERNS = [
    re.compile(
        r"original\s*prompt\s*:\s*(.*?)\r?\n\s*(?:output\s*prompt\s*:|please\s+put\s+your\s+changed\s+prompt\s+here\s*:)",
        flags=re.IGNORECASE | re.DOTALL,
    ),
    # Fallback if no trailing label was written (captures rest of text)
    re.compile(r"original\s*prompt\s*:\s*(.*)$", flags=re.IGNORECASE | re.DOTALL),
]

def extract_original_from_input_text(t: str) -> Optional[str]:
    if not isinstance(t, str):
        return None
    for rx in EXTRACT_PATTERNS:
        m = rx.search(t)
        if m:
            return m.group(1).strip()
    return None

# ---------------- y lookup (STRICT KEYS) ----------------
def build_lookup_from_file(path: str) -> Dict[str, str]:
    """
    Accepts ONLY: {"original prompt": "<PROMPT>", "output prompt": "<Y>"}
    Returns: norm(<PROMPT>) -> <Y>
    """
    lut = {}
    if not os.path.exists(path):
        print(f"[warn] missing y file: {path}")
        return lut
    for r in load_jsonl(path):
        p = r.get("original prompt")
        y = r.get("output prompt")
        if isinstance(p, str) and isinstance(y, str) and p.strip() and y.strip():
            key = norm_text(p)
            # keep longest target if duplicate
            if key not in lut or len(y) > len(lut[key]):
                lut[key] = y.strip()
    print(f"[y] {os.path.basename(path)} -> {len(lut)} entries")
    return lut

# ---------------- chat templating ----------------
def build_chat_from_user_text(tokenizer, user_text: str, target_text: str) -> str:
    """
    Prefer tokenizer.apply_chat_template; if unavailable, fall back to ChatML-like format
    used by Qwen chat models.
    """
    messages = [
        {"role": "system", "content": "You rewrite user prompts into improved prompts following the requested style."},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": target_text},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        # Fallback ChatML string
        return (
            "<|im_start|>system\nYou rewrite user prompts into improved prompts following the requested style.<|im_end|>\n"
            f"<|im_start|>user\n{user_text}<|im_end|>\n"
            f"<|im_start|>assistant\n{target_text}<|im_end|>\n"
        )

# --------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()

    # Inputs
    ap.add_argument("--inputs_file", default="sft_Qwen_traindata.jsonl",
                    help="X JSONL with a 'text' field (X is not modified).")
    ap.add_argument("--moderate_rewrites", default="sft_moderate_excludeD_changedprompts.jsonl",
                    help="Y (moderate): requires {'original prompt','output prompt'}")
    ap.add_argument("--severe_rewrites",   default="sft_severe_excludeHard_changedprompts.jsonl",
                    help="Y (severe): requires {'original prompt','output prompt'}")

    # Debug outputs
    ap.add_argument("--joined_preview_out", default="sft_joined_pairs_preview.jsonl")
    ap.add_argument("--unmatched_out", default="sft_unmatched_inputs.jsonl")

    # Train hyperparams
    ap.add_argument("--skip_train", action="store_true", help="Build/validate pairs only; do not train.")
    ap.add_argument("--model_id", default="Qwen/Qwen1.5-1.8B-Chat",
                    help="HF repo id of the base model (default: Qwen/Qwen1.5-1.8B-Chat).")
    ap.add_argument("--output_dir", default="qwen-sft-output")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_seq_len", type=int, default=1024)

    # Multi-GPU / H100 toggles
    ap.add_argument("--bf16", action="store_true", help="Enable bf16 mixed precision (recommended on H100).")
    ap.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    ap.add_argument("--deepspeed_config", default="", help="Path to Deepspeed JSON (optional).")

    # Logging
    ap.add_argument("--report_to", default="none", choices=["none", "wandb", "tensorboard"],
                    help="Where to report metrics.")
    ap.add_argument("--run_name", default="qwen-sft", help="Experiment/run name for loggers.")

    args = ap.parse_args()

    # 1) Load X (unchanged)
    X = load_jsonl(args.inputs_file)
    print(f"[inputs] loaded {len(X)} rows from {args.inputs_file}")

    # 2) Build y lookups (both files use 'original prompt' -> 'output prompt')
    lut = {}
    lut_mod = build_lookup_from_file(args.moderate_rewrites)
    lut_sev = build_lookup_from_file(args.severe_rewrites)
    lut.update(lut_mod)
    for k, v in lut_sev.items():
        lut.setdefault(k, v)

    # 3) Join X with y
    joined_pairs: List[Tuple[str, str]] = []
    unmatched: List[Dict] = []
    for i, rec in enumerate(X, 1):
        user_text = rec.get("text")
        if not isinstance(user_text, str):
            unmatched.append({"line": i, "reason": "no_text"}); continue
        orig = extract_original_from_input_text(user_text)
        if not orig:
            unmatched.append({"line": i, "reason": "cannot_extract_original",
                              "head": user_text[:120]}); continue
        y = lut.get(norm_text(orig))
        if y:
            joined_pairs.append((user_text, y))
        else:
            unmatched.append({"line": i, "original": orig})

    print(f"[join] matched={len(joined_pairs)}  unmatched={len(unmatched)}")

    # Debug dumps
    if joined_pairs:
        preview = [{"text": ut, "target": yt} for ut, yt in joined_pairs[:50]]
        write_jsonl(args.joined_preview_out, preview)
        print(f"[debug] wrote preview -> {args.joined_preview_out} ({len(preview)} rows)")
    if unmatched:
        write_jsonl(args.unmatched_out, unmatched)
        print(f"[debug] wrote unmatched -> {args.unmatched_out} ({len(unmatched)} rows)")

    if args.skip_train:
        print("[note] --skip_train set; not starting SFT.")
        return
    if not joined_pairs:
        print("[train] No matched pairs — aborting.")
        return

    # 4) Train
    from datasets import Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from trl import SFTTrainer, SFTConfig

    # tokenizer & model (explicit) — trust_remote_code for Qwen chat
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)

    model_kwargs = {"trust_remote_code": True}
    if args.bf16:
        import torch
        model_kwargs["torch_dtype"] = torch.bfloat16  # accelerate will handle device placement
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    texts = [build_chat_from_user_text(tok, ut, yt) for (ut, yt) in joined_pairs]
    ds = Dataset.from_list([{"text": t} for t in texts])

    cfg_kwargs = dict(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=200,
        max_seq_length=args.max_seq_len,
        packing=False,
        run_name=args.run_name,
    )
    if args.report_to != "none":
        cfg_kwargs["report_to"] = [args.report_to]
    if args.bf16:
        cfg_kwargs["bf16"] = True
    if args.gradient_checkpointing:
        cfg_kwargs["gradient_checkpointing"] = True
    if args.deepspeed_config:
        cfg_kwargs["deepspeed"] = args.deepspeed_config

    cfg = SFTConfig(**cfg_kwargs)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        train_dataset=ds,
        dataset_text_field="text",
        args=cfg,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"✅ Training finished. Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
