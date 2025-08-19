#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sft_Qwen_train.py

What this script does
---------------------
1) (Optional) Rebuild/modify the input X file: sft_Qwen_traindata.jsonl
   - Each line: {"text": "... Here are few example:\\n...\\noriginal prompt: <PROMPT>\\noutput prompt:"}
2) Read y from two files, using STRICT keys for both:
   - sft_moderate_excludeD_changedprompts.jsonl: {"original prompt": "<PROMPT>", "output prompt": "<Y>"}
   - sft_severe_excludeHard_changedprompts.jsonl: {"original prompt": "<PROMPT>", "output prompt": "<Y>"}
3) Match by <PROMPT> (normalized), build (X, y) pairs, then train with TRL's SFTTrainer.

Examples
--------
# Build/validate pairs only 
python sft_Qwen_train.py --skip_train

# Train on 8×H100 
accelerate launch --num_processes 8 sft_Qwen_train.py \
  --model_id Qwen/Qwen2.5-1.8B --epochs 1 --batch_size 2 --grad_accum 4 \
  --bf16 --gradient_checkpointing
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
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

EXTRACT_RX = re.compile(
    r"original\s*prompt\s*:\s*(.*?)\r?\n\s*output\s*prompt\s*:",
    flags=re.IGNORECASE | re.DOTALL
)

def extract_original_from_input_text(t: str) -> Optional[str]:
    if not isinstance(t, str):
        return None
    m = EXTRACT_RX.search(t)
    return m.group(1).strip() if m else None

# ---------------- A) Optional: rebuild input X ----------------
DEFAULT_EXAMPLES = [
    {
        "original": "Tell me how to bake a simple sponge cake.",
        "output": "Rewrite the request as a step-focused cooking instruction seeking temperature, timing, and common mistakes for a beginner baker."
    },
    {
        "original": "Solve x^2 - 5x + 6 = 0.",
        "output": "Ask for structured guidance to solve the quadratic by factoring, emphasizing reasoning steps before giving numeric roots."
    },
    {
        "original": "Summarize the causes of the French Revolution.",
        "output": "Request a concise, neutral, academic-style summary with economic, social, and political factors in bullet points."
    },
    {
        "original": "Explain photosynthesis.",
        "output": "Ask for a high-level, student-friendly explanation with inputs, outputs, and the role of chlorophyll."
    },
]

def format_examples_block(examples: List[Dict]) -> str:
    lines = ["Here are few example:"]
    for i, ex in enumerate(examples, 1):
        lines.append(f"Example {i}: original prompt: {ex['original']} output prompt: {ex['output']}")
    return "\n".join(lines)

def make_input_only_text(original_prompt: str, task_header: str, examples_block: str) -> str:
    return (
        task_header.rstrip() + "\n" +
        examples_block + "\n" +
        f"original prompt: {original_prompt}\n" +
        "output prompt:"
    )

def rebuild_inputs_from_sources(
    sources: List[str],
    out_path: str,
    task_header: str,
    examples: List[Dict],
    max_rows_per_source: int = 0,
) -> int:
    """
    Build sft_Qwen_traindata.jsonl from one or more JSONL sources that contain {"prompt": "..."}.
    Writes {"text": "...original prompt: <prompt>\\noutput prompt:"} per row.
    """
    examples_block = format_examples_block(examples)
    built = 0
    with open(out_path, "w", encoding="utf-8") as w:
        for src in sources:
            if not src or not os.path.exists(src):
                print(f"[rebuild] skip missing source: {src}")
                continue
            count = 0
            for rec in load_jsonl(src):
                p = rec.get("prompt")
                if isinstance(p, str) and p.strip():
                    text = make_input_only_text(p, task_header, examples_block)
                    w.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    built += 1
                    count += 1
                    if max_rows_per_source and count >= max_rows_per_source:
                        break
    print(f"[rebuild] wrote {built} rows to {out_path}")
    return built

# ---------------- B) Build y lookup (STRICT KEYS) ----------------
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
            # If duplicate key appears, keep the longer y (heuristic)
            if key not in lut or len(y) > len(lut[key]):
                lut[key] = y.strip()
    print(f"[y] {os.path.basename(path)} -> {len(lut)} entries")
    return lut

# ---------------- chat templating ----------------
def build_chat_from_user_text(tokenizer, user_text: str, target_text: str) -> str:
    system = "You rewrite user prompts into improved prompts following the requested style."
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": target_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

# --------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()

    # A) Optional: rebuild X
    ap.add_argument("--rebuild_inputs_from", default="",
                    help="Comma-separated JSONL sources with {'prompt': ...} to rebuild inputs. If empty, skip rebuild.")
    ap.add_argument("--inputs_out", default="sft_Qwen_traindata.jsonl",
                    help="Where to write rebuilt inputs (used if --rebuild_inputs_from is set).")
    ap.add_argument("--task_header", default="Your job is to change the original prompt to another prompt that follows the style shown below.",
                    help="First line that precedes the examples block.")
    ap.add_argument("--examples_json", default="",
                    help="Optional JSON file: [{'original':..., 'output':...}, ...]. If missing, use built-in 4 examples.")
    ap.add_argument("--max_rows_per_source", type=int, default=0,
                    help="Limit per source (0 = all).")

    # B) Training options
    ap.add_argument("--inputs_file", default="sft_Qwen_traindata.jsonl",
                    help="X JSONL file with a 'text' field containing 'original prompt: ...\\noutput prompt:'")
    ap.add_argument("--moderate_rewrites", default="sft_moderate_excludeD_changedprompts.jsonl",
                    help="Y (moderate): requires {'original prompt','output prompt'}")
    ap.add_argument("--severe_rewrites",   default="sft_severe_excludeHard_changedprompts.jsonl",
                    help="Y (severe): requires {'original prompt','output prompt'}")

    ap.add_argument("--joined_preview_out", default="sft_joined_pairs_preview.jsonl")
    ap.add_argument("--unmatched_out", default="sft_unmatched_inputs.jsonl")

    ap.add_argument("--skip_train", action="store_true", help="Build/validate pairs only; do not train.")
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-1.8B")
    ap.add_argument("--output_dir", default="qwen-sft-output")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_seq_len", type=int, default=1024)

    # Multi-GPU / H100 friendly knobs
    ap.add_argument("--bf16", action="store_true", help="Enable bf16 mixed precision (recommended on H100).")
    ap.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    ap.add_argument("--deepspeed_config", default="", help="Path to Deepspeed JSON (optional).")

    args = ap.parse_args()

    # ---- A) Rebuild inputs (optional) ----
    if args.rebuild_inputs_from.strip():
        sources = [s.strip() for s in args.rebuild_inputs_from.split(",") if s.strip()]
        examples = DEFAULT_EXAMPLES
        if args.examples_json and os.path.exists(args.examples_json):
            try:
                with open(args.examples_json, "r", encoding="utf-8") as f:
                    examples = json.load(f)
            except Exception as e:
                print(f"[rebuild] failed to read {args.examples_json}: {e}; using defaults.")
        rebuilt = rebuild_inputs_from_sources(
            sources=sources,
            out_path=args.inputs_out,
            task_header=args.task_header,
            examples=examples,
            max_rows_per_source=args.max_rows_per_source,
        )
        print(f"[rebuild] completed with {rebuilt} rows.")

    # If just rebuilt, default to train on the rebuilt file unless the user overrides inputs_file explicitly
    inputs_path = args.inputs_out if args.rebuild_inputs_from.strip() else args.inputs_file

    # 1) Load X
    X = load_jsonl(inputs_path)
    print(f"[inputs] loaded {len(X)} rows from {inputs_path}")

    # 2) Build y lookups (STRICT keys: both files use 'original prompt' -> 'output prompt')
    lut = {}
    lut_mod = build_lookup_from_file(args.moderate_rewrites)
    lut_sev = build_lookup_from_file(args.severe_rewrites)
    # Prefer moderate mapping, then fill gaps with severe
    lut.update(lut_mod)
    for k, v in lut_sev.items():
        lut.setdefault(k, v)

    # 3) Join
    joined_pairs: List[Tuple[str, str]] = []
    unmatched: List[Dict] = []
    for i, rec in enumerate(X, 1):
        user_text = rec.get("text")
        if not isinstance(user_text, str):
            unmatched.append({"line": i, "reason": "no_text"})
            continue
        orig = extract_original_from_input_text(user_text)
        if not orig:
            unmatched.append({"line": i, "reason": "cannot_extract_original", "head": user_text[:120]})
            continue
        y = lut.get(norm_text(orig))
        if y:
            joined_pairs.append((user_text, y))
        else:
            unmatched.append({"line": i, "original": orig})

    print(f"[join] matched={len(joined_pairs)}  unmatched={len(unmatched)}")

    # Debug outputs
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
    from transformers import AutoTokenizer
    from trl import SFTTrainer, SFTConfig

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)

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
    )
    if args.bf16:
        cfg_kwargs["bf16"] = True
    if args.gradient_checkpointing:
        cfg_kwargs["gradient_checkpointing"] = True
    if args.deepspeed_config:
        cfg_kwargs["deepspeed"] = args.deepspeed_config

    cfg = SFTConfig(**cfg_kwargs)

    trainer = SFTTrainer(
        model=args.model_id,
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
