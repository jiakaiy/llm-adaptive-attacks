#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_batch_moderateB.py  (CN defaults, DeepSeek R1, uses max_tokens)

Build a Batch JSONL from:
  - moderate_phaseA_prompts_only.jsonl  (each line: {"index": int, "prompt": str})
using your local template "catattack_trivia_tail" via prompts.get_universal_manual_prompt().

Example:
  python build_batch_moderateB.py \
    --input moderate_phaseA_prompts_only.jsonl \
    --out moderateB_batch_input.jsonl \
    --runtime moderateB_runtime.json \
    --template catattack_trivia_tail \
    --model deepseek-r1 \
    --max-tokens 350 \
    --region cn
"""

import argparse, json, sys
from pathlib import Path
from typing import List, Tuple

# Import your prompt builder
try:
    from prompts import get_universal_manual_prompt
except Exception as e:
    print("[FATAL] Could not import prompts.get_universal_manual_prompt. Run from repo root.", e)
    sys.exit(2)

def pick_base_url_by_region(region: str) -> str:
    region = (region or "cn").lower()
    if region == "cn":
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"
    return "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

def read_prompts_only(path: Path) -> List[Tuple[int, str]]:
    rows: List[Tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                continue
            idx = obj.get("index")
            pr  = obj.get("prompt")
            if idx is None or pr is None:
                continue
            try:
                rows.append((int(idx), str(pr)))
            except Exception:
                continue
    seen = set(); uniq: List[Tuple[int, str]] = []
    for i, p in rows:
        if i in seen: continue
        uniq.append((i, p)); seen.add(i)
    return uniq

def build_user_message(template_name: str, target_str: str, goal: str) -> str:
    # Do not lowercase goal; let your template handle it.
    return get_universal_manual_prompt(template_name, target_str or "", goal)

def make_batch_line(custom_id: int, user_msg: str, model: str,
                    max_tokens: int, temperature: float, top_p: float,
                    system_msg: str) -> dict:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": float(temperature),
        "top_p": float(top_p),
        # IMPORTANT: Bailian Batch validator expects max_tokens, even for R1
        "max_tokens": int(max_tokens),
    }
    return {
        "custom_id": str(custom_id),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="moderate_phaseA_prompts_only.jsonl",
                    help='Each line: {"index": int, "prompt": str}')
    ap.add_argument("--out", default="moderateB_batch_input.jsonl",
                    help="Output batch JSONL")
    ap.add_argument("--runtime", default="moderateB_runtime.json",
                    help="Runtime JSON for submit step (base_url/endpoint/model)")

    # Template/model
    ap.add_argument("--template", default="catattack_trivia_tail")
    ap.add_argument("--target-str", default="", help="Leave blank to let the template decide.")
    ap.add_argument("--model", default="deepseek-r1")
    ap.add_argument("--max-tokens", type=int, default=350)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--system", default="You are a concise assistant.")

    # Base URL info for submitter (CN default)
    ap.add_argument("--region", choices=["intl", "cn"], default="cn",
                    help="Bailian compatible endpoint region; default cn")

    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"[FATAL] input file not found: {in_path.resolve()}")

    pairs = read_prompts_only(in_path)
    if not pairs:
        raise SystemExit("[FATAL] No {index,prompt} rows found.")

    out_path = Path(args.out)
    n = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for idx, pr in pairs:
            user_msg = build_user_message(args.template, args.target_str, pr)
            rec = make_batch_line(idx, user_msg, args.model,
                                  args.max_tokens, args.temperature, args.top_p, args.system)
            line = json.dumps(rec, ensure_ascii=False)
            if len(line.encode("utf-8")) > 950_000:
                print(f"[warn] skip index={idx}: line too large after templating.")
                continue
            fout.write(line + "\n")
            n += 1

    runtime = {
        "base_url": pick_base_url_by_region(args.region),
        "endpoint": "/v1/chat/completions",
        "model": args.model,
        "template": args.template,
        "target_str": args.target_str,  # usually ""
        "count": n,
    }
    Path(args.runtime).write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] built batch input: {out_path.resolve()}  ({n} lines)")
    print(f"[OK] wrote runtime file: {Path(args.runtime).resolve()}")

if __name__ == "__main__":
    main()
