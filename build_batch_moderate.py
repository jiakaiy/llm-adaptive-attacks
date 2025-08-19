#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a JSONL file for Alibaba Cloud Model Studio Batch API (Bailian).

- Model: deepseek-r1 (Batch-safe; equivalent capability to the 0528 release)
- Accepts input as JSONL (fields: prompt|goal|text) or plain text (one prompt per line)
- Writes runtime JSON (base_url + endpoint + model)
- Uses prompts.get_universal_manual_prompt with template_id=refined_best_simplified by default

Typical use for the MODERATE set:
  python build_batch_input.py \
    --infile filtered_moderate_prompts.jsonl \
    --outfile batch_input_moderate.jsonl

Smoke test (no inference; validator-only endpoint):
  python build_batch_input.py --infile test.jsonl --outfile test_batch.jsonl --test
"""

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Iterable

# ---------------- helpers ----------------
def pick_base_url(region: str) -> str:
    region = (region or "cn").lower()
    if region == "intl":
        return "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    if region == "cn":
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"
    raise SystemExit("region must be one of: intl, cn")

def extract_prompt_from_line(line: str) -> str:
    """
    Accept JSONL with keys {'prompt' | 'goal' | 'text'} or treat as plain text.
    """
    line = (line or "").strip()
    if not line:
        return ""
    try:
        obj = json.loads(line)
        for k in ("prompt", "goal", "text"):
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
    except json.JSONDecodeError:
        return line

def lazy_import_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except Exception as e:
        print(f"[warn] could not import module {module_name}: {e}")
        return None

def list_possible_templates(mod) -> Iterable[str]:
    names = []
    if not mod:
        return names
    # Heuristic: common dict containers of manual templates
    for attr in ("TEMPLATES", "templates", "MANUAL_TEMPLATES", "ALL_TEMPLATES"):
        if hasattr(mod, attr) and isinstance(getattr(mod, attr), dict):
            names.extend(sorted(getattr(mod, attr).keys()))
    return names

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile",  required=True, help="Input: JSONL or TXT with one prompt per line")
    ap.add_argument("--outfile", required=True, help="Output JSONL for Batch API")
    ap.add_argument("--region",  choices=["intl","cn"], default="cn",
                    help="Sets base_url for submit script (cn=国内百炼, intl=国际站)")
    ap.add_argument("--runtime_json", default="batch_runtime.json",
                    help="Where to write base_url + endpoint + model")

    # Batch validator toggle (no real inference)
    ap.add_argument("--test", action="store_true",
                    help="Target the Batch test endpoint with batch-test-model")

    # Model / generation knobs
    ap.add_argument("--model", default="deepseek-r1",
                    help="Batch model name (DeepSeek R1 -> deepseek-r1). Test runs force batch-test-model.")
    ap.add_argument("--system", default="You are a concise assistant.",
                    help="System message for each request")
    ap.add_argument("--max_tokens", type=int, default=150)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--custom_id_offset", type=int, default=0,
                    help="Start index for custom_id (useful when sharding)")

    # Optional plain prefix (kept safe)
    ap.add_argument("--prepend", default="",
                    help="(Optional) safe prefix to prepend to each user message")

    # Template plumbing (MODERATE default)
    ap.add_argument("--template_module", default="prompts", help="Python module name")
    ap.add_argument("--template_name",   default="get_universal_manual_prompt", help="Function name")
    ap.add_argument("--template_id",     default="refined_best_simplified", help="Template id to pass to your function")
    ap.add_argument("--target_str",      default="Sure, here is a tutorial",
                    help="target_str arg for your template function")

    # Your data uses {"prompt": ...}
    ap.add_argument("--goal_field",      default="prompt",
                    help="Primary field name to read from input JSON lines")

    args = ap.parse_args()

    base_url = pick_base_url(args.region)
    # Endpoint must match batches.create(endpoint=...) in the submit script
    endpoint = "/v1/chat/ds-test" if args.test else "/v1/chat/completions"
    model = "batch-test-model" if args.test else args.model

    # Import template module + function
    tpl_mod = lazy_import_module(args.template_module)
    tpl_fn = None
    if tpl_mod:
        try:
            tpl_fn = getattr(tpl_mod, args.template_name)
        except Exception as e:
            print(f"[warn] could not import {args.template_module}.{args.template_name}: {e}")
            tpl_fn = None

        # Optional: verify template_id exists if module exposes a templates dict
        available = set(list_possible_templates(tpl_mod))
        if available and args.template_id not in available:
            print(f"[err] template_id '{args.template_id}' not found in {args.template_module}.")
            print("      Available template ids:", ", ".join(sorted(available)) or "(none)")
            sys.exit(2)
    else:
        print("[warn] template module not found; will use raw prompt as user content.")

    in_path  = Path(args.infile)
    out_path = Path(args.outfile)
    if not in_path.exists():
        raise SystemExit(f"[err] infile not found (did you mean .jsonl, not .josnl?): {in_path}")

    # Small preview to sanity-check the input
    try:
        with in_path.open("r", encoding="utf-8") as f:
            preview = []
            for _ in range(3):
                l = f.readline()
                if not l:
                    break
                preview.append(l.strip()[:200])
        if preview:
            print("[preview] first lines:")
            for i, ln in enumerate(preview, 1):
                print(f"  {i:>2}: {ln}")
    except Exception:
        pass

    n = 0
    skipped = 0
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for i, raw in enumerate(fin):
            raw = raw.rstrip("\n")
            if not raw:
                continue

            # Prefer explicit goal_field; else fall back to generic extractor
            goal = ""
            try:
                obj = json.loads(raw)
                goal = obj.get(args.goal_field) or ""
                if not isinstance(goal, str):
                    goal = ""
            except Exception:
                pass
            if not goal:
                goal = extract_prompt_from_line(raw)
            if not goal:
                skipped += 1
                continue

            # Compose user content
            if tpl_fn:
                try:
                    user_content = tpl_fn(args.template_id, args.target_str, goal)
                except TypeError:
                    user_content = tpl_fn(args.template_id, args.target_str, goal=goal)
            else:
                user_content = goal

            if args.prepend:
                user_content = f"{args.prepend}\n{user_content}"

            # Build the OpenAI-compatible request body
            body = {
                "model": model,
                "messages": [
                    {"role": "system", "content": args.system},
                    {"role": "user",   "content": user_content}
                ],
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "max_tokens": int(args.max_tokens),
            }

            rec = {
                "custom_id": str(args.custom_id_offset + i),
                "method": "POST",
                "url": endpoint,      # must match batches.create(endpoint=...)
                "body": body
            }

            line = json.dumps(rec, ensure_ascii=False)
            # Keep a conservative comfort limit; Batch allows up to ~1 MB per line.
            if len(line.encode("utf-8")) > 950_000:
                print(f"[warn] skipping line {i}: too large after templating (~>{len(line)//1024} KiB)")
                skipped += 1
                continue

            fout.write(line + "\n")
            n += 1

    runtime = {
        "base_url": base_url,
        "endpoint": endpoint,
        "model": model,
        "count": n,
        "skipped": skipped
    }
    Path(args.runtime_json).write_text(json.dumps(runtime, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Wrote {n} requests → {out_path}")
    if skipped:
        print(f"[OK] Skipped {skipped} empty/oversize lines")
    print(f"[OK] Runtime → {args.runtime_json}: {runtime}")

if __name__ == "__main__":
    main()
