#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-file dedupe for prompts (JSONL).

Goal:
- Keep filtered_moderate_unique_prompts.jsonl unchanged.
- Remove from filtered_severe_unique_prompts.jsonl any record whose *prompt*
  also appears in the moderate file (based on content equality).
- Write the remaining severe records to filtered_severe_unique_final_prompts.jsonl.

Defaults:
  --moderate  filtered_moderate_unique_prompts.jsonl
  --severe_in filtered_severe_unique_prompts.jsonl
  --severe_out filtered_severe_unique_final_prompts.jsonl

Matching:
- By default uses Unicode NFKC normalization and collapses whitespace.
- Use --strict for byte-for-byte equality (no normalization).

Usage:
  python find_final_unique_severe.py
  # with options:
  # python find_final_unique_severe.py --moderate path/mod.jsonl \
  #   --severe_in path/sev.jsonl --severe_out path/sev_final.jsonl --strict
"""

import argparse, json, re, sys, unicodedata
from typing import List, Dict

def _normalize_prompt(s: str) -> str:
    """Unicode-normalize + trim + collapse whitespace."""
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: line {i} invalid JSON: {e}") from e
            records.append(obj)
    return records

def write_jsonl(records: List[Dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")

def main():
    ap = argparse.ArgumentParser(description="Remove cross-file duplicate prompts (keep moderate, drop from severe).")
    ap.add_argument("--moderate", default="filtered_moderate_unique_prompts.jsonl",
                    help="Path to moderate unique JSONL (kept as ground truth).")
    ap.add_argument("--severe_in", default="filtered_severe_unique_prompts.jsonl",
                    help="Path to severe unique JSONL (will be filtered).")
    ap.add_argument("--severe_out", default="filtered_severe_unique_final_prompts.jsonl",
                    help="Output path for final severe JSONL after cross-file dedupe.")
    ap.add_argument("--strict", action="store_true",
                    help="Match prompts strictly (no Unicode/whitespace normalization).")
    args = ap.parse_args()

    # Load moderate and build prompt set
    mod = load_jsonl(args.moderate)
    if args.strict:
        mod_keys = { (r.get("prompt") or "") for r in mod if "prompt" in r and isinstance(r.get("prompt"), str) }
    else:
        mod_keys = { _normalize_prompt(r.get("prompt", "")) for r in mod if "prompt" in r and isinstance(r.get("prompt"), str) }

    # Load severe and filter those that are NOT in moderate
    sev = load_jsonl(args.severe_in)
    kept, removed, skipped = [], 0, 0
    for r in sev:
        p = r.get("prompt")
        if not isinstance(p, str):
            skipped += 1
            continue
        key = p if args.strict else _normalize_prompt(p)
        if key in mod_keys:
            removed += 1
            continue
        kept.append(r)

    # Write output
    write_jsonl(kept, args.severe_out)

    # Report
    print(f"[moderate] {len(mod)} records (prompt keys: {len(mod_keys)}) from {args.moderate}")
    print(f"[severe_in] {len(sev)} records from {args.severe_in}")
    print(f"[filter]   removed={removed} (duplicates vs moderate), skipped_no_prompt={skipped}")
    print(f"[severe_out] kept={len(kept)} -> {args.severe_out}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
