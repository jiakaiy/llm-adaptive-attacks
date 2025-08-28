#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sft_Qwen_examine_test_results.py

Analyzes a JSONL file where each line looks like:
  {"original prompt": "...", "predicted output prompt": "..."}

Tasks (default terms hardcoded below):
1) Find line numbers containing "Example" in predicted output prompt; print count.
2) Find line numbers containing "cat"     in predicted output prompt; print count.
3) Find line numbers containing "175"     in predicted output prompt; print count.
4) Find line numbers containing "1 + 1 > 2" in predicted output prompt; print count.

NEW:
- --make_deepseek_final will read a DeepSeek results JSONL of rows:
    {"original prompt": "...", "predicted output prompt": "...", "deepseek_response": "..."}
  and write a new JSONL that keeps ONLY:
    {"original prompt": "...", "deepseek_response": "..."}
  to sft_Qwen_test_deepseek_results_final.jsonl (or a custom path).

Notes:
- Matching is case-insensitive by default. Use --case-sensitive to switch.
- Line numbers are 1-based (first line is 1).
"""

import argparse
import json
import sys
from typing import List, Dict

DEFAULT_TERMS = [
    "Example",
    "cat",
    "175",
    "1 + 1 > 2",
]

def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                rows.append({"__line__": i, "__err__": "empty"})
                continue
            try:
                obj = json.loads(s)
                obj["__line__"] = i  # keep original line number
                rows.append(obj)
            except json.JSONDecodeError as e:
                rows.append({"__line__": i, "__err__": f"json_error: {e}"})
    return rows

def write_jsonl(path: str, records: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")

def find_matches(rows: List[Dict], term: str, case_sensitive: bool) -> List[int]:
    matches = []
    needle = term if case_sensitive else term.lower()
    for r in rows:
        line_no = r.get("__line__", None)
        if line_no is None:
            continue
        text = r.get("predicted output prompt", "")
        if not isinstance(text, str):
            continue
        haystack = text if case_sensitive else text.lower()
        if needle in haystack:
            matches.append(line_no)
    return matches

def make_deepseek_final(in_path: str, out_path: str) -> None:
    """
    Read DeepSeek results JSONL with keys:
      "original prompt", "predicted output prompt", "deepseek_response"
    Write JSONL keeping only:
      "original prompt", "deepseek_response"
    """
    rows = load_jsonl(in_path)
    total = len(rows)
    kept = 0
    out_rows = []
    for r in rows:
        if "__err__" in r:
            # keep line numbers aligned but drop invalid content
            out_rows.append({
                "original prompt": "",
                "deepseek_response": "",
            })
            continue
        out_rows.append({
            "original prompt": r.get("original prompt", ""),
            "deepseek_response": r.get("deepseek_response", ""),
        })
        kept += 1
    write_jsonl(out_path, out_rows)
    print(f"[deepseek] read: {in_path}")
    print(f"[deepseek] total lines: {total}, wrote: {len(out_rows)} -> {out_path}")

def main():
    ap = argparse.ArgumentParser()

    # Analysis of predicted outputs
    ap.add_argument("--file", default="sft_Qwen_test_predictedy.jsonl",
                    help="JSONL with predicted outputs to analyze.")
    ap.add_argument("--case-sensitive", action="store_true",
                    help="Use case-sensitive matching (default: case-insensitive).")
    ap.add_argument("--terms", nargs="*", default=DEFAULT_TERMS,
                    help="Terms to search for in predicted output prompt.")

    # NEW: DeepSeek finalization
    ap.add_argument("--make_deepseek_final", action="store_true",
                    help="Create a filtered DeepSeek results JSONL with only original prompt and deepseek_response.")
    ap.add_argument("--deepseek_file", default="sft_Qwen_test_deepseek_results.jsonl",
                    help="Input DeepSeek results JSONL.")
    ap.add_argument("--deepseek_final_out", default="sft_Qwen_test_deepseek_results_final.jsonl",
                    help="Output JSONL (only original prompt + deepseek_response).")

    args = ap.parse_args()

    # --- Part A: analyze predicted outputs ---
    rows = load_jsonl(args.file)
    total_lines = len(rows)
    invalid = sum(1 for r in rows if "__err__" in r)
    print(f"[info] analyzed file: {args.file}")
    print(f"[info] total lines: {total_lines} (invalid: {invalid}, valid: {total_lines - invalid})")
    print(f"[info] matching is {'CASE-SENSITIVE' if args.case_sensitive else 'case-insensitive'}\n")

    for term in args.terms:
        lines = find_matches(rows, term, case_sensitive=args.case_sensitive)
        lines_sorted = sorted(lines)
        print(f'Term: "{term}"')
        print(f"  count: {len(lines_sorted)}")
        if lines_sorted:
            preview = ", ".join(map(str, lines_sorted[:50]))
            print(f"  line numbers (first {min(50, len(lines_sorted))}): {preview}")
            if len(lines_sorted) > 50:
                print(f"  ... and {len(lines_sorted) - 50} more")
        else:
            print("  line numbers: (none)")
        print()

    # --- Part B (NEW): filter DeepSeek results if requested ---
    if args.make_deepseek_final:
        make_deepseek_final(args.deepseek_file, args.deepseek_final_out)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
