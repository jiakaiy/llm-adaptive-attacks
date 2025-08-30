#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sft_Qwen_data_process.py

What this script does:

1) For each line in sft_Qwen_traindata.jsonl ({"text": "..."}),
   strip EVERYTHING starting from the substring
       "Here are few example:"
   (INCLUSIVE) up to
       "\noriginal prompt:"
   (EXCLUSIVE), then write the cleaned object to
   sft_Qwen_traindata_zero.jsonl.

2) From the CLEANED text, extract the substring between
       "\noriginal prompt:"  and  "\nPlease put your changed prompt here:"
   This is the "original prompt" key used for matching.

3) Load both mapping files:
   - sft_moderate_excludeD_changedprompts.jsonl
   - sft_severe_excludeHard_changedprompts.jsonl
   Each line has: {"original prompt": ..., "output prompt": ...}
   Match by the extracted "original prompt", and write
   sft_Qwen_traindata_final.jsonl containing:
      {"text": <cleaned text>, "output prompt": <matched output>}

Notes:
- Lines with no match are SKIPPED in the final file. A small count is printed.
- Newlines are normalized to '\n' before processing to make marker matching robust.
"""

import json
import re
from typing import Dict, List, Optional, Tuple

# --------- Helpers ---------

EXAMPLES_BLOCK = re.compile(
    r"Here are few example:.*?(?=\noriginal prompt:)",  # remove up to BEFORE '\noriginal prompt:'
    flags=re.DOTALL,
)

ORIG_PROMPT_SPAN = re.compile(
    r"\noriginal prompt:\s*(.*?)(?=\nPlease put your changed prompt here:)",
    flags=re.DOTALL,
)

def normalize_newlines(s: str) -> str:
    """Convert CRLF/CR to LF so our '\n...' markers match reliably."""
    return s.replace("\r\n", "\n").replace("\r", "\n")

def strip_examples_section(text: str) -> str:
    """
    Remove everything starting from 'Here are few example:' (inclusive)
    up to just before '\noriginal prompt:' (exclusive).
    """
    text = normalize_newlines(text)
    # If pattern not found, text is returned unchanged.
    return EXAMPLES_BLOCK.sub("", text)

def extract_original_prompt(clean_text: str) -> Optional[str]:
    """
    Grab the content between '\noriginal prompt:' and
    '\nPlease put your changed prompt here:'.
    """
    m = ORIG_PROMPT_SPAN.search(clean_text)
    if not m:
        return None
    return m.group(1).strip()

def load_lookup(paths: List[str]) -> Dict[str, str]:
    """
    Build a dict: original prompt -> output prompt from multiple JSONL files.
    If duplicates exist, later files override earlier ones.
    """
    table: Dict[str, str] = {}
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                orig = (obj.get("original prompt") or "").strip()
                outp = (obj.get("output prompt") or "").strip()
                if orig and outp:
                    table[orig] = outp
    return table

# --------- Main pipeline ---------

def main():
    in_path = "sft_Qwen_traindata.jsonl"
    zero_path = "sft_Qwen_traindata_zero.jsonl"
    final_path = "sft_Qwen_traindata_final.jsonl"

    moderate_path = "sft_moderate_excludeD_changedprompts.jsonl"
    severe_path = "sft_severe_excludeHard_changedprompts.jsonl"

    # Step 1: clean & write zero file, while caching extracted original prompts
    processed: List[Tuple[str, Optional[str]]] = []  # (clean_text, original_prompt or None)

    with open(in_path, "r", encoding="utf-8") as fin, \
         open(zero_path, "w", encoding="utf-8") as fzero:

        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            cleaned = strip_examples_section(text)
            obj["text"] = cleaned
            fzero.write(json.dumps(obj, ensure_ascii=False) + "\n")

            orig_prompt = extract_original_prompt(cleaned)
            processed.append((cleaned, orig_prompt))

    # Step 3: load original->output lookup
    lookup = load_lookup([moderate_path, severe_path])

    # Step 4: write final file (skip unmatched)
    total = 0
    matched = 0
    unmatched = 0

    with open(final_path, "w", encoding="utf-8") as ff:
        for cleaned, orig in processed:
            total += 1
            if not orig:
                unmatched += 1
                continue
            outp = lookup.get(orig)
            if outp is None:
                unmatched += 1
                continue
            ff.write(json.dumps({"text": cleaned, "output prompt": outp}, ensure_ascii=False) + "\n")
            matched += 1

    print(f"[DONE] Wrote: {zero_path}")
    print(f"[DONE] Wrote: {final_path}")
    print(f"Total lines: {total} | Matched: {matched} | Unmatched (skipped): {unmatched}")

if __name__ == "__main__":
    main()
