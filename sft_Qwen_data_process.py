#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sft_Qwen_data_process.py

- TRAIN zeroing -> sft_Qwen_traindata_zero.jsonl
- TEST zeroing  -> sft_Qwen_testdata_zero.jsonl   (delete old, same semantics, more robust)
- TRAIN match join -> sft_Qwen_traindata_final.jsonl
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

# ---- Robust markers (same semantics, tolerant to spaces/case/newline) ----
START_RE = re.compile(r"here\s+are\s+(?:a\s+)?few\s+example[s]?\s*:", re.IGNORECASE)
END_RE   = re.compile(r"(?:\r?\n)?\s*original\s+prompt\s*:", re.IGNORECASE)

ORIG_PROMPT_SPAN = re.compile(
    r"\n\s*original\s+prompt\s*:\s*(.*?)(?=\n\s*Please\s+put\s+your\s+changed\s+prompt\s+here\s*:)",
    flags=re.DOTALL | re.IGNORECASE,
)

def normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")

def strip_examples_section(text: str) -> Tuple[str, bool]:
    """
    Remove every block from 'Here are few example:' (inclusive)
    up to just before the next 'original prompt:' (exclusive).
    Returns (cleaned_text, changed_flag).
    """
    s = normalize_newlines(text)
    out = []
    pos = 0
    changed = False

    while True:
        m_start = START_RE.search(s, pos)
        if not m_start:
            out.append(s[pos:])
            break

        # keep everything before the start
        out.append(s[pos:m_start.start()])

        # find the first 'original prompt:' AFTER the start
        m_end = END_RE.search(s, m_start.end())
        if not m_end:
            # no end marker after start -> nothing to strip per spec; keep the rest
            out.append(s[m_start.start():])
            break

        # strip the block [start, end)
        pos = m_end.start()
        changed = True

    return ("".join(out), changed)

def extract_original_prompt(clean_text: str) -> Optional[str]:
    m = ORIG_PROMPT_SPAN.search(clean_text)
    if not m:
        return None
    return m.group(1).strip()

def load_lookup(paths: List[str]) -> Dict[str, str]:
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

def write_zero_file(in_path: str, out_path: str, remove_existing: bool) -> Tuple[int, int, int]:
    """
    Writes cleaned JSONL.
    Returns (written_lines, stripped_lines, start_hits).
    """
    if remove_existing and os.path.exists(out_path):
        os.remove(out_path)

    written = stripped = start_hits = 0
    with open(in_path, "r", encoding="utf-8") as fin, \
         open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text", "")

            # quick stat: does this line even contain the start phrase (robust)?
            if START_RE.search(normalize_newlines(text)):
                start_hits += 1

            cleaned, changed = strip_examples_section(text)
            if changed:
                stripped += 1

            obj["text"] = cleaned
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            written += 1
    return written, stripped, start_hits

def main():
    # TRAIN files
    in_path   = "sft_Qwen_traindata.jsonl"
    zero_path = "sft_Qwen_traindata_zero.jsonl"
    final_path = "sft_Qwen_traindata_final.jsonl"

    # TEST files
    test_in_path   = "sft_Qwen_testdata.jsonl"
    test_zero_path = "sft_Qwen_testdata_zero.jsonl"

    # mapping files
    moderate_path = "sft_moderate_excludeD_changedprompts.jsonl"
    severe_path   = "sft_severe_excludeHard_changedprompts.jsonl"

    # ---- TRAIN zero + cache original prompts
    processed: List[Tuple[str, Optional[str]]] = []

    with open(in_path, "r", encoding="utf-8") as fin, \
         open(zero_path, "w", encoding="utf-8") as fzero:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            cleaned, _ = strip_examples_section(text)  # same semantics as test
            obj["text"] = cleaned
            fzero.write(json.dumps(obj, ensure_ascii=False) + "\n")
            processed.append((cleaned, extract_original_prompt(cleaned)))

    print(f"[DONE] Wrote: {zero_path}")

    # ---- TEST zero (delete old first, same semantics)
    if not os.path.exists(test_in_path):
        raise FileNotFoundError(f"{test_in_path} not found")
    written, stripped, start_hits = write_zero_file(test_in_path, test_zero_path, remove_existing=True)
    print(f"[DONE] Wrote: {test_zero_path} (lines={written}, start_hits={start_hits}, stripped={stripped})")

    # ---- TRAIN final join
    lookup = load_lookup([moderate_path, severe_path])

    total = matched = unmatched = 0
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

    print(f"[DONE] Wrote: {final_path}")
    print(f"Total lines (train): {total} | Matched: {matched} | Unmatched (skipped): {unmatched}")

if __name__ == "__main__":
    main()
