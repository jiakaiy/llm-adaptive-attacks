#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deduplicate prompts *within each file separately* based on prompt text,
keeping the record with the lowest `index`. Works with JSONL or JSON-array.

Inputs (defaults):
  - filtered_moderate_prompts_indexed.jsonl
  - filtered_severe_prompts_indexed.json

Outputs:
  - filtered_moderate_unique_prompts.jsonl
  - filtered_severe_unique_prompts.jsonl

Usage:
  python find_unique_prompts.py
  # or with custom paths / strict equality:
  # python find_unique_prompts.py --moderate_in path/to/mod.jsonl \
  #   --severe_in path/to/sev.json --strict
"""

import argparse, json, re, unicodedata

def _normalize_prompt(s: str) -> str:
    """Unicode-normalize and collapse whitespace for robust matching."""
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s

def _to_int(x):
    try:
        return int(x)
    except Exception:
        return float("inf")

def load_records(path: str):
    """Load JSONL or a single JSON array from path."""
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()
    stripped = data.lstrip()
    records = []
    if stripped.startswith("["):  # JSON array
        obj = json.loads(stripped)
        if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
            records = obj["data"]
        elif isinstance(obj, list):
            records = obj
        else:
            raise ValueError(f"{path}: unsupported JSON structure")
    else:  # JSONL
        for i, line in enumerate(data.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: line {i} invalid JSON: {e}") from e
    return records

def dedupe(records, strict: bool = False):
    """Return (unique_records_sorted, removed_count)."""
    best_idx_by_key = {}
    best_rec_by_key = {}
    removed = 0

    for rec in records:
        prompt = rec.get("prompt")
        if prompt is None:
            # skip malformed rows
            removed += 1
            continue
        key = prompt if strict else _normalize_prompt(prompt)
        idx = _to_int(rec.get("index"))
        if key not in best_idx_by_key or idx < best_idx_by_key[key]:
            best_idx_by_key[key] = idx
            best_rec_by_key[key] = rec

    unique_recs = list(best_rec_by_key.values())
    unique_recs.sort(key=lambda r: _to_int(r.get("index")))
    removed += len(records) - len(unique_recs)
    return unique_recs, removed

def write_jsonl(records, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False))
            f.write("\n")

def main():
    ap = argparse.ArgumentParser(
        description="Deduplicate prompts in each file by content, keeping lowest index."
    )
    ap.add_argument("--moderate_in", default="filtered_moderate_prompts_indexed.jsonl")
    ap.add_argument("--severe_in", default="filtered_severe_prompts_indexed.json")
    ap.add_argument("--moderate_out", default="filtered_moderate_unique_prompts.jsonl")
    ap.add_argument("--severe_out", default="filtered_severe_unique_prompts.jsonl")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Use byte-for-byte prompt equality (no Unicode/whitespace normalization).",
    )
    args = ap.parse_args()

    for tag, in_path, out_path in [
        ("moderate", args.moderate_in, args.moderate_out),
        ("severe", args.severe_in, args.severe_out),
    ]:
        recs = load_records(in_path)
        uniques, removed = dedupe(recs, strict=args.strict)
        write_jsonl(uniques, out_path)
        print(f"[{tag}] input={len(recs)} unique={len(uniques)} removed={removed} -> {out_path}")

if __name__ == "__main__":
    main()
