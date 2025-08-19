# This file is intentionally left empty 

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, re, sys
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Set

# ---------- Config (filenames) ----------
RESULT_GLOB = "result_batch*_moderate.jsonl"
FILTERED_OUT = "result_allError_moderate.jsonl"

ERROR_GLOB = "error_batch*_moderate.jsonl"
MERGED_PHASEA_OUT = "error_moderate_phaseA.jsonl"

MOD_PROMPTS_IN = "filtered_moderate_prompts.jsonl"
MOD_PROMPTS_INDEXED = "filtered_moderate_prompts_indexed.jsonl"

PROMPTS_ONLY_OUT = "moderate_phaseA_prompts_only.jsonl"

# Pattern: optional "<text>" then "Sure, here is a tutorial:" then (blank lines) then "Step 1:"
START_RE = re.compile(
    r"^\s*(?:<text>\s*)?sure,\s*here\s+is\s+a\s+tutorial\s*:\s*(?:\r?\n\s*)*(?:step\s*1\s*:)",
    re.IGNORECASE
)

# ---------- Helpers ----------
def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                yield {"__raw__": s, "__file__": str(path), "__line__": ln}

def content_starts_with_tutorial_step1(content: str) -> bool:
    return bool(START_RE.search(content or ""))

# ---------- Step 1: filter results ----------
def filter_moderate_results() -> Tuple[int, int]:
    cwd = Path(".")
    files = sorted(cwd.glob(RESULT_GLOB))
    if not files:
        print(f"[ERR] No files match {RESULT_GLOB} in {cwd.resolve()}", file=sys.stderr)
        return 0, 0

    total, kept = 0, 0
    with Path(FILTERED_OUT).open("w", encoding="utf-8") as out:
        for fp in files:
            for obj in iter_jsonl(fp):
                total += 1
                if "__raw__" in obj:
                    # If malformed JSON, keep for inspection
                    out.write(obj["__raw__"] + "\n")
                    kept += 1
                    continue
                try:
                    content = obj["response"]["body"]["choices"][0]["message"]["content"]
                except Exception:
                    out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    kept += 1
                    continue
                if not content_starts_with_tutorial_step1(content):
                    out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    kept += 1
    print(f"[OK] filter_moderate_results: kept {kept} of {total} -> {FILTERED_OUT}")
    return kept, total

# ---------- Step 2: merge errors + filtered (dedupe) ----------
def merge_phaseA() -> int:
    cwd = Path(".")
    files = [Path(FILTERED_OUT)] + sorted(cwd.glob(ERROR_GLOB))
    files = [p for p in files if p.exists()]
    if not files:
        print(f"[WARN] No inputs to merge into {MERGED_PHASEA_OUT}")
        Path(MERGED_PHASEA_OUT).write_text("", encoding="utf-8")
        return 0

    seen_cids: Set[str] = set()
    seen_raw: Set[str] = set()
    n = 0
    with Path(MERGED_PHASEA_OUT).open("w", encoding="utf-8") as out:
        for fp in files:
            for obj in iter_jsonl(fp):
                if "__raw__" in obj:
                    raw = obj["__raw__"]
                    if raw in seen_raw:
                        continue
                    seen_raw.add(raw)
                    out.write(raw + "\n")
                    n += 1
                    continue
                # Try to extract custom_id (top-level or nested)
                cid = obj.get("custom_id")
                if cid is None and isinstance(obj, dict):
                    sub = obj
                    for k in ("response", "body", "error", "meta", "data"):
                        if isinstance(sub, dict) and k in sub:
                            sub = sub[k]
                    if isinstance(sub, dict):
                        cid = sub.get("custom_id")
                cid_str = str(cid) if cid is not None else None
                if cid_str:
                    if cid_str in seen_cids:
                        continue
                    seen_cids.add(cid_str)
                else:
                    raw = json.dumps(obj, ensure_ascii=False)
                    if raw in seen_raw:
                        continue
                    seen_raw.add(raw)
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n += 1
    print(f"[OK] merge_phaseA: wrote {n} lines -> {MERGED_PHASEA_OUT}")
    return n

# ---------- Step 3: extract custom_ids (in-memory only) ----------
CUSTOM_ID_RE = re.compile(r'"custom_id"\s*:\s*"?(?P<cid>\d+)"?')

def extract_custom_ids(from_path: Path) -> List[int]:
    ids: Set[int] = set()
    for obj in iter_jsonl(from_path):
        if "__raw__" in obj:
            m = CUSTOM_ID_RE.search(obj["__raw__"])
            if m:
                ids.add(int(m.group("cid")))
            continue
        cid = obj.get("custom_id")
        if cid is None and isinstance(obj, dict):
            sub = obj
            for k in ("response", "body", "error", "meta", "data"):
                if isinstance(sub, dict) and k in sub:
                    sub = sub[k]
            if isinstance(sub, dict):
                cid = sub.get("custom_id")
        if isinstance(cid, (int, str)):
            try:
                ids.add(int(cid))
            except Exception:
                pass
    srt = sorted(ids)
    print(f"[OK] extract_custom_ids: {len(srt)} ids (in-memory)")
    return srt

# ---------- Step 4: index original prompts (index FIRST) ----------
def index_moderate_prompts(src: Path, dst: Path) -> int:
    if not src.exists():
        print(f"[WARN] {src} not found; skipping prompt indexing.")
        return 0
    i = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                obj = {"prompt": s}
            # ensure "index" FIRST
            indexed_obj = {"index": i}
            for k, v in obj.items():
                if k == "index":
                    continue
                indexed_obj[k] = v
            fout.write(json.dumps(indexed_obj, ensure_ascii=False) + "\n")
            i += 1
    print(f"[OK] index_moderate_prompts: indexed {i} -> {dst}")
    return i

# ---------- Step 5: map ids -> prompts (output ONLY index+prompt) ----------
def load_indexed_prompts(path: Path) -> Dict[int, str]:
    d: Dict[int, str] = {}
    for obj in iter_jsonl(path):
        if "__raw__" in obj:
            # try parse
            try:
                obj = json.loads(obj["__raw__"])
            except Exception:
                continue
        idx = obj.get("index")
        pr  = obj.get("prompt")
        if idx is None or pr is None:
            continue
        try:
            d[int(idx)] = str(pr)
        except Exception:
            pass
    return d

def write_prompts_only(ids: List[int], indexed_path: Path, out_jsonl: Path) -> Tuple[int, int]:
    idx2p = load_indexed_prompts(indexed_path)
    n_have = 0
    with out_jsonl.open("w", encoding="utf-8") as f_out:
        for cid in ids:
            pr = idx2p.get(cid)
            if pr is None:
                continue
            rec = {"index": cid, "prompt": pr}
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_have += 1
    print(f"[OK] write_prompts_only: wrote {n_have}/{len(ids)} -> {out_jsonl}")
    return n_have, len(ids)

# ---------- main ----------
def main():
    # 1) filter results
    filter_moderate_results()

    # 2) merge errors + filtered
    merge_phaseA()

    # 3) extract custom_ids (in-memory)
    ids = extract_custom_ids(Path(MERGED_PHASEA_OUT))

    # 4) index moderate prompts (index FIRST)
    index_moderate_prompts(Path(MOD_PROMPTS_IN), Path(MOD_PROMPTS_INDEXED))

    # 5) output ONLY index+prompt
    write_prompts_only(ids, Path(MOD_PROMPTS_INDEXED), Path(PROMPTS_ONLY_OUT))

if __name__ == "__main__":
    main()
