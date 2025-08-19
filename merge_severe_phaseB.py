
#!/usr/bin/env python3
import json, re, sys
from pathlib import Path
from typing import Iterable

# --------- Config ----------
RESULT_GLOB = "result_batch*_severe.jsonl"
MERGED_OUT = "merged_severe_not_sure.jsonl"

PROMPTS_IN = "filtered_severe_prompts.jsonl"
PROMPTS_OUT = "filtered_severe_prompts_indexed.jsonl"

ERROR_GLOB = "error_batch*_severe.jsonl"
ERROR_OUT = "error_severe_total.jsonl"

SURE_TUTORIAL_RE = re.compile(r"^\s*(?:<text>\s*)?sure,\s*here\s+is\s+a\s+tutorial\b", re.I)

# --------- Helpers ----------
def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception as e:
                sys.stderr.write(f"[WARN] {path}:{ln}: JSON parse error: {e}\n")

def content_starts_with_sure_tutorial(content: str) -> bool:
    return bool(SURE_TUTORIAL_RE.search(content or ""))

# --------- Tasks ----------
def merge_result_shards():
    cwd = Path(".")
    files = sorted(cwd.glob(RESULT_GLOB))
    if not files:
        print(f"[ERROR] No files match {RESULT_GLOB} in {cwd.resolve()}")
        return 1

    total, kept = 0, 0
    with Path(MERGED_OUT).open("w", encoding="utf-8") as out:
        for fp in files:
            for obj in iter_jsonl(fp):
                total += 1
                try:
                    content = obj["response"]["body"]["choices"][0]["message"]["content"]
                except Exception:
                    # keep malformed entries so you can inspect them
                    out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    kept += 1
                    continue
                if not content_starts_with_sure_tutorial(content):
                    out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    kept += 1
    print(f"[OK] Wrote {kept} of {total} entries -> {MERGED_OUT}")
    return 0

def index_prompts():
    pin = Path(PROMPTS_IN)
    if not pin.exists():
        print(f"[WARN] {PROMPTS_IN} not found; skipping indexing.")
        return 0
    i = 0
    with pin.open("r", encoding="utf-8") as fin, Path(PROMPTS_OUT).open("w", encoding="utf-8") as fout:
        for line in fin:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                obj = {"prompt": s}
            # ensure "index" comes first
            indexed_obj = {"index": i}
            for k, v in obj.items():
                if k == "index":
                    continue
                indexed_obj[k] = v
            fout.write(json.dumps(indexed_obj, ensure_ascii=False) + "\n")
            i += 1
    print(f"[OK] Indexed {i} prompts -> {PROMPTS_OUT}")
    return 0

def merge_error_shards():
    cwd = Path(".")
    files = sorted(cwd.glob(ERROR_GLOB))
    if not files:
        print(f"[WARN] No files match {ERROR_GLOB} in {cwd.resolve()} — skipping.")
        return 0
    count_lines = 0
    with Path(ERROR_OUT).open("w", encoding="utf-8") as out:
        for fp in files:
            with fp.open("r", encoding="utf-8") as fin:
                for ln, line in enumerate(fin, 1):
                    s = line.strip()
                    if not s:
                        continue
                    # Normalize to valid JSON if possible; otherwise write raw
                    try:
                        obj = json.loads(s)
                        out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    except Exception:
                        out.write(s + "\n")
                    count_lines += 1
    print(f"[OK] Concatenated {count_lines} lines from {len(files)} files -> {ERROR_OUT}")
    return 0



FINAL_COMBINED_IN_1 = MERGED_OUT              # "merge_severe_filtered.jsonl"
FINAL_COMBINED_IN_2 = ERROR_OUT               # "error_severe_total.jsonl"
FINAL_COMBINED_OUT  = "severe_phaseB_total.jsonl"

def combine_filtered_and_errors():
    in1 = Path(FINAL_COMBINED_IN_1)
    in2 = Path(FINAL_COMBINED_IN_2)
    if not in1.exists() and not in2.exists():
        print(f"[WARN] Neither {in1} nor {in2} exists; skipping final combine.")
        return 0

    seen_ids = set()
    seen_lines = set()
    out = Path(FINAL_COMBINED_OUT).open("w", encoding="utf-8")

    def add_path(p: Path, label: str):
        nonlocal seen_ids, seen_lines
        if not p.exists():
            print(f"[WARN] {label} file {p} missing; skipping.")
            return 0
        cnt = 0
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                    cid = obj.get("custom_id")
                    if cid is not None:
                        key = ("cid", str(cid))
                        if key in seen_ids:
                            continue
                        seen_ids.add(key)
                    else:
                        if s in seen_lines:
                            continue
                        seen_lines.add(s)
                    out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                except Exception:
                    if s in seen_lines:
                        continue
                    seen_lines.add(s)
                    out.write(s + "\n")
                cnt += 1
        return cnt

    n1 = add_path(in1, "filtered")
    n2 = add_path(in2, "errors")
    out.close()
    print(f"[OK] Wrote combined file with {n1 or 0} + {n2 or 0} lines -> {FINAL_COMBINED_OUT}")
    return 0


def main():
    rc0 = merge_error_shards()
    rc1 = merge_result_shards()
    rc2 = index_prompts()
    rc3 = combine_filtered_and_errors()
    if rc0 or rc1 or rc2 or rc3:
        # Non-fatal warnings use return code 0; only explicit 1s will trigger exit 1.
        # rc0/rc2 return 0 even when skipping is fine.
        pass

if __name__ == "__main__":
    main()
