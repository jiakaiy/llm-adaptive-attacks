#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase D pipeline (catattack_estimate_hint) — STREAMING VERSION (overwrite output)

- Starts from retry_catnonsense_phaseC_input.jsonl (or .josnl)
- KEEP rows whose index is NOT in EXCLUDE_IDS
- Maps index -> prompt from filtered_moderate_prompts_indexed.jsonl
  and writes retry_moderate_phaseD_catattackestimatehint_input.jsonl
- Re-runs prompts with template "catattack_estimate_hint" via your local main.py,
  using multiple DeepSeek keys in parallel (round-robin).
- **OVERWRITES** retry_moderate_phaseD_catattackestimatehint_output.jsonl at start,
  then writes results incrementally (one line per finished job).

Run (example):
  export DEEPSEEK_API_KEYS="sk-KEY1,sk-KEY2,sk-KEY3"
  python3 moderate_phaseD_catattackestimatehint.py \
    --phasec-input retry_catnonsense_phaseC_input.jsonl \
    --indexed filtered_moderate_prompts_indexed.jsonl \
    --base-url https://api.deepseek.com \
    --max-workers 3
"""

import argparse, json, os, re, sys, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Set
from concurrent.futures import ProcessPoolExecutor, as_completed

# import your local runner
try:
    from main import main as run_one, get_default_args
except Exception as e:
    print("[FATAL] Could not import main.py (need main(), get_default_args()).", e)
    sys.exit(2)

PHASEC_INPUT_DEFAULT = "retry_catnonsense_phaseC_input.jsonl"   # also accepts .josnl
INDEXED_PROMPTS = "filtered_moderate_prompts_indexed.jsonl"
PHASED_INPUT = "retry_moderate_phaseD_catattackestimatehint_input.jsonl"
PHASED_OUTPUT = "retry_moderate_phaseD_catattackestimatehint_output.jsonl"
TEMPLATE_NAME = "catattack_estimate_hint"

# IDs to EXCLUDE. We KEEP indices that are NOT in this set.
EXCLUDE_IDS: Set[int] = {
    860, 1371, 5242, 5497, 7366, 8363, 9372, 10318,
    11591, 11703, 11897, 13804, 15065, 15083
}

CUSTOM_ID_RE = re.compile(r'"custom_id"\s*:\s*"?(?P<cid>\d+)"?')
INDEX_RE     = re.compile(r'"index"\s*:\s*"?(?P<idx>\d+)"?')

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.rstrip("\n")
            if s:
                yield ln, s

def load_indexed_prompts(path: Path) -> Dict[int, str]:
    idx2p: Dict[int, str] = {}
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
                idx2p[int(idx)] = str(pr)
            except Exception:
                continue
    return idx2p

def _get_index_from_obj_or_line(obj: Optional[dict], line: str) -> Optional[int]:
    # prefer explicit "index" from parsed JSON
    if isinstance(obj, dict):
        if "index" in obj:
            try: return int(str(obj["index"]))
            except Exception: pass
        # common alternate fields
        if "custom_id" in obj:
            try: return int(str(obj["custom_id"]))
            except Exception: pass
        # look one level deeper for common containers
        cur = obj
        for k in ("response", "body", "meta", "data", "attrs", "metadata"):
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
        if isinstance(cur, dict):
            if "index" in cur:
                try: return int(str(cur["index"]))
                except Exception: pass
            if "custom_id" in cur:
                try: return int(str(cur["custom_id"]))
                except Exception: pass
    # fallback: regex scan the raw line
    m = INDEX_RE.search(line or "")
    if m:
        try: return int(m.group("idx"))
        except Exception: pass
    m = CUSTOM_ID_RE.search(line or "")
    if m:
        try: return int(m.group("cid"))
        except Exception: pass
    return None

def parse_api_keys(args) -> List[str]:
    keys: List[str] = []
    if args.api_keys:
        keys += [k.strip() for k in args.api_keys.split(",") if k.strip()]
    if args.api_keys_file:
        p = Path(args.api_keys_file)
        if not p.exists():
            print(f"[FATAL] --api-keys-file not found: {p}")
            sys.exit(2)
        keys += [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not keys:
        env_multi = os.getenv("DEEPSEEK_API_KEYS", "")
        if env_multi:
            keys = [k.strip() for k in env_multi.split(",") if k.strip()]
    if not keys:
        env_single = os.getenv("DEEPSEEK_API_KEY", "")
        if env_single:
            keys = [env_single.strip()]
    # dedupe order-preserving
    seen=set(); uniq=[]
    for k in keys:
        if k not in seen:
            uniq.append(k); seen.add(k)
    return uniq

def build_args_for_prompt(prompt: str, target_model: str, max_out_tokens: int, template_name: str):
    args = get_default_args()
    args.goal = prompt
    args.goal_modified = ""
    args.prompt_template = template_name
    args.target_model = target_model
    args.target_max_n_tokens = int(max_out_tokens)
    args.judge_model = "no-judge"
    args.debug = True
    args.n_iterations = 1
    args.n_restarts = 1
    args.n_chars_change_max = 0
    args.n_tokens_change_max = 0
    args.target_str = ""  # let template decide
    return args

def _worker_run_one(index: int, prompt: str, target_model: str, max_out_tokens: int,
                    api_key: str, base_url: Optional[str], template_name: str) -> dict:
    try:
        # Per-worker auth & base URL (mirrors Phase C)
        if api_key:
            os.environ["DEEPSEEK_API_KEY"] = api_key
            os.environ["OPENAI_API_KEY"] = api_key
        if base_url:
            os.environ["DEEPSEEK_API_BASE"] = base_url
            os.environ["OPENAI_BASE_URL"] = base_url
            os.environ["OPENAI_API_BASE"] = base_url
        else:
            for v in ("DEEPSEEK_API_BASE","OPENAI_BASE_URL","OPENAI_API_BASE"):
                os.environ.pop(v, None)

        args = build_args_for_prompt(prompt, target_model, max_out_tokens, template_name)
        noadv, orig, final = run_one(args)
        return {
            "custom_id": index,
            "index": index,
            "prompt": prompt,
            "template": template_name,
            "target_model": target_model,
            "noadv": noadv,
            "orig": orig,
            "final": final,
            "error": None,
        }
    except Exception as e:
        return {
            "custom_id": index,
            "index": index,
            "prompt": prompt,
            "template": template_name,
            "target_model": target_model,
            "error": str(e),
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phasec-input", default=PHASEC_INPUT_DEFAULT)
    ap.add_argument("--indexed", default=INDEXED_PROMPTS)
    ap.add_argument("--prep-out", default=PHASED_INPUT)
    ap.add_argument("--out", default=PHASED_OUTPUT)
    ap.add_argument("--template", default=TEMPLATE_NAME)
    ap.add_argument("--target-model", default="deepseek-reasoner")
    ap.add_argument("--max-out-tokens", type=int, default=350)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-run", action="store_true")

    # keys / concurrency / base-url
    ap.add_argument("--api-keys", default="")
    ap.add_argument("--api-keys-file", default="")
    ap.add_argument("--max-workers", type=int, default=0)
    ap.add_argument("--base-url", default="")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # accept typo .josnl
    phasec_path = Path(args.phasec_input)
    if not phasec_path.exists() and phasec_path.suffix == ".jsonl":
        alt = phasec_path.with_suffix(".josnl")
        if alt.exists():
            phasec_path = alt
    if not phasec_path.exists() and phasec_path.suffix == ".josnl":
        alt = phasec_path.with_suffix(".jsonl")
        if alt.exists():
            phasec_path = alt
    if not phasec_path.exists():
        raise SystemExit(f"[FATAL] could not find Phase-C input: {args.phasec_input}")

    # 1) Collect indices present in Phase-C, then EXCLUDE the given IDs
    present: Set[int] = set()
    for _, line in iter_jsonl(phasec_path):
        obj = None
        try:
            obj = json.loads(line)
        except Exception:
            pass
        idx = _get_index_from_obj_or_line(obj, line)
        if isinstance(idx, int):
            present.add(idx)

    keep_ids = sorted([i for i in present if i not in EXCLUDE_IDS])
    if args.debug:
        print(f"[debug] Phase-C present={len(present)}; keep(after exclude)={len(keep_ids)} -> {keep_ids[:40]}")

    if not keep_ids:
        print("[INFO] No remaining indices after exclusion. Nothing to do.")
        # build a fresh (empty) output file so old content is gone
        Path(args.out).open("w", encoding="utf-8").close()
        # still create empty prep file for traceability
        Path(args.prep_out).write_text("", encoding="utf-8")
        return

    # 2) Map index -> prompt from indexed prompts
    idx2prompt = load_indexed_prompts(Path(args.indexed))
    pairs: List[Tuple[int,str]] = []
    missing = 0
    for idx in keep_ids:
        pr = idx2prompt.get(idx)
        if pr is None:
            missing += 1
            continue
        pairs.append((idx, pr))
    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]

    # 3) Write Phase-D input file
    prep_path = Path(args.prep_out)
    with prep_path.open("w", encoding="utf-8") as f:
        for idx, pr in pairs:
            f.write(json.dumps({"index": idx, "prompt": pr}, ensure_ascii=False) + "\n")
    print(f"[OK] wrote {len(pairs)} prompts -> {prep_path} (missing={missing})")

    if args.no_run or not pairs:
        print("[INFO] stop after input build (--no-run or 0 prompts).")
        # fresh empty output
        Path(args.out).open("w", encoding="utf-8").close()
        return

    # 4) Prepare keys / workers / base-url
    keys = parse_api_keys(args)
    if not keys:
        raise SystemExit("[FATAL] provide DeepSeek keys via --api-keys/--api-keys-file or DEEPSEEK_API_KEYS env.")
    n_workers = args.max_workers if args.max_workers > 0 else len(keys)
    n_workers = max(1, min(n_workers, len(keys), 8))
    base_url = args.base_url.strip() or ""

    print(f"[RUN] {len(pairs)} prompt(s); {len(keys)} key(s); {n_workers} worker(s); model={args.target_model}; template={args.template}")

    # round-robin key assignment
    tasks = [(idx, pr, keys[i % len(keys)]) for i, (idx, pr) in enumerate(pairs)]

    # Echo inputs for reproducibility
    echo_in = prep_path.with_name(prep_path.stem + "_echo.jsonl")
    with echo_in.open("w", encoding="utf-8") as f:
        for idx, pr, _k in tasks:
            f.write(json.dumps({"index": idx, "prompt": pr}, ensure_ascii=False) + "\n")
    print(f"[OK] wrote input echo -> {echo_in}")

    out_path = Path(args.out)
    # **OVERWRITE** output file at start of run (fresh)
    out_path.open("w", encoding="utf-8").close()

    t0 = time.time()
    done = ok = err = 0

    # 5) Launch + STREAM writes as results complete
    with ProcessPoolExecutor(max_workers=n_workers) as ex, out_path.open("a", encoding="utf-8") as fout:
        futs = [ex.submit(_worker_run_one, idx, pr, args.target_model,
                          args.max_out_tokens, key, base_url or None, args.template)
                for (idx, pr, key) in tasks]

        for fut in as_completed(futs):
            rec = fut.result()
            # write one line immediately
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()

            done += 1
            if rec.get("error"):
                err += 1
            else:
                ok += 1
            if done % 10 == 0:
                print(f"[progress] {done}/{len(futs)} written… (ok={ok}, err={err})")

    print(f"[DONE] wrote {done} records -> {out_path.resolve()} in {time.time()-t0:.1f}s (ok={ok}, err={err})")

if __name__ == "__main__":
    main()
