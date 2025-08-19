#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
moderate_phaseD.py

Phase D (moderate → template 'refined_best_175'):
1) Filter specific indices from filtered_severe_prompts_indexed.jsonl
   and write them to retry_moderate_phaseD_175_input.jsonl
   (each line: {"index": int, "prompt": str})

2) Run those prompts via your local main.py pipeline using DeepSeek,
   with template "refined_best_175", rotating across multiple API keys
   in parallel. Stream each finished result line-by-line into
   retry_moderate_phaseD_175_output.jsonl.

Default indices (from your message):
  201,209,482,566,729,839,901,1749,1839,1945,2070,2147,2231,2238,2242,2270,2230,
  41944,4252,4278,4318

Usage (your 3 keys):
  python3 moderate_phaseD.py \
    --indexed filtered_severe_prompts_indexed.jsonl \
    --api-keys "sk-188e5289696d4c0686791f28c54f5c8f,sk-494b356ccd154879901ed596fc8d6327,sk-2e9db914d0e1452cad7470618c859f31" \
    --max-workers 3

Notes:
  • Do NOT hardcode keys in the file. Pass via --api-keys / --api-keys-file or env DEEPSEEK_API_KEYS.
  • Results are streamed: each completion is appended immediately to the output JSONL.
"""

import argparse, json, os, sys, time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---- import your runner ----
try:
    from main import main as run_one, get_default_args
except Exception as e:
    print("[FATAL] Could not import main.py (need main(), get_default_args()).", e)
    sys.exit(2)

DEFAULT_INDICES = [
    201, 209, 482, 566, 729, 839, 901, 1749, 1839, 1945, 2070, 2147,
    2231, 2238, 2242, 2270, 2230, 41944, 4252, 4278, 4318
]

# ---------- IO helpers ----------
def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                yield ln, json.loads(s)
            except Exception as e:
                sys.stderr.write(f"[WARN] {path}:{ln} JSON parse error: {e}\n")

def load_indexed_prompts(indexed_path: Path) -> Dict[int, str]:
    m: Dict[int, str] = {}
    for _, obj in iter_jsonl(indexed_path):
        idx = obj.get("index")
        pr  = obj.get("prompt")
        if isinstance(idx, int) and isinstance(pr, str):
            if idx not in m:
                m[idx] = pr
    return m

def write_prompts_only(pairs: List[Tuple[int, str]], out_path: Path) -> int:
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for idx, pr in pairs:
            f.write(json.dumps({"index": idx, "prompt": pr}, ensure_ascii=False) + "\n")
            n += 1
    return n

# ---------- build args for each prompt ----------
def build_args_for_prompt(prompt: str, target_model: str, max_out_tokens: int):
    args = get_default_args()
    args.goal = prompt
    args.goal_modified = ""                 # let main() copy
    args.prompt_template = "refined_best_175"
    args.target_model = target_model
    args.target_max_n_tokens = int(max_out_tokens)
    args.judge_model = "no-judge"
    args.debug = True

    # fast one-shot
    args.n_iterations = 1
    args.n_restarts = 1
    args.n_chars_change_max = 0
    args.n_tokens_change_max = 0

    # let template supply its own target string if any
    args.target_str = ""
    return args

# ---------- keys parsing ----------
def parse_api_keys(args) -> List[str]:
    keys: List[str] = []
    if args.api_keys:
        keys.extend([k.strip() for k in args.api_keys.split(",") if k.strip()])
    if args.api_keys_file:
        p = Path(args.api_keys_file)
        if p.exists():
            keys.extend([ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()])
    if not keys:
        env_multi = os.getenv("DEEPSEEK_API_KEYS", "")
        if env_multi:
            keys = [k.strip() for k in env_multi.split(",") if k.strip()]
    if not keys:
        env_single = os.getenv("DEEPSEEK_API_KEY", "")
        if env_single:
            keys = [env_single]

    # dedupe preserve order
    seen = set(); uniq = []
    for k in keys:
        if k not in seen:
            uniq.append(k); seen.add(k)
    return uniq

# ---------- worker ----------
def _worker_run_one(index: int, prompt: str, target_model: str, max_out_tokens: int,
                    api_key: str, base_url: Optional[str]) -> dict:
    try:
        if api_key:
            os.environ["DEEPSEEK_API_KEY"] = api_key
            os.environ["OPENAI_API_KEY"]   = api_key  # some stacks use this
        # Optional base url override (rarely needed for deepseek.com)
        if base_url:
            os.environ["DEEPSEEK_API_BASE"] = base_url
            os.environ["OPENAI_BASE_URL"]   = base_url
            os.environ["OPENAI_API_BASE"]   = base_url

        args = build_args_for_prompt(prompt, target_model, max_out_tokens)
        noadv, orig, final = run_one(args)

        return {
            "custom_id": index,
            "index": index,
            "prompt": prompt,
            "template": "refined_best_175",
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
            "template": "refined_best_175",
            "target_model": target_model,
            "error": str(e),
        }

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indexed", default="filtered_severe_prompts_indexed.jsonl",
                    help="Source file with {'index': int, 'prompt': str} per line.")
    ap.add_argument("--indices", default="",
                    help="Comma-separated indices; if empty, uses the default list in this script.")
    ap.add_argument("--prep-out", default="retry_moderate_phaseD_175_input.jsonl",
                    help="Where to write filtered prompts (input for Phase D).")
    ap.add_argument("--out", default="retry_moderate_phaseD_175_output.jsonl",
                    help="Where to stream results.")
    ap.add_argument("--target-model", default="deepseek-reasoner",
                    help="Must map to DeepSeekReasonerWrapper in conversers.py")
    ap.add_argument("--max-out-tokens", type=int, default=350)
    ap.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", ""),
                    help="Optional override; usually leave blank for deepseek.com")
    ap.add_argument("--api-keys", default="", help="Comma-separated DeepSeek keys.")
    ap.add_argument("--api-keys-file", default="", help="File with one key per line.")
    ap.add_argument("--max-workers", type=int, default=0,
                    help="Process workers; default = number of keys.")
    args = ap.parse_args()

    # ---- load prompts ----
    indexed_path = Path(args.indexed)
    if not indexed_path.exists():
        raise SystemExit(f"[FATAL] indexed file not found: {indexed_path.resolve()}")

    idx2prompt = load_indexed_prompts(indexed_path)

    # ---- indices to run ----
    if args.indices.strip():
        try:
            wanted = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
        except Exception:
            raise SystemExit("[FATAL] --indices must be comma-separated integers")
    else:
        wanted = list(DEFAULT_INDICES)

    # Build (index, prompt) list and report missing
    pairs: List[Tuple[int, str]] = []
    missing: List[int] = []
    for i in wanted:
        pr = idx2prompt.get(i)
        if isinstance(pr, str):
            pairs.append((i, pr))
        else:
            missing.append(i)

    if missing:
        print(f"[WARN] {len(missing)} indices not found in {indexed_path.name}: {missing[:12]}{'...' if len(missing)>12 else ''}")

    # ---- write Phase D input file ----
    prep_out = Path(args.prep_out)
    n_written = write_prompts_only(pairs, prep_out)
    print(f"[OK] Wrote {n_written} prompts -> {prep_out}")

    if n_written == 0:
        print("[DONE] Nothing to run. Exiting.")
        return

    # ---- set up keys + workers ----
    keys = parse_api_keys(args)
    if not keys:
        raise SystemExit("[FATAL] No API key provided. Use --api-keys / --api-keys-file / DEEPSEEK_API_KEYS env.")
    n_workers = args.max_workers if args.max_workers > 0 else len(keys)
    n_workers = max(1, min(n_workers, len(keys)))
    print(f"[INFO] Running {len(pairs)} prompts with {len(keys)} key(s), {n_workers} worker(s)")
    print(f"[INFO] target_model={args.target_model}  base_url={args.base_url or '(default)'}")

    # Round-robin assign keys
    tasks = [(idx, pr, keys[i % len(keys)]) for i, (idx, pr) in enumerate(pairs)]

    # ---- stream output as each finishes ----
    out_path = Path(args.out)
    out_path.write_text("", encoding="utf-8")  # truncate/create

    # Also write an input echo next to output
    echo_in = out_path.with_name(out_path.stem + "_input.jsonl")
    with echo_in.open("w", encoding="utf-8") as f_in:
        for idx, pr, _k in tasks:
            f_in.write(json.dumps({"index": idx, "prompt": pr}, ensure_ascii=False) + "\n")
    print(f"[OK] Echoed inputs -> {echo_in}")

    t0 = time.time()
    ok = err = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [
            ex.submit(_worker_run_one, idx, pr, args.target_model, args.max_out_tokens,
                      key, args.base_url or None)
            for (idx, pr, key) in tasks
        ]
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            if rec.get("error"):
                err += 1
            else:
                ok += 1
            # append immediately
            with out_path.open("a", encoding="utf-8") as fout:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()

            cid = rec.get("custom_id")
            status = "OK" if not rec.get("error") else f"ERR: {rec.get('error')}"
            print(f"[stream] {i}/{len(futs)}  custom_id={cid}  {status}")

    print(f"[DONE] wrote {ok+err} records -> {out_path}  (ok={ok}, err={err})  in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
