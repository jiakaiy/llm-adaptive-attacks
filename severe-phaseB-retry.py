# ty file - severe-phaseB-retry.py
#!/usr/bin/env python3
"""
Phase B retry (concurrent, multi-key):
- Read custom_ids from error-severe-phaseA(.jsonl)
- Map to prompts by index in filtered_severe_prompts_indexed.jsonl
- Re-run with template "refined_best_catfact" against DeepSeek
- Distribute requests across multiple DeepSeek API keys in parallel

Usage (single key):
  export DEEPSEEK_API_KEY=sk-...   # prefer env var, not CLI
  python severe-phaseB-retry.py --errors error-severe-phaseA --indexed filtered_severe_prompts_indexed.jsonl

Usage (two keys, via CLI):
  python severe-phaseB-retry.py \
    --errors error-severe-phaseA.jsonl \
    --indexed filtered_severe_prompts_indexed.jsonl \
    --api-keys "sk-KEY1,sk-KEY2" \
    --out retry_catfact_results.jsonl \
    --target-model deepseek-reasoner \
    --max-out-tokens 350 \
    --max-workers 2

Usage (two keys, via file — one key per line):
  python severe-phaseB-retry.py --api-keys-file keys.txt ...

Notes:
  * We run one-shot per prompt (n_iterations=1, n_restarts=1) to maximize throughput.
  * For safety: only use with benign prompts or to log refusals in evaluation settings.
"""

import argparse, json, re, sys, time, os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import your existing runner bits (used inside worker)
try:
    from main import main as run_one, get_default_args
except Exception as e:
    print("[FATAL] Could not import main.py (need main(), get_default_args()).", e)
    sys.exit(2)

CUSTOM_ID_RE = re.compile(r'"custom_id"\s*:\s*"?(?P<cid>\d+)"?')

def iter_jsonl_lines(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            yield ln, s

def extract_custom_ids(err_path: Path) -> List[int]:
    ids: List[int] = []
    for ln, s in iter_jsonl_lines(err_path):
        try:
            obj = json.loads(s)
            cid = obj.get("custom_id")
            if cid is None and isinstance(obj, dict):
                sub = obj
                for k in ("response", "body", "error", "meta", "data"):
                    if isinstance(sub, dict) and k in sub:
                        sub = sub[k]
                if isinstance(sub, dict):
                    cid = sub.get("custom_id")
            if cid is None:
                m = CUSTOM_ID_RE.search(s)
                cid = int(m.group("cid")) if m else None
            if cid is None:
                continue
            ids.append(int(cid))
        except Exception:
            m = CUSTOM_ID_RE.search(s)
            if m:
                ids.append(int(m.group("cid")))
    return sorted(set(ids))

def load_indexed_prompts(path: Path) -> Dict[int, str]:
    idx2p: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                obj = {"prompt": s}
            idx = obj.get("index")
            pr  = obj.get("prompt")
            if idx is None or pr is None:
                continue
            idx2p[int(idx)] = str(pr)
    return idx2p

def build_args_for_prompt(prompt: str, target_model: str, max_out_tokens: int):
    args = get_default_args()
    args.goal = prompt
    args.goal_modified = ""
    args.prompt_template = "refined_best_catfact"
    args.target_model = target_model
    args.target_max_n_tokens = int(max_out_tokens)
    args.judge_model = "no-judge"
    args.debug = True

    # one-shot, no search/mutations
    args.n_iterations = 1
    args.n_restarts = 1
    args.n_chars_change_max = 0
    args.n_tokens_change_max = 0
    return args

def resolve_path(p: str) -> Path:
    cand = Path(p)
    if cand.exists():
        return cand
    cand2 = Path(p + ".jsonl")
    if cand2.exists():
        return cand2
    print(f"[FATAL] File not found: {p}")
    sys.exit(2)

def parse_api_keys(args) -> List[str]:
    keys: List[str] = []
    if args.api_keys:
        keys.extend([k.strip() for k in args.api_keys.split(",") if k.strip()])
    if args.api_keys_file:
        path = Path(args.api_keys_file)
        if not path.exists():
            print(f"[FATAL] --api-keys-file not found: {path}")
            sys.exit(2)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    keys.append(s)
    # Fallback to env
    env_single = os.getenv("DEEPSEEK_API_KEY")
    if env_single and not keys:
        keys = [env_single]
    env_multi = os.getenv("DEEPSEEK_API_KEYS")
    if env_multi and not keys:
        keys = [k.strip() for k in env_multi.split(",") if k.strip()]

    # Dedupe while preserving order
    seen = set()
    uniq = []
    for k in keys:
        if k not in seen:
            uniq.append(k); seen.add(k)
    return uniq

def _worker_run_one(cid: int, prompt: str, target_model: str, max_out_tokens: int, api_key: str, base_url: Optional[str]) -> dict:
    """
    Runs in a separate process so environment variables don't clash across workers.
    """
    try:
        # Set env for this process only
        if api_key:
            os.environ["DEEPSEEK_API_KEY"] = api_key
            os.environ["OPENAI_API_KEY"] = api_key  # some stacks read from this
        if base_url:
            os.environ["DEEPSEEK_API_BASE"] = base_url
            os.environ["OPENAI_BASE_URL"] = base_url
            os.environ["OPENAI_API_BASE"] = base_url

        args = build_args_for_prompt(prompt, target_model, max_out_tokens)
        noadv, orig, final = run_one(args)
        return {
            "custom_id": cid,
            "prompt": prompt,
            "template": "refined_best_catfact",
            "target_model": target_model,
            "noadv": noadv,
            "orig": orig,
            "final": final,
            "error": None,
        }
    except Exception as e:
        return {
            "custom_id": cid,
            "prompt": prompt,
            "template": "refined_best_catfact",
            "target_model": target_model,
            "error": str(e),
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--errors", default="error-severe-phaseA")
    ap.add_argument("--indexed", default="filtered_severe_prompts_indexed.jsonl")
    ap.add_argument("--out", default="retry_catfact_results.jsonl")
    ap.add_argument("--target-model", default="deepseek-reasoner",
                    help="Your conversers.py must map this to DeepSeekReasonerWrapper")
    ap.add_argument("--max-out-tokens", type=int, default=350)
    ap.add_argument("--limit", type=int, default=0, help="Run only first N IDs; 0 means no limit.")

    # Multi-key & concurrency
    ap.add_argument("--api-keys", default="", help="Comma-separated DeepSeek keys (avoid shell history; prefer --api-keys-file or env).")
    ap.add_argument("--api-keys-file", default="", help="Path to a file with one key per line.")
    ap.add_argument("--max-workers", type=int, default=0, help="Number of worker processes. Default = number of keys (or 1).")
    ap.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", ""), help="Optional custom base URL (e.g., Bailian).")

    args = ap.parse_args()

    err_path = resolve_path(args.errors)
    idx_path = resolve_path(args.indexed)
    out_path = Path(args.out)

    keys = parse_api_keys(args)
    if not keys:
        print("[FATAL] No API key provided. Use --api-keys, --api-keys-file, or set DEEPSEEK_API_KEY / DEEPSEEK_API_KEYS.")
        sys.exit(2)

    n_workers = args.max_workers if args.max_workers > 0 else len(keys)
    n_workers = max(1, min(n_workers, len(keys)))
    print(f"[INFO] Using {n_workers} worker process(es) with {len(keys)} API key(s).")

    print(f"[INFO] Reading custom_ids from {err_path}")
    ids = extract_custom_ids(err_path)
    print(f"[INFO] Found {len(ids)} unique custom_ids")

    if args.limit and args.limit > 0:
        ids = ids[: args.limit]
        print(f"[INFO] Limiting to first {len(ids)} ids due to --limit")

    print(f"[INFO] Loading indexed prompts from {idx_path}")
    idx2prompt = load_indexed_prompts(idx_path)
    print(f"[INFO] Loaded {len(idx2prompt)} indexed prompts")

    # Prepare tasks (round-robin assign keys)
    tasks: List[Tuple[int, str, str]] = []
    for idx, cid in enumerate(ids):
        pr = idx2prompt.get(cid)
        if pr is None:
            continue
        key = keys[idx % len(keys)]
        tasks.append((cid, pr, key))

    # Input echo
    echo_in = Path(out_path.stem + "_input.jsonl")
    with echo_in.open("w", encoding="utf-8") as f_in:
        for cid, pr, _k in tasks:
            f_in.write(json.dumps({"custom_id": cid, "prompt": pr}, ensure_ascii=False) + "\n")
    print(f"[OK] Wrote input echo -> {echo_in}")

    # Run concurrently
    n_ok = n_err = 0
    t0 = time.time()
    with out_path.open("w", encoding="utf-8") as fout:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = []
            for cid, pr, key in tasks:
                fut = ex.submit(_worker_run_one, cid, pr, args.target_model, args.max_out_tokens, key, args.base_url or None)
                futs.append(fut)
            for i, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                if rec.get("error"):
                    n_err += 1
                else:
                    n_ok += 1
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if i % 10 == 0 or i == len(futs):
                    print(f"[progress] {i}/{len(futs)} written... (ok={n_ok}, err={n_err})")

    dt = time.time() - t0
    print(f"[DONE] wrote {n_ok} ok, {n_err} errors -> {out_path}  in {dt:.1f}s")

if __name__ == "__main__":
    main()
