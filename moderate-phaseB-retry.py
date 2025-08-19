#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retry Phase B (moderate): run prompts from moderate_phaseA_prompts_only.jsonl
with template "catattack_trivia_tail" against DeepSeek, using multiple API keys
in parallel for speed. Uses your local main.py pipeline (no batch API).

Usage (recommended: keys file):
  # keys.txt: one DEEPSEEK key per line
  python retry_moderate_phaseB.py \
    --input moderate_phaseA_prompts_only.jsonl \
    --api-keys-file keys.txt \
    --target-model deepseek-reasoner \
    --out retry_moderate_phaseB_results.jsonl \
    --max-out-tokens 350 \
    --max-workers 3 \
    --sorted-output

Or via env:
  export DEEPSEEK_API_KEYS="sk-KEY1,sk-KEY2,sk-KEY3"
  python retry_moderate_phaseB.py --input moderate_phaseA_prompts_only.jsonl --max-workers 3
"""

import argparse, json, os, sys, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# Import your existing runner bits
try:
    from main import main as run_one, get_default_args
except Exception as e:
    print("[FATAL] Could not import main.py (need main(), get_default_args()).", e)
    sys.exit(2)


def load_pairs_from_prompts_only(path: Path) -> List[Tuple[int, str]]:
    """
    Expect each line: {"index": <int>, "prompt": "<text>"}
    """
    pairs: List[Tuple[int, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
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
                pairs.append((int(idx), str(pr)))
            except Exception:
                continue
    # dedupe by index, keep first
    seen = set()
    uniq = []
    for idx, pr in pairs:
        if idx in seen:
            continue
        uniq.append((idx, pr)); seen.add(idx)
    return uniq


def build_args_for_prompt(prompt: str, target_model: str, max_out_tokens: int):
    args = get_default_args()
    # Override what we need
    args.goal = prompt
    args.goal_modified = ""  # let main() copy goal into goal_modified
    args.prompt_template = "catattack_trivia_tail"
    args.target_model = target_model
    args.target_max_n_tokens = int(max_out_tokens)
    args.judge_model = "no-judge"  # avoid extra judge calls
    args.debug = True              # skip W&B and internal save_run

    # One-shot, no mutations
    args.n_iterations = 1
    args.n_restarts = 1
    args.n_chars_change_max = 0
    args.n_tokens_change_max = 0

    # Benign target_str for this trivia template
    args.target_str = "Let's answer the trivia question"
    return args


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
    # Fallbacks
    env_multi = os.getenv("DEEPSEEK_API_KEYS")
    if env_multi and not keys:
        keys = [k.strip() for k in env_multi.split(",") if k.strip()]
    env_single = os.getenv("DEEPSEEK_API_KEY")
    if env_single and not keys:
        keys = [env_single]

    # Dedupe preserving order
    seen = set(); uniq = []
    for k in keys:
        if k not in seen:
            uniq.append(k); seen.add(k)
    return uniq


def _worker_run_one(index: int, prompt: str, target_model: str, max_out_tokens: int,
                    api_key: str, base_url: Optional[str]) -> dict:
    """
    Runs in a separate process so env vars don't collide across workers.
    """
    try:
        if api_key:
            os.environ["DEEPSEEK_API_KEY"] = api_key
            os.environ["OPENAI_API_KEY"] = api_key  # some stacks read this
        if base_url:
            os.environ["DEEPSEEK_API_BASE"] = base_url
            os.environ["OPENAI_BASE_URL"] = base_url
            os.environ["OPENAI_API_BASE"] = base_url

        args = build_args_for_prompt(prompt, target_model, max_out_tokens)
        noadv, orig, final = run_one(args)
        return {
            "custom_id": index,
            "index": index,
            "prompt": prompt,
            "template": "catattack_trivia_tail",
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
            "template": "catattack_trivia_tail",
            "target_model": target_model,
            "error": str(e),
        }


def resolve_out_path(p: str) -> Path:
    out = Path(p)
    if out.suffix == "":
        out = out.with_suffix(".jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    return out.resolve()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="moderate_phaseA_prompts_only.jsonl",
                    help="JSONL with lines of {'index': int, 'prompt': str}")
    ap.add_argument("--out", default="retry_moderate_phaseB_results.jsonl")
    ap.add_argument("--target-model", default="deepseek-reasoner",
                    help="Must map to DeepSeekReasonerWrapper() in conversers.py")
    ap.add_argument("--max-out-tokens", type=int, default=350)
    ap.add_argument("--limit", type=int, default=0, help="Run only first N; 0 means no limit.")

    # Multi-key + concurrency
    ap.add_argument("--api-keys", default="", help="Comma-separated keys (prefer file or env).")
    ap.add_argument("--api-keys-file", default="", help="File with one key per line.")
    ap.add_argument("--max-workers", type=int, default=0,
                    help="Number of worker processes. Default = number of keys (or 1).")
    ap.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", ""),
                    help="Optional custom base URL (e.g., Bailian-compatible endpoint).")

    # Optional sorted copy
    ap.add_argument("--sorted-output", action="store_true",
                    help="Also write *_sorted.jsonl by custom_id asc.")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise SystemExit(f"[FATAL] input not found: {in_path.resolve()}")

    keys = parse_api_keys(args)
    if not keys:
        raise SystemExit("[FATAL] No API key provided. Use --api-keys, --api-keys-file, or DEEPSEEK_API_KEYS env.")

    pairs = load_pairs_from_prompts_only(in_path)
    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]
    n_total = len(pairs)
    if not n_total:
        raise SystemExit("[FATAL] No prompts found.")

    # Prepare tasks: round-robin assign keys
    tasks: List[Tuple[int, str, str]] = []
    for i, (idx, pr) in enumerate(pairs):
        key = keys[i % len(keys)]
        tasks.append((idx, pr, key))

    # Resolve and pre-create output files
    out_path = resolve_out_path(args.out)
    echo_in = out_path.with_name(out_path.stem + "_input.jsonl")
    # Touch results file so it exists even if something crashes later
    out_path.write_text("", encoding="utf-8")

    # Input echo
    with echo_in.open("w", encoding="utf-8") as f_in:
        for idx, pr, _key in tasks:
            f_in.write(json.dumps({"index": idx, "prompt": pr}, ensure_ascii=False) + "\n")
    print(f"[OK] wrote input echo -> {echo_in}")
    print(f"[INFO] streaming results to -> {out_path}")

    n_workers = args.max_workers if args.max_workers > 0 else len(keys)
    n_workers = max(1, min(n_workers, len(keys)))
    print(f"[INFO] Using {n_workers} worker(s) over {len(keys)} key(s); target_model={args.target_model}")

    # Run concurrently and STREAM results as they arrive
    t0 = time.time()
    recs: List[dict] = []
    with out_path.open("a", encoding="utf-8") as fout:
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = []
            for idx, pr, key in tasks:
                fut = ex.submit(_worker_run_one, idx, pr, args.target_model,
                                args.max_out_tokens, key, args.base_url or None)
                futs.append(fut)
            for i, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                recs.append(rec)
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                if i % 20 == 0 or i == len(futs):
                    ok = sum(1 for r in recs if not r.get("error"))
                    err = sum(1 for r in recs if r.get("error"))
                    print(f"[progress] {i}/{len(futs)} written... (ok={ok}, err={err})")

    print(f"[DONE] wrote {len(recs)} records -> {out_path} in {time.time()-t0:.1f}s")

    if args.sorted_output:
        sorted_path = out_path.with_name(out_path.stem + "_sorted.jsonl")
        recs_sorted = sorted(recs, key=lambda r: int(r.get("custom_id", 10**12)))
        with sorted_path.open("w", encoding="utf-8") as f2:
            for r in recs_sorted:
                f2.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[DONE] also wrote sorted copy -> {sorted_path}")


if __name__ == "__main__":
    main()
