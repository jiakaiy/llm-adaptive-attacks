# Empty file 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
moderate_phaseB_pay.py

Fix up DeepSeek 402 Payment Required rows in retry_catfact_results.jsonl:
 - detect rows whose "error" mentions 402 / Payment Required
 - re-run those prompts locally via your main.py pipeline
   using template "catattack_trivia_tail" and target_model "deepseek-reasoner"
 - merge fresh results with the non-error rows
 - write retry_catfact_results_final.jsonl

Usage:
  # preferred: env var
  export DEEPSEEK_API_KEY=sk-...   # your DeepSeek key
  python moderate_phaseB_pay.py

  # or pass the key explicitly (avoid in shared terminals):
  python moderate_phaseB_pay.py --api-key sk-...

Flags:
  --in    (default: retry_catfact_results.jsonl)
  --out   (default: retry_catfact_results_final.jsonl)
  --limit (int, 0 = no limit) only retry first N errors
  --max-out-tokens (default 350)
  --base-url  (optional) override DeepSeek base URL
  --sorted-output  also write *_sorted.jsonl by custom_id
"""

import argparse, json, os, re, sys, time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union, Optional

# Import your local runner
try:
    from main import main as run_one, get_default_args
except Exception as e:
    print("[FATAL] Could not import main.py (need main(), get_default_args()).", e)
    sys.exit(2)

PAYMENT_RE = re.compile(r"(402|payment\s*required)", re.I)

# ---------- IO helpers ----------

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception as e:
                sys.stderr.write(f"[WARN] {path}:{ln}: bad JSON: {e}\n")

def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------- logic ----------

def is_payment_error(rec: Dict[str, Any]) -> bool:
    err = rec.get("error", "")
    if not isinstance(err, str):
        return False
    return bool(PAYMENT_RE.search(err))

def load_results(path: Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))

def build_args_for_prompt(prompt: str, max_out_tokens: int, target_model: str = "deepseek-reasoner"):
    args = get_default_args()
    # minimal, fast, one-shot:
    args.goal = prompt
    args.goal_modified = ""
    args.prompt_template = "catattack_trivia_tail"
    args.target_model = target_model
    args.target_max_n_tokens = int(max_out_tokens)
    args.judge_model = "no-judge"
    args.debug = True

    args.n_iterations = 1
    args.n_restarts = 1
    args.n_chars_change_max = 0
    args.n_tokens_change_max = 0

    # benign target string for this template (template can ignore it if not needed)
    args.target_str = ""   # let the template provide its own

    return args

def retry_one(index: int, prompt: str, max_out_tokens: int, target_model: str, base_url: Optional[str]) -> Dict[str, Any]:
    # Isolate env for the local DeepSeek wrapper
    if base_url:
        os.environ["DEEPSEEK_API_BASE"] = base_url
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["OPENAI_API_BASE"] = base_url

    args = build_args_for_prompt(prompt, max_out_tokens, target_model=target_model)
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
    }

def merge_replace_by_custom_id(original: List[Dict[str, Any]], updates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id: Dict[int, Dict[str, Any]] = {}
    order: List[int] = []
    for r in original:
        cid = r.get("custom_id")
        try:
            cid = int(cid)
        except Exception:
            cid = None
        if cid is None:
            # keep non-conforming rows by appending with no key replace
            order.append(None)  # type: ignore
            by_id[id(r)] = r    # unique key to avoid collision
        else:
            if cid not in by_id:
                order.append(cid)
            by_id[cid] = r
    # apply updates
    for u in updates:
        cid = u.get("custom_id")
        if cid is None:
            continue
        try:
            cid = int(cid)
        except Exception:
            continue
        if cid not in by_id:
            order.append(cid)
        by_id[cid] = u
    # rebuild in original order (updated rows replace originals)
    out: List[Dict[str, Any]] = []
    seen_keys = set()
    for k in order:
        if k is None:
            # find the stored "id(r)" placeholders
            # collect any we haven't emitted
            for any_key, val in list(by_id.items()):
                if isinstance(any_key, int) and any_key not in seen_keys and "custom_id" not in val:
                    out.append(val); seen_keys.add(any_key)
        else:
            if k in by_id and k not in seen_keys:
                out.append(by_id[k]); seen_keys.add(k)
    # also append any leftovers (shouldn't happen, but safe)
    for k, v in by_id.items():
        if k not in seen_keys:
            out.append(v)
    return out

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="retry_catfact_results.jsonl")
    ap.add_argument("--out", dest="out_path", default="retry_catfact_results_final.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="Only retry first N errors (0 = all)")
    ap.add_argument("--max-out-tokens", type=int, default=350)
    ap.add_argument("--target-model", default="deepseek-reasoner")
    ap.add_argument("--api-key", default="", help="DeepSeek API key (prefer env DEEPSEEK_API_KEY)")
    ap.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", ""),
                    help="Optional custom base URL; leave blank to use https://api.deepseek.com")
    ap.add_argument("--sorted-output", action="store_true",
                    help="Also write *_sorted.jsonl sorted by custom_id")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise SystemExit(f"[FATAL] input not found: {in_path}")

    # Key setup (do NOT hardcode in the script)
    api_key = args.api_key.strip() or os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("[FATAL] Missing DeepSeek key. Pass --api-key or export DEEPSEEK_API_KEY=sk-...")
    os.environ["DEEPSEEK_API_KEY"] = api_key
    os.environ["OPENAI_API_KEY"] = api_key  # some stacks read this

    base_url = (args.base_url or "").strip() or None
    if base_url:
        os.environ["DEEPSEEK_API_BASE"] = base_url
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["OPENAI_API_BASE"] = base_url

    all_rows = load_results(in_path)
    errs = [r for r in all_rows if is_payment_error(r)]
    print(f"[INFO] loaded {len(all_rows)} rows, found {len(errs)} Payment-Required errors.")

    if args.limit and args.limit > 0:
        errs = errs[: args.limit]
        print(f"[INFO] limiting retries to first {len(errs)} error rows due to --limit")

    # sanity & echo
    retry_pairs: List[Tuple[int, str]] = []
    for r in errs:
        cid = r.get("custom_id")
        prm = r.get("prompt")
        try:
            cid = int(cid)
        except Exception:
            cid = None
        if cid is None or not isinstance(prm, str) or not prm.strip():
            continue
        retry_pairs.append((cid, prm.strip()))

    echo_in = Path(args.out_path).with_name(Path(args.out_path).stem.replace(".jsonl", "") + "_retry_input.jsonl")
    with echo_in.open("w", encoding="utf-8") as f:
        for idx, pr in retry_pairs:
            f.write(json.dumps({"custom_id": idx, "prompt": pr}, ensure_ascii=False) + "\n")
    print(f"[OK] wrote retry input echo -> {echo_in} (n={len(retry_pairs)})")

    # run sequentially (few failures usually)
    fresh: List[Dict[str, Any]] = []
    t0 = time.time()
    for i, (idx, pr) in enumerate(retry_pairs, 1):
        try:
            rec = retry_one(idx, pr, args.max_out_tokens, args.target_model, base_url)
            fresh.append(rec)
        except Exception as e:
            fresh.append({
                "custom_id": idx,
                "index": idx,
                "prompt": pr,
                "template": "catattack_trivia_tail",
                "target_model": args.target_model,
                "error": str(e),
            })
        if i % 10 == 0 or i == len(retry_pairs):
            ok = sum(1 for r in fresh if "error" not in r)
            err = sum(1 for r in fresh if "error" in r)
            print(f"[progress] retried {i}/{len(retry_pairs)} ... ok={ok} err={err}")

    # merge: keep all non-402 rows from original, replace 402 rows with fresh
    final_rows = merge_replace_by_custom_id(
        [r for r in all_rows if not is_payment_error(r)],
        fresh
    )

    out_path = Path(args.out_path)
    write_jsonl(out_path, final_rows)
    print(f"[DONE] wrote {len(final_rows)} rows -> {out_path}  in {time.time()-t0:.1f}s")

    if args.sorted_output:
        sorted_path = out_path.with_name(out_path.stem + "_sorted.jsonl")
        try:
            final_sorted = sorted(final_rows, key=lambda r: int(r.get("custom_id", 10**12)))
        except Exception:
            final_sorted = final_rows
        write_jsonl(sorted_path, final_sorted)
        print(f"[DONE] also wrote sorted copy -> {sorted_path}")

if __name__ == "__main__":
    main()
