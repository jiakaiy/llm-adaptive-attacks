#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase C pipeline (moderate) — STREAMING VERSION
- Filters result_moderateB_shard*.jsonl for rows whose assistant content does NOT start with:
    "<text>\\nFun fact: ...\\n\\nStep 1:"
- Maps custom_id -> {index,prompt} from filtered_moderate_prompts_indexed.jsonl
  and writes retry_catnonsense_phaseC_input.jsonl
- Re-runs those prompts with template "catnonsense" via your local main.py,
  using multiple DeepSeek keys in parallel.
- **Writes retry_catnonsense_phaseC_output.jsonl incrementally** as each job finishes.

Run:
  export DEEPSEEK_API_KEYS="sk-KEY1,sk-KEY2"
  python3 moderate_phaseC_pipeline.py \
    --results-glob "result_moderateB_shard*.jsonl" \
    --indexed filtered_moderate_prompts_indexed.jsonl
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

DEFAULT_RESULTS_GLOB = "result_moderateB_shard*.jsonl"
INDEXED_PROMPTS = "filtered_moderate_prompts_indexed.jsonl"
PHASEC_INPUT = "retry_catnonsense_phaseC_input.jsonl"
PHASEC_OUTPUT = "retry_catnonsense_phaseC_output.jsonl"

CATFACT_STEP1_RE = re.compile(
    r"^\s*(?:<text>\s*)?fun\s*fact\s*:\s.*?(?:\r?\n){0,3}\s*step\s*1\s*:",
    re.IGNORECASE | re.DOTALL,
)
CUSTOM_ID_RE = re.compile(r'"custom_id"\s*:\s*"?(?P<cid>\d+)"?')

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

def extract_message_content(obj: dict) -> str:
    try:
        choice = obj["response"]["body"]["choices"][0]
        msg = choice.get("message", {})
        return (msg.get("content") or msg.get("reasoning_content") or "")
    except Exception:
        return ""

def get_custom_id_from_obj_or_line(obj: Optional[dict], line: str) -> Optional[int]:
    if isinstance(obj, dict):
        cid = obj.get("custom_id")
        if cid is None:
            cur = obj
            for k in ("response", "body", "error", "meta", "data"):
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
            if isinstance(cur, dict):
                cid = cur.get("custom_id")
        if cid is not None:
            try: return int(str(cid))
            except Exception: pass
    m = CUSTOM_ID_RE.search(line or "")
    if m:
        try: return int(m.group("cid"))
        except Exception: return None
    return None

def is_good_funfact_start(content: str) -> bool:
    return bool(CATFACT_STEP1_RE.search(content or ""))

def collect_bad_custom_ids(results_paths: List[Path]) -> List[int]:
    bad: Set[int] = set()
    total = 0
    for rp in results_paths:
        for _, line in iter_jsonl(rp):
            total += 1
            obj = None
            try:
                obj = json.loads(line)
            except Exception:
                pass
            content = extract_message_content(obj or {})
            if not content or not is_good_funfact_start(content):
                cid = get_custom_id_from_obj_or_line(obj, line)
                if cid is not None:
                    bad.add(cid)
    print(f"[filter] scanned {total} row(s) across {len(results_paths)} file(s); selected {len(bad)} to retry")
    return sorted(bad)

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

def build_args_for_prompt(prompt: str, target_model: str, max_out_tokens: int):
    args = get_default_args()
    args.goal = prompt
    args.goal_modified = ""
    args.prompt_template = "catnonsense"
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
                    api_key: str, base_url: Optional[str]) -> dict:
    try:
        if api_key:
            os.environ["DEEPSEEK_API_KEY"] = api_key
            os.environ["OPENAI_API_KEY"] = api_key
        # Set/clear base URLs
        if base_url:
            os.environ["DEEPSEEK_API_BASE"] = base_url
            os.environ["OPENAI_BASE_URL"] = base_url
            os.environ["OPENAI_API_BASE"] = base_url
        else:
            for v in ("DEEPSEEK_API_BASE","OPENAI_BASE_URL","OPENAI_API_BASE"):
                os.environ.pop(v, None)

        args = build_args_for_prompt(prompt, target_model, max_out_tokens)
        noadv, orig, final = run_one(args)
        return {
            "custom_id": index,
            "index": index,
            "prompt": prompt,
            "template": "catnonsense",
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
            "template": "catnonsense",
            "target_model": target_model,
            "error": str(e),
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-glob", default=DEFAULT_RESULTS_GLOB)
    ap.add_argument("--indexed", default=INDEXED_PROMPTS)
    ap.add_argument("--prep-out", default=PHASEC_INPUT)
    ap.add_argument("--out", default=PHASEC_OUTPUT)
    ap.add_argument("--target-model", default="deepseek-reasoner")
    ap.add_argument("--max-out-tokens", type=int, default=350)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-run", action="store_true")

    # keys / concurrency / base-url
    ap.add_argument("--api-keys", default="")
    ap.add_argument("--api-keys-file", default="")
    ap.add_argument("--max-workers", type=int, default=0)
    ap.add_argument("--base-url", default="")
    args = ap.parse_args()

    files = sorted(Path(".").glob(args.results_glob))
    if not files:
        raise SystemExit(f"[FATAL] no files match {args.results_glob!r}")
    bad_ids = collect_bad_custom_ids(files)

    idx2prompt = load_indexed_prompts(Path(args.indexed))
    pairs: List[Tuple[int,str]] = []
    missing = 0
    for cid in bad_ids:
        pr = idx2prompt.get(cid)
        if pr is None:
            missing += 1
            continue
        pairs.append((cid, pr))
    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]

    prep_path = Path(args.prep_out)
    with prep_path.open("w", encoding="utf-8") as f:
        for idx, pr in pairs:
            f.write(json.dumps({"index": idx, "prompt": pr}, ensure_ascii=False) + "\n")
    print(f"[OK] wrote {len(pairs)} prompts -> {prep_path} (missing={missing})")

    if args.no_run or not pairs:
        print("[INFO] stop after input build ( --no-run or 0 prompts ).")
        return

    keys = parse_api_keys(args)
    if not keys:
        raise SystemExit("[FATAL] provide DeepSeek keys via --api-keys/--api-keys-file or DEEPSEEK_API_KEYS env.")

    n_workers = args.max_workers if args.max_workers > 0 else len(keys)
    n_workers = max(1, min(n_workers, len(keys), 8))
    base_url = args.base_url.strip() or ""  # default: DeepSeek public

    print(f"[RUN] {len(pairs)} prompt(s); {len(keys)} key(s); {n_workers} worker(s); model={args.target_model}")

    # round-robin key assignment
    tasks = [(idx, pr, keys[i % len(keys)]) for i, (idx, pr) in enumerate(pairs)]

    # echo inputs
    echo_in = prep_path.with_name(prep_path.stem + "_echo.jsonl")
    with echo_in.open("w", encoding="utf-8") as f:
        for idx, pr, _k in tasks:
            f.write(json.dumps({"index": idx, "prompt": pr}, ensure_ascii=False) + "\n")
    print(f"[OK] wrote input echo -> {echo_in}")

    out_path = Path(args.out)
    # create/truncate output BEFORE launching workers
    with out_path.open("w", encoding="utf-8") as fout:
        fout.write("")  # ensure file exists & is empty

    t0 = time.time()
    done = 0
    ok = err = 0

    # launch + STREAM writes as results come back
    with ProcessPoolExecutor(max_workers=n_workers) as ex, out_path.open("a", encoding="utf-8") as fout:
        futs = [ex.submit(_worker_run_one, idx, pr, args.target_model,
                          args.max_out_tokens, key, base_url or None)
                for (idx, pr, key) in tasks]

        for fut in as_completed(futs):
            rec = fut.result()
            # write one line immediately
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()  # make it visible right away
            done += 1
            if rec.get("error"):
                err += 1
            else:
                ok += 1
            if done % 10 == 0:
                print(f"[progress] {done}/{len(futs)} written… (ok={ok}, err={err})")

    print(f"[DONE] wrote {done} records -> {out_path} in {time.time()-t0:.1f}s (ok={ok}, err={err})")

if __name__ == "__main__":
    main()
# 