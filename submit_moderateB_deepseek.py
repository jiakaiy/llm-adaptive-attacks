#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
submit_moderateB_deepseek.py  (CN defaults)

Submit an OpenAI-compatible batch JSONL to Alibaba Cloud Model Studio (Bailian),
splitting into N shards and optionally submitting ALL shards at once.

If a shard fails validation, this script downloads and prints the first few
lines of the error file so you can see why.
"""

import os, time, json, argparse, math, random
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
from openai import OpenAI
from openai import RateLimitError

TERMINAL = {"completed", "failed", "expired", "cancelled"}

def pick_base_url_by_region(region: str) -> str:
    region = (region or "cn").lower()
    if region == "cn":
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"
    return "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

def load_runtime(runtime_json: Optional[str], base_url_arg: Optional[str], region_arg: Optional[str]) -> Tuple[str, str]:
    base_url, endpoint = "", "/v1/chat/completions"
    if runtime_json and Path(runtime_json).exists():
        try:
            obj = json.loads(Path(runtime_json).read_text(encoding="utf-8"))
            base_url = (obj.get("base_url") or "").rstrip("/")
            endpoint = obj.get("endpoint") or endpoint
        except Exception:
            pass
    if base_url_arg:
        base_url = base_url_arg.rstrip("/")
    if not base_url and region_arg:
        base_url = pick_base_url_by_region(region_arg)
    if not base_url:
        base_url = pick_base_url_by_region("cn")
    return base_url, endpoint

def load_api_key(args) -> str:
    if args.api_key:
        return args.api_key.strip()
    if args.api_key_file:
        p = Path(args.api_key_file)
        if p.exists():
            return p.read_text(encoding="utf-8").strip().splitlines()[0]
    return os.getenv(args.api_key_env, "").strip()

def count_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for _ in f: n += 1
    return n

def split_into_n_shards(input_path: Path, shards: int, out_dir: Path, prefix: str) -> List[Tuple[Path, int]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    total = count_lines(input_path)
    shards = max(1, min(shards, total))
    per = math.ceil(total / shards)

    shard_files: List[Tuple[Path, int]] = []
    fouts = []
    for i in range(shards):
        sp = out_dir / f"{prefix}{i:02d}.jsonl"
        fout = sp.open("w", encoding="utf-8")
        shard_files.append((sp, 0))
        fouts.append(fout)

    i_shard = 0
    written = 0
    with input_path.open("r", encoding="utf-8") as fin:
        for _, line in enumerate(fin):
            if written >= per and i_shard < shards - 1:
                fouts[i_shard].close()
                spath, _ = shard_files[i_shard]
                shard_files[i_shard] = (spath, written)
                i_shard += 1
                written = 0
            fouts[i_shard].write(line); written += 1

    for j in range(i_shard, shards):
        try: fouts[j].close()
        except Exception: pass
    spath, _ = shard_files[i_shard]
    shard_files[i_shard] = (spath, written)
    for k in range(i_shard + 1, shards):
        sp, _ = shard_files[k]
        shard_files[k] = (sp, 0)

    return [(p, n) for (p, n) in shard_files if n > 0]

def sniff_first_url(input_path: Path) -> str:
    try:
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s: continue
                obj = json.loads(s)
                if isinstance(obj, dict) and "url" in obj:
                    return obj["url"]
                break
    except Exception:
        pass
    return ""

def extract_counts(batch) -> Dict[str, int]:
    for key in ("request_counts", "counts"):
        rc = getattr(batch, key, None)
        if rc:
            try:
                if isinstance(rc, dict):
                    d = rc
                else:
                    d = rc.model_dump() if hasattr(rc, "model_dump") else rc.dict()
                return {
                    "total": int(d.get("total", 0)),
                    "completed": int(d.get("completed", 0)),
                    "failed": int(d.get("failed", 0)),
                    "running": int(d.get("running", d.get("processing", 0))),
                }
            except Exception:
                pass
    return {"total": 0, "completed": 0, "failed": 0, "running": 0}

def create_batch_with_backoff(client: OpenAI, shard_path: Path, endpoint: str, completion_window: str):
    retry = 0
    while True:
        try:
            with open(shard_path, "rb") as fh:
                fobj = client.files.create(file=fh, purpose="batch")
            batch = client.batches.create(
                input_file_id=fobj.id,
                endpoint=endpoint,
                completion_window=completion_window
            )
            return fobj, batch
        except RateLimitError:
            delay = min(60, (2 ** retry) + random.uniform(0, 1))
            print(f"[429] create for {shard_path.name} backoff {delay:.1f}s")
            time.sleep(delay); retry += 1; continue
        except Exception as e:
            msg = str(e)
            if "401" in msg or "invalid_api_key" in msg.lower():
                raise SystemExit(
                    "[FATAL] 401 Unauthorized while creating batch.\n"
                    "  • Ensure DASHSCOPE_API_KEY is a **CN** key.\n"
                    "  • base_url must be CN: https://dashscope.aliyuncs.com/compatible-mode/v1\n"
                    f"  • Raw error: {msg}"
                )
            delay = min(30, 5 + random.uniform(0, 2))
            print(f"[warn] create for {shard_path.name} err: {e}; retry in {delay:.1f}s")
            time.sleep(delay); continue

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", default="moderateB_batch_input.jsonl")
    ap.add_argument("--runtime", default="moderateB_runtime.json")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--region", choices=["intl","cn"], default=None)

    # Key
    ap.add_argument("--api_key", default=None)
    ap.add_argument("--api_key_file", default=None)
    ap.add_argument("--api_key_env", default="DASHSCOPE_API_KEY")

    # Shards / concurrency
    ap.add_argument("--shards", type=int, default=5)
    ap.add_argument("--submit-all", action="store_true")
    ap.add_argument("--max-parallel", type=int, default=5)
    ap.add_argument("--shard-dir", default="shards_moderateB")
    ap.add_argument("--shard-prefix", default="modB_")

    # Batch controls
    ap.add_argument("--completion-window", default="24h")
    ap.add_argument("--poll-secs", type=int, default=60)
    ap.add_argument("--progress-secs", type=int, default=60)

    # Output prefixes
    ap.add_argument("--out-prefix", default="result_moderateB")
    ap.add_argument("--err-prefix", default="error_moderateB")

    args = ap.parse_args()

    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise SystemExit(f"[err] input_jsonl not found: {input_path.resolve()}")

    base_url, endpoint = load_runtime(args.runtime, args.base_url, args.region)
    print(f"[info] base_url={base_url}")
    print(f"[info] endpoint={endpoint}")

    first_url = sniff_first_url(input_path)
    if first_url and first_url != endpoint:
        print(f"[warn] JSONL 'url' is {first_url!r} but batches.create endpoint is {endpoint!r}. They MUST match.")

    api_key = load_api_key(args)
    if not api_key or not api_key.startswith("sk-"):
        raise SystemExit("[FATAL] Missing/invalid DASHSCOPE_API_KEY (CN).")

    # neutralize OpenAI vars
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("OPENAI_BASE_URL", None)
    os.environ.pop("OPENAI_API_BASE", None)

    client = OpenAI(api_key=api_key, base_url=base_url)

    # Split into N shards
    shard_dir = Path(args.shard_dir)
    shards = split_into_n_shards(input_path, args.shards, shard_dir, args.shard_prefix)
    total = sum(n for _, n in shards)
    print(f"[split] total={total} -> shards={len(shards)} in {shard_dir}")

    active: List[Dict[str, Any]] = []
    finished: List[Dict[str, Any]] = []
    pending: List[Tuple[Path, int]] = shards[:]
    start_ts = time.time()
    next_progress_ts = start_ts

    def start_next_if_slots():
        while pending and len(active) < max(1, args.max_parallel):
            idx = len(active) + len(finished)
            spath, nlines = pending.pop(0)
            fobj, batch = create_batch_with_backoff(client, spath, endpoint, args.completion_window)
            job = {
                "i": idx, "path": spath, "n": nlines,
                "file_id": fobj.id, "batch_id": batch.id,
                "status": batch.status, "downloaded": False,
                "counts": {"total": nlines, "completed": 0, "failed": 0, "running": 0},
                "next_poll_ts": time.time() + (idx % 3) * 5,
            }
            active.append(job)
            print(f"[create] shard#{idx:02d} file_id={fobj.id} batch_id={batch.id} status={batch.status}")

    if args.submit_all:
        print("[mode] submit-all: creating ALL shards up-front...")
        while pending:
            idx = len(active) + len(finished)
            spath, nlines = pending.pop(0)
            fobj, batch = create_batch_with_backoff(client, spath, endpoint, args.completion_window)
            job = {
                "i": idx, "path": spath, "n": nlines,
                "file_id": fobj.id, "batch_id": batch.id,
                "status": batch.status, "downloaded": False,
                "counts": {"total": nlines, "completed": 0, "failed": 0, "running": 0},
                "next_poll_ts": time.time() + (idx % 3) * 5,
            }
            active.append(job)
            print(f"[create] shard#{idx:02d} file_id={fobj.id} batch_id={batch.id} status={batch.status}")
    else:
        start_next_if_slots()

    while active:
        now = time.time()
        due = [j["next_poll_ts"] for j in active]
        due.append(next_progress_ts)
        next_due = min(due)

        if now < next_due:
            time.sleep(min(args.poll_secs, max(0.0, next_due - now)))
            now = time.time()

        for j in list(active):
            if time.time() < j["next_poll_ts"]:
                continue

            retry = 0
            while True:
                try:
                    b = client.batches.retrieve(batch_id=j["batch_id"])
                    break
                except RateLimitError:
                    delay = min(60, (2 ** retry) + random.uniform(0, 1))
                    print(f"[429] shard#{j['i']:02d} poll backoff {delay:.1f}s")
                    time.sleep(delay); retry += 1; continue
                except Exception as e:
                    delay = min(30, 5 + random.uniform(0, 2))
                    print(f"[warn] shard#{j['i']:02d} poll err: {e}; retry in {delay:.1f}s")
                    time.sleep(delay); continue

            j["status"] = b.status
            j["counts"] = extract_counts(b)
            j["next_poll_ts"] = time.time() + args.poll_secs
            c = j["counts"]
            print(f"[poll] shard#{j['i']:02d} {j['status']} completed={c['completed']}/{c['total']} failed={c['failed']} running={c['running']}")

            if b.status == "failed":
                # Try to download validator error file
                try:
                    efid = getattr(b, "error_file_id", None) or (b.get("error_file_id") if isinstance(b, dict) else None)
                except Exception:
                    efid = None
                if efid:
                    err_name = f"{args.err_prefix}_shard{j['i']:02d}_{b.id}.jsonl"
                    try:
                        client.files.content(efid).write_to_file(err_name)
                        print(f"[done] wrote {err_name}")
                        try:
                            # Print first few lines for quick diagnosis
                            with open(err_name, "r", encoding="utf-8") as fh:
                                print("[error-head]")
                                for k in range(3):
                                    line = fh.readline().strip()
                                    if not line: break
                                    print("  ", line[:400])
                        except Exception:
                            pass
                    except Exception as e:
                        print(f"[warn] could not download error file for shard#{j['i']:02d}: {e}")

            if b.status in TERMINAL and not j["downloaded"]:
                try:
                    ofid = getattr(b, "output_file_id", None) or (b.get("output_file_id") if isinstance(b, dict) else None)
                except Exception:
                    ofid = None
                if ofid:
                    out_name = f"{args.out_prefix}_shard{j['i']:02d}_{b.id}.jsonl"
                    try:
                        client.files.content(ofid).write_to_file(out_name)
                        print(f"[done] wrote {out_name}")
                    except Exception as e:
                        print(f"[warn] shard#{j['i']:02d} cannot download output: {e}")

                j["downloaded"] = True
                active.remove(j); finished.append(j)
                print(f"[terminal] shard#{j['i']:02d} -> {b.status}")

                if not args.submit_all:
                    start_next_if_slots()

        if time.time() >= next_progress_ts:
            elapsed = int(time.time() - start_ts)
            overall_total = overall_completed = overall_failed = overall_running = 0
            for j in finished + active:
                c = j["counts"]
                overall_total     += c.get("total", 0)
                overall_completed += c.get("completed", 0)
                overall_failed    += c.get("failed", 0)
                overall_running   += c.get("running", 0)
            print(f"[progress] completed={overall_completed}/{overall_total} "
                  f"failed={overall_failed} running={overall_running} "
                  f"elapsed={elapsed}s (active={len(active)} finished={len(finished)})")
            next_progress_ts = time.time() + args.progress_secs

    ok = sum(1 for j in finished if j["status"] == "completed")
    print(f"[summary] batches={len(finished)} completed={ok} not_completed={len(finished)-ok}")
    for j in finished:
        c = j["counts"]
        print(f"  - shard#{j['i']:02d} {j['status']} {c.get('completed',0)}/{c.get('total',0)} failed={c.get('failed',0)}")

if __name__ == "__main__":
    main()
