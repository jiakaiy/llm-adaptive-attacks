#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Submit one or more Batch jobs to Alibaba Cloud Model Studio (Bailian) OpenAI-compatible API,
with parallel sharding, robust polling/backoff, and throttled progress printing.

Usage:
  export DASHSCOPE_API_KEY=sk-...
  python submit_batch_deepseek.py \
    --input_jsonl batch_input_deepseek.jsonl \
    --runtime_json batch_runtime.json \
    --completion_window 24h \
    --shards 5 \
    --poll_secs 60 \
    --progress_secs 60
"""

import os, time, json, argparse, math, random
from pathlib import Path
from typing import Tuple, Dict, Any, List
from openai import OpenAI
from openai import RateLimitError

# ---------------------------- helpers: endpoint/base_url/key ----------------------------

def pick_base_url_by_region(region: str) -> str:
    region = (region or "cn").lower()
    return "https://dashscope-intl.aliyuncs.com/compatible-mode/v1" if region == "intl" \
           else "https://dashscope.aliyuncs.com/compatible-mode/v1"

def resolve_base_and_endpoint(runtime_json: str, base_url_arg: str, region_arg: str) -> Tuple[str, str]:
    # Priority: runtime_json > base_url_arg > region_arg > default CN
    if runtime_json and Path(runtime_json).exists():
        try:
            obj = json.loads(Path(runtime_json).read_text(encoding="utf-8"))
        except Exception:
            obj = {}
        base_url = (obj.get("base_url") or "").rstrip("/")
        endpoint = obj.get("endpoint") or "/v1/chat/completions"
        if base_url:
            return base_url, endpoint
    if base_url_arg:
        return base_url_arg.rstrip("/"), "/v1/chat/completions"
    if region_arg:
        return pick_base_url_by_region(region_arg), "/v1/chat/completions"
    return "https://dashscope.aliyuncs.com/compatible-mode/v1", "/v1/chat/completions"

def load_api_key(args) -> str:
    if args.api_key:
        return args.api_key.strip()
    if args.api_key_file:
        p = Path(args.api_key_file)
        if p.exists():
            return p.read_text(encoding="utf-8").strip().splitlines()[0]
    return os.getenv(args.api_key_env, "").strip()

def sniff_first_url_in_jsonl(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict) and "url" in obj:
                    return obj["url"]
                break
    except Exception:
        pass
    return ""

# ---------------------------- helpers: splitting ----------------------------

def count_lines(path: Path) -> int:
    n = 0
    with path.open("r", encoding="utf-8") as f:
        for _ in f:
            n += 1
    return n

def split_jsonl_contiguous(input_path: Path, shards: int, out_dir: Path, prefix: str):
    """
    Split input JSONL into 'shards' contiguous pieces.
    Returns: list[(shard_path, n_lines_in_shard)], total_count
    """
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
                shard_path, _ = shard_files[i_shard]
                shard_files[i_shard] = (shard_path, written)
                i_shard += 1
                written = 0
            fouts[i_shard].write(line)
            written += 1

    for j in range(i_shard, shards):
        try:
            fouts[j].close()
        except Exception:
            pass
    shard_path, _ = shard_files[i_shard]
    shard_files[i_shard] = (shard_path, written)
    for k in range(i_shard + 1, shards):
        sp, _ = shard_files[k]
        shard_files[k] = (sp, 0)

    return shard_files, total

# ---------------------------- helpers: progress ----------------------------

def extract_counts(batch) -> Dict[str, int]:
    """
    Normalizes counts across slight schema differences.
    """
    for key in ("request_counts", "counts"):
        rc = getattr(batch, key, None)
        if rc:
            if isinstance(rc, dict):
                return {
                    "total": int(rc.get("total", 0)),
                    "completed": int(rc.get("completed", 0)),
                    "failed": int(rc.get("failed", 0)),
                    "running": int(rc.get("running", rc.get("processing", 0))),
                }
            try:
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

def safe_attr(obj: Any, *names: str):
    for n in names:
        v = getattr(obj, n, None)
        if v:
            return v
    return None

# ---------------------------- main ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_jsonl", required=True, help="Batch input JSONL created by build_batch_input.py")
    ap.add_argument("--runtime_json", default="batch_runtime.json",
                    help="Runtime file from build_batch_input.py (base_url/endpoint)")
    ap.add_argument("--base_url", default=None, help="Override base URL (rarely needed)")
    ap.add_argument("--region", choices=["intl","cn"], default=None,
                    help="If set, picks a default base_url for intl/cn")
    ap.add_argument("--completion_window", default="24h", help="Batch time window, e.g., 4h, 8h, 24h")

    # Key loading options
    ap.add_argument("--api_key", default=None, help="Paste your key here (not recommended)")
    ap.add_argument("--api_key_file", default=None, help="Path to a file containing the key")
    ap.add_argument("--api_key_env", default="DASHSCOPE_API_KEY", help="Environment variable name for the key")

    # Parallel/sharding
    ap.add_argument("--shards", type=int, default=5, help="Number of parallel batches to run")
    ap.add_argument("--shard_dir", default="shards", help="Directory to write shard files")
    ap.add_argument("--shard_prefix", default="shard_", help="Filename prefix for shard files")

    # Cadence controls
    ap.add_argument("--poll_secs", type=int, default=60,
                    help="How often to poll each shard for status (seconds)")
    ap.add_argument("--progress_secs", type=int, default=60,
                    help="How often to print the combined progress line (seconds)")

    ap.add_argument("--out_prefix", default="result", help="Prefix for per-shard result files")
    ap.add_argument("--err_prefix", default="error", help="Prefix for per-shard error files")
    args = ap.parse_args()

    # ----- key, endpoint, client
    api_key = load_api_key(args)
    if not api_key:
        raise SystemExit("Missing API key. Provide --api_key / --api_key_file or export DASHSCOPE_API_KEY=sk-...")

    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise SystemExit(f"[err] input_jsonl not found: {input_path}")

    base_url, endpoint = resolve_base_and_endpoint(args.runtime_json, args.base_url, args.region)
    print(f"[info] base_url: {base_url}")
    print(f"[info] endpoint: {endpoint}")

    # Warn if JSONL url mismatches endpoint
    first_url = sniff_first_url_in_jsonl(input_path)
    if first_url and first_url != endpoint:
        print(f"[warn] The first JSONL 'url' is {first_url!r} but batches.create endpoint is {endpoint!r}.")
        print("       They MUST match.")

    client = OpenAI(api_key=api_key, base_url=base_url)

    # ----- split into shards
    shard_dir = Path(args.shard_dir)
    shard_files, total_prompts = split_jsonl_contiguous(input_path, args.shards, shard_dir, args.shard_prefix)
    nonempty = [(p, n) for (p, n) in shard_files if n > 0]
    if not nonempty:
        raise SystemExit("[err] nothing to submit (all shards empty?)")

    print(f"[split] total={total_prompts} → shards={len(nonempty)} (dir={shard_dir})")
    for i, (p, n) in enumerate(nonempty):
        print(f"        - {p.name}: {n} lines")

    # ----- create batches (one per shard)
    jobs = []  # dict with shard info
    for i, (spath, nlines) in enumerate(nonempty):
        fobj = client.files.create(file=spath, purpose="batch")
        batch = client.batches.create(
            input_file_id=fobj.id,
            endpoint=endpoint,
            completion_window=args.completion_window
        )
        # Stagger initial poll times a bit so we don't hit the route all at once
        next_poll_ts = time.time() + (i % 3) * 10
        jobs.append({
            "i": i,
            "path": spath,
            "n": nlines,
            "file_id": fobj.id,
            "batch_id": batch.id,
            "status": batch.status,
            "downloaded": False,
            "counts": {"total": 0, "completed": 0, "failed": 0, "running": 0},
            "next_poll_ts": next_poll_ts,
        })
        print(f"[create] shard#{i:02d} file_id={fobj.id} batch_id={batch.id} status={batch.status}")

    # ----- poll all until terminal
    terminal = {"completed", "failed", "expired", "cancelled"}
    done = set()
    start_ts = time.time()
    next_progress_ts = start_ts  # print first progress immediately

    while len(done) < len(jobs):
        now = time.time()

        # Compute the next due time among all shards and the next progress print
        due_times = [j["next_poll_ts"] for j in jobs if j["batch_id"] not in done]
        if not due_times:
            # nothing pending to poll; ensure we still print progress on cadence
            due_times = [next_progress_ts]
        next_due = min(min(due_times), next_progress_ts)

        if now < next_due:
            # Sleep until either next poll or next progress print (whichever is sooner),
            # but never more than poll_secs to keep the loop responsive to new terminals.
            sleep_for = max(0.0, min(args.poll_secs, next_due - now))
            time.sleep(sleep_for)

        overall_total = overall_completed = overall_failed = overall_running = 0

        # poll shards that are due
        for j in jobs:
            if j["batch_id"] in done:
                overall_total     += j["counts"].get("total", j["n"])
                overall_completed += j["counts"].get("completed", 0)
                overall_failed    += j["counts"].get("failed", 0)
                continue

            if time.time() < j["next_poll_ts"]:
                continue  # not yet time to poll this shard

            # poll with exponential backoff on 429
            retry = 0
            while True:
                try:
                    b = client.batches.retrieve(batch_id=j["batch_id"])
                    break
                except RateLimitError:
                    delay = min(60, (2 ** retry) + random.uniform(0, 1))
                    print(f"[rate-limit] shard#{j['i']:02d} 429; backoff {delay:.1f}s")
                    time.sleep(delay)
                    retry += 1
                    continue
                except Exception as e:
                    delay = min(30, 5 + random.uniform(0, 2))
                    print(f"[warn] shard#{j['i']:02d} poll error: {e}; retry in {delay:.1f}s")
                    time.sleep(delay)
                    continue

            j["status"] = b.status
            rc = extract_counts(b)
            if rc.get("total", 0) == 0:
                rc["total"] = j["n"]
            j["counts"] = rc

            # schedule next poll for this shard
            j["next_poll_ts"] = time.time() + args.poll_secs

            overall_total     += rc["total"]
            overall_completed += rc["completed"]
            overall_failed    += rc["failed"]
            overall_running   += rc["running"]

            # per-shard status
            print(f"[poll] shard#{j['i']:02d} {j['status']} "
                  f"completed={rc['completed']}/{rc['total']} failed={rc['failed']} running={rc['running']}")

            # if failed, print reason and try to download error file
            if b.status == "failed":
                err_msg = safe_attr(b, "error", "last_error")
                err_file = safe_attr(b, "error_file_id")
                print(f"[fail-reason] shard#{j['i']:02d} batch_id={b.id} error={err_msg} err_file={err_file}")
                if err_file:
                    try:
                        err_name = f"{args.err_prefix}_shard{j['i']:02d}_{b.id}.jsonl"
                        client.files.content(err_file).write_to_file(err_name)
                        print(f"[done] wrote {err_name}")
                    except Exception as e:
                        print(f"[warn] could not download error file for shard#{j['i']:02d}: {e}")

            # if terminal, download outputs/errors
            if b.status in terminal and not j["downloaded"]:
                out_name = f"{args.out_prefix}_shard{j['i']:02d}_{b.id}.jsonl"
                try:
                    out_file = safe_attr(b, "output_file_id")
                    if out_file:
                        client.files.content(out_file).write_to_file(out_name)
                        print(f"[done] wrote {out_name}")
                except Exception as e:
                    print(f"[warn] shard#{j['i']:02d} could not download results: {e}")

                try:
                    err_file = safe_attr(b, "error_file_id")
                    if err_file:
                        err_name = f"{args.err_prefix}_shard{j['i']:02d}_{b.id}.jsonl"
                        client.files.content(err_file).write_to_file(err_name)
                        print(f"[done] wrote {err_name}")
                except Exception as e:
                    print(f"[warn] shard#{j['i']:02d} could not download errors: {e}")

                j["downloaded"] = True
                done.add(j["batch_id"])
                print(f"[terminal] shard#{j['i']:02d} → {b.status}")

        # combined progress (throttled by progress_secs)
        now = time.time()
        if now >= next_progress_ts:
            elapsed = int(now - start_ts)
            # recompute overall from current job states (covers shards we didn't poll this tick)
            overall_total = overall_completed = overall_failed = overall_running = 0
            for j in jobs:
                c = j["counts"]
                overall_total     += c.get("total", j["n"])
                overall_completed += c.get("completed", 0)
                overall_failed    += c.get("failed", 0)
                overall_running   += c.get("running", 0)
            print(f"[progress] overall completed={overall_completed}/{overall_total} "
                  f"failed={overall_failed} running={overall_running} elapsed={elapsed}s")
            next_progress_ts = now + args.progress_secs

    # ----- final summary
    final_ok = sum(1 for j in jobs if j["status"] == "completed")
    final_fail = len(jobs) - final_ok
    print(f"[summary] batches={len(jobs)} completed={final_ok} not_completed={final_fail}")
    for j in jobs:
        print(f"  - shard#{j['i']:02d} {j['status']} "
              f"completed={j['counts'].get('completed',0)}/{j['counts'].get('total',j['n'])} "
              f"failed={j['counts'].get('failed',0)}")

if __name__ == "__main__":
    main()
