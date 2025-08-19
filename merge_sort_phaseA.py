 #!/usr/bin/env python3
"""
Append Phase-A shard rows (>=560) into run_outputs_severe.jsonl,
keeping existing rows (e.g., 0..559) unchanged.

- Dedupe by prompt_idx (shard rows overwrite any existing same-index row)
- Maintains sorted order
"""

import json, sys
from pathlib import Path

MASTER = Path("run_outputs_severe.jsonl")

def load_rows(fn):
    out = []
    with open(fn, "r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            out.append(row)
    return out

def main():
    if len(sys.argv) < 2:
        print("Usage: python merge_sort_phaseA.py <shard files...>")
        sys.exit(1)

    if not MASTER.exists():
        sys.exit(f"❌ {MASTER} not found — expected to already contain prompts 0..559.")

    # 1) load master and separate into "keep" (<560) and "replaceable" (>=560)
    master_rows = load_rows(MASTER)
    keep_rows = [r for r in master_rows if isinstance(r.get("prompt_idx"), int) and r["prompt_idx"] < 560]

    # 2) load shard rows
    shard_rows = []
    for fn in sys.argv[1:]:
        shard_rows.extend(load_rows(fn))

    # 3) dedupe shard rows by prompt_idx
    seen = set()
    deduped_shards = []
    for r in shard_rows:
        k = r.get("prompt_idx")
        if isinstance(k, int) and k not in seen:
            seen.add(k)
            deduped_shards.append(r)

    # 4) combine keep_rows + deduped_shards and sort
    combined = keep_rows + deduped_shards
    combined.sort(key=lambda r: r.get("prompt_idx", 1 << 60))

    # 5) write back to MASTER
    with MASTER.open("w", encoding="utf-8") as f:
        for r in combined:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"📄 Appended {len(deduped_shards)} shard rows to {MASTER.resolve()}")
    print(f"✅ Total rows now: {len(combined)}")

if __name__ == "__main__":
    main()
