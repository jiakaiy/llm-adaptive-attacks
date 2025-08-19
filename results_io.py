"""
results_io.py – append and load one-line-per-run JSON logs
Python 3.9-compatible version
"""
import json
from pathlib import Path
from typing import Dict, Generator, Union

DEFAULT_PATH = Path("run_outputs.jsonl")


def append(run_record: Dict, out_path: Union[Path, str] = DEFAULT_PATH) -> None:
    out_path = Path(out_path)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(run_record, ensure_ascii=False) + "\n")


def load(out_path: Union[Path, str] = DEFAULT_PATH) -> Generator[Dict, None, None]:
    out_path = Path(out_path)
    if not out_path.exists():
        raise FileNotFoundError(f"No log file found at {out_path.resolve()}")
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)
