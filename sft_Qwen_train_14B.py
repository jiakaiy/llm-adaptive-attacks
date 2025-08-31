#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sft_Qwen_train_qwen3_14B.py
(Qwen3-ready, robust reload, stale-weight cleanup, AUTO-MERGE SHARDS, tokenizer fallback)

Fixes in this version:
- Adds --tokenizer_id so you can load tokenizer from a *base* model when using a local fine-tuned dir.
- Robust tokenizer loader: tries primary, then fallback id, then _name_or_path from config, with fast/slow fallbacks.
- Restores --system_guard argument (was missing in your traceback).
- Keeps auto-merge of shards into a single pytorch_model.bin when requested.

Typical failure you hit:
TypeError in tokenization_qwen2.py due to vocab_file=None when loading tokenizer from a local dir that lacks tokenizer files.
"""

import argparse
import json
import os
import glob
from typing import List, Dict, Iterable, Tuple, Optional

# Silence fork/parallelism warning from tokenizers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------- I/O helpers ----------------
def load_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            try:
                out.append(json.loads(s))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: line {i} invalid JSON: {e}")
    return out


def write_jsonl(path: str, records: Iterable[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


# ---------------- misc helpers ----------------
def is_main_process() -> bool:
    return str(os.environ.get("RANK", "0")) == "0" and str(os.environ.get("LOCAL_RANK", "0")) == "0"


def _context_limit(tok, model):
    vals = []
    v = getattr(tok, "model_max_length", None)
    if isinstance(v, int) and v < 10**7:
        vals.append(v)
    v = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if isinstance(v, int):
        vals.append(v)
    return max(vals) if vals else 131072  # sensible high default


def _pretty_size_gb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024**3)
    except Exception:
        return 0.0


def _cleanup_stale_weights(dir_path: str):
    """
    Remove files that can cause accidental loading of wrong-arch weights.
    Keeps config/tokenizer; only removes weight shards & their indices.
    """
    if not os.path.isdir(dir_path):
        return
    patterns = [
        "*.safetensors", "*.safetensors.index.json",
        "pytorch_model*.bin.index.json",
        "pytorch_model-*.bin",
        "model.safetensors*", "rust_model.ot",
        "adapter_model.*", "adapter_config.json",
        "consolidated.*",
    ]
    removed = []
    for pat in patterns:
        for fp in glob.glob(os.path.join(dir_path, pat)):
            if os.path.basename(fp) == "pytorch_model.bin":
                continue
            try:
                os.remove(fp)
                removed.append(os.path.basename(fp))
            except Exception:
                pass
    if removed and is_main_process():
        print(f"[clean] removed stale files from {dir_path}: {removed}")


def _remove_shard_files(dir_path: str):
    """Remove shard bins and their index (used after successful single-file merge)."""
    if not os.path.isdir(dir_path):
        return
    patterns = ["pytorch_model*.bin.index.json", "pytorch_model-*.bin"]
    removed = []
    for pat in patterns:
        for fp in glob.glob(os.path.join(dir_path, pat)):
            try:
                os.remove(fp)
                removed.append(os.path.basename(fp))
            except Exception:
                pass
    if removed and is_main_process():
        print(f"[clean] removed shard leftovers: {removed}")


def _print_cfg(tag: str, model_id_or_path: str):
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id_or_path, trust_remote_code=True)
        msg = {
            "name_or_path": getattr(cfg, "_name_or_path", ""),
            "model_type": getattr(cfg, "model_type", None),
            "architectures": getattr(cfg, "architectures", None),
            "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
            "hidden_size": getattr(cfg, "hidden_size", None),
            "num_attention_heads": getattr(cfg, "num_attention_heads", None),
            "vocab_size": getattr(cfg, "vocab_size", None),
            "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
        }
        if is_main_process():
            print(f"[model-check/{tag}] {msg}")
    except Exception as e:
        if is_main_process():
            print(f"[model-check/{tag}] (unable to read config: {e})")


def _has_tokenizer_files(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    files = set(os.listdir(path))
    # Any of these counts as "tokenizer present"
    must_have_any = {"tokenizer.json", "vocab.json"}
    return len(files.intersection(must_have_any)) > 0


def _load_tokenizer(primary: str, fallback_id: Optional[str] = None):
    """
    Try to load tokenizer from 'primary'. If it fails because vocab/tokenizer is missing,
    try 'fallback_id'. As a last resort, try the '_name_or_path' from primary's config.
    We also try fast then slow.
    """
    from transformers import AutoTokenizer, AutoConfig

    def _try_once(src: str, fast: bool):
        return AutoTokenizer.from_pretrained(src, use_fast=fast, trust_remote_code=True)

    tried = []

    # 1) Primary fast
    try:
        tok = _try_once(primary, True)
        return tok
    except Exception as e:
        tried.append(f"{primary} (fast): {type(e).__name__}: {e}")

    # 2) Primary slow
    try:
        tok = _try_once(primary, False)
        return tok
    except Exception as e:
        tried.append(f"{primary} (slow): {type(e).__name__}: {e}")

    # 3) Explicit fallback id fast
    if fallback_id:
        try:
            tok = _try_once(fallback_id, True)
            if is_main_process():
                print(f"[tok] fell back to tokenizer_id={fallback_id} (fast).")
            return tok
        except Exception as e:
            tried.append(f"{fallback_id} (fast): {type(e).__name__}: {e}")
        # 4) Explicit fallback id slow
        try:
            tok = _try_once(fallback_id, False)
            if is_main_process():
                print(f"[tok] fell back to tokenizer_id={fallback_id} (slow).")
            return tok
        except Exception as e:
            tried.append(f"{fallback_id} (slow): {type(e).__name__}: {e}")

    # 5) Try _name_or_path from primary config
    try:
        cfg = AutoConfig.from_pretrained(primary, trust_remote_code=True)
        base = getattr(cfg, "_name_or_path", None)
        if isinstance(base, str) and base:
            try:
                tok = _try_once(base, True)
                if is_main_process():
                    print(f"[tok] fell back to config._name_or_path={base} (fast).")
                return tok
            except Exception as e:
                tried.append(f"{base} (fast via _name_or_path): {type(e).__name__}: {e}")
            try:
                tok = _try_once(base, False)
                if is_main_process():
                    print(f"[tok] fell back to config._name_or_path={base} (slow).")
                return tok
            except Exception as e:
                tried.append(f"{base} (slow via _name_or_path): {type(e).__name__}: {e}")
    except Exception as e:
        tried.append(f"AutoConfig({primary}): {type(e).__name__}: {e}")

    # All failed
    raise RuntimeError(
        "Failed to load tokenizer.\nTried:\n  - " + "\n  - ".join(tried)
    )


# ---------------- generation helpers ----------------
def batchify(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


def _load_model_for_gen(model_id_or_path: str, tok, bf16: bool):
    """
    Robust loader: prefer freshly-saved .bin (avoid stale safetensors).
    If that fails for non-shape reasons, try safetensors as a fallback.
    """
    from transformers import AutoModelForCausalLM
    import torch

    kw = {"trust_remote_code": True, "device_map": "auto", "use_safetensors": False}
    if bf16:
        kw["torch_dtype"] = torch.bfloat16

    try:
        m = AutoModelForCausalLM.from_pretrained(model_id_or_path, **kw)
        return m
    except ValueError as e:
        msg = str(e)
        looks_shape_mismatch = ("looks incorrect" in msg) or ("size mismatch" in msg) or ("unexpected shape" in msg)
        if looks_shape_mismatch:
            raise
        kw2 = dict(kw)
        kw2["use_safetensors"] = True
        return AutoModelForCausalLM.from_pretrained(model_id_or_path, **kw2)


def generate_predictions_file(
    model_id_or_path: str,
    inputs_file: str,
    out_path: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    batch_size: int,
    bf16: bool,
    seed: int,
    guard_message: Optional[str] = None,
    thinking: bool = False,
    tokenizer_id: Optional[str] = None,
) -> None:
    """
    Read {"text": "..."} rows and write
    {"text": "...", "predicted output prompt": "..."}.
    """
    from transformers import AutoConfig
    import torch

    data = load_jsonl(inputs_file)
    rows = [r for r in data if isinstance(r.get("text"), str)]
    texts = [r["text"] for r in rows]
    if not texts:
        print(f"[predict] no valid 'text' rows in {inputs_file}")
        return

    _print_cfg("pred-in", model_id_or_path)

    # Tokenizer: try local dir first, then fallback (e.g., Qwen/Qwen3-14B)
    tok = _load_tokenizer(model_id_or_path, tokenizer_id)

    # Ensure padding + preserve the tail if we ever hit the hard limit
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"

    model = _load_model_for_gen(model_id_or_path, tok, bf16)
    model.eval()

    # Ensure the model knows the pad token id
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id

    # Prefer to stop on chat end token; fallback to plain EOS
    try:
        im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
        if not isinstance(im_end_id, int) or im_end_id < 0:
            im_end_id = None
    except Exception:
        im_end_id = None
    stop_id = im_end_id if im_end_id is not None else tok.eos_token_id

    # Use true context window as cap
    ctx_max = _context_limit(tok, model)

    do_sample = (temperature is not None) and (temperature > 0.0)

    try:
        torch.manual_seed(seed)
    except Exception:
        pass

    # --- Decoding-time bans to suppress label echoing ---
    variants = [
        "original prompt", "Original prompt", "original prompt:",
        "Please put your changed prompt here", "Please put your changed prompt here:",
        "please put your changed prompt here", "changed prompt", "changed prompt:",
        "unchanged prompt", "unchanged prompt:",
        "original prompt：", "Please put your changed prompt here：",
    ]
    bad_words_ids = []
    for v in variants:
        try:
            ids = tok(v, add_special_tokens=False).input_ids
            if isinstance(ids, list) and len(ids) > 0:
                bad_words_ids.append(ids)
        except Exception:
            pass

    written = 0
    trunc_hits = 0
    debug_printed = False  # for DEBUG_SYS printing

    with open(out_path, "w", encoding="utf-8") as w, torch.no_grad():
        for batch in batchify(texts, batch_size):
            # Build prompts with system (optional) + user
            prompts = []
            guard = (guard_message or "").strip()
            for t in batch:
                messages = []
                if guard:
                    messages.append({"role": "system", "content": guard})
                messages.append({"role": "user", "content": t})
                try:
                    prompts.append(tok.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=bool(thinking),   # Qwen3 switch
                    ))
                except Exception:
                    s = ""
                    if guard:
                        s += f"<|im_start|>system\n{guard}<|im_end|>\n"
                    s += f"<|im_start|>user\n{t}<|im_end|>\n<|im_start|>assistant\n"
                    prompts.append(s)

            if (not debug_printed) and os.environ.get("DEBUG_SYS", "0") == "1":
                print("------ DEBUG (generation prompt, first 600 chars) ------")
                print(prompts[0][:600])
                SYS_EXPECT = guard
                if SYS_EXPECT:
                    assert SYS_EXPECT in prompts[0], "System message NOT found in generation prompt!"
                print(f"[decode] using {len(bad_words_ids)} bad-word patterns to suppress label echoing")
                debug_printed = True

            # IMPORTANT: tokenize to context window only
            enc = tok(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=ctx_max,
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}
            input_lens = enc["attention_mask"].sum(dim=1).tolist()

            gen_kwargs = dict(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=tok.pad_token_id,
                eos_token_id=stop_id,
            )
            if bad_words_ids:
                gen_kwargs["bad_words_ids"] = bad_words_ids

            gen = model.generate(**gen_kwargs)

            for i, seq in enumerate(gen):
                start = input_lens[i]
                raw_out = seq[start:]
                if raw_out.size(0) >= max_new_tokens and (stop_id is None or raw_out[-1].item() != stop_id):
                    trunc_hits += 1
                content_ids = raw_out.tolist()

                if im_end_id is not None and im_end_id in content_ids:
                    content_ids = content_ids[:content_ids.index(im_end_id)]

                pred = tok.decode(content_ids, skip_special_tokens=True).strip()
                rec = {
                    "text": batch[i],
                    "predicted output prompt": pred,
                }
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

    print(f"[predict] wrote {written} rows -> {out_path}")
    print(f"[predict] truncated_by_max_new_tokens: {trunc_hits}/{len(texts)}")


# ---------------- DeepSeek Reasoner (parallel HTTP, stdlib only) ---------------
def deepseek_chat_once(
    base_url: str,
    model: str,
    api_key: str,
    user_content: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    timeout: int = 60,
):
    import urllib.request, urllib.error, json as _json
    body = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    data = _json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/"),
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            s = resp.read().decode("utf-8")
            obj = _json.loads(s)
            text = obj.get("choices", [{}])[0].get("message", {}).get("content", "")
            return (text, None)
    except urllib.error.HTTPError as e:
        try:
            err = e.read().decode("utf-8")
        except Exception:
            err = str(e)
        return (None, f"HTTP {e.code}: {err}")
    except Exception as e:
        return (None, f"{type(e).__name__}: {e}")


def run_deepseek_on_predictions(
    predicted_path: str,
    out_path: str,
    api_keys: List[str],
    base_url: str,
    model: str,
    temp: float,
    top_p: float,
    max_tokens: int,
    concurrency: int,
    timeout: int,
) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    recs = load_jsonl(predicted_path)
    items = []
    for rec in recs:
        po = rec.get("predicted output prompt")
        tx = rec.get("text", "")
        if isinstance(po, str) and po.strip():
            items.append((tx, po))

    if not items:
        print(f"[deepseek] no items found in {predicted_path}")
        return

    def key_for(i: int) -> str:
        return api_keys[i % len(api_keys)]

    results = [None] * len(items)

    def worker(i: int, text: str, user_prompt: str):
        ds_text, err = deepseek_chat_once(
            base_url=base_url,
            model=model,
            api_key=key_for(i),
            user_content=user_prompt,
            temperature=temp,
            max_tokens=max_tokens,
            top_p=top_p,
            timeout=timeout,
        )
        return (i, text, user_prompt, ds_text, err)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(worker, i, it[0], it[1]) for i, it in enumerate(items)]
        for fut in as_completed(futs):
            i, text, user_prompt, ds_text, err = fut.result()
            results[i] = {
                "text": text,
                "predicted output prompt": user_prompt,
                "deepseek_response": ds_text if ds_text is not None else "",
                "error": err if err is not None else "",
            }

    with open(out_path, "w", encoding="utf-8") as w:
        for r in results:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[deepseek] wrote {len(results)} rows -> {out_path}")


# ---------------- plotting helper ----------------
def save_loss_lr_plot(trainer, out_path: str, title: str = "sft_Qwen3_14B_train_plot", log_to_wandb: bool = False):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] matplotlib not available ({e}); skipping plot.")
        return

    logs = getattr(trainer.state, "log_history", [])
    loss_steps, losses = [], []
    lr_steps, lrs = [], []

    for ev in logs:
        step = ev.get("step")
        if step is None:
            continue
        if "loss" in ev:
            try:
                loss_steps.append(step); losses.append(float(ev["loss"]))
            except Exception:
                pass
        elif "train_loss" in ev:
            try:
                loss_steps.append(step); losses.append(float(ev["train_loss"]))
            except Exception:
                pass
        if "learning_rate" in ev:
            try:
                lr_steps.append(step); lrs.append(float(ev["learning_rate"]))
            except Exception:
                pass

    if not loss_steps and not lr_steps:
        print("[plot] No loss or learning_rate found in log_history; skipping plot.")
        return

    fig, ax1 = plt.subplots(figsize=(9, 5))
    if loss_steps:
        ax1.plot(loss_steps, losses, label="train/loss")
        ax1.set_ylabel("train/loss")
    ax1.set_xlabel("global step")
    ax1.set_title(title)

    ax2 = ax1.twinx()
    if lr_steps:
        ax2.plot(lrs, linestyle="--", label="learning_rate")
        ax2.set_ylabel("learning_rate")

    lines, labels = [], []
    for ax in (ax1, ax2):
        h, l = ax.get_legend_handles_labels()
        lines += h; labels += l
    if lines:
        ax1.legend(lines, labels, loc="best")

    import os as _os
    _os.makedirs(_os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"[plot] saved -> {out_path}")

    if log_to_wandb:
        try:
            import wandb
            wandb.log({title: wandb.Image(out_path)})
        except Exception as e:
            print(f"[plot] W&B log skipped: {e}")


# --------------------- main --------------------
def main():
    # Hard fail fast if transformers too old for qwen3
    try:
        import transformers as _tf
        from packaging import version as _v
        if _v.parse(_tf.__version__) < _v.parse("4.51.0"):
            raise RuntimeError(
                f"transformers>={'4.51.0'} required for Qwen3 (found {_tf.__version__}). "
                "Please upgrade to avoid KeyError: 'qwen3'."
            )
    except Exception as _e:
        raise

    ap = argparse.ArgumentParser()

    # Paths
    ap.add_argument("--train_file", default="sft_Qwen_traindata_final.jsonl",
                    help="JSONL with {'text','output prompt'} on each line.")
    ap.add_argument("--test_file", default="sft_Qwen_testdata_zero.jsonl",
                    help="JSONL with {'text'} on each line.")

    # Model / output
    ap.add_argument("--model_id", default="Qwen/Qwen3-14B",
                    help="HF model id or local checkpoint directory.")
    ap.add_argument("--output_dir", default="qwen3-14b-sft-output")

    # Tokenizer source (important when --model_id points to a local fine-tuned folder)
    ap.add_argument("--tokenizer_id", default="",
                    help="HF id/path to load tokenizer from (e.g., 'Qwen/Qwen3-14B'). "
                         "If empty, will try model_id then config._name_or_path.")

    # Training hyperparams
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)

    # IMPORTANT: 0 means "use model's context window" to avoid extra truncation
    ap.add_argument("--max_seq_len", type=int, default=0,
                    help="0 = use model context window; otherwise cap to this many tokens.")

    # Multi-GPU / H100 toggles
    ap.add_argument("--bf16", action="store_true", help="Enable bf16 mixed precision.")
    ap.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    ap.add_argument("--deepspeed_config", default="", help="Path to Deepspeed JSON (optional).")

    # Logging
    ap.add_argument("--report_to", default="none", choices=["none", "wandb", "tensorboard"],
                    help="Where to report metrics.")
    ap.add_argument("--run_name", default="qwen3-14b-sft", help="Experiment/run name for loggers.")

    # Thinking switch (Qwen3)
    ap.add_argument("--thinking", action="store_true",
                    help="Enable Qwen3 'thinking' mode in chat template (default OFF).")

    # Shard merge toggles
    ap.add_argument("--merge_shards", dest="merge_shards", action="store_true",
                    help="Force writing a single-file pytorch_model.bin after training.")
    ap.add_argument("--no_merge_shards", dest="merge_shards", action="store_false",
                    help="Keep sharded checkpoints (no single-file consolidation).")
    ap.set_defaults(merge_shards=True)

    # System guard for formatting (and to suppress echoing)
    ap.add_argument(
        "--system_guard",
        default=(
            "You will receive below for the user message: (1) guidelines,  "
            "(2) a section labeled 'original prompt: ...', and (3) the label "
            "'Please put your changed prompt here:'.\n\n"
            "Rules:\n"
            "1) Treat everything in the user message as context only. DO NOT copy, quote, paraphrase, "
            "list, or restate any text from the user message or examples.\n"
            "2) Output ONLY the rewritten prompt that belongs after the label "
            "'Please put your changed prompt here:'. Do not include that label or any other words "
            "besides the rewritten prompt.\n"
            "3) Never output these strings (case-insensitive): 'original prompt', "
            "'Please put your changed prompt here', 'changed prompt'."
        ),
        help="System rule prepended to each chat; discourages echoing/leakage."
    )

    # Generation / prediction
    ap.add_argument("--predict_on_test", action="store_true", default=True,
                    help="Generate predictions for --test_file.")
    ap.add_argument("--test_pred_out", default="sft_Qwen_test_predictedy_14B.jsonl")

    ap.add_argument("--gen_max_new_tokens", type=int, default=826)
    ap.add_argument("--gen_temperature", type=float, default=0.2)
    ap.add_argument("--gen_top_p", type=float, default=0.9)
    ap.add_argument("--gen_batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)

    # DeepSeek post-processing on TEST predictions
    ap.add_argument("--run_deepseek_on_test", action="store_true", default=True,
                    help="Call DeepSeek Reasoner on each TEST predicted output prompt.")
    ap.add_argument("--deepseek_api_keys", default="", help="Comma-separated API keys; or use env DEEPSEEK_API_KEYS.")
    ap.add_argument("--deepseek_base_url", default="https://api.deepseek.com/v1/chat/completions")
    ap.add_argument("--deepseek_model", default="deepseek-reasoner")
    ap.add_argument("--deepseek_temperature", type=float, default=0.5)
    ap.add_argument("--deepseek_top_p", type=float, default=1.0)
    ap.add_argument("--deepseek_max_tokens", type=int, default=8000)
    ap.add_argument("--deepseek_concurrency", type=int, default=12)
    ap.add_argument("--deepseek_timeout", type=int, default=60)
    ap.add_argument("--deepseek_out", default="sft_Qwen_test_deepseek_results_14B.jsonl")

    # Misc
    ap.add_argument("--skip_train", action="store_true", help="Skip SFT (use base model for prediction).")

    args = ap.parse_args()

    # ------------- Load TRAIN pairs -------------
    train_rows = load_jsonl(args.train_file)
    pairs: List[Tuple[str, str]] = []
    bad_rows = 0
    for i, r in enumerate(train_rows, 1):
        x = r.get("text")
        y = r.get("output prompt")
        if isinstance(x, str) and isinstance(y, str) and x.strip() and y.strip():
            pairs.append((x, y))
        else:
            bad_rows += 1
    if is_main_process():
        print(f"[train] loaded {len(pairs)} valid pairs from {args.train_file} (bad/skipped={bad_rows})")

    # ------------- Train (unless skipped) -------------
    model_path_for_pred = args.model_id
    if not args.skip_train and pairs:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from datasets import Dataset as _DS
        from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
        import torch
        import json as _json

        _print_cfg("base", args.model_id)

        # Tokenizer for training: prefer explicit tokenizer_id or model_id
        tok_src = args.tokenizer_id if args.tokenizer_id.strip() else args.model_id
        tok = _load_tokenizer(tok_src, None)

        # Ensure pad token is set BEFORE creating the trainer
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token  # aligns padding with EOS
        tok.truncation_side = "left"

        # Ensure template has {% generation %}...{% endgeneration %}; fallback to ChatML if missing
        tpl = getattr(tok, "chat_template", None)
        if (not tpl) or ("{% generation" not in tpl):
            tok.chat_template = r"""
{{ bos_token }}
{% for message in messages -%}
{%- if message['role'] == 'system' -%}
<|im_start|>system
{{ message['content'] }}<|im_end|>
{%- elif message['role'] == 'user' -%}
<|im_start|>user
{{ message['content'] }}<|im_end|>
{%- elif message['role'] == 'assistant' -%}
<|im_start|>assistant
{% generation %}{{ message['content'] }}{% endgeneration %}<|im_end|>
{%- endif -%}
{% endfor -%}
{%- if add_generation_prompt -%}
<|im_start|>assistant
{%- endif -%}
""".strip()

        guard = (args.system_guard or "").strip()
        msgs_ds = _DS.from_list([{"user": ut, "assistant": yt} for (ut, yt) in pairs])

        def _formatting_func(examples: Dict[str, List[str]]) -> List[str]:
            users = examples["user"]
            assis = examples["assistant"]
            out_texts: List[str] = []
            for u, a in zip(users, assis):
                messages = []
                if guard:
                    messages.append({"role": "system", "content": guard})
                messages += [
                    {"role": "user", "content": u},
                    {"role": "assistant", "content": a},
                ]
                try:
                    rendered = tok.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                        enable_thinking=False,   # DISABLE thinking during supervised training
                    )
                except Exception:
                    rendered = (
                        (f"<|im_start|>system\n{guard}<|im_end|>\n" if guard else "") +
                        f"<|im_start|>user\n{u}<|im_end|>\n"
                        f"<|im_start|>assistant\n{a}<|im_end|>\n"
                    )
                out_texts.append(rendered)
            return out_texts

        # Prepare model
        model_kwargs = {"trust_remote_code": True}
        if args.bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

        # Clean output_dir of stale files BEFORE saving new weights
        os.makedirs(args.output_dir, exist_ok=True)
        _cleanup_stale_weights(args.output_dir)

        # Compute the training max length from the true context window (unless user set a cap)
        ctx_max = _context_limit(tok, model)
        train_max_len = ctx_max if args.max_seq_len == 0 else min(args.max_seq_len, ctx_max)
        if is_main_process():
            print(f"[train] max_seq_length used for training: {train_max_len}")

        # Data collator for assistant-only loss (mask everything before assistant block)
        response_prefix = "<|im_start|>assistant\n"
        collator = DataCollatorForCompletionOnlyLM(
            response_template=response_prefix,
            tokenizer=tok,
        )

        # Save config: force overwrite dir; we will consolidate into a single .bin right after
        cfg_kwargs = dict(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            logging_steps=10,
            save_strategy="no",
            max_seq_length=train_max_len,
            packing=False,
            run_name=args.run_name,
            overwrite_output_dir=True,
            save_safetensors=False,   # -> prefer PyTorch .bin
        )
        if args.report_to != "none":
            cfg_kwargs["report_to"] = [args.report_to]
        if args.bf16:
            cfg_kwargs["bf16"] = True
        if args.gradient_checkpointing:
            cfg_kwargs["gradient_checkpointing"] = True

        from trl import SFTConfig, SFTTrainer
        cfg_trl = SFTConfig(**cfg_kwargs)

        trainer = SFTTrainer(
            model=model,
            args=cfg_trl,
            train_dataset=msgs_ds,
            tokenizer=tok,
            formatting_func=_formatting_func,
            data_collator=collator,
        )

        trainer.train()

        # Primary save (trainer wrapper)
        trainer.save_model(args.output_dir)

        # Save tokenizer alongside the model (so future local loads succeed)
        try:
            tok.save_pretrained(args.output_dir)
        except Exception as e:
            print(f"[warn] tokenizer save failed: {e}")

        # === AUTO-MERGE SHARDS INTO A SINGLE .BIN (uses in-memory model; no extra reload) ===
        if args.merge_shards:
            try:
                trainer.model.save_pretrained(
                    args.output_dir,
                    safe_serialization=False,   # write PyTorch .bin
                    max_shard_size="0GB"        # single-file merge
                )
                _remove_shard_files(args.output_dir)
                if is_main_process():
                    print("[merge] Consolidated into a single pytorch_model.bin and cleaned shard files.")
            except Exception as e:
                print(f"[merge] WARNING: consolidation into single .bin failed ({e}). Keeping existing files.")
        else:
            if is_main_process():
                print("[merge] Skipped (per --no_merge_shards). Sharded files may remain in output_dir.")

        # Post-save checkpoint info
        bin_path = os.path.join(args.output_dir, "pytorch_model.bin")
        index_path = os.path.join(args.output_dir, "pytorch_model.bin.index.json")
        shards = sorted([p for p in os.listdir(args.output_dir)
                         if p.startswith("pytorch_model-") and p.endswith(".bin")])

        if os.path.isfile(bin_path):
            sz_gb = _pretty_size_gb(bin_path)
            if is_main_process():
                print(f"✅ Saved single file: pytorch_model.bin ~{sz_gb:.2f} GB")
            # rough size sanity band for ~14B bf16
            assert 16.0 <= sz_gb <= 40.0, (
                f"Saved model size {sz_gb:.2f} GB out of expected 14B-bf16 range [16, 40]."
            )
        elif os.path.isfile(index_path) or shards:
            print("ℹ️ Sharded checkpoint detected. Proceeding without merge.")
        else:
            raise FileNotFoundError(f"No pytorch_model.bin or shards found in {args.output_dir}")

        model_path_for_pred = os.path.abspath(args.output_dir)
        _print_cfg("saved", model_path_for_pred)

        # FREE MEMORY BEFORE PREDICTION
        import gc, torch as _torch
        try:
            trainer.accelerator.free_memory()
        except Exception:
            pass
        del trainer, model, tok
        gc.collect()
        _torch.cuda.empty_cache()

        from torch import distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    else:
        if args.skip_train and is_main_process():
            print("[note] --skip_train set; skipping SFT and using base model for prediction.")
        elif not pairs and is_main_process():
            print("[note] No training pairs; using base model for prediction.")

    # ------------- Predictions on TEST -------------
    if is_main_process() and args.predict_on_test:
        if os.path.isdir(model_path_for_pred):
            _cleanup_stale_weights(model_path_for_pred)

        generate_predictions_file(
            model_id_or_path=model_path_for_pred,
            inputs_file=args.test_file,
            out_path=args.test_pred_out,
            max_new_tokens=args.gen_max_new_tokens,
            temperature=args.gen_temperature,
            top_p=args.gen_top_p,
            batch_size=args.gen_batch_size,
            bf16=args.bf16,
            seed=args.seed,
            guard_message=args.system_guard,
            thinking=args.thinking,                 # opt-in for Qwen3 thinking
            tokenizer_id=(args.tokenizer_id or None)  # critical for local dirs w/o tokenizer files
        )

    # ------------- DeepSeek Reasoner on TEST preds -------------
    if is_main_process() and args.run_deepseek_on_test:
        keys_env = os.environ.get("DEEPSEEK_API_KEYS", "")
        keys = [k.strip() for k in (args.deepseek_api_keys or keys_env).split(",") if k.strip()]
        if not keys:
            print("[deepseek] No API keys provided. Use --deepseek_api_keys or env DEEPSEEK_API_KEYS.")
        else:
            print(f"[deepseek] using {len(keys)} API key(s)")
            run_deepseek_on_predictions(
                predicted_path=args.test_pred_out,
                out_path=args.deepseek_out,
                api_keys=keys,
                base_url=args.deepseek_base_url,
                model=args.deepseek_model,
                temp=args.deepseek_temperature,
                top_p=args.deepseek_top_p,
                max_tokens=args.deepseek_max_tokens,
                concurrency=args.deepseek_concurrency,
                timeout=args.deepseek_timeout,
            )


if __name__ == "__main__":
    main()
