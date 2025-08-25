#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
sft_Qwen_train.py

Pipeline:
1) Read X from sft_Qwen_traindata.jsonl
   {"text": "... original prompt: <PROMPT>\\n(output prompt:|Please put your changed prompt here:) ..."}
2) Read y from two files (STRICT keys):
   - sft_moderate_excludeD_changedprompts.jsonl: {"original prompt": "<PROMPT>", "output prompt": "<Y>"}
   - sft_severe_excludeHard_changedprompts.jsonl: {"original prompt": "<PROMPT>", "output prompt": "<Y>"}
3) Join by normalized <PROMPT>, train with TRL SFTTrainer.
4) Generate predictions:
   - TRAIN X  -> sft_Qwen_train_predictedy.jsonl
   - TEST  X  -> sft_Qwen_test_predictedy.jsonl
5) (Optional) Send each TEST "predicted output prompt" to DeepSeek Reasoner in parallel and save:
   - sft_Qwen_test_deepseek_results.jsonl

NOTE: Do NOT hardcode secrets. Pass DeepSeek keys via
  --deepseek_api_keys sk-aaa,sk-bbb,sk-ccc
or environment variable
  export DEEPSEEK_API_KEYS="sk-aaa,sk-bbb,sk-ccc"
"""

import argparse
import json
import os
import re
import unicodedata
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

# ---------------- text utils ----------------
def norm_text(s: str) -> str:
    """NFKC normalize, trim, collapse whitespace, lowercase."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s.lower()

# Accept both labels after the original prompt
EXTRACT_PATTERNS = [
    re.compile(
        r"original\s*prompt\s*:\s*"          # left label
        r"(.*?)"                              # capture the original prompt (non-greedy)
        r"\r?\n\s*"                           # newline to the next label
        r"Please\s+put\s+your\s+changed\s+prompt\s+here\s*:\s*",  # right label
        flags=re.IGNORECASE | re.DOTALL,
    ),
]

def extract_original_from_input_text(t: str) -> Optional[str]:
    if not isinstance(t, str):
        return None
    for rx in EXTRACT_PATTERNS:
        m = rx.search(t)
        if m:
            return m.group(1).strip()
    return None


# ---------------- prompt-length helpers ----------------
def _context_limit(tok, model):
    """
    Resolve the model's true context window, ignoring HF's huge sentinel values.
    """
    vals = []
    # return max length of tok
    v = getattr(tok, "model_max_length", None)
    if isinstance(v, int) and v < 10**7:
        vals.append(v)
    v = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if isinstance(v, int):
        vals.append(v)
    return max(vals) if vals else 131072  # sensible default for Qwen 2.5
    # pick the max context window


# ---------------- y lookup, takes path, return dictioanry mapping
def build_lookup_from_file(path: str) -> Dict[str, str]:
    """
    Accepts ONLY: {"original prompt": "<PROMPT>", "output prompt": "<Y>"}
    Returns: norm(<PROMPT>) -> <Y>
    """
    lut = {}
    rows = load_jsonl(path)  # assume present
    for r in rows:
        p = r.get("original prompt")
        y = r.get("output prompt")
        if isinstance(p, str) and isinstance(y, str) and p.strip() and y.strip():
            key = norm_text(p)
            if key not in lut or len(y) > len(lut[key]):  # keep longest target if duplicate
                lut[key] = y.strip()
    print(f"[y] {os.path.basename(path)} -> {len(lut)} entries")
    return lut

















# ---------------- chat templating (string fallback) ----------------, didn't use, 
def chat_text(tokenizer, user_text: str, target_text: str) -> str:
    """
    Build a chat-style training sample WITHOUT any extra/system text.
    """
    messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": target_text},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except Exception:
        # Minimal ChatML-style fallback (no system text)
        return (
            f"<|im_start|>user\n{user_text}<|im_end|>\n"
            f"<|im_start|>assistant\n{target_text}<|im_end|>\n"
        )

# ---------------- generation helpers ----------------
def is_main_process() -> bool:
    return str(os.environ.get("RANK", "0")) == "0" and str(os.environ.get("LOCAL_RANK", "0")) == "0"

def batchify(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def generate_file(model_id_or_path: str,
                  inputs_file: str,
                  out_path: str,
                  max_new_tokens: int,
                  temperature: float,
                  top_p: float,
                  batch_size: int,
                  bf16: bool,
                  seed: int) -> None:
    """
    Tokenize to the model's true context window (no smaller cap),
    left-truncate only if absolutely necessary (to preserve assistant header),
    and stop on <|im_end|> if present.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    data = load_jsonl(inputs_file)
    rows = [r for r in data if isinstance(r.get("text"), str)]
    texts = [r["text"] for r in rows]
    if not texts:
        print(f"[predict] no valid 'text' rows in {inputs_file}")
        return


    #It provides a chat_template Jinja snippet so apply_chat_template can turn structured messages into strings the model was trained on.
    tok = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=True, trust_remote_code=True)
    # return input ids and attention masks 

    # Ensure padding + preserve the tail if we ever hit the hard limit
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"

    kw = {"trust_remote_code": True, "device_map": "auto"}    ## kw is python dictonary
    if bf16:
        kw["torch_dtype"] = torch.bfloat16
    #**kw unpacks your dictionary into keyword arguments.
    model = AutoModelForCausalLM.from_pretrained(model_id_or_path, **kw)
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
  
    #how many prediction records you successfully write to the output file.
    written = 0 
    with open(out_path, "w", encoding="utf-8") as w, torch.no_grad():
        for batch in batchify(texts, batch_size):
            # Build prompts with ONLY a user message (no system text)
            prompts = []
            for t in batch:
                messages = [{"role": "user", "content": t}]
                try:
                    prompts.append(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
                except Exception:
                    prompts.append(f"<|im_start|>user\n{t}<|im_end|>\n<|im_start|>assistant\n")

            # IMPORTANT: truncate only to the model's real context window
            enc = tok(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=ctx_max,
            )
            enc = {k: v.to(model.device) for k, v in enc.items()}

            input_lens = enc["attention_mask"].sum(dim=1).tolist()

            ## print each line of gen, download model from huggging face, colab, 做infer 
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                pad_token_id=tok.pad_token_id,
                eos_token_id=stop_id,
            )

            for i, seq in enumerate(gen):
                start = input_lens[i]
                content_ids = seq[start:].tolist()

                # hard-trim at first <|im_end|> if present (for clean text only)
                if im_end_id is not None and im_end_id in content_ids:
                    content_ids = content_ids[:content_ids.index(im_end_id)]

                pred = tok.decode(content_ids, skip_special_tokens=True).strip()
                orig = extract_original_from_input_text(batch[i]) or ""
                rec = {
                    "original prompt": orig,
                    "predicted output prompt": pred,
                }
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

    print(f"[predict] wrote {written} rows -> {out_path}")

# ---------------- DeepSeek Reasoner (parallel HTTP, stdlib only) ---------------
def deepseek_chat_once(base_url: str, model: str, api_key: str,
                       user_content: str, temperature: float,
                       max_tokens: int, top_p: float, timeout: int = 60) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (text, error). Uses urllib from stdlib to avoid new deps.
    """
    import urllib.request, urllib.error

    body = {
        "model": model,
        "messages": [
            {"role": "user", "content": user_content}
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    data = json.dumps(body).encode("utf-8")
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
            obj = json.loads(s)
            text = (
                obj.get("choices", [{}])[0]
                   .get("message", {})
                   .get("content", "")
            )
            return (text, None)
    except urllib.error.HTTPError as e:
        try:
            err = e.read().decode("utf-8")
        except Exception:
            err = str(e)
        return (None, f"HTTP {e.code}: {err}")
    except Exception as e:
        return (None, f"{type(e).__name__}: {e}")

def run_deepseek_on_predictions(predicted_path: str,
                                out_path: str,
                                api_keys: List[str],
                                base_url: str,
                                model: str,
                                temp: float,
                                top_p: float,
                                max_tokens: int,
                                concurrency: int,
                                timeout: int) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    recs = load_jsonl(predicted_path)
    items = []
    for rec in recs:
        po = rec.get("predicted output prompt")
        op = rec.get("original prompt", "")
        if isinstance(po, str) and po.strip():
            items.append((op, po))

    if not items:
        print(f"[deepseek] no items found in {predicted_path}")
        return

    # Round-robin keys across tasks
    def key_for(i: int) -> str:
        return api_keys[i % len(api_keys)]

    results = [None] * len(items)

    def worker(i: int, original: str, user_prompt: str):
        text, err = deepseek_chat_once(
            base_url=base_url,
            model=model,
            api_key=key_for(i),
            user_content=user_prompt,
            temperature=temp,
            max_tokens=max_tokens,
            top_p=top_p,
            timeout=timeout,
        )
        return (i, original, user_prompt, text, err)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(worker, i, it[0], it[1]) for i, it in enumerate(items)]
        for fut in as_completed(futs):
            i, original, user_prompt, text, err = fut.result()
            results[i] = {
                "original prompt": original,
                "predicted output prompt": user_prompt,
                "deepseek_response": text if text is not None else "",
                "error": err if err is not None else ""
            }

    # write in original order
    with open(out_path, "w", encoding="utf-8") as w:
        for r in results:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[deepseek] wrote {len(results)} rows -> {out_path}")

# ---------------- plotting helper (NEW) ----------------
def save_loss_lr_plot(trainer, out_path: str, title: str = "sft_Qwen2.5_7B_train_plot", log_to_wandb: bool = False):
    """
    Build a dual-axis plot from trainer.state.log_history showing train/loss and learning_rate vs global step.
    Saves a PNG to out_path and optionally logs it to Weights & Biases.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
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
        ax2.plot(lr_steps, lrs, linestyle="--", label="learning_rate")
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
            wandb.log({"sft_Qwen2.5_7B_train_plot": wandb.Image(out_path)})
        except Exception as e:
            print(f"[plot] W&B log skipped: {e}")

# --------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()

    # Inputs
    ap.add_argument("--inputs_file", default="sft_Qwen_traindata.jsonl",
                    help="X JSONL with a 'text' field (X is not modified).")
    ap.add_argument("--moderate_rewrites", default="sft_moderate_excludeD_changedprompts.jsonl",
                    help="Y (moderate): requires {'original prompt','output prompt'}")
    ap.add_argument("--severe_rewrites",   default="sft_severe_excludeHard_changedprompts.jsonl",
                    help="Y (severe): requires {'original prompt','output prompt'}")

    # Debug outputs
    ap.add_argument("--joined_preview_out", default="sft_joined_pairs_preview.jsonl")
    ap.add_argument("--unmatched_out", default="sft_unmatched_inputs.jsonl")

    # Training
    ap.add_argument("--skip_train", action="store_true", help="Build/validate pairs only; still can run prediction.")
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct",
                    help="HF repo id of the base model.")
    ap.add_argument("--output_dir", default="qwen-sft-output")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=1)   # per-GPU micro-batch
    ap.add_argument("--grad_accum", type=int, default=1)   # micro-steps accumulated before one optimizer step
    ap.add_argument("--lr", type=float, default=2e-5)

    # IMPORTANT: 0 means "use model's context window" to avoid extra truncation, tranining 
    ap.add_argument("--max_seq_len", type=int, default=0,
                    help="0 = use model context window (no extra training truncation); otherwise cap to this many tokens.")

    # Multi-GPU / H100 toggles
    ap.add_argument("--bf16", action="store_true", help="Enable bf16 mixed precision (recommended on H100).")
    ap.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    ap.add_argument("--deepspeed_config", default="", help="Path to Deepspeed JSON (optional).")

    # Logging
    ap.add_argument("--report_to", default="none", choices=["none", "wandb", "tensorboard"],
                    help="Where to report metrics.")
    ap.add_argument("--run_name", default="qwen-sft", help="Experiment/run name for loggers.")

    # Generation / prediction
    ap.add_argument("--predict_on_train", action="store_true",
                    help="Generate predictions for the training X.")
    ap.add_argument("--train_pred_out", default="sft_Qwen_train_predictedy.jsonl")
    ap.add_argument("--predict_on_test", action="store_true",
                    help="Generate predictions for --test_inputs_file.")
    ap.add_argument("--test_inputs_file", default="sft_Qwen_testdata.jsonl")
    ap.add_argument("--test_pred_out", default="/home/hubing/llm-adaptive-attacks/data_to_upload/sft_Qwen_test_predictedy.jsonl")


    ap.add_argument("--gen_max_new_tokens", type=int, default=8000)
    ap.add_argument("--gen_temperature", type=float, default=0.5)
    ap.add_argument("--gen_top_p", type=float, default=1.0)
    ap.add_argument("--gen_batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)

    # DeepSeek post-processing on TEST predictions
    ap.add_argument("--run_deepseek_on_test", action="store_true",
                    help="Call DeepSeek Reasoner on each TEST predicted output prompt.")
    ap.add_argument("--deepseek_api_keys", default="", help="Comma-separated API keys; or use env DEEPSEEK_API_KEYS.")
    ap.add_argument("--deepseek_base_url", default="https://api.deepseek.com/v1/chat/completions")
    ap.add_argument("--deepseek_model", default="deepseek-reasoner")
    ap.add_argument("--deepseek_temperature", type=float, default=0.5)
    ap.add_argument("--deepseek_top_p", type=float, default=1.0)
    ap.add_argument("--deepseek_max_tokens", type=int, default=8000)
    ap.add_argument("--deepseek_concurrency", type=int, default=12)
    ap.add_argument("--deepseek_timeout", type=int, default=60)
    ap.add_argument("--deepseek_out", default="/home/hubing/llm-adaptive-attacks/data_to_upload/sft_Qwen_test_deepseek_results.jsonl")



    args = ap.parse_args()

    # 1) Load X (unchanged)
    X = load_jsonl(args.inputs_file)
    print(f"[inputs] loaded {len(X)} rows from {args.inputs_file}")

    # 2) Build y lookups (both files use 'original prompt' -> 'output prompt')
    # inserts k in severe if it is not present in mod
    lut: Dict[str, str] = {}
    lut_mod = build_lookup_from_file(args.moderate_rewrites)
    lut_sev = build_lookup_from_file(args.severe_rewrites)
    lut.update(lut_mod)
    for k, v in lut_sev.items():
        lut.setdefault(k, v)

    # 3) Join X with y
    joined_pairs: List[Tuple[str, str]] = []
    unmatched: List[Dict] = []
    for i, rec in enumerate(X, 1):
        user_text = rec.get("text")
        if not isinstance(user_text, str):
            unmatched.append({"line": i, "reason": "no_text"}); continue
        orig = extract_original_from_input_text(user_text)
        if not orig:
            unmatched.append({"line": i, "reason": "cannot_extract_original",
                              "head": user_text[:120]}); continue
        y = lut.get(norm_text(orig))
        if y:
            joined_pairs.append((user_text, y))
        else:
            unmatched.append({"line": i, "original": orig})

    print(f"[join] matched={len(joined_pairs)}  unmatched={len(unmatched)}")

    # Debug dumps
    if joined_pairs:
        preview = [{"text": ut, "target": yt} for ut, yt in joined_pairs[:50]]
        write_jsonl(args.joined_preview_out, preview)
        print(f"[debug] wrote preview -> {args.joined_preview_out} ({len(preview)} rows)")
    if unmatched:
        write_jsonl(args.unmatched_out, unmatched)
        print(f"[debug] wrote unmatched -> {args.unmatched_out} ({len(unmatched)} rows)")

    # 4) Train (unless skip)
    model_path_for_pred = args.model_id
    if not args.skip_train and joined_pairs:
        from datasets import Dataset
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from trl import SFTTrainer, SFTConfig
        import torch

        tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)

        # Ensure pad token is set BEFORE creating the trainer
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token  # aligns padding with EOS

        # If we ever hit a limit, keep the assistant tail
        tok.truncation_side = "left"

        # Ensure the template has `{% generation %}...{% endgeneration %}`
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

        model_kwargs = {"trust_remote_code": True}
        if args.bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

        # Compute the training max length from the true context window (unless user set a cap)
        ctx_max = _context_limit(tok, model)
        train_max_len = ctx_max if args.max_seq_len == 0 else min(args.max_seq_len, ctx_max)

        # Conversational dataset so assistant_only_loss works
        msgs_ds = Dataset.from_list([
            {"messages": [
                {"role": "user", "content": ut},
                {"role": "assistant", "content": yt},
            ]}
            for (ut, yt) in joined_pairs
        ])

        cfg_kwargs = dict(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            logging_steps=10,
            save_strategy="no",
            max_seq_length=train_max_len,   # <= now uses model context by default
            packing=False,
            run_name=args.run_name,
            assistant_only_loss=True,
        )

        if args.report_to != "none":
            cfg_kwargs["report_to"] = [args.report_to]
        if args.bf16:
            cfg_kwargs["bf16"] = True
        if args.gradient_checkpointing:
            cfg_kwargs["gradient_checkpointing"] = True
        if args.deepspeed_config:
            cfg_kwargs["deepspeed"] = args.deepspeed_config

        cfg = SFTConfig(**cfg_kwargs)

        trainer = SFTTrainer(
            model=model,
            args=cfg,
            train_dataset=msgs_ds,
            processing_class=tok,   # pass tokenizer for chat template & masking
        )

        trainer.train()
        trainer.save_model(args.output_dir)
        tok.save_pretrained(args.output_dir)
        model_path_for_pred = args.output_dir

        # -------- save plot (main process only) --------
        plot_path = os.path.join(args.output_dir, "sft_Qwen2.5_7B_train_plot.png")
        try:
            is_world_zero = (getattr(trainer, "args", None) is not None and getattr(trainer.args, "process_index", 0) == 0)
        except Exception:
            is_world_zero = True
        if is_world_zero:
            save_loss_lr_plot(
                trainer,
                out_path=plot_path,
                title="sft_Qwen2.5_7B_train_plot",
                log_to_wandb=(args.report_to == "wandb"),
            )
        # -----------------------------------------------

        print(f"✅ Training finished. Model saved to {args.output_dir}")
    else:
        if args.skip_train:
            print("[note] --skip_train set; skipping SFT and using base model for prediction.")
        elif not joined_pairs:
            print("[note] No training pairs; using base model for prediction.")

    # 5) Predictions (main process only)
    if is_main_process():
        if args.predict_on_train:
            generate_file(
                model_id_or_path=model_path_for_pred,
                inputs_file=args.inputs_file,
                out_path=args.train_pred_out,
                max_new_tokens=args.gen_max_new_tokens,
                temperature=args.gen_temperature,
                top_p=args.gen_top_p,
                batch_size=args.gen_batch_size,
                bf16=args.bf16,
                seed=args.seed,
            )

        if args.predict_on_test:
            generate_file(
                model_id_or_path=model_path_for_pred,
                inputs_file=args.test_inputs_file,
                out_path=args.test_pred_out,
                max_new_tokens=args.gen_max_new_tokens,
                temperature=args.gen_temperature,
                top_p=args.gen_top_p,
                batch_size=args.gen_batch_size,
                bf16=args.bf16,
                seed=args.seed,
            )

        # 6) DeepSeek post-processing (optional)
        if args.run_deepseek_on_test:
            keys = [k.strip() for k in (args.deepseek_api_keys or os.environ.get("DEEPSEEK_API_KEYS", "")).split(",") if k.strip()]
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
