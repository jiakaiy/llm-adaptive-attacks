# conversers.py
import os
import requests
import torch
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer

from language_models import GPT, HuggingFace

# ─────────────────────────────────────────────────────────────
# helper: load any HF model into HuggingFace wrapper
# ─────────────────────────────────────────────────────────────
def hf_loader(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mod = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device).eval()
    return HuggingFace(model_id, mod, tok)

# ─────────────────────────────────────────────────────────────
# main.py expects this
# ─────────────────────────────────────────────────────────────
def load_target_model(args):
    name = getattr(args, "target_model", "")
    if name == "deepseek-chat":
        return DeepSeekWrapper()
    elif name == "deepseek-reasoner":
        return DeepSeekReasonerWrapper()
    elif name == "stair-qwen":
        return TargetLM("stair-qwen", temp=0.7, top_p=0.95)
    else:
        return TargetLM(name or "gpt-3.5-turbo", temp=0.7, top_p=0.95)

# ─────────────────────────────────────────────────────────────
# Minimal TargetLM wrapper (adapts to our inner concrete models)
# ─────────────────────────────────────────────────────────────
class TargetLM:
    def __init__(self, name: str, temp: float = 0.7, top_p: float = 0.95):
        self.temperature = temp
        self.top_p       = top_p
        self.model       = load_indiv_model(name)
        # ensure tokenizer exists for main.py
        self.tokenizer   = getattr(self.model, "tokenizer", None)
        if self.tokenizer is None:
            # extremely small fallback tokenizer, if needed
            class _Tok:
                vocab_size = 1000000
                def encode(self, s): return [ord(c) % 256 for c in s]
                def decode(self, ids): return "".join(chr(i) for i in ids)
            self.tokenizer = _Tok()
        # stats expected by main.py
        self.n_input_tokens  = 0
        self.n_output_tokens = 0
        self.n_input_chars   = 0
        self.n_output_chars  = 0

    def generate(self, prompts: List[str], max_n_tokens: int = 1024):
        return self.model.generate(
            prompts,
            max_n_tokens=max_n_tokens,
            temperature=self.temperature,
            top_p=self.top_p
        )

    def get_response(self, prompts: List[str], max_n_tokens=1024, temperature=None):
        if temperature is None:
            temperature = self.temperature
        outs = self.model.generate(
            prompts,
            max_n_tokens=max_n_tokens,
            temperature=temperature,
            top_p=self.top_p
        )
        # accumulate stats (best‑effort)
        for o in outs:
            self.n_input_chars   += sum(len(p) for p in prompts)
            self.n_output_chars  += len(o.get("text", ""))
            self.n_input_tokens  += o.get("n_input_tokens", 0)
            self.n_output_tokens += o.get("n_output_tokens", 0)
        return outs

# ─────────────────────────────────────────────────────────────
# Factory returning concrete model wrapper
# ─────────────────────────────────────────────────────────────
def load_indiv_model(name: str):
    nm = (name or "").lower()
    if "gpt" in nm or "together" in nm:          # OpenAI / Together
        return GPT(name)
    if nm == "hf_falcon_rw_1b":                  # Falcon attacker
        return hf_loader("tiiuae/falcon-rw-1b")
    if nm == "stair-qwen":                       # STAIR-Qwen target
        return hf_loader("thu-ml/STAIR-Qwen2-7B-DPO-3")
    if nm in ("deepseek_chat", "deepseek-chat"):
        return DeepSeekWrapper()
    if nm in ("deepseek_reasoner", "deepseek-reasoner"):
        return DeepSeekReasonerWrapper()
    raise ValueError(f"Unknown model name '{name}'")

# ─────────────────────────────────────────────────────────────
# DeepSeek base class and wrappers
# ─────────────────────────────────────────────────────────────
class _DeepSeekBase:
    """
    Non‑streaming wrapper around DeepSeek's OpenAI‑compatible /v1/chat/completions.

    Fixes for R1:
      - uses max_output_tokens (not max_tokens)
      - falls back to reasoning_content if content is empty
    """
    def __init__(self, model_name: str, api_key_env: str = "DEEPSEEK_API_KEY"):
        self.api_key = os.getenv(api_key_env) or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(f"{api_key_env} environment variable not set")
        self.model_name = model_name

        # Base URL precedence: DEEPSEEK_API_BASE > OPENAI_BASE_URL > OPENAI_API_BASE > default
        self.base_url = (
            os.getenv("DEEPSEEK_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("OPENAI_API_BASE")
            or "https://api.deepseek.com"
        ).rstrip("/")

        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        })

        # tokenizer expected by main.py; prefer local HF tokenizer if available
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        except Exception:
            class _Tok:
                vocab_size = 1000000
                def encode(self, s): return [ord(c) % 256 for c in s]
                def decode(self, ids): return "".join(chr(i) for i in ids)
            self.tokenizer = _Tok()

        # mimic attributes main.py expects
        self.model = self
        self.n_input_tokens  = 0
        self.n_output_tokens = 0
        self.n_input_chars   = 0
        self.n_output_chars  = 0

    def _post_json(self, payload, timeout=120):
        url = f"{self.base_url}/v1/chat/completions"
        resp = self._session.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _extract_text(choice_message: dict) -> str:
        """
        Prefer the final answer in message.content; if empty, fall back to
        message.reasoning_content (R1 often fills this during/after reasoning).
        """
        if not isinstance(choice_message, dict):
            return ""
        text = (choice_message.get("content") or "")  # final answer
        if not text:
            rc = choice_message.get("reasoning_content") or ""
            text = rc
        return text or ""

    def generate(self, prompts, max_n_tokens, temperature, top_p):
        results = []
        for prompt in prompts:
            # Build request payload; do NOT stream here.
            payload = {
                "model":    self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": float(temperature),
                "top_p": float(top_p),
                "stream": False,
            }

            # Critical: reasoner needs max_output_tokens; chat uses max_tokens
            if "reasoner" in self.model_name:
                payload["max_output_tokens"] = int(max_n_tokens)
                # Optional knob: reasoning effort (low/medium/high)
                payload["reasoning"] = {"effort": "medium"}
            else:
                payload["max_tokens"] = int(max_n_tokens)

            j = self._post_json(payload)
            choice = j.get("choices", [{}])[0]
            msg = choice.get("message", {})
            text = self._extract_text(msg)

            # update stats (best‑effort)
            self.n_input_chars   += len(prompt)
            self.n_output_chars  += len(text)
            # crude token approximations for logging
            in_tok  = max(1, len(prompt.split()))
            out_tok = max(1, len(text.split())) if text else 0
            self.n_input_tokens  += in_tok
            self.n_output_tokens += out_tok

            results.append({
                "text":            text,
                "logprobs":        None,       # DeepSeek API doesn't expose token logprobs
                "n_input_tokens":  in_tok,
                "n_output_tokens": out_tok,
            })
        return results

    def get_response(self, prompts, max_n_tokens=1024, temperature=None):
        if temperature is None:
            temperature = 0.7
        return self.generate(prompts, max_n_tokens, temperature, 0.95)

class DeepSeekWrapper(_DeepSeekBase):
    def __init__(self):
        super().__init__(model_name="deepseek-chat")

class DeepSeekReasonerWrapper(_DeepSeekBase):
    def __init__(self):
        super().__init__(model_name="deepseek-reasoner")
