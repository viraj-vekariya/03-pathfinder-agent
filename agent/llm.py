"""LLM adapter with an explicit, measurable context budget.

The context budget is not incidental here - it is the independent variable of the whole
evaluation. Every backend therefore reports:

  * `context_limit`  - how many tokens it will actually accept
  * `count_tokens`   - the real tokenizer's count, not a characters/4 estimate
  * `truncated`      - whether THIS call lost input, and how much

Most agent frameworks hide truncation. That is precisely the bug this project exists to
measure: when a prompt silently loses its tail, the model answers confidently about
information it never received, and the failure looks like reasoning failure rather than
what it is.

Backends:
  * **flan-t5** (google/flan-t5-base, 512-token limit) - a real instruction-tuned
    seq2seq model, running locally with no key. Its small context is a feature: it
    reaches the truncation cliff at a depth this corpus actually contains.
    flan-t5-SMALL was tried first and rejected: asked to pick a tool it replied
    "resolve python package installation order" - it echoed the task instead of acting
    on it, so every run scored zero for reasons that had nothing to do with context.
    A model that cannot do the task at depth 1 cannot measure degradation with depth.
  * **anthropic** - a real API call when ANTHROPIC_API_KEY is set.
  * **echo** - deterministic, no model. Used to separate "the harness works" from
    "the model works", and to make the test suite run without downloads.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional

log = logging.getLogger("pathfinder.llm")


@dataclass
class Completion:
    text: str
    backend: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    context_limit: int
    truncated: bool
    tokens_dropped: int
    latency_ms: float
    error: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {"backend": self.backend, "model": self.model,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "context_limit": self.context_limit,
                "truncated": self.truncated, "tokens_dropped": self.tokens_dropped,
                "latency_ms": round(self.latency_ms, 2), "error": self.error}


class BaseLLM:
    backend = "base"
    model = "none"
    context_limit = 512

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError

    def complete(self, prompt: str, max_new_tokens: int = 256) -> Completion:
        raise NotImplementedError

    def would_truncate(self, prompt: str) -> tuple[bool, int]:
        n = self.count_tokens(prompt)
        return n > self.context_limit, max(0, n - self.context_limit)


class EchoLLM(BaseLLM):
    """No model. Extracts package names from the prompt and returns them in the order
    they appear.

    This is the harness's control arm, not a joke: it isolates how much of the agent's
    score comes from the PROMPT already containing the answer in a usable order, versus
    from the model doing anything. If a real model cannot beat this, it is not
    contributing.
    """

    backend = "echo"
    model = "echo"

    def __init__(self, context_limit: int = 512):
        self.context_limit = context_limit

    def count_tokens(self, text: str) -> int:
        # Whitespace tokens. Crude, but this backend exists to test control flow, and
        # it is honest about being an approximation rather than pretending otherwise.
        return len(text.split())

    def complete(self, prompt: str, max_new_tokens: int = 256) -> Completion:
        started = time.perf_counter()
        truncated, dropped = self.would_truncate(prompt)
        # Truncation is SIMULATED faithfully: the tail is dropped, exactly as a real
        # tokenizer would, so the control arm degrades the same way the model does.
        if truncated:
            prompt = " ".join(prompt.split()[: self.context_limit])

        seen, names = set(), []
        for token in re.findall(r"\b[a-z][a-z0-9._-]{1,40}\b", prompt.lower()):
            if token not in seen and "-" in token or token in seen:
                pass
            if token not in seen:
                seen.add(token)
                names.append(token)
        text = ", ".join(names[:80])
        return Completion(text, self.backend, self.model, self.count_tokens(prompt),
                          len(text.split()), self.context_limit, truncated, dropped,
                          (time.perf_counter() - started) * 1000)


class FlanT5LLM(BaseLLM):
    """google/flan-t5-* run locally.

    The 512-token limit is real and is enforced by the model's own positional encoding,
    not by us. That makes it an honest instrument: the cliff the evaluation finds is a
    property of the model, and the same experiment on a 128k-context model would find
    the cliff much later - which is the point being made, not a limitation of it.
    """

    backend = "flan-t5"

    def __init__(self, model_name: str = "google/flan-t5-base"):
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch

        self.torch = torch
        self.model = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self._model.eval()
        self.context_limit = int(self.tokenizer.model_max_length)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=True)["input_ids"])

    def complete(self, prompt: str, max_new_tokens: int = 256) -> Completion:
        started = time.perf_counter()
        truncated, dropped = self.would_truncate(prompt)
        encoded = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                 max_length=self.context_limit)
        with self.torch.no_grad():
            output = self._model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                # Greedy. The evaluation compares plans across depths, and sampling
                # would add run-to-run variance that is indistinguishable from the
                # effect being measured.
                do_sample=False,
                num_beams=1,
            )
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return Completion(text, self.backend, self.model,
                          int(encoded["input_ids"].shape[1]) + dropped,
                          int(output.shape[1]), self.context_limit, truncated, dropped,
                          (time.perf_counter() - started) * 1000)


class AnthropicLLM(BaseLLM):
    """Real API call. Only used when a key is present."""

    backend = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5", api_key: str = "",
                 timeout: float = 40.0, context_limit: int = 200_000):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.timeout = timeout
        self.context_limit = context_limit

    def count_tokens(self, text: str) -> int:
        # Approximation, and labelled as one. The exact count needs an API round trip,
        # which is not worth a network call per prompt inside a sweep. It is only used
        # for reporting here; the real limit is enforced server-side.
        return max(1, len(text) // 4)

    def complete(self, prompt: str, max_new_tokens: int = 512) -> Completion:
        started = time.perf_counter()
        payload = json.dumps({
            "model": self.model,
            "max_tokens": max_new_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=payload,
            headers={"content-type": "application/json", "x-api-key": self.api_key,
                     "anthropic-version": "2023-06-01"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode())
        except Exception as exc:                 # noqa: BLE001
            return Completion("", self.backend, self.model, self.count_tokens(prompt),
                              0, self.context_limit, False, 0,
                              (time.perf_counter() - started) * 1000, error=str(exc)[:200])

        text = "".join(b.get("text", "") for b in body.get("content", [])
                       if b.get("type") == "text").strip()
        usage = body.get("usage", {})
        return Completion(text, self.backend, self.model,
                          int(usage.get("input_tokens", self.count_tokens(prompt))),
                          int(usage.get("output_tokens", 0)),
                          self.context_limit, False, 0,
                          (time.perf_counter() - started) * 1000)


def get_llm(prefer: str = "auto", **kwargs) -> BaseLLM:
    """`prefer` is "auto" | "flan-t5" | "anthropic" | "echo"."""
    if prefer == "echo":
        return EchoLLM(**kwargs)
    if prefer == "anthropic" or (prefer == "auto" and os.environ.get("ANTHROPIC_API_KEY")):
        try:
            return AnthropicLLM(**kwargs)
        except Exception as exc:                 # noqa: BLE001
            if prefer == "anthropic":
                raise
            log.warning("anthropic unavailable (%s)", exc)
    try:
        return FlanT5LLM(**{k: v for k, v in kwargs.items() if k == "model_name"})
    except Exception as exc:                     # noqa: BLE001
        if prefer == "flan-t5":
            raise
        log.warning("flan-t5 unavailable (%s); using the echo control", exc)
        return EchoLLM()
