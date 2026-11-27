"""LLM boundary — the seam that keeps traces reproducible.

Two rules:

1. A node never talks to a provider SDK directly; it calls :class:`LLM`.
2. Every LLM call is *recorded* into the trace as a ``node_token`` stream plus
   its final text, so a run can be replayed without the network.

That is what makes ``MockLLM`` useful beyond testing: a recorded real run and a
mock run produce the same trace shape, so the lens cannot tell them apart.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Protocol


@dataclass(slots=True)
class Message:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass(slots=True)
class LLMResponse:
    text: str
    model: str
    #: True when this came from a mock — the lens badges these differently.
    mocked: bool = False
    usage: dict[str, int] = field(default_factory=dict)


class LLM(Protocol):
    name: str

    def complete(self, messages: list[Message], *, on_token: Callable[[str], None] | None = None) -> LLMResponse: ...


# --------------------------------------------------------------------------
# Mock LLM — deterministic, offline, and scriptable
# --------------------------------------------------------------------------
@dataclass(slots=True)
class ScriptedRule:
    """If any message contains ``contains``, reply with ``reply``."""

    contains: str
    reply: str


class MockLLM:
    """Deterministic stand-in with three modes, checked in this order:

    1. explicit ``script`` rules (substring match on the joined prompt)
    2. ``replies`` FIFO (pop one per call) — lets a script drive a multi-turn loop
    3. ``default`` reply

    Token streaming is simulated by chunking the reply, so ``node_token``
    events appear exactly as they would with a real provider.
    """

    def __init__(
        self,
        *,
        name: str = "mock-llm",
        script: list[ScriptedRule] | None = None,
        replies: list[str] | None = None,
        default: str = "ok",
        chunk_size: int = 12,
        delay_s: float = 0.0,
    ) -> None:
        self.name = name
        self.script = script or []
        self.replies = list(replies or [])
        self.default = default
        self.chunk_size = max(1, chunk_size)
        self.delay_s = delay_s
        self.calls: list[list[Message]] = []

    def _pick(self, prompt: str) -> str:
        for rule in self.script:
            if rule.contains in prompt:
                return rule.reply
        if self.replies:
            return self.replies.pop(0)
        return self.default

    def complete(
        self,
        messages: list[Message],
        *,
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        self.calls.append(list(messages))
        prompt = "\n".join(m.content for m in messages)
        text = self._pick(prompt)
        if on_token is not None:
            for i in range(0, len(text), self.chunk_size):
                if self.delay_s:
                    time.sleep(self.delay_s)
                on_token(text[i : i + self.chunk_size])
        return LLMResponse(text=text, model=self.name, mocked=True)


class RecordingLLM:
    """Wrap any LLM and cache responses by prompt hash under ``cache_dir``.

    Ship a cache file with the repo and your examples replay offline at zero
    cost — while still being *real* model output.
    """

    def __init__(self, inner: LLM, cache_dir: str = "llm_cache") -> None:
        self.inner = inner
        self.name = getattr(inner, "name", "llm")
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    @staticmethod
    def _key(messages: list[Message], model: str) -> str:
        payload = json.dumps(
            {"model": model, "messages": [m.to_dict() for m in messages]},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]

    def complete(
        self,
        messages: list[Message],
        *,
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        key = self._key(messages, self.name)
        path = os.path.join(self.cache_dir, f"{key}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            text = raw["text"]
            if on_token:
                for i in range(0, len(text), 12):
                    on_token(text[i : i + 12])
            return LLMResponse(
                text=text,
                model=raw.get("model", self.name),
                # Not mocked: this is a real model response being replayed. Only
                # the words "no model was called *this time*" differ, and the
                # lens badges a mocked response as "not a real model" — which
                # would be a lie here.
                mocked=False,
                usage={"cached": 1},
            )
        response = self.inner.complete(messages, on_token=on_token)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"text": response.text, "model": response.model}, fh, ensure_ascii=False)
        return response


class OpenAICompatibleLLM:
    """Adapter for any OpenAI-compatible ``/chat/completions`` endpoint.

    Uses only the standard library so the kernel has zero dependencies —
    which matters for a project whose point is to be readable.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        name: str | None = None,
        temperature: float = 0.0,
        timeout_s: float = 60.0,
    ) -> None:
        self.model = model
        self.name = name or model
        self.base_url = (base_url or os.environ.get("LENS_LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key or os.environ.get("LENS_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.temperature = temperature
        self.timeout_s = timeout_s
        if not self.api_key:
            raise RuntimeError(
                "no API key: set LENS_LLM_API_KEY (or OPENAI_API_KEY), "
                "or use MockLLM instead"
            )

    def complete(
        self,
        messages: list[Message],
        *,
        on_token: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        import urllib.request

        body = json.dumps(
            {
                "model": self.model,
                "messages": [m.to_dict() for m in messages],
                "temperature": self.temperature,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        text = raw["choices"][0]["message"]["content"] or ""
        if on_token:
            for i in range(0, len(text), 24):
                on_token(text[i : i + 24])
        usage = raw.get("usage") or {}
        return LLMResponse(
            text=text,
            model=raw.get("model", self.model),
            mocked=False,
            usage={k: v for k, v in usage.items() if isinstance(v, int)},
        )


def load_llm(spec: str | None = None, **kwargs: Any) -> LLM:
    """Build an LLM from a short spec: ``mock`` | ``openai:<model>`` | ``auto``.

    ``auto`` prefers a recorded cache, then a real provider if a key exists,
    then falls back to the mock. Printed once by the CLI so the user always
    knows whether they are seeing real model output.
    """
    spec = (spec or os.environ.get("LENS_LLM") or "mock").strip()
    if spec == "mock":
        return MockLLM(**kwargs)
    if spec.startswith("openai:"):
        return OpenAICompatibleLLM(model=spec.split(":", 1)[1], **kwargs)
    if spec == "auto":
        if os.environ.get("LENS_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"):
            try:
                return OpenAICompatibleLLM(
                    model=os.environ.get("LENS_LLM_MODEL", "gpt-4o-mini"), **kwargs
                )
            except RuntimeError:
                pass
        return MockLLM(**kwargs)
    raise ValueError(f"unknown LLM spec {spec!r}; use mock | auto | openai:<model>")
