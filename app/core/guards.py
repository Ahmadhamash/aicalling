"""Fast, regex-only guardrails for the live voice path.

Two guards, **no LLM calls anywhere** (a second model on the hot path would
blow the sub-500ms TTFT budget):

* :func:`scan_input` — runs once *before* the stream opens (<1ms). It blocks
  prompt-injection attempts, obviously unsafe requests, and clearly off-topic
  turns. A blocked turn is answered with a polite Arabic redirect.

* :func:`check_output_sentence` — runs *during* the stream, once per completed
  Arabic sentence. It catches fake guarantees, leaked internal/system info, and
  invented numeric prices. The latency cost is one sentence's worth of buffering
  rather than the whole reply.

All patterns are precompiled at import time.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# INPUT GUARD
# --------------------------------------------------------------------------- #

# Prompt-injection / jailbreak attempts, English + Arabic phrasings.
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE | re.UNICODE)
    for p in (
        r"ignore\s+(all\s+)?(your\s+)?(previous\s+)?instructions",
        r"disregard\s+(the\s+)?(above|previous|prior)",
        r"forget\s+(everything|all|your\s+instructions)",
        r"you\s+are\s+now\s+(a|an|no\s+longer)",
        r"act\s+as\s+(a|an|if)",
        r"pretend\s+(to\s+be|you\s+are)",
        r"system\s+prompt",
        r"developer\s+mode",
        r"jailbreak",
        r"reveal\s+(your\s+)?(prompt|instructions|system)",
        r"print\s+(your\s+)?(prompt|instructions|system)",
        # Arabic equivalents
        r"تجاهل\s+(كل\s+)?(تعليمات|التعليمات|الأوامر)",
        r"انسى\s+(كل\s+)?(تعليمات|التعليمات)",
        r"تصرف\s+(كأنك|مثل)",
        r"انت\s+هلأ",
        r"اطبع\s+(تعليمات|التعليمات|البرومبت)",
        r"البرومبت",
        r"نظام\s+المطور",
    )
]

# Unsafe / disallowed content requests (kept intentionally small & high-signal).
_UNSAFE_PATTERNS = [
    re.compile(p, re.IGNORECASE | re.UNICODE)
    for p in (
        r"\b(kill|bomb|explosive|weapon|hack|malware|ransomware)\b",
        r"\bhow\s+to\s+(make|build)\s+(a\s+)?(bomb|weapon|drug)",
        r"قنبلة|متفجرات|سلاح|مخدرات|اخترق|تهكير",
    )
]

# Cheap "off-topic" signal: mentions of unrelated big-tech assistants or
# "write me code / essay" style abuse of a customer-service line.
_OFFTOPIC_PATTERNS = [
    re.compile(p, re.IGNORECASE | re.UNICODE)
    for p in (
        r"\bwrite\s+(me\s+)?(a\s+)?(poem|essay|code|program|script|story)\b",
        r"\b(chatgpt|openai|gpt-?4|language\s+model)\b",
        r"اكتبلي\s+(قصيدة|مقال|كود|برنامج)",
    )
]

BlockReason = str


@dataclass(frozen=True)
class InputVerdict:
    """Result of the input guard scan."""

    blocked: bool
    reason: BlockReason | None = None

    @classmethod
    def ok(cls) -> "InputVerdict":
        return cls(blocked=False)

    @classmethod
    def block(cls, reason: BlockReason) -> "InputVerdict":
        return cls(blocked=True, reason=reason)


def scan_input(text: str) -> InputVerdict:
    """Scan a user utterance. Returns a verdict in well under a millisecond.

    Ordering matters: injection > unsafe > off-topic. Empty input is allowed
    (Vapi occasionally sends empty turns; the model handles them gracefully).
    """

    if not text:
        return InputVerdict.ok()

    for rx in _INJECTION_PATTERNS:
        if rx.search(text):
            return InputVerdict.block("prompt_injection")
    for rx in _UNSAFE_PATTERNS:
        if rx.search(text):
            return InputVerdict.block("unsafe")
    for rx in _OFFTOPIC_PATTERNS:
        if rx.search(text):
            return InputVerdict.block("off_topic")
    return InputVerdict.ok()


# --------------------------------------------------------------------------- #
# OUTPUT GUARD (sentence level)
# --------------------------------------------------------------------------- #

# Arabic + Latin sentence terminators. We split *after* the terminator so the
# punctuation stays attached to the sentence that is spoken.
_SENTENCE_BOUNDARY = re.compile(r"[،.؟!\n]")

# Fake promises / guarantees the agent must never make on its own.
_FAKE_PROMISE_PATTERNS = [
    re.compile(p, re.IGNORECASE | re.UNICODE)
    for p in (
        r"\b(guarantee[d]?|promise|100%|for\s+sure)\b",
        r"مية\s*بالمية|مئة\s*بالمئة|١٠٠\s*٪|100\s*٪",
        r"أضمن(لك| لك)?|بضمن(لك| لك)?|أوعدك|بوعدك|أكيد\s*مية",
    )
]

# Internal / system information that must never leak into the spoken reply.
_LEAK_PATTERNS = [
    re.compile(p, re.IGNORECASE | re.UNICODE)
    for p in (
        r"system\s+prompt|instructions?\s*:|you\s+are\s+نوا",
        r"\b(api[_\s-]?key|sk-[a-z0-9]|bearer\s+|password|secret)\b",
        r"\b(qdrant|redis|openai|embedding|vector|prompt|token)\b",
        r"retrieved\s+context|السياق\s+المسترجع|التعليمات\s+الداخلية",
    )
]

# Invented numeric prices. Because the system prompt forces *spoken* numbers
# (words, not digits), any digit sequence near a currency word is a strong
# signal of an invented/leaked figure and is blocked.
_PRICE_PATTERNS = [
    re.compile(p, re.IGNORECASE | re.UNICODE)
    for p in (
        r"\d+[\d.,]*\s*(دينار|دنانير|jd|jod|شيكل|دولار|\$|٪|%)",
        r"(دينار|دنانير|jd|jod|شيكل|دولار|\$)\s*\d",
        r"[٠-٩]+\s*(دينار|دنانير|شيكل|دولار)",
    )
]


@dataclass(frozen=True)
class OutputVerdict:
    """Result of a sentence-level output check."""

    ok: bool
    reason: BlockReason | None = None


def check_output_sentence(sentence: str) -> OutputVerdict:
    """Validate one completed sentence of model output.

    Returns ``ok=False`` with a reason if the sentence makes a fake promise,
    leaks internal info, or states a numeric price. Runs in microseconds.
    """

    if not sentence.strip():
        return OutputVerdict(ok=True)

    for rx in _FAKE_PROMISE_PATTERNS:
        if rx.search(sentence):
            return OutputVerdict(ok=False, reason="fake_promise")
    for rx in _LEAK_PATTERNS:
        if rx.search(sentence):
            return OutputVerdict(ok=False, reason="info_leak")
    for rx in _PRICE_PATTERNS:
        if rx.search(sentence):
            return OutputVerdict(ok=False, reason="invented_price")
    return OutputVerdict(ok=True)


def find_sentence_boundary(buffer: str) -> int:
    """Return the index just past the first sentence terminator, or ``-1``.

    Used by the streaming output guard to decide when a full sentence has
    accumulated and can be checked + flushed.
    """

    match = _SENTENCE_BOUNDARY.search(buffer)
    return match.end() if match else -1
