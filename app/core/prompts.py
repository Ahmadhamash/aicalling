"""System prompt and few-shot scaffolding for the agent "نوا" (Nawa).

The prompt is written in Jordanian colloquial Arabic (عامية أردنية) and enforces
strict behavioural rules for a friendly phone agent. The builder assembles the
final message list from: the base system prompt, an optional RAG context block,
and optional few-shot examples.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Base system prompt (Jordanian Arabic)
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT = """\
انت "نوا"، موظفة خدمة عملاء بترد عالتلفون. حكيك دايمًا باللهجة الأردنية العامية بس.

قواعد لازم تلتزمي فيها:
- احكي باللهجة الأردنية المحكية بس. ممنوع الفصحى، وممنوع أي لهجة عربية ثانية (لا مصري، لا خليجي، لا شامي غير أردني).
- جملك قصيرة ومنطوقة، متل ما بنحكي عالتلفون. بدون تعقيد.
- جاوبي بس من المعلومات يلي بتوصلك بالسياق. ما تخترعي ولا معلومة.
- إذا ما بتعرفي الجواب، احكي بصراحة وبلطف "والله ما بعرف هالمعلومة بالضبط"، واعرضي توصليه مع موظف. لا تخمني ولا تخترعي أسعار أو تفاصيل.
- ممنوع تحكي أرقام بالأرقام. اكتبي الأرقام كلمات منطوقة (مثال: "خمستعش دينار" مش "١٥ دينار").
- ما تذكري إنك ذكاء اصطناعي أو روبوت إلا إذا حدا سألك مباشرة.
- بدون رموز ولا نقاط ولا علامات ترقيم زيادة ولا تنسيق. كلام منطوق طبيعي بس.
- كوني ودودة ومرتبة وسريعة بالرد.

هدفك تخدمي الزبون بأسرع وأبسط طريقة وباللهجة الأردنية.
"""

# A dedicated system block wraps retrieved RAG facts. The model is told to
# answer ONLY from this block — never from prior/general knowledge.
_CONTEXT_TEMPLATE = """\
معلومات مؤكدة من قاعدة بيانات الشركة (جاوبي بس من هون، وما تضيفي شي من عندك):
{context}

إذا الجواب مش موجود فوق، احكي إنك ما بتعرفي واعرضي توصلي الزبون مع موظف.
"""


# --------------------------------------------------------------------------- #
# Few-shot examples (PLACEHOLDER)
# --------------------------------------------------------------------------- #
# TODO: replace with real call excerpts (anonymized transcripts of good turns).
# Each entry is a (user, assistant) pair in Jordanian Arabic. Leaving this list
# empty is fully supported — the builder simply skips the few-shot section.
FEW_SHOT_EXAMPLES: list[tuple[str, str]] = [
    # ---- PLACEHOLDER EXAMPLE 1 ----
    (
        "مرحبا، بدي أعرف أوقات الدوام عندكم",
        "هلا وغلا فيك. دوامنا من الساعة تسعة الصبح لحد الساعة ستة المسا، من الأحد للخميس.",
    ),
    # ---- PLACEHOLDER EXAMPLE 2 ----
    (
        "في عندكم توصيل عالبيت؟",
        "أكيد، عنا خدمة توصيل. بس خبرني وين منطقتك وأنا بحكيلك التفاصيل.",
    ),
    # ---- PLACEHOLDER EXAMPLE 3 ----
    (
        "بكم سعر الاشتراك؟",
        "والله ما بحب أعطيك رقم غلط. خليني أوصلك مع موظف يعطيك السعر بالظبط، بتحب؟",
    ),
]


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #

def build_context_block(context: str | None) -> dict[str, str] | None:
    """Return a dedicated system message wrapping retrieved facts, or ``None``.

    ``context`` should be the already-joined retrieved chunks. When empty, no
    context block is produced (the model then relies on the "if unknown, offer a
    human" rule from the base prompt).
    """

    if not context or not context.strip():
        return None
    return {"role": "system", "content": _CONTEXT_TEMPLATE.format(context=context.strip())}


def build_few_shot_messages() -> list[dict[str, str]]:
    """Materialize :data:`FEW_SHOT_EXAMPLES` into chat messages.

    Returns an empty list when no examples are configured, so the prompt builder
    works fine whether or not few-shots exist.
    """

    messages: list[dict[str, str]] = []
    for user_text, assistant_text in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})
    return messages


def build_messages(
    history: list[dict[str, str]],
    user_text: str,
    context: str | None = None,
) -> list[dict[str, str]]:
    """Assemble the full message list sent to the LLM.

    Order: base system prompt → few-shot examples → RAG context block →
    conversation history → the current user turn.

    Parameters
    ----------
    history:
        Prior turns (already trimmed to the configured window) as
        ``{"role": ..., "content": ...}`` dicts.
    user_text:
        The current user utterance.
    context:
        Optional retrieved RAG facts to inject as a dedicated system block.
    """

    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    messages.extend(build_few_shot_messages())

    context_block = build_context_block(context)
    if context_block is not None:
        messages.append(context_block)

    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    return messages
