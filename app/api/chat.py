"""The Vapi Custom LLM endpoint: ``POST /chat/completions``.

Accepts the OpenAI chat-completions body (plus Vapi's ``call`` object) and
returns an OpenAI-format SSE stream. The per-turn pipeline is:

1. Bind the call id for logging and start the TTFT timer.
2. **Input guard** (regex, <1ms). If blocked, stream a polite Arabic redirect.
3. **RAG** (before opening the stream, ~80ms budget) + load recent history.
4. Open the stream and flush **guarded** sentences as OpenAI delta chunks.
5. After ``[DONE]``, persist the turn to Redis (off the latency path).
"""

from __future__ import annotations

from typing import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.config import settings
from app.core import guards, llm, rag, state
from app.core.prompts import build_messages
from app.schemas.vapi import ChatCompletionRequest
from app.utils import sse
from app.utils.logging import TurnTimer, get_logger, set_call_id

logger = get_logger(__name__)

router = APIRouter()

# Headers that keep proxies/ALB from buffering the SSE stream.
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # disable nginx/ingress buffering
}


async def _text_source(
    req: ChatCompletionRequest,
    user_text: str,
) -> AsyncIterator[str]:
    """Yield the reply text pieces for this turn (guarded).

    When the input guard blocks the turn, this is a single polite redirect;
    otherwise it is the guarded model stream augmented with RAG + history.
    """

    verdict = guards.scan_input(user_text)
    if verdict.blocked:
        logger.info("input blocked: %s", verdict.reason)
        yield settings.blocked_input_reply
        return

    # RAG + history both run BEFORE we open the model stream.
    context = await rag.retrieve_context(user_text)
    history = await state.get_recent_history(req.call_id())
    messages = build_messages(history=history, user_text=user_text, context=context)

    async for piece in llm.stream_guarded(messages):
        yield piece


async def _event_stream(
    req: ChatCompletionRequest,
    user_text: str,
    timer: TurnTimer,
) -> AsyncIterator[str]:
    """Produce the OpenAI-format SSE event stream for one turn."""

    completion_id = sse.new_completion_id()
    model = settings.llm_model
    call_id = req.call_id()

    # Announce the assistant role first (standard OpenAI stream preamble).
    yield sse.role_chunk(completion_id, model)

    assistant_text_parts: list[str] = []
    try:
        async for piece in _text_source(req, user_text):
            if not piece:
                continue
            timer.mark_first_token()
            assistant_text_parts.append(piece)
            yield sse.content_chunk(completion_id, model, piece)
    finally:
        # Always close the stream cleanly so Vapi never hangs on a dead call.
        yield sse.finish_chunk(completion_id, model)
        yield sse.done()

    assistant_text = "".join(assistant_text_parts)
    timer.finish(chars=len(assistant_text))

    # Persist the turn AFTER [DONE] so state I/O never affects TTFT.
    try:
        await state.append_turn(call_id, user_text, assistant_text)
    except Exception:
        logger.warning("failed to persist turn", exc_info=True)


@router.post("/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request) -> StreamingResponse:
    """Vapi Custom LLM entrypoint. Always returns a ``text/event-stream``."""

    call_id = req.call_id()
    set_call_id(call_id)
    user_text = req.last_user_text()
    timer = TurnTimer(logger, call_id)

    logger.info("chat turn start (chars=%d)", len(user_text))

    return StreamingResponse(
        _event_stream(req, user_text, timer),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
    )
