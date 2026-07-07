"""Pydantic models for the Vapi Custom LLM request contract.

Vapi calls a Custom LLM exactly like the OpenAI ``POST /chat/completions``
endpoint: it sends the standard ``messages[]`` / ``model`` / ``stream`` body,
and additionally embeds a ``call`` object describing the live phone call. We
model the fields we rely on and keep ``extra="allow"`` everywhere so unknown
Vapi/OpenAI fields never cause a 422 on the live path.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class Message(BaseModel):
    """A single chat message in the OpenAI schema."""

    model_config = ConfigDict(extra="allow")

    role: Literal["system", "user", "assistant", "tool"] = "user"
    # ``content`` may be null for tool/assistant messages carrying tool calls.
    content: str | None = None
    name: str | None = None

    def text(self) -> str:
        """Return the message content as plain text (never ``None``)."""

        return self.content or ""


class VapiCall(BaseModel):
    """The subset of Vapi's ``call`` object we need.

    The only field we truly depend on is ``id`` (used to key per-call state in
    Redis). Everything else is preserved but optional.
    """

    model_config = ConfigDict(extra="allow")

    id: str | None = None
    org_id: str | None = Field(None, alias="orgId")
    type: str | None = None
    assistant_id: str | None = Field(None, alias="assistantId")
    customer: dict[str, Any] | None = None
    phone_number: dict[str, Any] | None = Field(None, alias="phoneNumber")


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request as sent by Vapi.

    Lenient by design: unknown fields are accepted and ignored so upstream
    changes in Vapi/OpenAI never break the endpoint.
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    model: str | None = None
    messages: list[Message] = Field(default_factory=list)
    stream: bool = True
    temperature: float | None = None
    max_tokens: int | None = None
    # Vapi-specific: the live call context.
    call: VapiCall | None = None
    metadata: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    def call_id(self) -> str:
        """Best-effort stable identifier for this call.

        Prefers the Vapi ``call.id``; falls back to a metadata id; finally a
        constant sentinel so state operations degrade gracefully instead of
        raising on the live path.
        """

        if self.call and self.call.id:
            return self.call.id
        if self.metadata and isinstance(self.metadata.get("call_id"), str):
            return self.metadata["call_id"]
        return "unknown-call"

    def last_user_text(self) -> str:
        """Return the text of the most recent user message (or empty)."""

        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.text().strip()
        return ""
