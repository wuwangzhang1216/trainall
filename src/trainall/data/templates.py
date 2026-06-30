"""Chat templating + prompt masking.

A :class:`ChatTemplate` turns a list of :class:`~trainall.types.Message` into
the exact string a base model was trained on (ChatML / Llama-3 / plain), and
can report the character spans of the assistant turns so a tokenizer-aware
caller can build a loss mask that learns only on completions.

:func:`mask_prompt` builds causal-LM ``labels`` from already-tokenised prompt
and response ids, masking the prompt to ``-100`` so the loss ignores it.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

from ..types import Message

__all__ = ["ChatTemplate", "apply_template", "mask_prompt"]


def _to_messages(messages: Sequence[Any]) -> List[Message]:
    out: List[Message] = []
    for m in messages:
        if isinstance(m, Message):
            out.append(m)
        elif isinstance(m, dict):
            out.append(Message(role=str(m.get("role", "user")), content=str(m.get("content", ""))))
        else:
            raise TypeError(f"unsupported message type: {type(m)!r}")
    return out


class ChatTemplate:
    """Renders conversations in a named chat format.

    Supported styles:

    * ``"chatml"``  — ``<|im_start|>role\\n...content...<|im_end|>`` (OpenAI/Qwen).
    * ``"llama3"``  — Llama-3 ``<|start_header_id|>`` / ``<|eot_id|>`` markers.
    * ``"plain"``   — ``Role: content`` lines, generation primed with ``Assistant:``.
    """

    def __init__(self, style: str = "chatml", add_generation_prompt: bool = False) -> None:
        style = style.lower()
        if style not in {"chatml", "llama3", "plain"}:
            raise ValueError(f"unknown chat template style: {style!r}")
        self.style = style
        self.add_generation_prompt = add_generation_prompt

    # -- per-style turn formatting ------------------------------------------ #
    def _open(self, role: str) -> str:
        if self.style == "chatml":
            return f"<|im_start|>{role}\n"
        if self.style == "llama3":
            return f"<|start_header_id|>{role}<|end_header_id|>\n\n"
        return f"{role.capitalize()}: "

    def _close(self) -> str:
        if self.style == "chatml":
            return "<|im_end|>\n"
        if self.style == "llama3":
            return "<|eot_id|>"
        return "\n"

    def _gen_prefix(self) -> str:
        return self._open("assistant")

    def render(self, messages: Sequence[Any]) -> str:
        """Render ``messages`` to the full prompt string."""
        text, _ = self.render_with_mask(messages)
        return text

    def render_with_mask(
        self, messages: Sequence[Any]
    ) -> Tuple[str, List[Tuple[int, int]]]:
        """Render and return ``(text, assistant_spans)``.

        ``assistant_spans`` are ``(start, end)`` character offsets covering the
        *content* of each assistant turn — the supervised region.
        """
        msgs = _to_messages(messages)
        parts: List[str] = []
        spans: List[Tuple[int, int]] = []
        pos = 0

        def emit(s: str) -> None:
            nonlocal pos
            parts.append(s)
            pos += len(s)

        if self.style == "llama3":
            emit("<|begin_of_text|>")

        for m in msgs:
            emit(self._open(m.role))
            if m.role == "assistant":
                start = pos
                emit(m.content)
                spans.append((start, pos))
            else:
                emit(m.content)
            emit(self._close())

        if self.add_generation_prompt:
            emit(self._gen_prefix())

        return "".join(parts), spans


def apply_template(messages: Sequence[Any], template: str = "chatml", **kwargs: Any) -> str:
    """Convenience: render ``messages`` with a freshly built :class:`ChatTemplate`."""
    return ChatTemplate(style=template, **kwargs).render(messages)


def mask_prompt(
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    ignore_index: int = -100,
) -> Tuple[List[int], List[int]]:
    """Concatenate prompt + response ids and build masked causal-LM ``labels``.

    The prompt positions are set to ``ignore_index`` (``-100``) so the loss is
    computed only over the response — the standard completion-only SFT recipe.

    Returns ``(input_ids, labels)`` as plain python lists.
    """
    prompt_ids = list(prompt_ids)
    response_ids = list(response_ids)
    input_ids = prompt_ids + response_ids
    labels = [ignore_index] * len(prompt_ids) + list(response_ids)
    return input_ids, labels
