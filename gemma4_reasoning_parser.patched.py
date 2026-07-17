# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Local patch over upstream vllm/reasoning/gemma4_reasoning_parser.py.
# Fixes a streaming-mode bug: with Gemma 4's chat template the open token
# <|channel> is emitted as part of the prompt, so the upstream streaming
# parser (which requires start_token_id in previous/delta_token_ids)
# falls through to "all content" and never emits reasoning deltas.
# See gemma4bug.md in the build context for the full bug report.

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tokenizers import TokenizerLike

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

_THOUGHT_PREFIX = "thought\n"


class Gemma4ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for Google Gemma4 thinking models.

    Gemma4 uses <|channel>...<channel|> tokens to delimit reasoning. The
    chat template pre-emits ``<|channel>thought\\n`` at the end of the
    assistant turn header, so in production the start token never appears
    in the generated stream — only the end token ``<channel|>`` does.
    Streaming defaults to "reasoning in progress" to handle this; the
    harness (vllm/parser/abstract_parser.py) only routes deltas here when
    ``is_reasoning_end(prompt_token_ids)`` says reasoning is open, so this
    default never fires for prompts that don't pre-emit the open token.
    """

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self._in_reasoning: bool = True
        self._reasoning_text: str = ""
        self._prefix_stripped: bool = False
        self.new_turn_token_id = self.vocab["<|turn>"]
        self.tool_call_token_id = self.vocab["<|tool_call>"]
        self.tool_response_token_id = self.vocab["<|tool_response>"]

    def adjust_request(
        self, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> "ChatCompletionRequest | ResponsesRequest":
        request.skip_special_tokens = False
        return request

    @property
    def start_token(self) -> str:
        return "<|channel>"

    @property
    def end_token(self) -> str:
        return "<channel|>"

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        start_token_id = self.start_token_id
        end_token_id = self.end_token_id
        new_turn_token_id = self.new_turn_token_id
        tool_call_token_id = self.tool_call_token_id
        tool_response_token_id = self.tool_response_token_id

        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == start_token_id:
                return False
            if input_ids[i] == tool_call_token_id:
                return True
            if input_ids[i] in (new_turn_token_id, tool_response_token_id):
                return False
            if input_ids[i] == end_token_id:
                return True
        return False

    # ------------------------------------------------------------------
    # Non-streaming path (unchanged from upstream)
    # ------------------------------------------------------------------

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        if self.start_token not in model_output and self.end_token not in model_output:
            return None, model_output

        reasoning, content = super().extract_reasoning(model_output, request)
        if reasoning is not None:
            reasoning = _strip_thought_label(reasoning)
        return reasoning, content

    # ------------------------------------------------------------------
    # Streaming path (rewritten)
    # ------------------------------------------------------------------

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        start_id = self.start_token_id
        end_id = self.end_token_id
        tool_call_id = self.tool_call_token_id
        new_turn_id = self.new_turn_token_id
        tool_response_id = self.tool_response_token_id

        # Pure-special-token deltas are the common boundary carriers in
        # streaming (one token per chunk). Drive state off the token id
        # rather than string-matching delta_text, per the bug doc.
        if len(delta_token_ids) == 1:
            tok = delta_token_ids[0]
            if tok == start_id:
                self._reset_span()
                self._in_reasoning = True
                return None
            if tok == end_id:
                self._in_reasoning = False
                return None
            if tok == tool_call_id:
                # Tool-call shortcut: reasoning is over even without
                # an explicit <channel|>. Pass the token's text through
                # as content so downstream tool parsers can see it.
                self._in_reasoning = False
                if delta_text:
                    return DeltaMessage(content=delta_text)
                return None
            if tok in (new_turn_id, tool_response_id):
                # Begin a fresh reasoning span — mirrors is_reasoning_end's
                # handling of these tokens.
                self._reset_span()
                self._in_reasoning = True
                return None

        if not self._in_reasoning:
            # Outside reasoning. Re-entry triggers reset the span state.
            for trig in (start_id, new_turn_id, tool_response_id):
                if trig in delta_token_ids:
                    self._reset_span()
                    self._in_reasoning = True
                    # Conservatively emit any text in this delta as content;
                    # the next delta will carry the start of the new span.
                    if delta_text:
                        return DeltaMessage(content=delta_text)
                    return None
            if delta_text:
                return DeltaMessage(content=delta_text)
            return None

        # In reasoning.

        if tool_call_id in delta_token_ids:
            self._in_reasoning = False
            if delta_text:
                return DeltaMessage(content=delta_text)
            return None

        if end_id in delta_token_ids:
            # End-of-reasoning boundary. The end-token chunk typically has
            # empty delta_text in vLLM streaming, so split-on-string is a
            # best-effort fallback for multi-token deltas.
            if self.end_token in delta_text:
                idx = delta_text.find(self.end_token)
                left = delta_text[:idx]
                right = delta_text[idx + len(self.end_token):]
            else:
                left = ""
                right = ""

            emitted_reasoning = self._consume(left, force_flush=True)
            self._in_reasoning = False
            if not emitted_reasoning and not right:
                return None
            return DeltaMessage(
                reasoning=emitted_reasoning or None,
                content=right or None,
            )

        # Steady-state reasoning, no boundary in this delta.
        if not delta_text:
            return None
        out = self._consume(delta_text, force_flush=False)
        if out is None:
            return None
        return DeltaMessage(reasoning=out)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _reset_span(self) -> None:
        self._reasoning_text = ""
        self._prefix_stripped = False

    def _consume(self, fragment: str, force_flush: bool) -> str | None:
        """Accumulate a reasoning fragment and lazily strip the leading
        ``thought\\n`` role-label prefix.

        Returns the text to emit as reasoning (possibly empty string), or
        ``None`` if buffering should continue.
        """
        if self._prefix_stripped:
            return fragment

        prev_len = len(self._reasoning_text)
        self._reasoning_text += fragment
        prefix_len = len(_THOUGHT_PREFIX)

        if self._reasoning_text.startswith(_THOUGHT_PREFIX):
            self._prefix_stripped = True
            if prev_len >= prefix_len:
                return fragment
            return fragment[prefix_len - prev_len:]

        if _THOUGHT_PREFIX.startswith(self._reasoning_text):
            if force_flush:
                self._prefix_stripped = True
                return self._reasoning_text
            return None

        # Divergence: flush accumulated text once, then pass through.
        self._prefix_stripped = True
        return self._reasoning_text


def _strip_thought_label(text: str) -> str:
    if text.startswith(_THOUGHT_PREFIX):
        return text[len(_THOUGHT_PREFIX):]
    return text
