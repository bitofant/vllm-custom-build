# vLLM gemma4 reasoning parser — streaming bug

> **STATUS: Fixed locally** in `vllm-custom:latest` on 2026-05-23. See
> [Fix](#fix) section at the bottom for what changed and how it was
> verified. The text below is the original bug handover, kept for
> reference; upstream is still broken at HEAD as of the fix date.

Handover for a coding agent fixing a streaming-mode bug in the
`gemma4` reasoning parser. Read the whole document before touching code;
the fix is small but the conditions that surface the bug are subtle.

## TL;DR

The `Gemma4ReasoningParser` works correctly for **non-streaming**
chat completions but emits **no `reasoning` field at all in streaming
mode** — every reasoning token is delivered through `delta.content`
instead. The cause is that the streaming code path inherits the base
parser's logic unchanged, and that base logic assumes the model
generates the reasoning **open** token (`<|channel>`). Gemma 4's
chat template instead places the open token in the **prompt**, so the
streaming parser never sees it in the generated output and falls
through to the "no thinking start token" branch that treats everything
as content.

Both the entire chain of thought and the final answer therefore arrive
in the user-visible content stream, with no marker between them.

## Reproducer

Environment used to confirm:

- vLLM build: `vllm/vllm-openai:gemma4` (`v0.19.1.dev6+g6d4a8e6d2`).
- Model: `RedHatAI/gemma-4-31B-it-NVFP4` (`compressed-tensors`,
  `kv-cache-dtype=fp8`, `max-model-len=175000`).
- Server flags (relevant subset):
  ```
  --reasoning-parser gemma4
  --tool-call-parser gemma4
  --chat-template /etc/vllm/chat-templates/gemma4-force-think.jinja
  --enable-auto-tool-choice
  ```
- Chat template `gemma4-force-think.jinja`, where the
  `add_generation_prompt` block emits the open-thought channel marker
  as part of the prompt:
  ```jinja
  {{- '<|turn>model\n' -}}
  {%- if thinking_on -%}
      {{- '<|channel>thought\n' -}}
  {%- else -%}
      {{- '<|channel>thought\n<channel|>' -}}
  {%- endif -%}
  ```

### Request that exposes the bug

```bash
curl -s -N -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "RedHatAI/gemma-4-31B-it-NVFP4",
        "messages": [{"role":"user","content":"What is 47 * 53? Think step by step."}],
        "stream": true,
        "max_tokens": 600,
        "chat_template_kwargs": {"enable_thinking": true}
      }'
```

### Observed streaming response (abridged)

Every chunk carries `delta.content`. The string `"reasoning"` does not
appear in any chunk of the SSE stream. The reasoning text and the
final answer flow together with no separator at all — for example:

```
… 4. Final result.To solve **47 * 53**, you can use the …
                  ^^ no space, no marker, transition from thought to answer
```

A summarized view of the delta shapes seen across ~600 chunks:

```
"delta":{"role":"assistant","content":""}
"delta":{"content":"The"}
"delta":{"content":" user"}
… hundreds of content deltas …
"delta":{"content":" Final"}
"delta":{"content":" result"}
"delta":{"content":"."}
"delta":{"content":"To"}        ← reasoning ends, answer begins, no marker
"delta":{"content":" solve"}
…
```

`grep -c reasoning` on the SSE stream returns 0.

### Same request with `stream=false` works correctly

```jsonc
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "To solve $47 \\times 53$, …",
        "reasoning": "The user wants to find the product of 47 and 53.\n\n  *  Method 1: …"
      }
    }
  ]
}
```

The non-streaming path populates `message.reasoning` (note: the field
is named `reasoning`, not `reasoning_content`) and keeps the visible
answer in `message.content`. Clients that route reasoning into a
separate UI lane therefore work correctly in non-streaming mode and
silently break in streaming mode.

### Diagnostic: the boundary token IS in the stream, just not surfaced

Re-running with `"skip_special_tokens": false` and `"return_token_ids": true`
shows that the model emits token id `101` (`<channel|>`, the
reasoning-end token) as a separate, empty-content chunk at the exact
boundary between thought and answer:

```
…
"delta":{"content":"."},"token_ids":[236761]
"delta":{},"token_ids":[101]          ← <channel|> arrives here
"delta":{"content":"2"},"token_ids":[236778]   ← "2,491" begins
"delta":{"content":","}
"delta":{"content":"4"}
"delta":{"content":"9"}
…
```

So the information needed to split reasoning from content is present
in the generated token stream. The bug is purely in how
`Gemma4ReasoningParser` interprets it during the streaming phase.

The prompt token IDs decode (with `skip_special_tokens=False`) to:

```
<bos><|turn>system\n<|think|>\n<turn|>\n<|turn>user\nWhat is 47 * 53?<turn|>\n<|turn>model\n<|channel>thought\n
```

confirming that `<|channel>` (token 100) is delivered to the model as
part of the prompt and **never appears in the model's generated
token IDs** — only `<channel|>` (token 101) does, when reasoning ends.

## Why this is a bug

The contract for a reasoning parser is: separate reasoning from
content regardless of whether the client streams. Today the
`gemma4` parser fulfills that contract for the non-streaming path
and silently violates it for the streaming path. Concretely:

1. The Jinja template ships with the open token already attached to
   the assistant turn (`<|turn>model\n<|channel>thought\n`). This is
   the *intended* shape for Gemma 4's thinking mode — the open token
   is treated like a role label, not like content the model has to
   generate. The non-streaming `extract_reasoning` in the gemma4
   parser handles this explicitly: if neither token is present in the
   output it short-circuits, otherwise it lets the base parser split
   on `<channel|>` and then trims a `thought\n` role-label prefix.
2. The streaming `extract_reasoning_streaming` inherited from
   `BaseThinkingReasoningParser` does not have the equivalent
   fallback. Its branches require `start_token_id` to be present in
   `previous_token_ids` or `delta_token_ids`. For Gemma 4 with this
   chat template, neither is ever true: the start token is in the
   prompt, not in the generated output. Every delta therefore falls
   into the final `else` branch and returns
   `DeltaMessage(content=delta_text)`.
3. The `is_reasoning_end` override in the gemma4 parser already
   acknowledges that the start token is sometimes only seen in the
   prompt (it scans backwards looking for either token and is happy
   to return `False` when neither appears) — so the parser is
   internally inconsistent: its end-detection logic is template-aware,
   but its streaming extraction is not.
4. The result is observable user harm: any client wired to surface
   `reasoning` separately from `content` (in OpenAI-compatible UIs,
   agent frameworks, etc.) ends up posting the entire chain of
   thought to end users when streaming. The `--reasoning-parser`
   flag becomes a no-op for streaming and the non-streaming behavior
   makes the regression easy to miss in manual testing.

## Fix hypothesis (conceptual)

Override `extract_reasoning_streaming` in
`vllm/reasoning/gemma4_reasoning_parser.py` so it treats output as
"reasoning by default" until it observes `<channel|>` (the end token)
in the generated stream. The override mirrors the non-streaming
parser's existing fallback policy. In rough pseudocode (not final
code, the implementor should re-derive against the current class):

- Maintain instance state for whether reasoning is still in progress.
  Start `True` for a fresh request — the chat template guarantees
  the model resumes inside the thought channel.
- On each delta:
  - If `end_token_id` is in `delta_token_ids`:
    - Split `delta_text` on the `end_token` string. (When
      `skip_special_tokens=False`, `delta_text` for that chunk is
      typically empty; the split degenerates to two empty halves and
      that's fine.)
    - Emit `DeltaMessage(reasoning=left_half or None,
      content=right_half or None)`. Also account for the `thought\n`
      role-label prefix using the existing
      `_THOUGHT_PREFIX`/`_prefix_stripped` machinery the class
      already carries.
    - Flip the instance flag so subsequent deltas are treated as
      content.
  - Else if reasoning still in progress: return
    `DeltaMessage(reasoning=delta_text)` after the same prefix
    stripping the class already does.
  - Else: return `DeltaMessage(content=delta_text)`.
- Reset the instance flag and `_reasoning_text`/`_prefix_stripped` at
  the start of each new request (the class is reinstantiated per
  request in vLLM, but verify — if it isn't, do it in `adjust_request`).
- Keep `adjust_request` as-is so `skip_special_tokens` stays `False`;
  the end token is what lets the override detect the boundary.
- Special cases worth covering with tests:
  - Tool-call paths: `is_reasoning_end` already treats the
    tool-call token as ending reasoning; the streaming override
    needs the same shortcut so a tool-call sequence flips the flag
    even without a `<channel|>` token.
  - New-turn / tool-response tokens: if they appear mid-stream the
    parser should treat the next deltas as a fresh reasoning span
    (mirrors the existing `is_reasoning_end` behavior).
  - Edge case: the model unexpectedly emits a `<|channel>` start
    token mid-stream. Flip the flag back to "in reasoning" so the
    behavior is symmetric with the prompt-supplied open token.
  - Edge case: chat template not pre-emitting the open token (e.g.
    a future template variant). If the first generated token is
    `<|channel>`, the override should still produce the right
    splits; treating "reasoning in progress" as the default is fine
    because the open token chunk has empty `delta_text` under
    `skip_special_tokens=False`.

## How to validate the fix

1. Run the reproducer above against the rebuilt server.
   Expectation: streaming chunks now contain `"reasoning":"…"`
   deltas while reasoning is in progress, then a single chunk with
   `"reasoning":null, "content":"…"` (or just a `content` delta)
   when the model crosses `<channel|>`, and `content` deltas
   thereafter.
2. Re-run with `stream=false` to confirm the existing non-streaming
   behavior is unchanged.
3. Drive a tool-calling prompt to confirm the parser flips to
   content mode before the `<|tool_call>` token rather than after.
4. Run the existing unit tests for the gemma4 parser and the
   reasoning-parser test suite; add a streaming test covering the
   prompt-supplied open-token case if one is missing — that gap is
   what allowed this bug to ship in the first place.

## Files likely involved

- `vllm/reasoning/gemma4_reasoning_parser.py` — primary fix.
- `vllm/reasoning/basic_parsers.py` — read-only reference for the
  inherited streaming logic; do **not** modify, other parsers
  depend on its current shape (DeepSeek, Seed, etc.).
- Tests under `tests/reasoning/` (path may differ in this checkout)
  — add a streaming regression test that asserts a `reasoning`
  delta is produced when the prompt ends with `<|channel>thought\n`
  and the generated stream contains `<channel|>` but not
  `<|channel>`.

## What NOT to do

- Do **not** "fix" this by mutating the chat template so the model
  has to generate `<|channel>` itself. Gemma 4 was trained to expect
  the open token to be present in the prompt; flipping that is a
  product/training contract change disguised as a bug fix.
- Do **not** widen the base `BaseThinkingReasoningParser` to
  default-on reasoning — other parsers (DeepSeek-R1, Seed-think,
  etc.) intentionally require the model to emit the open token and
  rely on the current "no start token ⇒ content" branch.
- Do **not** rely on `delta_text` containing the literal
  `<channel|>` substring. With `skip_special_tokens=False` the end
  token's text *is* `<channel|>`, but the chunk that carries it has
  `delta.content == ""` in practice — drive the decision off
  `delta_token_ids` containing `self.end_token_id`, not off string
  matching on `delta_text`.

## Fix

Applied 2026-05-23, against upstream `main` HEAD `f0feb15e7fc5`. The
fix lives outside `vllm/` per the vendored-clone rule; it is baked into
the Docker image at build time, not in the upstream tree.

### Approach

The patched parser defaults `_in_reasoning=True` (the harness only
routes deltas to `extract_reasoning_streaming` when
`is_reasoning_end(prompt_token_ids)` says reasoning is open, so this
default never fires for prompts that don't pre-emit the open token).
Boundary detection is driven entirely off `delta_token_ids`, never off
string-matching `<|channel>`/`<channel|>` in `delta_text`. A small
`_consume()` helper retains the existing lazy `thought\n` prefix
stripping. Tool-call, new-turn, and tool-response tokens flip the flag
the same way `is_reasoning_end` already does for the non-streaming
path.

### Files changed

- `gemma4_reasoning_parser.patched.py` (top-level, build context root)
  — full replacement for `vllm/reasoning/gemma4_reasoning_parser.py`.
- `Dockerfile` (builder stage) — `COPY` + `RUN` that overwrites the
  vendored parser before `setup.py bdist_wheel`. The `RUN` is guarded
  by `grep -q 'if result.reasoning is None:' …` — if upstream ever
  lands its own fix (the short-circuit signature disappears), the
  build fails loud and the patch should be deleted.
- `Dockerfile` (runtime stage) — `pip install
  humming-kernels[cu13]==0.1.0 tokenspeed-mla==0.1.2 tilelang==0.1.9
  fastsafetensors==0.2.2`. Unrelated to the parser fix but blocked
  the rebuild: `humming` is eagerly imported from the quant registry,
  so any model with a quantization config (gemma4 NVFP4 included)
  crashes at startup if it's missing.

Nothing in `vllm/` was edited directly.

### Verification

Reproducer from the "Request that exposes the bug" section above, run
against `RedHatAI/gemma-4-31B-it-NVFP4` on `vllm-custom:latest` with
the gemma4-force-think chat template:

| | Before (`vllm/vllm-openai:gemma4`) | After (`vllm-custom:latest`) |
|---|---|---|
| `grep -c '"reasoning":'` | 0 | 433 |
| `grep -c '"delta":\{"content":'` | 301 (everything) | 95 |
| `finish_reason` | length | stop |
| Boundary | — | clean split at `<channel|>` |

Streaming chunks during the chain of thought now carry
`"reasoning":"…"`, the model crosses `<channel|>`, and subsequent
chunks carry `"content":"…"`. Non-streaming behavior is unchanged
(`message.reasoning` populated, `message.content` is the visible
answer).

### Side effects on upstream tests

The original test suite at
`tests/reasoning/test_gemma4_reasoning_parser.py` exercises the
working code path only — it encodes `<|channel>` into the *generated*
output via `gemma4_encode_output`. Three streaming cases encoded the
broken behavior as their expected value and now diverge from it:

- `INVALID_SIMPLE_STREAMING` — was `reasoning=None, content=<everything>`;
  now matches `INVALID_SIMPLE_NONSTREAMING`
  (`reasoning="This is a reasoning section", content="This is the rest"`).
- `INVALID_COMPLETE_STREAMING` — same pattern; now matches its
  non-streaming sibling.
- `NEW_LINE_STREAMING` — pre-`<|channel>` text is now treated as
  reasoning instead of content, matching the non-streaming behavior.

These tests don't ship in the runtime wheel, so they don't break
production. The fix makes streaming and non-streaming behavior
*consistent*, which is the contract this bug doc argues for; the
broken-baseline test expectations would need updating before any
upstream PR.
