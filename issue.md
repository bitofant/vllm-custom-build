# Engine livelocks after period of normal use: requests accumulate in waiting queue, scheduler never schedules them, no errors emitted

## Summary

After a fresh container start, vLLM serves the model correctly. After some minutes to hours of normal traffic (a mix of short chats, streamed responses, and tool-use conversations), inference endpoints (`/v1/chat/completions`, `/v1/completions`) stop returning. Requests accumulate in the scheduler's waiting queue but are never moved to running, even though KV cache is 0% used and no errors are logged. Non-inference endpoints (`/health`, `/v1/models`, `/metrics`) continue to respond normally. A container restart fully recovers; the same probe battery that passes on a fresh container hangs after the wedge.

## Environment

| Item | Value |
|------|-------|
| vLLM | `0.19.1.dev0+g2a69949bd.d20260418` |
| Base image | `nvcr.io/nvidia/vllm:26.01-py3` (custom rebuild from source) |
| Python | 3.12.3 |
| Model | `cyankiwi/gemma-4-31B-it-AWQ-4bit` |
| Resolved architecture | `Gemma4ForConditionalGeneration` |
| Quantization | `compressed-tensors` |
| GPU | RTX 5090 (Blackwell, sm_120), 32 GiB |
| Driver | 580.105.08 |
| Clients in use | OpenWebUI, OpenClaw (both via host port 8000) |

Launch flags:

```
vllm serve cyankiwi/gemma-4-31B-it-AWQ-4bit \
  --quantization compressed-tensors \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.915 \
  --dtype auto \
  --enable-prefix-caching \
  --kv-cache-dtype fp8 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4
```

Engine reports at startup:

```
Gemma4 model has heterogeneous head dimensions (head_dim=256, global_head_dim=512).
  Forcing TRITON_ATTN backend to prevent mixed-backend numerical divergence.
Available KV cache memory: 7.65 GiB
GPU KV cache size: 16,688 tokens
Maximum concurrency for 65,536 tokens per request: 2.08x
Asynchronous scheduling is enabled.
init engine (profile, create kv cache, warmup model) took 69.27 seconds
```

## Observed behavior

### Before the hang (fresh container)

An 11-test probe battery passed cleanly:

| Test | Description | Elapsed | Bytes | TTFB |
|------|-------------|---------|-------|------|
| health | `GET /health` | <10 ms | 0 | 1 ms |
| models | `GET /v1/models` | <10 ms | 499 | 1 ms |
| simple | Short chat, non-stream | 0.90 s | 587 | 0.90 s |
| simple_stream | Short chat, stream | 0.14 s | 2468 | 6 ms |
| tools | Short chat + tool def | 0.24 s | 720 | 0.24 s |
| long_1k | ~1k-token prompt | 0.34 s | 598 | 0.33 s |
| long_4k | ~4k-token prompt | 0.91 s | 600 | 0.91 s |
| long_10k | ~10k-token prompt | 1.69 s | 600 | 1.69 s |
| long_16k | ~16k-token prompt | 1.81 s | 602 | 1.80 s |
| long_20k | ~20k-token prompt (over KV cache size) | 1.31 s | 602 | 1.31 s |
| tools_long_4k | ~4k-token prompt + tool def | 1.24 s | 724 | 1.23 s |

`tools` and `tools_long_4k` returned a structurally correct tool call for the `gemma4` parser:

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [{
    "id": "chatcmpl-tool-...",
    "type": "function",
    "function": { "name": "get_weather", "arguments": "{\"city\": \"Paris\"}" }
  }]
}
```

### After the hang

- `POST /v1/chat/completions` with payload `{"messages":[{"role":"user","content":"hi"}],"max_tokens":3}` accepts the TCP connection, reads the full body, and never writes any response bytes. A 120-second curl produced zero bytes received.
- `GET /health` returns in ~4 ms.
- `GET /v1/models` returns full JSON in ~4 ms.
- `GET /metrics` returns in ~4 ms.
- `docker logs` shows no error, warning, or traceback during the hang. `INFO` throughput logs from `[loggers.py:259]` stop being emitted while the engine is wedged, though they resume when new probe requests arrive (same line, 0 tokens/s, `Running: 0 reqs, Waiting: N reqs`).
- Client abort (curl timeout, closed connection) does not clear the request from the waiting queue; the count monotonically increases as more clients hit the wedged server.

## Observed state during the hang

### Metrics snapshot (from `/metrics`)

```
vllm:num_requests_running          = 0
vllm:num_requests_waiting          = 3
vllm:num_preemptions_total         = 0
vllm:kv_cache_usage_perc           = 0.0
vllm:prefix_cache_queries_total    = 6.1968266979e+10    # ~62 billion
vllm:prefix_cache_hits_total       = 55520
vllm:request_success_total{stop}   = 19
vllm:request_success_total{length} = 0
vllm:request_success_total{abort}  = 0                   # despite many client disconnects
vllm:request_success_total{error}  = 0
vllm:iteration_tokens_total_bucket{le=1.0}  = 3215
vllm:iteration_tokens_total_bucket{le=8.0}  = 3349
vllm:iteration_tokens_total_bucket{le=16.0} = 3349
# i.e. 96% of engine iterations scheduled ≤1 token
```

Note the queries/hits ratio: 62e9 queries vs 55k hits (hits happened during normal pre-hang work; queries continue to increment while wedged).

### Process state (from `ps` inside the container)

```
  PID  PPID COMMAND         STAT %CPU %MEM
    1     0 vllm            Ssl   1.4  2.3
  434     1 python          S     0.0  0.0
  435     1 VLLM::EngineCor Sl    7.7  4.1
```

EngineCore has 154 threads; thread-state histogram by `wchan`:

```
  137 futex_wait_queue
   11 ep_poll
    4 do_poll.constprop.0
    2 hrtimer_nanosleep
```

No threads in uninterruptible (`D`) state; no kernel-level deadlock.

### GPU state

```
memory.used   = 31110 MiB / 32607 MiB   (95.4%)
utilization   = 0% (GPU idle)
```

### TCP state on port 8000

```
LISTEN  0 4096   0.0.0.0:8000
ESTAB   0    0   172.17.0.1:55002   172.17.0.3:8000   # host -> vllm
ESTAB   0    0   172.17.0.1:8000    172.17.0.2:59206  # owui -> vllm (via host DNAT)
ESTAB   0    0   172.17.0.1:58796   172.17.0.3:8000
ESTAB   0    0   172.17.0.1:8000    172.17.0.2:46248
```

All `Recv-Q` / `Send-Q` are 0. The server has consumed request bodies but has no bytes pending to send.

### py-spy dumps (captured with sidecar container, `--cap-add SYS_PTRACE`, `--pid container:<name>`)

**APIServer (PID 1), 8 rapid samples during a hung request — every sample identical:**

```
Thread 1 (idle): "MainThread"
    run (asyncio/runners.py:118)
    run (asyncio/runners.py:194)
    run (uvloop/__init__.py:96)
    cmd (vllm/entrypoints/cli/serve.py:122)
    main (vllm/entrypoints/cli/main.py:75)
    <module> (vllm:6)
Thread 1712 (idle): "MPClientEngineMonitor"
    select (selectors.py:415)
    wait (multiprocessing/connection.py:1136)
    monitor_engine_cores (vllm/v1/engine/core_client.py:659)
Thread 1713 (idle): "ThreadPoolExecutor-2_0"
    _worker (concurrent/futures/thread.py:89)
Thread 2365 (idle): "Thread-1"
    run (zmq/utils/garbage.py:46)
```

No handler coroutine stack is ever observed on the APIServer; the main thread sits in the uvloop run. (Note: py-spy cannot show parked asyncio coroutines, so a handler awaiting a future is not necessarily visible.)

**EngineCore (PID 435), 8 rapid samples — 7 identical idle, 1 mid-schedule:**

7 / 8 samples:

```
Thread 435 (idle): "MainThread"
    _process_engine_step (vllm/v1/engine/core.py:1193)
    run_busy_loop       (vllm/v1/engine/core.py:1142)
    run_engine_core     (vllm/v1/engine/core.py:1101)
```

1 / 8 samples:

```
Thread 435 (idle): "MainThread"
    find_longest_cache_hit (vllm/v1/core/single_type_kv_cache_manager.py:531)
    find_longest_cache_hit (vllm/v1/core/kv_cache_coordinator.py:514)
    get_computed_blocks    (vllm/v1/core/kv_cache_manager.py:203)
    schedule               (vllm/v1/core/sched/scheduler.py:613)
    step_with_batch_queue  (vllm/v1/engine/core.py:449)
    _process_engine_step   (vllm/v1/engine/core.py:1181)
    run_busy_loop          (vllm/v1/engine/core.py:1142)
    run_engine_core        (vllm/v1/engine/core.py:1101)
```

All other EngineCore threads idle (zmq `poll` on input socket, `queue.get` on output socket, `concurrent.futures` workers, tqdm monitors, usage reporter).

### strace on APIServer (PID 1) for 3 s during the hang

Syscall histogram:

```
 % time      calls syscall
 36.33       5366  epoll_pwait
 30.52      16119  getpid
 27.47      13437  poll
  5.68       2682  read
 ------------------------
100.00      37604  total
```

Zero `accept*`, `recv*`, `send*`, `write` to network sockets. A detailed trace shows a repeating pattern at roughly 5 kHz:

```
[pid 431] recvfrom(48, "...num_running_reqs...", 8192, 0, NULL, NULL) = 424
[pid 431] write(34, "\x01\x00\x00\x00\x00\x00\x00\x00", 8) = 8
[pid   1] read(34, "\x01\x00\x00\x00\x00\x00\x00\x00", 8) = 8
[pid 431] epoll_wait(31, ...)
```

(i.e. engine stats messages from EngineCore → APIServer, delivered via eventfd wakeup; main loop reads the wakeup and returns to epoll with no further work.)

### Other observations

- `/proc/<pid>/stack` returns `Operation not permitted` inside the running container (no `SYS_PTRACE` capability). Stacks were obtained by spawning a sidecar with `--cap-add SYS_PTRACE --pid container:<name>`.
- `/proc/<pid>/syscall` is similarly blocked.
- `_process_engine_step` at `core.py:1193` followed by the `time.sleep(0.001)` path described at `core.py:1189-1193` is consistent with a request existing in `has_unfinished_requests()` but `model_executed` never becoming `True`.

## Reproduction

1. Start a vLLM container with the flags above on the listed hardware.
2. Run the included probe (`gemma_probe.sh`) — all 11 tests pass.
3. Use OpenWebUI and OpenClaw against the server for an indefinite period (chat, streamed responses, tool-use conversations, client disconnects). Exact trigger has not been isolated; the wedge appeared within ~30 min of light real-world use in our case.
4. Re-run the probe: it hangs on the first inference request, and the stack / metrics match those shown above.

A container restart (`docker restart <container>`) returns the server to full health, at which point step 2 reproduces the issue again.

---

## Hypotheses (not yet verified)

These are ordered by our current confidence, but none has been conclusively tested against a patched binary.

1. **Prefix-cache state corruption.** The only live stack frame ever observed in EngineCore during the hang is `find_longest_cache_hit` called from `schedule()` via `get_computed_blocks`. 62 billion prefix-cache queries is consistent with a tight schedule-then-break loop where the prefix-cache path runs on every waiting request on every engine tick. Prefix caching is the most complex of the enabled features and a natural candidate for state corruption after abnormal request termination.

2. **Abort-handling leak.** `vllm:request_success_total{abort} = 0` despite many observed client timeouts and disconnections. This suggests that when a client disconnects, the request is not being transitioned to the aborted state; it stays in the waiting queue and (hypothetically) retains block-manager allocations or metadata that poison later scheduling decisions. This may be the root cause for (1).

3. **A `break` condition in `Scheduler.schedule()` that flips to always-true after some state transition.** The hung scheduler repeatedly reaches `get_computed_blocks` and then presumably hits one of the `break` statements further down (`chunked_prefill` budget check, `can_fit_full_sequence`, `allocate_slots is None`, or encoder input scheduling). Which one has not been determined from py-spy snapshots alone — a debug log at each break site would distinguish them.

4. **Gemma4 heterogeneous-head-dim path + `fp8` KV cache.** The model forces `TRITON_ATTN` because of `head_dim=256` vs `global_head_dim=512`. This path is newer and the interaction with `--kv-cache-dtype fp8` may be under-tested. This is speculative; we do not have evidence pointing specifically at the attention backend.

5. **A livelock compounded by the deliberate `time.sleep(0.001)` in `_process_engine_step`.** The comment at `core.py:1188-1193` describes that sleep as "yield the GIL briefly to allow background threads ... to make progress." In our case it allows the engine to burn 1 kHz of wasted scheduler iterations indefinitely without producing output or hitting a timeout. This is a symptom-amplifier rather than a cause, but contributes to the 62-billion-query metric.

## Follow-up: prefix caching ruled out as root cause

We re-ran the same setup with `--enable-prefix-caching` removed. The wedge reproduced. Observations differ from the first wedge in one important way:

- `vllm:num_requests_waiting = 0` (was `3`)
- `vllm:num_requests_running = 0`
- `vllm:kv_cache_usage_perc = 0.0`
- `vllm:request_success_total{abort} = 0` (unchanged)
- `vllm:prefix_cache_hits_total = 0` (expected; flag removed)
- EngineCore idle stack is now `_process_input_queue` at `core.py:1162` blocking on `queue.get()` (no more busy-loop in `_process_engine_step`) — the engine is waiting for input that never arrives.

In other words, without prefix caching the engine no longer busy-loops on un-schedulable waiting requests, but requests submitted to the server during the wedged period never reach the engine queue at all. The wedge also recovered on its own between the probe detecting it and our follow-up diagnostics — subsequent requests (including 20-way concurrent load) returned 200 in ~3.5 s each with no issue.

This shifts suspicion from the scheduler to the APIServer → EngineCore zmq path and/or the HTTP handler's interaction with the async engine client.

## What we have not yet tried (pre-Round 2)

- Running without `--kv-cache-dtype fp8`.
- Running with the v0 engine.
- Patching `Scheduler.schedule()` with a `logger.debug` at each `break` to identify which condition triggered during the first wedge.
- Instrumenting the APIServer → EngineCore zmq channel to verify whether wedged-period requests ever hit `process_input_sockets` in the engine.
- Testing with a different model of the same family (e.g. a non-Gemma4 model) under the same client mix, to isolate whether the bug is Gemma4-specific or a general lifecycle issue.

## Attachments we can provide

- Full `docker logs` from startup through the wedge (no errors).
- Full py-spy dumps of both processes.
- Raw `strace -c` and timeline output.
- Probe script (`gemma_probe.sh`) and its per-test artifacts (response bodies, GPU samples, log tails).
- Metrics snapshot.

---

## Round 2 — 2026-04-20

**Context:** Updated to vLLM latest main branch; bug persists. Full source code audit of the relevant paths.

### Source code audit — key file locations

All paths relative to `vllm/vllm/v1/`:

| File | Key function / lines |
|------|----------------------|
| `engine/core.py` | `run_busy_loop` ~1160, `_process_input_queue` ~1170, `_process_engine_step` ~1201 |
| `engine/core_client.py` | `AsyncMPClient.abort_requests_async` ~1063, `AsyncMPClient._send_input` ~1001 |
| `engine/async_llm.py` | `generate()` ~523, `add_request()` ~282, `_add_request()` ~399, `abort()` ~706 |
| `core/sched/scheduler.py` | `schedule()` ~348 — four break points listed below; `set_pause_state` ~1849 |
| `core/kv_cache_manager.py` | `can_fit_full_sequence` ~218, `allocate_slots` ~257 |
| `core/block_pool.py` | `get_num_free_blocks` ~478, `get_usage` ~486 |
| `entrypoints/utils.py` | `with_cancellation` ~56, `listen_for_disconnect` ~41 |

### Scenario A (with prefix caching): scheduler break analysis

The scheduler's waiting-request loop (`scheduler.py:567`) calls `get_computed_blocks` on each peek (line 613) and then evaluates four sequential break/continue conditions before reaching `allocate_slots`:

| # | Condition | Approx line | Notes |
|---|-----------|-------------|-------|
| B1 | `not enable_chunked_prefill and num_new_tokens > token_budget` | ~681 | `token_budget = max_num_batched_tokens`; short prompts shouldn't hit this |
| B2 | Mamba block alignment → `num_new_tokens == 0` | ~716 | Not a Mamba model; skip |
| B3 | `scheduler_reserve_full_isl and not can_fit_full_sequence(...)` | ~741 | **Default is `True`** (`config/scheduler.py:140`) |
| B4 | `allocate_slots(...) is None` | ~771 | Block pool returns None if `num_blocks_to_allocate > get_num_free_blocks()` |

**Break B3 — `scheduler_reserve_full_isl`:** This flag defaults `True`. `can_fit_full_sequence()` checks `num_blocks_to_allocate <= block_pool.get_num_free_blocks()`, where `num_blocks_to_allocate` is derived from `min(request.num_tokens, max_model_len)`. Since `request.num_tokens` is the *current* prompt length (not max_tokens), this should be small and the check should pass for typical prompts. **Not yet confirmed as trigger.**

**`kv_cache_usage_perc = 0.0` vs. allocation failure:** `get_usage()` = `1 - free_blocks/total_blocks` (`block_pool.py:497`). Both B3 and B4 use the same `get_num_free_blocks()` call from the same shared block pool. If usage is truly 0%, both should pass — *unless* `get_num_blocks_to_allocate()` returns a pathologically large value due to a bug in the coordinator's per-request block accounting.

**`prefix_cache_queries` counter is a symptom, not a cause:** `PrefixCacheStats.record()` adds `num_tokens` (not 1) per scheduling attempt (`metrics/stats.py:141`). At 1 kHz busy-loop with 3 waiting requests of ~1 k tokens each → ~3 M tokens/sec → 60 B tokens over ~20 ks. This fully explains the counter without requiring any separate bug.

**`PauseState` is not active during Scenario A:** Under `PAUSED_NEW`, the waiting loop is skipped (`scheduler.py:564`), but `has_unfinished_requests()` would return 0, so no `time.sleep(0.001)` → no busy-loop. Under `PAUSED_ALL`, same. Since Scenario A *shows* a busy-loop, `_pause_state` must be `UNPAUSED`.

### Scenario B (without prefix caching): API-server → EngineCore request gap

The EngineCore sits at `queue.get(block=True)` in `_process_input_queue` — requests never arrive. The strace from Scenario A confirms no `send*` syscalls from PID 1 during the hang: the APIServer never calls `_send_input()`.

Candidate blocking points between HTTP body receipt and `_send_input()`:

1. **`await self.get_supported_tasks()`** — cached after first call; near-zero latency after warmup.
2. **`await _send_input()` → zmq `send_multipart` on ROUTER socket** — returns an awaitable that resolves when zmq has buffered the message. If the ROUTER socket's SNDHWM (default 1000 messages) is hit because `process_input_sockets` on the EngineCore is not draining fast enough, this awaitable stalls indefinitely. **Self-recovery** (Scenario B cleared on its own) is consistent with the SNDHWM backlog draining once request pressure subsided.
3. **`run_engine_stats_update_task` at 5 kHz** — yields on every `await poller.poll()` but consumes significant event-loop slots. Unlikely to fully starve other coroutines, but may contribute to transient congestion.

### Abort metric explanation

`request_success_total{abort} = 0` is **not necessarily evidence of broken abort handling** on the single-GPU `AsyncMPClient` path. The abort path (`async_llm.py:714-715`) is structurally correct. The `{abort}` label counter is only incremented when a request finishes with `FINISHED_ABORTED` status *after* passing through the output processor — not for requests aborted while still in WAITING state. Requests stuck in WAITING and removed by an abort call exit silently from the metric's perspective. This is a **counter design gap**, not a functional abort-handling bug.

(The `DPLBAsyncMPClient.abort_requests_async` silent-drop bug noted in earlier analysis applies only to multi-engine data-parallel setups, not the single-GPU configuration in use.)

### Updated hypotheses (ordered by confidence)

1. **Break B3 or B4 is the proximate cause of Scenario A.** The scheduler enters the waiting loop, calls `get_computed_blocks`, then hits the `can_fit_full_sequence` check (B3, `scheduler_reserve_full_isl=True`) or the `allocate_slots → None` check (B4). Adding `logger.debug` at each break site will identify which one fires. **Highest priority action.**

2. **zmq ROUTER SNDHWM backpressure is the proximate cause of Scenario B.** When the EngineCore is in a Scenario A busy-loop (input socket not being drained), the ROUTER socket's send buffer fills and `await _send_input()` stalls indefinitely. Self-recovery happens when buffer drains. Fix: raise SNDHWM, or verify `process_input_sockets` always drains the socket even during scheduler busy-loops.

3. **Gemma4 heterogeneous KV cache groups may produce an asymmetric block-allocation failure.** If `KVCacheCoordinator.get_num_blocks_to_allocate()` sums blocks needed across BOTH local and global attention groups (head_dim 256 and 512), but `get_usage()` / `get_num_free_blocks()` only reports one group's pool, the allocation check can fail even when reported usage is 0%. Needs verification: is there one shared block pool or separate pools per KV cache group?

4. **`request_success_total{abort} = 0` is a counter design gap** — pre-execution aborts are not counted. Lower confidence that fixing abort handling will fix the livelock.

5. **`--enable-prefix-caching` is not the root cause** (confirmed). It amplifies Scenario A into a high-query busy-loop.

### Additional findings (Round 2, session 2)

**zmq SNDHWM = 0 (unlimited):** `make_zmq_socket` in `vllm/utils/network_utils.py:315` unconditionally sets `SNDHWM=0` on ROUTER sockets. zmq backpressure is **ruled out** as the cause of Scenario B — the send is fire-and-forget.

**Scenario B revised:** since `_send_input()` with SNDHWM=0 should be near-instant, the zero `send*` syscalls in the strace indicate the Python code never reached `add_request_async()`. Suspect: the `await preprocess_chat(...)` call in `render_chat_request` (serves as tokenization) blocking the event loop, combined with the 5 kHz stats update task starving new coroutines. Self-recovery is consistent with transient event-loop congestion. Low priority vs. Scenario A.

**Scheduler instrumented (done):** added `_warn_break_once()` helper and `_break_warned` set to `Scheduler.__init__` in `core/sched/scheduler.py`. On the next hang, grep docker logs for `[LIVELOCK-PROBE]` to immediately see which break condition (B1–B4) is firing. Logs once per (request_id, break_type) pair to avoid flooding during the busy-loop.

**`--no-scheduler-reserve-full-isl` flag confirmed:** uses `BooleanOptionalAction` — pass `--no-scheduler-reserve-full-isl` on the vllm serve command line to disable B3. If the hang clears, B3 is the cause.

### What to try next (Round 2)

- [ ] **Reproduce the hang with the instrumented build** and grep for `[LIVELOCK-PROBE]` in docker logs to identify the exact break condition.
- [ ] **Quick parallel test:** add `--no-scheduler-reserve-full-isl` to the vllm serve command. If Scenario A does not occur, Break B3 (`can_fit_full_sequence`) is confirmed.
- [ ] **Verify whether Gemma4 uses one or two block pools** by inspecting `KVCacheCoordinator` and checking what `kv_cache_manager.usage` reports vs. actual block pool state during the hang.
- [ ] Running without `--kv-cache-dtype fp8` (unchanged from prior list).
- [ ] Testing with a non-Gemma4 model under the same client mix.

---

## Round 3 — 2026-05-22

**Context:** Pulled 949 upstream commits onto `main`. The local `[LIVELOCK-PROBE]` instrumentation from Round 2 was discarded before the fast-forward; if needed, it will be re-applied against the new `scheduler.py`. One upstream commit looked superficially relevant to the investigation and was audited.

### Upstream commit audit: `f34623bf3ca`

`f34623bf3ca [bug] AsyncScheduler drops first post-resume token after pause_generation + clear_cache (#42117)` — matched the pause/clear_cache keywords from earlier hypotheses, so checked in detail.

**What it changes:** `Scheduler.reset_prefix_cache(reset_running_requests=True)` previously set `request.discard_latest_async_tokens = True` (a boolean) when force-preempting in-flight requests. With async scheduling, multiple output frames can be in flight per request (spec decode, pipeline parallel), so the single boolean undercounted. Fix replaces it with `request.async_tokens_to_discard = request.num_output_placeholders` (an integer counter) so all stale frames are discarded as they return. Touches `core/sched/scheduler.py:1905-1915`, `core/sched/async_scheduler.py:37-46`, `request.py:139-142`.

**Why it is NOT relevant to our livelock:**

1. **Symptom mismatch.** The upstream bug produces *one wrong token in the output stream* after a force-preempt+clear cycle. Our bug is *the scheduler stops dispatching anything* — no output at all, requests pile up in WAITING.
2. **Code path is unreachable in our setup.** `reset_prefix_cache(reset_running_requests=True)` — the only call site that triggers this code — is reached via:
    - `POST /reset_prefix_cache` (manual admin endpoint), or
    - `pause_generation(clear_cache=True)` from the RLHF API router (`vllm/entrypoints/serve/rlhf/api_router.py:31`), or
    - direct `LLM.reset_prefix_cache(...)` calls (in-proc API).
   None of these is invoked by OpenWebUI or OpenClaw traffic in our `vllm serve` configuration. There is no automatic / internal caller of `_reset_caches()` outside the RLHF pause/resume flow.
3. **State after the fix.** Even if this path *were* somehow triggered, the symptom of the OLD code was an extra retained token — not a request stuck in WAITING. The break conditions B1–B4 audited in Round 2 are not touched by this commit.

### Updated hypothesis rank

No change from Round 2. Hypothesis 1 (Break B3 or B4 in `Scheduler.schedule()`) remains the primary lead. The pause/clear_cache keyword overlap was a red herring.

### What to try next (Round 3)

- [ ] Re-apply the `[LIVELOCK-PROBE]` instrumentation against the new `scheduler.py` (line numbers have shifted; the four break sites need to be located by context, not line number).
- [ ] All Round 2 follow-ups still pending.
