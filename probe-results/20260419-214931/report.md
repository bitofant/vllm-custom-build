# gemma4 probe — 20260419-214931

- container: `vllm_gemma4-31b`
- model: `cyankiwi/gemma-4-31B-it-AWQ-4bit`
- host: http://localhost:8000
- engine core pid (at start): 435

## Initial state
```
memory.used [MiB], memory.total [MiB], utilization.gpu [%]
31104 MiB, 32607 MiB, 0 %
```

## Test: `health` — GET /health


## Test: `models` — GET /v1/models


## Test: `simple` — Short chat, no tools, non-stream


## Test: `simple_stream` — Short chat, no tools, stream


## Test: `tools` — Short chat with tool definition


## Test: `long_1k` — ~1k-token prompt, no tools


## Test: `long_4k` — ~4k-token prompt, no tools


## Test: `long_10k` — ~10k-token prompt, no tools


## Test: `long_16k` — ~16k-token prompt (near KV cache limit)


## Test: `long_20k` — ~20k-token prompt (over KV cache limit)


## Test: `tools_long_4k` — ~4k-token prompt with tools

