#!/bin/bash

################################################################################
# vLLM Model Management Script — interactive Docker menu for a vLLM server.
#
# Box: Ryzen 9 9900X / RTX 5090 (32GB VRAM) / 96GB RAM / Ubuntu 24.04.
# Needs docker (nvidia runtime), curl, python3. OpenAI API on port 8000.
# Only one model runs at a time (single GPU).
#
# Functions:
#   vllm()              interactive menu, or pass an option/alias
#   get_vllm_model_id() print the running server's model ID
#
# Menu (see the case statement for per-config details & quantization notes):
#   1  Qwen3.8-27B-NVFP4          (150k ctx; '1c' = 228k ctx, '1k' = bf16 KV @ 76k)
#   2  Qwen3.6-35B-A3B-NVFP4      (256k ctx, native MTP; custom image)
#   3  DeepSeek-R1-Distill-32B-AWQ(32k ctx)
#   4  Gemma-4-31B-IT-NVFP4       (nvidia, tiny ctx; '4m'/'4f' RedHatAI+MTP, '4mc' turbo text)
#   5  Nemotron-3.5-Lightning-30B-A3B-NVFP4 (Mamba/MoE hybrid, native MTP)
#
# Control verbs are WORDS, never numbers. The old numeric 5/6/7 (logs/test/stop)
# were retired when Nemotron took slot 5 — a digit now always means "start a
# model". Each verb is matched directly by its case branch, so the same token
# works both as an argument and as an answer to the interactive prompt:
#   logs | l     live logs
#   test | t     test inference against the running model
#   stop | s | kill | k    stop all running vLLM containers
#
# Usage:
#   vllm | vllm 1 | vllm 1c | vllm 5 | vllm logs | vllm stop
#   vllm status       list vllm_* containers (running + stopped) via `docker ps`
#   vllm ctx          show ctx length + KV-cache concurrency of running model
#   vllm rec [<cfg>]  recreate a container; bare form infers running config + confirms
#                     (-y / `yes |` skips), `vllm rec <cfg>` recreates without prompting
#
# CPU offload (--cpu-offload-gb): keeps weights in RAM for models > 32GB VRAM.
# Much slower — PCIe ~32 GB/s vs VRAM ~1000 GB/s. All current configs use 0.
#
# Notes:
# - Stopped containers are restarted (not recreated) if they exist.
# - Starting a model (fresh run, restart, or `rec`) first stops every OTHER
#   running vllm_* container and waits for its VRAM to drain — one GPU, one port.
#   So switching configs needs no manual `vllm stop`.
# - `vllm test` reads multi-line input, terminated by "END" on its own line.
# - vLLM sources for valid flags/quant names: ~/src/vllm/vllm
#   (e.g. quantization/__init__.py under model_executor/layers).
################################################################################

# Default image (a case branch may override per config). Images use `vllm serve`
# as ENTRYPOINT, so the model is a positional arg (CMD_PREFIX empty). Do NOT add
# `--model`: vLLM main deprecated it and crashes (IndexError in argparse_utils).
#   vllm/vllm-openai:latest          upstream stable, CUDA 13, has Gemma 4
#   nvcr.io/nvidia/vllm:26.03.post1-py3  NVIDIA-built, ~2 minors behind (no Gemma 4)
#   vllm-custom:latest               local CUDA 13.1 build of vLLM main (Blackwell)
DOCKER_IMAGE="vllm/vllm-openai:latest"
CMD_PREFIX=()

function sync_agent_models() {
  # Point BOTH agents — pi (~/.pi/agent) and OpenClaw (~/.openclaw/openclaw.json)
  # — at the model we just started, by delegating to update-agent-models.sh.
  #
  # Call this AFTER wait_for_healthy, never before: that script reads the id and
  # max_model_len actually being SERVED from /v1/models, which is the truth even
  # on the stopped-container restart path (where --max-model-len is baked in at
  # create time and can differ from CONTEXT_SIZE in this file). The old
  # pre-start, pi-only version wrote CONTEXT_SIZE and could disagree with the
  # running server. INPUT_MODES has to be passed in — /v1/models doesn't report
  # modalities — and CONTEXT_WINDOW is only a fallback for endpoints that omit
  # max_model_len (vLLM reports it; colibrì doesn't).
  local input_modes="$1" ctx="$2"
  local sync=~/scripts/update-agent-models.sh
  [ -x "$sync" ] || return 0
  INPUT_MODES="$input_modes" CONTEXT_WINDOW="$ctx" "$sync" \
    || echo "Warning: agent model sync failed — pi/OpenClaw configs left unchanged."
}

function log_vllm_concurrency() {
  # Surface vLLM's KV-cache concurrency log line ("Maximum concurrency for N
  # tokens per request: X"), i.e. how many parallel requests fit the KV cache.
  local container="$1"
  local CYAN='\033[96m' BOLD='\033[1m' RESET='\033[0m'
  local line
  line=$(docker logs "$container" 2>&1 | grep "Maximum concurrency" | tail -n1)
  [ -n "$line" ] && echo -e "${CYAN}${BOLD}${line#*\] }${RESET}"
}

function wait_for_healthy() {
  # Poll the server's /health endpoint OURSELVES, report elapsed time, append a row
  # to vllm-startups.csv, and surface KV concurrency. Args: container, start-epoch.
  # Shows a live elapsed-seconds counter (redrawn in place via \r) while waiting.
  #
  # This deliberately does NOT read `docker inspect .State.Health.Status`. That
  # field only changes when DOCKER runs a probe, and these containers are created
  # with --health-start-period 60s / --health-start-interval 1s / --health-interval
  # 60s: the 1s cadence applies ONLY inside the start period, so from t=60s the
  # status refreshes once a MINUTE. Every boot here takes longer than 60s, so the
  # old 1s loop spent its time re-reading a stale value and returned on the next
  # whole minute — quantizing every startup to a 60s grid. It shows up as 94 of 158
  # rows in vllm-startups.csv landing within 2s of a 60s multiple (121, 241, 301,
  # 361, 422, 601...). Measured on the 2026-08-27 gemma boot: ready at 69s
  # ("Application startup complete." in the log), reported at 120s — ~51s of pure
  # dead wait, ~30s on average, up to 59s. Probing /health ourselves is exact.
  #
  # It also takes effect on the SIX ALREADY-CREATED containers with no `vllm rec`,
  # which raising --health-start-period would not: healthcheck settings are baked
  # in at container-create time, this loop is not.
  #
  # The docker healthcheck is left in place on purpose — `vllm status` / `docker ps`
  # still report (healthy), just on their own lazy schedule. Nothing else in this
  # repo or ~/.bash_aliases reads .State.Health.Status (llama.sh has its own copy).
  #
  # Not reading Docker's verdict also means losing --health-retries as the failure
  # signal, so crashes are detected here instead. Containers run with `--restart
  # unless-stopped`, so a boot that dies is silently restarted and would otherwise
  # spin this loop forever: a bump in .RestartCount IS that crash. A state that is
  # neither running nor restarting is a container that stayed down.
  # The loop is otherwise unbounded, as before — 3am.sh wraps the call in `timeout`.
  local container="$1" start="$2" csv=~/scripts/vllm-startups.csv
  local dur state status restarts base_restarts
  # NB: RestartCount is TOP-LEVEL in the inspect JSON, not under .State — a
  # {{.State.RestartCount}} template fails with "map has no entry for key".
  base_restarts=$(docker inspect --format='{{.RestartCount}}' "$container" 2>/dev/null) || base_restarts=0
  while true; do
    dur=$(( $(date +%s) - start ))
    # One inspect per iteration covers both fields; a failure here means the
    # container is gone, per the empty-state branch.
    state=$(docker inspect --format='{{.State.Status}} {{.RestartCount}}' "$container" 2>/dev/null) || state=""
    status="${state%% *}"
    restarts="${state##* }"
    if [ -n "$state" ] && [ "$restarts" -gt "$base_restarts" ]; then
      printf "\r\033[KContainer crashed and was auto-restarted (%s -> %s) after %ds.\n" \
        "$base_restarts" "$restarts" "$dur"
      printf "Check logs with: docker logs %s\n" "$container"
      return 1
    fi
    case "$status" in
      running|restarting|created) ;;
      "")
        printf "\r\033[KContainer %s no longer exists.\n" "$container"
        return 1
        ;;
      *)
        printf "\r\033[KContainer %s is '%s' after %ds. Check logs with: docker logs %s\n" \
          "$container" "$status" "$dur" "$container"
        return 1
        ;;
    esac
    # Probe INSIDE the container's network namespace (same command the docker
    # healthcheck runs), not from the host. A host-side `curl localhost:8000`
    # only proves *something* holds the port, and would happily report a
    # half-started container healthy on the strength of a foreign listener —
    # observed while testing: a dummy container that had bound nothing was
    # declared healthy at 0s because the previous vLLM still answered on 8000.
    # wait_for_port_release makes that unlikely but not impossible (it gives up
    # and starts anyway after PORT_FREE_TIMEOUT). exec attributes the answer to
    # this container unconditionally, costs ~47ms, and fails fast (~7ms) when the
    # container isn't running. --max-time so a wedged server can't stall the loop.
    if [ "$status" = "running" ] \
       && docker exec "$container" curl -sf --max-time 3 http://localhost:8000/health >/dev/null 2>&1; then
      printf "\r\033[KContainer is healthy! (took %ds)\n" "$dur"
      [ -f "$csv" ] || echo "timestamp,container,duration_seconds" > "$csv"
      echo "$(date '+%Y-%m-%d %H:%M:%S'),$container,$dur" >> "$csv"
      log_vllm_concurrency "$container"
      return 0
    fi
    printf "\rWaiting for container to become healthy... (%ds)\033[K" "$dur"
    sleep 1
  done
}

function wait_for_gpu_release() {
  # Poll until the given host PIDs no longer hold CUDA memory, or GPU_FREE_TIMEOUT
  # (90s) passes. `docker stop` returns as soon as the container is gone, but the
  # card drains ASYNCHRONOUSLY — start vLLM into that window and it profiles its
  # KV pool against VRAM the old server still owns (same lesson as 3am.sh).
  # Only the PIDs handed in are watched, deliberately: other GPU users on this box
  # (ComfyUI, the SD WebUI) legitimately keep the card busy and must never block a
  # vLLM start the way a "wait for 0 MiB" check would.
  local pids="$1" deadline live pid busy
  [ -n "${pids// /}" ] || return 0
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  deadline=$(( $(date +%s) + ${GPU_FREE_TIMEOUT:-90} ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    live=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    busy=
    if [ -n "$live" ]; then
      for pid in $pids; do
        if echo "$live" | grep -qx -- "$pid"; then busy=1; break; fi
      done
    fi
    if [ -z "$busy" ]; then
      printf "\r\033[K"
      return 0
    fi
    printf "\rWaiting for the GPU to release VRAM from the stopped container...\033[K"
    sleep 1
  done
  printf "\r\033[KWarning: GPU still held after %ss — starting anyway.\n" "${GPU_FREE_TIMEOUT:-90}"
  return 0
}

function wait_for_port_release() {
  # Poll until the host TCP port is free, or PORT_FREE_TIMEOUT (30s) passes.
  # Same async-teardown lesson as wait_for_gpu_release, different resource:
  # `docker stop` returns before the daemon has torn down the port binding.
  #
  # Losing this race is far nastier than a failed start. Docker answers 500
  # ("Bind for 0.0.0.0:8000 failed: port is already allocated") and then EMPTIES
  # the container's persisted NetworkSettings.Networks map. That map — not
  # HostConfig.NetworkMode — is what a later `docker start` reads to decide what
  # to attach, so every subsequent start silently joins NOTHING: no sbJoin, no
  # error, HostConfig still showing `bridge` + the 8000 binding. The container
  # comes up with only `lo`, and vLLM dies in ModelConfig.__post_init__ resolving
  # huggingface.co ("Temporary failure in name resolution") — which reads as a
  # DNS/network fault rather than a Docker one. `docker start` will NEVER undo
  # it; only `docker rm` (i.e. `vllm rec <cfg>`) clears the map.
  # Cost on 2026-08-25: ~40min of misdiagnosis and ~120 crash-loop restarts.
  local port="${1:-8000}" deadline
  command -v ss >/dev/null 2>&1 || return 0
  deadline=$(( $(date +%s) + ${PORT_FREE_TIMEOUT:-30} ))
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if ! ss -tln "sport = :$port" 2>/dev/null | grep -q LISTEN; then
      printf "\r\033[K"
      return 0
    fi
    printf "\rWaiting for port %s to be released...\033[K" "$port"
    sleep 1
  done
  printf "\r\033[KWarning: port %s still bound after %ss — starting anyway.\n" "$port" "${PORT_FREE_TIMEOUT:-30}"
  return 0
}

function sample_token_usage() {
  # Fold the tokens served since the last cron sample into vllm-token-usage.json
  # BEFORE the server goes away. That script diffs the live /metrics counters
  # against its stored sample, and the counters reset to 0 when vLLM restarts —
  # so everything served between the last 10-minute cron tick and a stop is lost
  # unless we take one final sample here. Called on every path that stops a
  # container (stop verb, rm, rec, and stop_other_vllm_containers).
  #
  # Safe to call repeatedly: it accumulates deltas, so an extra sample splits a
  # window in two rather than double-counting it. Never fatal — a missing
  # script, missing jq, or an unreachable server must not block a stop.
  local script="$HOME/scripts/vllm-token-usage.sh"
  [ -x "$script" ] || return 0
  "$script" || true
}

function stop_other_vllm_containers() {
  # Single GPU and a single port 8000: only one vLLM server can run at a time, so
  # every OTHER running vllm_* container is stopped before this one starts. Called
  # on every start path (fresh `docker run`, stopped-container restart, and `rec`),
  # so switching configs never needs a manual `vllm stop` first. Arg: the container
  # about to be started — it is left alone. Containers are created with
  # `--restart unless-stopped`, so an explicit `docker stop` keeps them down.
  local keep="$1" others name pids
  others=$(docker ps --filter "name=vllm_" --format "{{.Names}}" | grep -vx -- "$keep")
  [ -n "$others" ] || return 0

  # Grab the host PIDs BEFORE stopping: once the container is gone `docker top`
  # has nothing left to report, and wait_for_gpu_release needs them to tell our
  # draining processes apart from unrelated CUDA apps.
  pids=""
  while read -r name; do
    [ -n "$name" ] || continue
    echo "Stopping other vLLM container: $name"
    pids+=" $(docker top "$name" -eo pid 2>/dev/null | tail -n +2 | tr '\n' ' ')"
  done <<< "$others"

  sample_token_usage
  echo "$others" | xargs -r docker stop
  wait_for_gpu_release "$pids"
  # The GPU and the port drain independently — a container can be off the card
  # while the daemon still holds its 8000 binding. Both waits are needed.
  wait_for_port_release 8000
}

function one_running_vllm() {
  # Echo the single running vLLM container name; error (to stderr) + return 1 if
  # none or more than one are running.
  local names
  names=$(docker ps --filter "name=vllm_" --format "{{.Names}}")
  if [ -z "$names" ]; then
    echo "Error: No running vLLM container." >&2
    return 1
  fi
  if [ "$(echo "$names" | wc -l)" -gt 1 ]; then
    echo "Error: Multiple vLLM containers running. Stop all but one first ('vllm stop')." >&2
    return 1
  fi
  echo "$names"
}

function rec_config_for_container() {
  # Reverse of the DOCKER_NAME assignments in the vllm() case statement: map a
  # running container name back to the menu token that produced it, so bare
  # `vllm rec` can recreate the exact configuration. The model id alone is
  # ambiguous (configs 1, 1c and 1k share unsloth/Qwen3.8-27B-NVFP4, and 4m/4f
  # share RedHatAI/gemma-4-31B-it-NVFP4), so we key off the unique container name.
  # Keep this in sync with the DOCKER_NAME values in the case statement below.
  case "$1" in
    vllm_qwen_3_8)               echo "1" ;;
    vllm_qwen_3_8_228k)          echo "1c" ;;
    vllm_qwen_3_8_bf16kv)        echo "1k" ;;
    vllm_qwen3_6_35b_a3b_nvfp4)  echo "2" ;;
    vllm_deepseek_r1_32b_awq)    echo "3" ;;
    vllm_gemma4-31b-nvfp4)       echo "4" ;;
    vllm_gemma4-31b-nvfp4-mtp)   echo "4m" ;;
    vllm_gemma4-31b-nvfp4-mtp-hc) echo "4mc" ;;
    vllm_gemma4-31b-nvfp4-mtp-fp4kv) echo "4f" ;;
    vllm_nemotron35_lightning_30b_a3b) echo "5" ;;
    *) return 1 ;;
  esac
}

function get_vllm_model_id() {
  # Print the loaded model's ID from /v1/models (used by aid() in ~/.bash_aliases).
  local host="http://localhost:8000/v1" model_id
  if ! curl -s --head "$host/models" > /dev/null; then
    echo "Error: Cannot connect to vLLM at $host (is it running?)" >&2
    exit 1
  fi
  model_id=$(curl -s "$host/models" | python3 -c "import sys, json; print(json.load(sys.stdin)['data'][0]['id'])")
  if [ -z "$model_id" ]; then
    echo "Error: Could not extract model ID." >&2
    exit 1
  fi
  echo "$model_id"
}

function vllm() {
  local RM_MODE=
  local REC_MODE=
  local REC_INFERRED=
  local REC_YES=
  local REC_REPLY=
  # Per-config extra `docker run` env flags (e.g. --env VLLM_ATTENTION_BACKEND=...).
  # Default empty; a case branch below may set it. Baked into the container at
  # create time, so the stopped-container restart path inherits it automatically.
  local ENV_ARGS=()

  # Color codes
  local CYAN='\033[96m'
  local GREEN='\033[92m'
  local YELLOW='\033[93m'
  local MAGENTA='\033[95m'
  local RED='\033[91m'
  local GRAY='\033[90m'
  local BOLD='\033[1m'
  local RESET='\033[0m'

  # Check if argument was provided
  if [ -n "$1" ]; then
    # Only the arg-shaped forms (rm/rec consume extra args) need pre-processing.
    # Control verbs (logs|l, test|t, stop|s|kill|k, status|st|ps) fall through to
    # `*)` and are matched verbatim by their own case branches below — which is
    # what lets the interactive prompt accept them too, since that path assigns
    # model_choice straight from `read` without passing through here.
    case "$1" in
      rm)
        RM_MODE=1
        model_choice="$2"
        ;;
      rec)
        REC_MODE=1
        # Consume 'rec'; remaining args are an optional config token + -y/--yes.
        # This shift mutates $@, but the only later read of $1 (the DETACHED
        # branch) is guarded by `-z "$REC_MODE"`, so rec mode never reaches it.
        shift
        for arg in "$@"; do
          case "$arg" in
            -y|--yes) REC_YES=1 ;;
            *) model_choice="$arg" ;;
          esac
        done
        ;;
      *)
        model_choice="$1"
        ;;
    esac
  else
    echo ""
    echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════════════╗${RESET}"
    echo -e "${CYAN}║${RESET}${BOLD}                        vLLM Model Selection${RESET}                                ${CYAN}║${RESET}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════════════╝${RESET}"
    echo ""
    echo -e "${BOLD}1)${RESET} ${CYAN}Qwen3.8-27B-NVFP4${RESET}"
    echo -e "   ${GRAY}150k context | fp8 KV cache | native MTP spec-decode${RESET}"
    echo -e "   ${GRAY}Qwen 3.8 27B dense hybrid-attention, NVFP4 (Blackwell)${RESET}"
    echo -e "   ${GRAY}(use '1c' for 228k max context, '1k' for bf16 KV cache @ 76k)${RESET}"
    echo ""
    echo -e "${BOLD}2)${RESET} ${CYAN}Qwen3.6-35B-A3B-NVFP4${RESET}"
    echo -e "   ${GRAY}256k context | MTP spec-decode | no offloading${RESET}"
    echo -e "   ${GRAY}Qwen 3.6 MoE model (NVFP4), strong reasoning${RESET}"
    echo ""
    echo -e "${BOLD}3)${RESET} ${CYAN}DeepSeek-R1-Distill-Qwen-32B-AWQ${RESET}"
    echo -e "   ${GRAY}32k context | ~74 tok/s | no offloading${RESET}"
    echo -e "   ${GRAY}R1 distilled reasoning model (shows <think> tags)${RESET}"
    echo ""
    echo -e "${BOLD}4)${RESET} ${CYAN}Gemma-4-31B-IT-NVFP4 (nvidia)${RESET}"
    echo -e "   ${GRAY}8k context | fp8 KV cache | barely fits 32GB (attn stays bf16)${RESET}"
    echo -e "   ${GRAY}Gemma 4 IT, nvidia NVFP4 (modelopt), multimodal (Blackwell)${RESET}"
    echo -e "   ${GRAY}(use '4m' for MTP spec-decode, '4mc' for text-only Turbo high-context,${RESET}"
    echo -e "   ${GRAY} '4f' for MTP + NVFP4 KV cache)${RESET}"
    echo ""
    echo -e "${BOLD}5)${RESET} ${CYAN}Nemotron-3.5-Lightning-30B-A3B-NVFP4${RESET}"
    echo -e "   ${GRAY}256k context | fp8 KV cache | native MTP spec-decode${RESET}"
    echo -e "   ${GRAY}NVIDIA Mamba-2/MoE hybrid, 3B active of 30B (text-only)${RESET}"
    echo ""
    echo -e "${BOLD}l)${RESET} ${YELLOW}Show live logs of running container${RESET}"
    echo ""
    echo -e "${BOLD}t)${RESET} ${GREEN}Test inference with running model${RESET}"
    echo ""
    echo -e "${BOLD}s)${RESET} ${RED}Stop all running vLLM containers${RESET}"
    echo ""
    echo -ne "${BOLD}Select model (1-5) or action (l/t/s):${RESET} "
    read model_choice
  fi

  # Bare `vllm rec` (no config token): infer which configuration is running from
  # the container name and recreate that one.
  if [ -n "$REC_MODE" ] && [ -z "$model_choice" ]; then
    local running
    running=$(docker ps --filter "name=vllm_" --format "{{.Names}}")
    if [ -z "$running" ]; then
      echo -e "${RED}Error: No running vLLM container to recreate.${RESET}"
      return 1
    fi
    if [ "$(echo "$running" | wc -l)" -gt 1 ]; then
      echo -e "${RED}Error: Multiple vLLM containers running; can't infer which to recreate. Specify a config, e.g. 'vllm rec 1'.${RESET}"
      return 1
    fi
    model_choice=$(rec_config_for_container "$running")
    if [ -z "$model_choice" ]; then
      echo -e "${RED}Error: Running container '$running' doesn't match a known configuration.${RESET}"
      return 1
    fi
    REC_INFERRED=1
    echo -e "${GRAY}Detected running container ${BOLD}$running${RESET}${GRAY} -> config '$model_choice'.${RESET}"
  fi

  case $model_choice in
    1)
      # unsloth NVFP4 checkpoint -> compressed-tensors (quant_method in
      # config.json, format "mixed-precision"), NOT the modelopt_fp4 the old
      # nvidia 3.6 checkpoint used: nvidia has published no 3.8 NVFP4, so this
      # tracks unsloth (same uploader as option 2). Weights ship as a single
      # shard + model_mtp.safetensors.
      # Architecture is UNCHANGED from 3.6 (Qwen3_5ForConditionalGeneration,
      # model_type qwen3_5, 64 layers / head_dim 256 / 262k native / vocab
      # 248320) — the registry already carries Qwen3_5ForConditionalGeneration
      # and Qwen3_5MTP, so the parsers and MTP flags below carry over verbatim.
      # Needs the Blackwell-capable local build (vllm-custom:latest) for SM120 NVFP4.
      # MTP is native (draft layers built into the checkpoint), so --speculative-config
      # needs no external drafter (unlike gemma 4m/4f). GPU_MEM_UTIL capped at 0.982:
      # the newer flashinfer's fp4 autotune warmup needs a ~336 MiB transient that
      # OOM-crash-looped at 0.983 on this image (same lesson as gemma 4m).
      # CONTEXT_SIZE 150000 is INHERITED from the 3.6 tuning (identical dims, so
      # the KV math should hold: 60k measured 2.59x -> ~155k KV total -> 150k @
      # ~1.03x) but has NOT been re-measured on this checkpoint — re-derive from
      # the "Maximum concurrency" log line on the first boot, as after any
      # image/util change. --max-num-seqs 2 to match (MTP favors low
      # concurrency); use option 1c for the full 228k.
      #
      # --num-gpu-blocks-override 112 PINS the KV pool (b20, 2026-09-04). Without
      # it this config no longer boots at all: b20's profiler sized 108 blocks
      # (5.67 GiB) and the engine hard-errors rather than clamping — "estimated
      # maximum model length is 143824", short of 150000.
      #
      # DERIVING THE NUMBER — pool blocks and ctx-eligible blocks are NOT the same
      # unit, and conflating them sent the first attempt the wrong way (override
      # 93 *lowered* the pool from 108 and made it worse, 119584 max len).
      #   - block_size is 1616 tokens here, not the usual 16: it's forced up by
      #     "attention page size >= mamba page size" on this hybrid model. So ctx
      #     moves in 1616-token steps.
      #   - ~19 blocks are held back from the max_model_len math (MTP draft layer
      #     + the padding-layer waste the 1k notes measured at ~6.25%). Measured
      #     twice, constant: 108 blocks -> 143824 = 89*1616; 93 -> 119584 = 74*1616.
      #   => usable ctx = (N - 19) * 1616. 150000 needs 93 usable, so N = 112,
      #      i.e. ~0.21 GiB beyond what profiling offered.
      # Treat that as a conservative estimator, not an identity: it predicted a
      # 150,288-token pool and the boot MEASURED 151,351 (1.01x @ 150k, engine
      # init 180s, 761 MiB VRAM left free — autotune survived). Re-measure the
      # 19 after an image bump; don't assume it holds.
      #
      # That is safe here for a specific reason: this vLLM now ESTIMATES cudagraph
      # memory and subtracts it from KV up front ("--gpu-memory-utilization=0.9820
      # is equivalent to 0.9624 without CUDA graph memory profiling") — a ~0.6 GiB
      # reservation against a capture that measured 0.04 GiB on this config. The
      # pin spends part of that over-estimate, not the fp4 autotune headroom that
      # forced the 0.982 cap. Leave GPU_MEM_UTIL alone; it guards a different
      # transient. If a future image makes 112 OOM during autotune, drop
      # CONTEXT_SIZE to 143000 and the override with it, rather than raising util.
      # Re-derive both after any image rebuild: read block_size + num_gpu_blocks
      # from `curl -s localhost:8000/metrics | grep cache_config_info`.
      #
      # b21 (2026-09-05) DID make 112 OOM -- but only on a COLD boot: the fp4
      # autotune transient left the MTP drafter's cudagraph warmup 32 MiB short
      # (b20 had 761 MiB free here; the 417->443 upstream bump ate it). Fixed by
      # persisting ~/.cache/vllm (see the docker run block) rather than by
      # spending context, since the warm path has always fit. If a future image
      # OOMs even WARM, that's the real signal to take the 143000 downgrade.
      MODEL_ID="unsloth/Qwen3.8-27B-NVFP4"
      QUANTIZATION="compressed-tensors"
      CONTEXT_SIZE=150000
      INPUT_MODES="text,image"
      DOCKER_NAME="vllm_qwen_3_8"
      DOCKER_IMAGE="vllm-custom:latest"
      CPU_OFFLOAD=0
      GPU_MEM_UTIL=0.982
      EXTRA=(--enable-prefix-caching --kv-cache-dtype fp8 --max-num-seqs 2 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --speculative-config '{"method":"mtp","num_speculative_tokens":4}' --num-gpu-blocks-override 112)
      ;;
    1c)
      # Same as option 1 but with maximized context (228K of the 256K native).
      # Native MTP + 0.982 util ceiling as in option 1 (see notes there).
      MODEL_ID="unsloth/Qwen3.8-27B-NVFP4"
      QUANTIZATION="compressed-tensors"
      CONTEXT_SIZE=228000
      INPUT_MODES="text,image"
      DOCKER_NAME="vllm_qwen_3_8_228k"
      DOCKER_IMAGE="vllm-custom:latest"
      CPU_OFFLOAD=0
      GPU_MEM_UTIL=0.982
      EXTRA=(--enable-prefix-caching --kv-cache-dtype fp8 --max-num-seqs 2 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --speculative-config '{"method":"mtp","num_speculative_tokens":4}')
      ;;
    1k)
      # Option 1 with an UNQUANTIZED (bf16) KV cache — a quality experiment, not a
      # capacity one. Weight quantization error is fixed and calibrated; KV
      # quantization error instead ACCUMULATES along the sequence and lands
      # directly on attention scores, so at long context it's plausibly the larger
      # quality lever on this checkpoint (whose bulk is already only 4-bit in the
      # MLPs — attention, lm_head and layers 56-63 are FP8, vision/MTP bf16).
      # `--kv-cache-dtype bfloat16` MUST be passed explicitly, and `auto` is a TRAP
      # here. This checkpoint bakes its own KV quantization into config.json
      # (`kv_cache_scheme`: num_bits 8, type float, strategy tensor, static
      # calibrated scales); CompressedTensorsKVCacheMethod applies that whatever
      # the CLI says, and logs NO warning. Verified on the 2026-08-15 boot of this
      # very config with `auto`: FlashInfer reported kv_cache_dtype=
      # torch.float8_e4m3fn, i.e. it was silently identical to option 1. Grep the
      # boot log for "FlashInfer resolved query dtypes" to confirm what you got —
      # the engine-config line echoes the CLI value and will lie to you.
      #
      # VERIFIED 2026-08-15: metrics report cache_dtype="bfloat16", so the explicit
      # flag DOES beat the checkpoint scheme. Note the "FlashInfer resolved query
      # dtypes" line disappears on this path — it only logs for the fp8 KV backend,
      # so check `curl -s localhost:8000/metrics | grep cache_config_info` instead.
      #
      # Sizing: the KV pool is ~5.98 GiB REGARDLESS of CONTEXT_SIZE — it's whatever
      # survives weights (22.9 GiB) + peak activation (1.91) + cudagraphs (0.04).
      # Lowering ctx frees NO memory; it only changes the concurrency ratio.
      # Measured: 74,556 tokens @ bf16 (~86 B/token) vs 130,121 @ fp8 (~49 B/token)
      # — a 1.75x ratio, NOT the 2.0x the dtype width suggests, because on this
      # hybrid the linear_attn/Mamba state is sized separately (mamba_block_size
      # 16, mamba_ssm_cache_dtype float32, both untouched by --kv-cache-dtype) and
      # ~6.25% is lost to "Add 3 padding layers". Don't predict this ratio; read it.
      # 64000 measured 1.16x, so CONTEXT_SIZE was raised to 74000: FINAL MEASURED
      # STATE is 76,715 pool tokens @ 1.04x, the ~1.0x single-user target the other
      # configs use. The pool moved (74,556 -> 76,715) even though block_size (816)
      # and num_gpu_blocks (113) did NOT — so re-read cache_config_info and the
      # concurrency line after any ctx change instead of assuming either holds.
      #
      # Native MTP is retained and costs no quality — rejection sampling makes
      # speculative decoding distribution-preserving. Dropping fp8 KV also moves
      # AWAY from the FlashInfer fp8-prefill + MTP crash noted on option 2, so this
      # is the safer path there, not the riskier one. GPU_MEM_UTIL stays at the
      # 0.982 ceiling for the same reason as option 1 (the flashinfer fp4 autotune
      # warmup transient OOM-crash-loops at 0.983) — that limit is about the weight
      # path and is unaffected by the KV dtype. CONTEXT_SIZE is part of the
      # torch.compile cache key, so the first boot pays a full recompile.
      MODEL_ID="unsloth/Qwen3.8-27B-NVFP4"
      QUANTIZATION="compressed-tensors"
      CONTEXT_SIZE=74000
      INPUT_MODES="text,image"
      DOCKER_NAME="vllm_qwen_3_8_bf16kv"
      DOCKER_IMAGE="vllm-custom:latest"
      CPU_OFFLOAD=0
      GPU_MEM_UTIL=0.982
      EXTRA=(--enable-prefix-caching --kv-cache-dtype bfloat16 --max-num-seqs 2 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --speculative-config '{"method":"mtp","num_speculative_tokens":4}')
      ;;
    2)
      # Qwen 3.6 35B-A3B MoE, NVFP4. MTP is native (draft layers built in), so
      # --speculative-config needs no external drafter (unlike gemma4 4m/4f).
      # compressed-tensors, NOT modelopt_fp4 (checkpoint advertises the former).
      # --max-num-seqs 2: keeps CUDA-graph capture small; the default (~512) OOMs
      #   during profiling (graph capture + NVFP4 weights + MTP draft layer).
      # --max-num-batched-tokens 4096: hybrid (Mamba) model; Mamba block_size 2160
      #   must be <= this, so it can't drop below ~2160.
      # RESOLVED (2026-07-22): the vllm-custom rebuild ships a newer FlashInfer
      #   whose fp8 prefill accepts the .run(kv_cache_sf=...) kwarg, so the crash
      #   that hit MTP + fp8 KV is gone — confirmed on the 27B NVFP4 MTP configs
      #   (options 1/1c). Not yet re-tested on this A3B checkpoint, but the
      #   FlashInfer-version blocker no longer applies.
      MODEL_ID="unsloth/Qwen3.6-35B-A3B-NVFP4"
      QUANTIZATION="compressed-tensors"
      CONTEXT_SIZE=262144
      INPUT_MODES="text"
      DOCKER_NAME="vllm_qwen3_6_35b_a3b_nvfp4"
      DOCKER_IMAGE="vllm-custom:latest"
      CPU_OFFLOAD=0
      GPU_MEM_UTIL=0.983
      EXTRA=(--enable-prefix-caching --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --max-num-seqs 2 --max-num-batched-tokens 4096 --speculative-config '{"method":"mtp","num_speculative_tokens":4}')
      ;;
    3)
      MODEL_ID="casperhansen/deepseek-r1-distill-qwen-32b-awq"
      QUANTIZATION="awq_marlin"
      CONTEXT_SIZE=32768
      INPUT_MODES="text"
      DOCKER_NAME="vllm_deepseek_r1_32b_awq"
      CPU_OFFLOAD=0
      GPU_MEM_UTIL=0.95
      EXTRA=(--enable-prefix-caching)
      ;;
    4)
      # nvidia's official NVFP4 checkpoint (modelopt_fp4, NOT compressed-tensors).
      # Unlike RedHatAI (used by 4m/4f), it keeps all 60 self_attn layers plus
      # vision/embed/lm_head at bf16, so the weights alone load to ~30.5 GB and do
      # NOT fit the 5090's 32 GB with room for KV/activations — hence tiny context,
      # no MTP, and a minimal prefill reservation (--max-num-batched-tokens 2560 is
      # the image floor from Gemma 4's bidirectional MM attention; --max-num-seqs 1).
      # CPU_OFFLOAD=4 is REQUIRED: without it the run OOM-crash-loops during weight
      # loading (torch.empty for qkv_proj fails with ~192 MiB free — the card is
      # genuinely full of weights, so expandable_segments has nothing to reclaim and
      # shrinking CONTEXT_SIZE can't help since the OOM is pre-KV). Spilling ~4 GB of
      # weights to RAM leaves headroom for KV + the flashinfer fp4 autotune transient;
      # inference is slower (PCIe-bound). '4mc' (text-only turbo) is a lighter,
      # offload-free alternative. Needs vllm-custom:latest for SM120 NVFP4. Re-derive
      # CONTEXT_SIZE from the "GPU KV cache size: N tokens" log line — 8192 is a safe
      # starting point. (2026-07-22)
      MODEL_ID="nvidia/Gemma-4-31B-IT-NVFP4"
      QUANTIZATION="modelopt_fp4"
      CONTEXT_SIZE=8192
      INPUT_MODES="text,image"
      DOCKER_NAME="vllm_gemma4-31b-nvfp4"
      DOCKER_IMAGE="vllm-custom:latest"
      CPU_OFFLOAD=4
      GPU_MEM_UTIL=0.983
      ENV_ARGS=(--env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
      EXTRA=(--enable-prefix-caching --kv-cache-dtype fp8 --max-num-batched-tokens 2560 --max-num-seqs 1 --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --chat-template /etc/vllm/chat-templates/gemma4-force-think.jinja)
      ;;
    4m)
      # Option 4 + MTP speculative decoding via Google's drafter
      # google/gemma-4-31B-it-assistant (~0.5B, model_type gemma4_assistant which
      # vLLM rewrites to gemma4_mtp; n_predict forced to 1). Drafter shares the
      # target KV cache, so ~1 GB extra weights only.
      #
      # REQUIRES transformers >= 5.12.1 in the image (gemma4_assistant isn't in
      # vLLM's registry; only transformers >=5.12.x recognizes it). Stock
      # vllm-custom:latest ships 5.5.4 -> crashes at config load until rebuilt.
      #
      # CONTEXT_SIZE is KV-bound and drifts with every image bump — re-derive it
      # from the "GPU KV cache size: N tokens" log line after any rebuild.
      # Mind GPU_MEM_UTIL headroom: the flashinfer fp4 autotune warmup needs a
      # ~336 MiB transient that OOM-crash-looped at 0.983 (2026-07-20) and, it
      # turned out, still at 0.982. Nudging GPU_MEM_UTIL is futile here: vLLM
      # sizes the KV pool to fill the whole budget (profiled ~114k tokens), so
      # lowering util just shrinks KV to refill the same sliver, and the
      # flashinfer fp4 autotune warmup (NOT counted in vLLM's mem profiling)
      # OOMs on the leftover. The real lever is PYTORCH_CUDA_ALLOC_CONF=
      # expandable_segments:True (see ENV_ARGS below): reclaims the ~185 MiB
      # PyTorch keeps "reserved but unallocated", covering the transient with no
      # context cost. (2026-07-22)
      #
      # 2026-08-02 image rebuild (0.26.1rc1.dev251+g0033211c0) ate that headroom
      # again and crash-looped: V2 Model Runner + TRITON_ATTN (FA4 unavailable for
      # Gemma 4's heterogeneous head dims) + cudagraph FULL_AND_PIECEWISE. The
      # autotune transient missed by ~9 MiB (needed 168, had 158.88 free) — and
      # note flashinfer swallows per-tactic OOMs ("falling back to default
      # tactic") but the fallback's own OOM escapes and kills engine init.
      # expandable_segments was already on and doing its job (only 129 MiB
      # reserved-unallocated), so the fix this time is to skip the autotune
      # outright via --kernel-config (see EXTRA): the cache file is populated and
      # hits ("Config cache hit for fp4_gemm"), so tuning buys ~nothing here.
      #
      # --num-gpu-blocks-override 7000 PINS the KV pool, and DELIBERATELY takes
      # more than vLLM's profiler offers. Two separate reasons:
      #
      # 1. Profiling UNDER-estimates. vLLM compiles BEFORE the profiling pass
      #    that sizes KV, so the compile working set is resident while the peak
      #    is measured — memory it is about to release. A cold boot therefore
      #    reports only 6.87 GiB available (-> 5629 blocks / 87,388 tokens) and
      #    leaves ~2.2 GiB of the budget unused at steady state. Overriding to
      #    7000 claims it: 119,182 tokens, 1.14x concurrency at 105,000 ctx.
      #
      # 2. It closes the restart path, which is what actually crash-looped. The
      #    torch.compile cache lives in the container's writable layer and
      #    survives `docker restart` (only `docker rm` clears it). So boot 1
      #    profiles cold at 6.87 GiB but every --restart unless-stopped restart
      #    after it profiles WARM at 9.14 GiB, handing 2.27 GiB more to KV and
      #    starving what runs AFTER profiling. Autotune OOM'd on the leftover
      #    158 MiB; cudagraph capture (0.21 GiB) would have too. Pinning makes
      #    every boot identical regardless of cache state.
      #
      # tokens/block is NOT linear — 5629 -> 15.5 tok/blk but 7000 -> 17.0.
      # Do not extrapolate; boot it and read vllm:cache_config_info.
      #
      # HEADROOM: this leaves only ~117 MiB free, which is fine but is NOT a
      # budget to spend further. It is stable rather than a countdown because
      # chunked prefill caps activations at --max-num-batched-tokens (4096) per
      # step, so neither prompt length nor concurrency moves the high-water
      # mark. Verified 2026-08-03 at a steady 117 MiB across: 100,027-token
      # prompt, 2 concurrent 40k prompts, repeat requests, and an image request
      # (896x896). Re-derive from a COLD boot after any image bump — and note
      # CONTEXT_SIZE is part of the torch.compile cache key, so changing it
      # forces a full recompile. (2026-08-03)
      MODEL_ID="RedHatAI/gemma-4-31B-it-NVFP4"
      QUANTIZATION="compressed-tensors"
      CONTEXT_SIZE=105000
      INPUT_MODES="text,image"
      DOCKER_NAME="vllm_gemma4-31b-nvfp4-mtp"
      DOCKER_IMAGE="vllm-custom:latest"
      CPU_OFFLOAD=0
      GPU_MEM_UTIL=0.982
      ENV_ARGS=(--env "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
      EXTRA=(--enable-prefix-caching --kv-cache-dtype fp8 --max-num-batched-tokens 4096 --max-num-seqs 2 --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --chat-template /etc/vllm/chat-templates/gemma4-force-think.jinja --speculative-config '{"method":"mtp","model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":4}' --kernel-config '{"enable_flashinfer_autotune":false}' --num-gpu-blocks-override 7000)
      ;;
    4mc)
      # High-context turbo: LilaRest turbo (modelopt_fp4, TEXT-ONLY), NOT the
      # RedHatAI compressed-tensors of 4m. Single-user config that shrinks the
      # prefill reservation to buy KV: 4m's defaults reserve ~3 GiB of
      # activation/NVFP4 matmul workspace (straight out of the KV pool);
      # --max-num-batched-tokens 2560 --max-num-seqs 1 frees most of it back into
      # KV for more context. (The old 2560 hard floor was Gemma 4's bidirectional
      # image attention needing one 2496-token image item per batch — it no longer
      # binds now that this is text-only, so 2560 can go lower to buy more KV.)
      # The google MTP drafter is kept but UNVERIFIED on this quant; drop
      # --speculative-config if engine init balks.
      # Tune CONTEXT_SIZE to ~1.0x KV concurrency (re-derive from the log line).
      MODEL_ID="LilaRest/gemma-4-31B-it-NVFP4-turbo"
      QUANTIZATION="modelopt_fp4"
      CONTEXT_SIZE=100000
      INPUT_MODES="text"
      DOCKER_NAME="vllm_gemma4-31b-nvfp4-mtp-hc"
      DOCKER_IMAGE="vllm-custom:latest"
      CPU_OFFLOAD=0
      GPU_MEM_UTIL=0.982
      EXTRA=(--enable-prefix-caching --kv-cache-dtype fp8 --max-num-batched-tokens 2560 --max-num-seqs 1 --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --chat-template /etc/vllm/chat-templates/gemma4-force-think.jinja --speculative-config '{"method":"mtp","model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":4}')
      ;;
    4f)
      # 4m but with NVFP4 KV cache. DOES NOT BOOT on the RTX 5090: NVFP4 KV needs
      # FlashInfer's trtllm-gen kernel, gated to SM100 (datacenter Blackwell);
      # the 5090 is SM120, so engine init dies "kv_cache_dtype not supported".
      # SM100-only placeholder — use 4m (fp8 KV) on this box.
      MODEL_ID="RedHatAI/gemma-4-31B-it-NVFP4"
      QUANTIZATION="compressed-tensors"
      CONTEXT_SIZE=128000
      INPUT_MODES="text,image"
      DOCKER_NAME="vllm_gemma4-31b-nvfp4-mtp-fp4kv"
      DOCKER_IMAGE="vllm-custom:latest"
      CPU_OFFLOAD=0
      GPU_MEM_UTIL=0.983
      EXTRA=(--enable-prefix-caching --kv-cache-dtype nvfp4 --kv-cache-dtype-skip-layers sliding_window --max-num-batched-tokens 4096 --max-num-seqs 2 --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --chat-template /etc/vllm/chat-templates/gemma4-force-think.jinja --speculative-config '{"method":"mtp","model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":1}')
      ;;
    5)
      # NVIDIA Nemotron 3.5 Lightning: Mamba-2/MoE hybrid, 3B active of 30B.
      # Only 6 of 52 layers are attention (indices 5/12/19/26/33/42) at 2 KV
      # heads x 128 head_dim, so fp8 KV costs ~3 KiB/token — an order of
      # magnitude cheaper than the dense configs above — so KV is NOT the
      # binding constraint here; weights (~18 GiB) are. MEASURED on the first
      # cold boot (2026-08-13): the pool came out at 1,671,967 tokens, i.e.
      # 6.38x concurrency at a 262,144 ctx — six times more headroom than the
      # ~1.0x single-user target the other configs are tuned to. CONTEXT_SIZE is
      # therefore the full native 1,048,576 (max_position_embeddings) — the real
      # ceiling is the position embeddings, not VRAM.
      #
      # Counter-intuitively the pool got BIGGER when ctx was raised: 262,144 ctx
      # -> 1,671,967 tokens (6.38x), but 1,048,576 ctx -> 2,046,288 tokens
      # (1.95x). vLLM derives block_size from max_model_len (4240 at 262k), so
      # raising ctx re-blocks the pool rather than eating it. Do NOT assume a
      # bigger ctx costs KV here — measure both ends.
      #
      # Unlike every other config in this file, the lever to watch is the
      # weights, not the KV pool: 0.95 util leaves ~1.9 GiB free at steady
      # state, so there is genuine slack (contrast 4m's 117 MiB).
      #
      # QUANTIZATION=modelopt_mixed, NOT modelopt_fp4. The checkpoint is
      # modelopt-produced but quant_algo is MIXED_PRECISION: routed/shared
      # experts are W4A16_NVFP4 (group_size 16) while the Mamba in_proj/out_proj
      # are FP8. vLLM maps that to ModelOptMixedPrecisionConfig — its
      # override_quantization_method() returns "modelopt_mixed" on seeing
      # MIXED_PRECISION, and modelopt_fp4 would mis-handle the FP8 layers.
      #
      # MTP is native: config.json carries num_nextn_predict_layers=1 and vLLM
      # registers NemotronHMTPModel, rewriting model_type nemotron_h ->
      # nemotron_h_mtp automatically. So --speculative-config takes no "model"
      # key (same as options 1/2, unlike gemma 4m/4f). The predictor asserts
      # exactly 1 MTP layer; num_speculative_tokens 3 reuses that head
      # autoregressively, which is the count NVIDIA's card recommends for
      # low-concurrency single-user serving.
      #
      # Mamba-specific flags, both off-default and both needed:
      #   --mamba-backend flashinfer  (default is TRITON)
      #   --mamba-cache-mode align    (default "none"; prefix caching on a hybrid
      #                                model requires "align" — it caches the
      #                                Mamba state of each scheduler step's last
      #                                token and pads to attention page size)
      # --moe-backend marlin matches the target's W4A16_NVFP4 expert kernels,
      # but it is a GLOBAL setting and the MTP draft layer's MoE is UNQUANTIZED
      # (the mtp_layers_block_type layers are absent from `quantized_layers`).
      # Marlin is quantized-only, so the drafter's UnquantizedFusedMoEMethod
      # rejects it at load: "moe_backend='marlin' is not supported for
      # unquantized MoE" — engine init dies AFTER the target model has already
      # loaded, which makes it look like a target-side failure. Hence
      # "moe_backend":"triton" inside --speculative-config: SpeculativeConfig
      # takes a per-drafter override for exactly this case (its docstring cites
      # "quantized generator with unquantized drafter"). Do NOT "simplify" by
      # collapsing the two backends — they are deliberately different. (2026-08-13)
      #
      # GPU_MEM_UTIL is a deliberately conservative 0.95 for the first boot:
      # weights leave ~14 GiB free, so unlike the gemma configs there is no need
      # to scrape the last 2% — the slack absorbs the Marlin/FlashInfer autotune
      # transients that crash-looped options 4m/1. Push it up only after a clean
      # cold boot, and re-derive from the "GPU KV cache size" log line.
      #
      # NOTE: the model card claims vLLM >= v0.27.1. vllm-custom:latest reports
      # 0.26.1rc1.devNNN, but that is just main's version string lagging: the
      # build (commit edbc4969a76) already registers NemotronHForCausalLM,
      # NemotronHMTPModel, the nemotron_v3 reasoning parser, modelopt_mixed, and
      # the mamba/moe backend flags. Verified against ~/src/vllm/vllm.
      #
      # BOOTED AND VERIFIED 2026-08-13: healthy in 242s (warm weights; the first
      # run also downloads ~21.6 GB), tool/reasoning parsing OK, MTP accepting
      # 848/1569 draft tokens (~54%, ~2.6 tok/step; still 39% at position 2,
      # which is what justifies num_speculative_tokens 3).
      #
      # Heads up when testing: this is a HEAVY reasoner and the nemotron_v3
      # parser puts the think block in `reasoning`, leaving `content` NULL until
      # it finishes. A 300-token cap returned content=None mid-thought; it needed
      # ~570 completion tokens for a one-sentence answer. `vllm test` caps at 500
      # and prints only `content`, so it can print "None" on a working server —
      # that is the harness being terse, not a broken model.
      MODEL_ID="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
      QUANTIZATION="modelopt_mixed"
      CONTEXT_SIZE=1048576
      INPUT_MODES="text"
      DOCKER_NAME="vllm_nemotron35_lightning_30b_a3b"
      DOCKER_IMAGE="vllm-custom:latest"
      CPU_OFFLOAD=0
      GPU_MEM_UTIL=0.95
      EXTRA=(--enable-prefix-caching --kv-cache-dtype fp8 --mamba-backend flashinfer --mamba-cache-mode align --moe-backend marlin --max-num-batched-tokens 4096 --max-num-seqs 2 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}')
      ;;
    logs|l)
      RUNNING_NAMES=$(docker ps --filter "name=vllm_" --format "{{.Names}}")
      if [ -n "$RUNNING_NAMES" ]; then
        if [ "$(echo "$RUNNING_NAMES" | wc -l)" -gt 1 ]; then
          echo "Error: Multiple vLLM containers are running. Please stop all first ('vllm stop')."
          return 1
        fi
        echo "Showing live logs for: $RUNNING_NAMES"
        echo "Press Ctrl+C to exit logs"
        docker logs -f "$RUNNING_NAMES"
        return 0
      fi

      # None running: show logs from the most recently stopped container.
      STOPPED=$(docker ps -a --filter "name=vllm_" --filter "status=exited" --format "{{.Names}}")
      if [ -z "$STOPPED" ]; then
        echo "Error: No vLLM containers found (running or stopped)."
        return 1
      fi

      MOST_RECENT=$(echo "$STOPPED" | while read -r name; do
        printf '%s\t%s\n' "$(docker inspect --format='{{.State.FinishedAt}}' "$name")" "$name"
      done | sort -r | head -n1 | awk '{print $2}')

      echo "No running vLLM container. Showing logs from most recently stopped: $MOST_RECENT"
      docker logs "$MOST_RECENT"
      return 0
      ;;
    test|t)
      one_running_vllm > /dev/null || return 1

      echo "Enter your prompt (type END on a new line when done, or Ctrl+C to cancel):"
      USER_PROMPT=""
      while IFS= read -r line; do
        [ "$line" = "END" ] && break
        if [ -z "$USER_PROMPT" ]; then
          USER_PROMPT="$line"
        else
          USER_PROMPT+=$'\n'"$line"
        fi
      done

      if [ -z "$USER_PROMPT" ]; then
        echo "Error: Empty prompt provided."
        return 1
      fi

      echo ""
      echo "Sending request to model..."
      echo ""

      # Build JSON request using Python to properly escape everything
      RESPONSE=$(python3 -c "
import json
import sys
import urllib.request

model_id = '$(get_vllm_model_id)'
prompt = '''$USER_PROMPT'''

data = {
    'model': model_id,
    'messages': [{'role': 'user', 'content': prompt}],
    'temperature': 0.7,
    'max_tokens': 500
}

req = urllib.request.Request(
    'http://localhost:8000/v1/chat/completions',
    data=json.dumps(data).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)

try:
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode('utf-8'))
        print(result['choices'][0]['message']['content'])
except Exception as e:
    sys.exit(1)
" 2>/dev/null)

      if [ $? -ne 0 ]; then
        echo "Error: Failed to get response from model. Is it still loading?"
        return 1
      fi

      echo "$RESPONSE"

      echo ""
      return 0
      ;;
    status|st|ps)
      # Shorthand for `docker ps` scoped to this script's containers. Show
      # stopped ones too (-a) so a crashed/exited server is visible, not silently
      # absent. Same `name=vllm_` filter the rest of the script keys off.
      docker ps -a --filter "name=vllm_" \
        --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
      return 0
      ;;
    ctx)
      RUNNING_NAMES=$(one_running_vllm) || return 1

      MODEL_INFO=$(curl -s http://localhost:8000/v1/models)
      if [ -z "$MODEL_INFO" ]; then
        echo "Error: Cannot reach vLLM API at http://localhost:8000 (still loading?)"
        return 1
      fi

      MODEL_ID=$(echo "$MODEL_INFO" | python3 -c "import sys, json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
      CTX_LEN=$(echo "$MODEL_INFO" | python3 -c "import sys, json; print(json.load(sys.stdin)['data'][0].get('max_model_len', '?'))" 2>/dev/null)

      echo -e "${BOLD}Container:${RESET}      $RUNNING_NAMES"
      echo -e "${BOLD}Model:${RESET}          $MODEL_ID"
      echo -e "${BOLD}Context length:${RESET} $CTX_LEN"
      log_vllm_concurrency "$RUNNING_NAMES"
      return 0
      ;;
    stop|s|kill|k)
      IDS=$(docker ps -q --filter "name=vllm_")
      if [ -z "$IDS" ]; then
        echo "No running vLLM containers found."
        return 0
      fi
      sample_token_usage
      echo "$IDS" | xargs -r docker stop
      return 0
      ;;
    *)
      return 0
      ;;
  esac

  # Bare `vllm rec` inferred the running config, so confirm before tearing it
  # down (unless -y / piped `yes`; the prompt reads a 'y' from stdin). Explicit
  # `vllm rec <cfg>` skips this — the caller already named the target.
  if [ -n "$REC_INFERRED" ]; then
    echo -e "${YELLOW}Recreate ${BOLD}$DOCKER_NAME${RESET}${YELLOW} (model: $MODEL_ID, ctx: $CONTEXT_SIZE)?${RESET}"
    if [ -n "$REC_YES" ]; then
      echo -e "${GRAY}(-y) Proceeding without confirmation.${RESET}"
    else
      echo -ne "${BOLD}Stop, remove, and start fresh? [y/N]:${RESET} "
      read -r REC_REPLY
      case "$REC_REPLY" in
        y|Y|yes|YES) ;;
        *) echo "Aborted."; return 0 ;;
      esac
    fi
  fi

  # Handle rm mode: stop (if running) then remove the container for the selected model
  if [ -n "$RM_MODE" ]; then
    if [ -n "$(docker ps -q -f name=^/${DOCKER_NAME}$)" ]; then
      sample_token_usage
      docker stop "$DOCKER_NAME"
    fi
    docker rm "$DOCKER_NAME"
    return 0
  fi

  # Handle rec mode: stop+rm existing container so the fresh start path below runs
  if [ -n "$REC_MODE" ]; then
    if [ -n "$(docker ps -q -f name=^/${DOCKER_NAME}$)" ]; then
      sample_token_usage
      docker stop "$DOCKER_NAME"
    fi
    if [ -n "$(docker ps -a -q -f name=^/${DOCKER_NAME}$)" ]; then
      docker rm "$DOCKER_NAME"
    fi
  fi

  # Single GPU + single port: clear out any other running vLLM server first. This
  # sits above BOTH start paths (restart + fresh run) on purpose, and after the
  # rm/rec handling above, so `rm` never touches an unrelated container.
  stop_other_vllm_containers "$DOCKER_NAME"

  # Container exists but is stopped -> just restart it (config is baked in).
  if [ "$(docker ps -a -q -f name=^/${DOCKER_NAME}$ -f status=exited)" ]; then
    echo "Starting existing Docker container: $DOCKER_NAME"
    ~/scripts/update-openclaw-wan-ip.sh
    START_TIME=$(date +%s)
    # Propagate the failure instead of falling through to wait_for_healthy, which
    # would otherwise poll a container that never started. A start that fails on
    # "port is already allocated" also bricks this container's networking for
    # good (see wait_for_port_release) — hence the `vllm rec` hint.
    if ! docker start "$DOCKER_NAME"; then
      echo "Error: docker start failed for $DOCKER_NAME"
      # NB: on a start that failed before the container ran, these lines are from
      # its PREVIOUS run — stale, but still the best clue when the process itself
      # is what died.
      echo "--- last 10 log lines from $DOCKER_NAME ---"
      docker logs --tail 10 "$DOCKER_NAME" 2>&1 | sed 's/^/  /'
      echo "-------------------------------------------"
      echo "If this was a port/network failure, recreate the container: vllm rec $model_choice"
      return 1
    fi
    wait_for_healthy "$DOCKER_NAME" "$START_TIME" || return 1
    sync_agent_models "$INPUT_MODES" "$CONTEXT_SIZE"
    return 0
  fi

  # Optional offload args as an array (zsh doesn't word-split unquoted expansions).
  OFFLOAD_ARGS=()
  if [ "$CPU_OFFLOAD" -gt 0 ]; then
    OFFLOAD_ARGS=(--cpu-offload-gb "$CPU_OFFLOAD")
  fi

  # HuggingFace token: read from the host's HF_TOKEN env var (never committed).
  # Only passed through when set; public models work without it.
  HF_ENV_ARGS=()
  if [ -n "${HF_TOKEN:-}" ]; then
    HF_ENV_ARGS=(--env "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}")
  fi

  # Always detached (-d) so the menu/direct-arg calls never block the shell.
  ~/scripts/update-openclaw-wan-ip.sh
  START_TIME=$(date +%s)
  # -v ~/.cache/vllm PERSISTS vLLM's startup caches across `rec` (2026-09-05).
  # Without it they live in the container's writable layer and every recreate
  # pays a COLD boot: ~28s torch.compile of backbone+eagle_head, plus a ~20s
  # FlashInfer fp4 autotune pass whose transient is what OOM-crashed config 1's
  # first b21 boot (32 MiB short, at the MTP drafter's cudagraph warmup; the
  # unless-stopped restart then succeeded warm). Contents: torch_compile_cache/
  # (~200 MB of AOT graphs), flashinfer_autotune_cache/ (the tuned
  # bf16_fp4_cute_dsl_gemm configs -- the memory-relevant one), modelinfos/.
  # Safe to share one host dir across ALL configs: every path is content-keyed
  # (flashinfer version, and hashes of vLLM version + compilation config), so
  # entries never collide and a stale one is never reused. Corollary: an image
  # rebuild is still cold ONCE (new hashes) -- expect one crash+auto-restart on
  # the first `rec` after a build. Nothing evicts: ~200 MB per (build x config),
  # so `rm -rf ~/.cache/vllm` occasionally.
  docker run -d --name $DOCKER_NAME --runtime nvidia --gpus all \
    --restart unless-stopped \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/.cache/vllm:/root/.cache/vllm \
    -v ~/scripts/chat-templates:/etc/vllm/chat-templates:ro \
    "${HF_ENV_ARGS[@]}" \
    "${ENV_ARGS[@]}" \
    -p 8000:8000 \
    --ipc=host \
    --health-cmd "curl -f http://localhost:8000/health || exit 1" \
    --health-interval 60s \
    --health-start-interval 1s \
    --health-timeout 5s \
    --health-retries 10 \
    --health-start-period 60s \
    "$DOCKER_IMAGE" \
    "${CMD_PREFIX[@]}" "$MODEL_ID" \
    --quantization "$QUANTIZATION" \
    --max-model-len "$CONTEXT_SIZE" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --dtype auto \
    "${EXTRA[@]}" \
    "${OFFLOAD_ARGS[@]}"

  wait_for_healthy "$DOCKER_NAME" "$START_TIME" || return 1
  sync_agent_models "$INPUT_MODES" "$CONTEXT_SIZE"
}

# Run by path (see ~/.bash_aliases: alias vllm, aid(), asyncservice()). Dispatch
# CLI args to vllm(), except `model-id`/`id` -> get_vllm_model_id (used by aid()).
# When sourced, `return 0` succeeds so dispatch is skipped and only the function
# definitions load (portable across bash and zsh).
if ! (return 0 2>/dev/null); then
  case "$1" in
    model-id|id|get_vllm_model_id)
      get_vllm_model_id
      ;;
    *)
      vllm "$@"
      ;;
  esac
fi
