#!/usr/bin/env bash
# gemma_probe.sh — diagnose hangs in vllm_gemma4-31b
#
# Runs a battery of requests of increasing complexity, samples GPU and engine
# state during each, and on a hang captures a stack trace of the EngineCore
# process. Produces a markdown report in ./probe-results/.
#
# Usage:
#   ./gemma_probe.sh [test_name ...]     # run selected tests (default: all)
#   ./gemma_probe.sh --list               # list tests
#   ./gemma_probe.sh --clean              # delete old reports
#
# Requires: docker, curl, jq, python3, nvidia-smi.

set -uo pipefail

CONTAINER="${CONTAINER:-vllm_gemma4-31b}"
MODEL="${MODEL:-cyankiwi/gemma-4-31B-it-AWQ-4bit}"
HOST="${HOST:-http://localhost:8000}"
REQ_TIMEOUT="${REQ_TIMEOUT:-60}"       # per-request timeout in seconds
HANG_THRESHOLD="${HANG_THRESHOLD:-20}" # seconds with no bytes before we call it hung
GPU_SAMPLE_INTERVAL="${GPU_SAMPLE_INTERVAL:-1}"

OUTDIR="$(dirname "$(readlink -f "$0")")/probe-results"
STAMP="$(date +%Y%m%d-%H%M%S)"
RUNDIR="$OUTDIR/$STAMP"
REPORT="$RUNDIR/report.md"

# ---------- prompt builders ----------

build_long_prompt() {
  # Deterministic filler content of roughly N tokens (~4 chars/token).
  local target_tokens="$1"
  local target_chars=$(( target_tokens * 4 ))
  python3 -c "
import sys
n = int(sys.argv[1])
base = 'The quick brown fox jumps over the lazy dog near the riverbank. '
out = (base * ((n // len(base)) + 2))[:n]
print(out)
" "$target_chars"
}

payload_simple() {
  jq -nc --arg m "$MODEL" '{model:$m,messages:[{role:"user",content:"Say hi in one word."}],max_tokens:8,stream:false}'
}

payload_simple_stream() {
  jq -nc --arg m "$MODEL" '{model:$m,messages:[{role:"user",content:"Count to three."}],max_tokens:32,stream:true}'
}

payload_with_tools() {
  jq -nc --arg m "$MODEL" '{
    model:$m,
    messages:[{role:"user",content:"What is the weather in Paris right now?"}],
    max_tokens:128,
    stream:false,
    tools:[{
      type:"function",
      function:{
        name:"get_weather",
        description:"Get the current weather for a city",
        parameters:{type:"object",properties:{city:{type:"string"}},required:["city"]}
      }
    }]
  }'
}

payload_long() {
  local toks="$1"
  local filler
  filler="$(build_long_prompt "$toks")"
  jq -nc --arg m "$MODEL" --arg c "$filler Summarise in one word." \
    '{model:$m,messages:[{role:"user",content:$c}],max_tokens:16,stream:false}'
}

payload_long_with_tools() {
  local toks="$1"
  local filler
  filler="$(build_long_prompt "$toks")"
  jq -nc --arg m "$MODEL" --arg c "$filler Given the context above, call get_weather for Paris." \
    '{
      model:$m,
      messages:[{role:"user",content:$c}],
      max_tokens:128,
      stream:false,
      tools:[{
        type:"function",
        function:{
          name:"get_weather",
          description:"Get the current weather for a city",
          parameters:{type:"object",properties:{city:{type:"string"}},required:["city"]}
        }
      }]
    }'
}

# ---------- tests ----------

declare -a TESTS=(
  "health:GET /health"
  "models:GET /v1/models"
  "simple:Short chat, no tools, non-stream"
  "simple_stream:Short chat, no tools, stream"
  "tools:Short chat with tool definition"
  "long_1k:~1k-token prompt, no tools"
  "long_4k:~4k-token prompt, no tools"
  "long_10k:~10k-token prompt, no tools"
  "long_16k:~16k-token prompt (near KV cache limit)"
  "long_20k:~20k-token prompt (over KV cache limit)"
  "tools_long_4k:~4k-token prompt with tools"
)

list_tests() {
  for t in "${TESTS[@]}"; do
    echo "  ${t%%:*}  — ${t#*:}"
  done
}

# ---------- diagnostics ----------

engine_core_pid() {
  # Engine core is a subprocess inside the container. Prefer the one whose
  # argv[0] matches "VLLM::EngineCore" (ps truncates, so grep broadly).
  docker exec "$CONTAINER" sh -c 'ps -eo pid,comm | awk "/EngineCore|VLLM/ {print \$1; exit}"'
}

apiserver_pid() {
  echo 1  # vllm runs as PID 1 in the container
}

sample_gpu() {
  # Background sampler: writes CSV rows until killed.
  local out="$1"
  echo "timestamp,mem_used_MiB,mem_total_MiB,gpu_util,mem_util,power_W,temp_C" > "$out"
  while true; do
    nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu \
      --format=csv,noheader,nounits >> "$out" 2>/dev/null
    sleep "$GPU_SAMPLE_INTERVAL"
  done
}

capture_stacks() {
  local dir="$1"
  local ecpid apid
  ecpid="$(engine_core_pid || true)"
  apid="$(apiserver_pid)"
  {
    echo "=== container ps ==="
    docker exec "$CONTAINER" ps -eo pid,ppid,comm,stat,pcpu,pmem 2>&1 || true
    echo
    echo "=== APIServer pid=$apid /proc/$apid/stack ==="
    docker exec "$CONTAINER" sh -c "cat /proc/$apid/stack 2>/dev/null || echo '(unavailable)'"
    echo
    if [[ -n "$ecpid" ]]; then
      echo "=== EngineCore pid=$ecpid /proc/$ecpid/stack ==="
      docker exec "$CONTAINER" sh -c "cat /proc/$ecpid/stack 2>/dev/null || echo '(unavailable)'"
      echo
      echo "=== EngineCore task stacks ==="
      docker exec "$CONTAINER" sh -c "for t in /proc/$ecpid/task/*/stack; do echo \"--- \$t ---\"; cat \$t 2>/dev/null || echo '(unavailable)'; done"
      echo
      echo "=== py-spy dump (if available) ==="
      docker exec "$CONTAINER" sh -c "command -v py-spy >/dev/null 2>&1 && py-spy dump --pid $ecpid 2>&1 || echo '(py-spy not installed — install with: docker exec $CONTAINER pip install py-spy)'"
    else
      echo "(could not locate EngineCore PID)"
    fi
  } > "$dir/stacks.txt" 2>&1
}

# ---------- request runner ----------

run_request() {
  # Args: test_name, method, path, body_file_or_empty
  local name="$1" method="$2" path="$3" body="$4"
  local tdir="$RUNDIR/$name"
  mkdir -p "$tdir"

  local gpu_csv="$tdir/gpu.csv"
  local logs_before="$tdir/logs-before.txt"
  local logs_after="$tdir/logs-after.txt"
  local resp_headers="$tdir/response-headers.txt"
  local resp_body="$tdir/response-body.txt"
  local timing="$tdir/timing.txt"

  # Snapshot docker log cursor (we'll capture only new lines after request).
  local cursor
  cursor="$(date -u +%Y-%m-%dT%H:%M:%S)"

  # Start GPU sampler in background.
  sample_gpu "$gpu_csv" &
  local sampler_pid=$!
  trap "kill $sampler_pid 2>/dev/null || true" RETURN

  # Build curl args.
  local -a curl_args=(
    -sS
    -o "$resp_body"
    -D "$resp_headers"
    --max-time "$REQ_TIMEOUT"
    --connect-timeout 5
    -w 'http_code=%{http_code}\nsize_download=%{size_download}\ntime_total=%{time_total}\ntime_starttransfer=%{time_starttransfer}\ntime_connect=%{time_connect}\n'
    -X "$method"
    "$HOST$path"
  )
  if [[ -n "$body" ]]; then
    curl_args+=(-H "Content-Type: application/json" --data-binary "@$body")
  fi

  local start end
  start="$(date +%s.%N)"
  curl "${curl_args[@]}" > "$timing" 2>&1
  local curl_exit=$?
  end="$(date +%s.%N)"
  local elapsed
  elapsed="$(python3 -c "print(f'{float('$end') - float('$start'):.2f}')")"

  # Stop sampler.
  kill $sampler_pid 2>/dev/null || true
  wait $sampler_pid 2>/dev/null || true
  trap - RETURN

  # Grab the new docker log lines since cursor.
  docker logs --since "$cursor" "$CONTAINER" > "$logs_after" 2>&1 || true

  # If the request looks hung (very slow + nothing generated), capture stacks.
  local ttfb
  ttfb="$(awk -F= '/^time_starttransfer/ {print $2}' "$timing" 2>/dev/null || echo 0)"
  local size
  size="$(awk -F= '/^size_download/ {print $2}' "$timing" 2>/dev/null || echo 0)"
  local hung=0
  if [[ "$curl_exit" -ne 0 ]] || [[ "${size:-0}" -eq 0 && "$(python3 -c "print(1 if float('$elapsed') > $HANG_THRESHOLD else 0)")" -eq 1 ]]; then
    hung=1
    capture_stacks "$tdir"
  fi

  # Write per-test summary line.
  {
    echo "name=$name"
    echo "elapsed_s=$elapsed"
    echo "curl_exit=$curl_exit"
    echo "hung=$hung"
    cat "$timing" 2>/dev/null || true
  } > "$tdir/summary.env"

  # Return non-zero if hung, so caller can bail early.
  [[ "$hung" -eq 0 ]]
}

# ---------- reporting ----------

write_test_section() {
  local name="$1" desc="$2"
  local tdir="$RUNDIR/$name"
  local sum="$tdir/summary.env"
  [[ -f "$sum" ]] || return 0

  # shellcheck disable=SC1090
  (
    source "$sum"
    {
      echo
      echo "## Test: \`$name\` — $desc"
      echo
      echo "- elapsed: **${elapsed_s}s**"
      echo "- curl_exit: $curl_exit"
      echo "- http_code: ${http_code:-?}"
      echo "- bytes: ${size_download:-0}"
      echo "- TTFB: ${time_starttransfer:-?}s"
      echo "- hung: $([[ $hung -eq 1 ]] && echo '**YES**' || echo no)"
      echo
      if [[ -s "$tdir/gpu.csv" ]]; then
        echo "### GPU samples (first/peak/last)"
        echo '```'
        {
          head -1 "$tdir/gpu.csv"
          # Skip header, then emit first, peak-by-mem, last.
          awk -F, 'NR>1 {
            if (first=="") first=$0;
            if ($2+0 > peak_mem+0) { peak_mem=$2; peak=$0 }
            last=$0
          } END {
            print "first: " first;
            print "peak:  " peak;
            print "last:  " last
          }' "$tdir/gpu.csv"
        }
        echo '```'
        echo
      fi
      if [[ -s "$tdir/response-headers.txt" ]]; then
        echo "### Response headers"
        echo '```'
        head -20 "$tdir/response-headers.txt"
        echo '```'
      fi
      if [[ -s "$tdir/response-body.txt" ]]; then
        echo "### Response body (first 1KB)"
        echo '```'
        head -c 1024 "$tdir/response-body.txt"
        echo
        echo '```'
      fi
      if [[ -s "$tdir/logs-after.txt" ]]; then
        echo "### Engine log during request (last 40 lines, non-health)"
        echo '```'
        grep -v 'GET /health' "$tdir/logs-after.txt" | tail -40
        echo '```'
      fi
      if [[ -f "$tdir/stacks.txt" ]]; then
        echo "### Stack traces (captured because hang detected)"
        echo '```'
        head -200 "$tdir/stacks.txt"
        echo '```'
      fi
    } >> "$REPORT"
  )
}

# ---------- main ----------

case "${1:-}" in
  --list)
    list_tests
    exit 0
    ;;
  --clean)
    rm -rf "$OUTDIR"
    echo "cleaned $OUTDIR"
    exit 0
    ;;
esac

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "container '$CONTAINER' is not running" >&2
  exit 1
fi

mkdir -p "$RUNDIR"

# Pick tests.
selected_tests=()
if [[ $# -gt 0 ]]; then
  for t in "$@"; do
    selected_tests+=("$t")
  done
else
  for t in "${TESTS[@]}"; do
    selected_tests+=("${t%%:*}")
  done
fi

# Report header.
{
  echo "# gemma4 probe — $STAMP"
  echo
  echo "- container: \`$CONTAINER\`"
  echo "- model: \`$MODEL\`"
  echo "- host: $HOST"
  echo "- engine core pid (at start): $(engine_core_pid)"
  echo
  echo "## Initial state"
  echo '```'
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
  echo '```'
} > "$REPORT"

# Run tests, bailing out after first hang (engine is likely wedged after that).
for name in "${selected_tests[@]}"; do
  # Look up description.
  desc=""
  for t in "${TESTS[@]}"; do
    if [[ "${t%%:*}" == "$name" ]]; then desc="${t#*:}"; break; fi
  done
  [[ -z "$desc" ]] && { echo "unknown test: $name" >&2; continue; }

  echo "→ running: $name ($desc)"
  body_file=""

  case "$name" in
    health)        method=GET  path=/health ;;
    models)        method=GET  path=/v1/models ;;
    simple)        method=POST path=/v1/chat/completions; body_file="$RUNDIR/$name.json"; payload_simple > "$body_file" ;;
    simple_stream) method=POST path=/v1/chat/completions; body_file="$RUNDIR/$name.json"; payload_simple_stream > "$body_file" ;;
    tools)         method=POST path=/v1/chat/completions; body_file="$RUNDIR/$name.json"; payload_with_tools > "$body_file" ;;
    long_1k)       method=POST path=/v1/chat/completions; body_file="$RUNDIR/$name.json"; payload_long 1000  > "$body_file" ;;
    long_4k)       method=POST path=/v1/chat/completions; body_file="$RUNDIR/$name.json"; payload_long 4000  > "$body_file" ;;
    long_10k)      method=POST path=/v1/chat/completions; body_file="$RUNDIR/$name.json"; payload_long 10000 > "$body_file" ;;
    long_16k)      method=POST path=/v1/chat/completions; body_file="$RUNDIR/$name.json"; payload_long 16000 > "$body_file" ;;
    long_20k)      method=POST path=/v1/chat/completions; body_file="$RUNDIR/$name.json"; payload_long 20000 > "$body_file" ;;
    tools_long_4k) method=POST path=/v1/chat/completions; body_file="$RUNDIR/$name.json"; payload_long_with_tools 4000 > "$body_file" ;;
    *) echo "unknown test: $name" >&2; continue ;;
  esac

  if run_request "$name" "$method" "$path" "$body_file"; then
    echo "   ok"
  else
    echo "   HUNG — captured stacks. Bailing out (engine likely wedged)."
    write_test_section "$name" "$desc"
    {
      echo
      echo "## Aborted — engine hung on \`$name\`"
      echo
      echo "Remaining tests skipped. Restart container with \`vllm rec 4\` and re-run."
    } >> "$REPORT"
    echo
    echo "report: $REPORT"
    exit 2
  fi

  write_test_section "$name" "$desc"
done

echo
echo "report: $REPORT"
