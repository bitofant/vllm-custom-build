#!/bin/bash

################################################################################
# Async vLLM Build Launcher
#
# Runs build.sh in the background via nohup, survives SSH disconnects.
# Logs go to build.log in this directory.
#
# Usage:
#   ./build-async.sh          # start the build
#   ./build-async.sh status   # check if build is running + tail log
#   ./build-async.sh status --json  # machine-readable JSON status (for agents)
#   ./build-async.sh wait     # block until build finishes; exits with build's exit code
#   ./build-async.sh log      # tail -f the log
#   ./build-async.sh stop     # kill the running build
#
# Agent workflow:
#   ./build-async.sh start && ./build-async.sh wait
#   # or poll: ./build-async.sh status --json
#
# Key files (all in this directory):
#   build.log   full stdout+stderr of the build
#   build.pid   PID of the running build process (deleted on completion)
#   build.exit  exit code written on completion (0 = success, non-zero = failure)
################################################################################

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$SCRIPT_DIR/build.log"
PID_FILE="$SCRIPT_DIR/build.pid"
EXIT_FILE="$SCRIPT_DIR/build.exit"

is_running() {
  if [ -f "$PID_FILE" ]; then
    local pid
    pid=$(cat "$PID_FILE")
    if kill -0 "$pid" 2>/dev/null; then
      echo "$pid"
      return 0
    fi
  fi
  return 1
}

get_exit_code() {
  if [ -f "$EXIT_FILE" ]; then
    cat "$EXIT_FILE"
  else
    echo "null"
  fi
}

cmd="${1:-start}"

case "$cmd" in
  start)
    if pid=$(is_running); then
      echo "Build is already running (PID $pid). Use '$0 status' or '$0 stop'."
      exit 1
    fi

    # Clear previous exit code
    rm -f "$EXIT_FILE"

    echo "BUILD_STARTED at $(date)" | tee "$LOG_FILE"
    echo "Log: $LOG_FILE"

    # Wrapper captures exit code and writes log markers
    nohup bash -c '
      bash '"$SCRIPT_DIR"'/build.sh
      rc=$?
      echo $rc > '"$EXIT_FILE"'
      rm -f '"$PID_FILE"'
      if [ $rc -eq 0 ]; then
        echo "BUILD_SUCCESS at $(date)"
      else
        echo "BUILD_FAILED (exit $rc) at $(date)"
      fi
    ' >> "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE"
    echo "Build started (PID $(cat "$PID_FILE")). Detached from terminal."
    echo "  Check status : $0 status"
    echo "  Follow logs  : $0 log"
    echo "  Wait (agent) : $0 wait"
    ;;

  status)
    pid=$(is_running) && running=true || running=false
    exit_code=$(get_exit_code)

    if [ "${2:-}" = "--json" ]; then
      echo "{"
      echo "  \"running\": $running,"
      if $running; then
        echo "  \"pid\": $pid,"
      fi
      echo "  \"exit_code\": $exit_code,"
      echo "  \"log\": \"$LOG_FILE\","
      echo "  \"pid_file\": \"$PID_FILE\","
      echo "  \"exit_file\": \"$EXIT_FILE\""
      echo "}"
    else
      if $running; then
        echo "Build is RUNNING (PID $pid)"
      else
        if [ "$exit_code" = "null" ]; then
          echo "Build is NOT running (never started or log cleared)"
        elif [ "$exit_code" = "0" ]; then
          echo "Build is DONE — SUCCESS (exit 0)"
        else
          echo "Build is DONE — FAILED (exit $exit_code)"
        fi
      fi
      echo ""
      echo "--- Last 20 lines of $LOG_FILE ---"
      tail -n 20 "$LOG_FILE" 2>/dev/null || echo "(no log file)"
    fi
    ;;

  wait)
    # Block until build finishes; exit with the build's exit code.
    # Useful for agents: ./build-async.sh start && ./build-async.sh wait
    if ! [ -f "$PID_FILE" ] && ! [ -f "$EXIT_FILE" ]; then
      echo "No build in progress and no recorded exit code. Start one with '$0 start'." >&2
      exit 1
    fi

    echo "Waiting for build to complete..." >&2
    while pid=$(is_running); do
      sleep 10
    done

    exit_code=$(get_exit_code)
    if [ "$exit_code" = "null" ]; then
      echo "Build finished but no exit code recorded." >&2
      exit 1
    fi

    if [ "$exit_code" = "0" ]; then
      echo "BUILD_SUCCESS" >&2
    else
      echo "BUILD_FAILED (exit $exit_code)" >&2
    fi
    exit "$exit_code"
    ;;

  log)
    if ! [ -f "$LOG_FILE" ]; then
      echo "No log file found at $LOG_FILE"
      exit 1
    fi
    echo "Following $LOG_FILE (Ctrl+C to stop following, build continues)..."
    tail -f "$LOG_FILE"
    ;;

  stop)
    if pid=$(is_running); then
      echo "Stopping build (PID $pid)..."
      kill "$pid"
      rm -f "$PID_FILE"
      echo "Sent SIGTERM to PID $pid."
    else
      echo "No running build found."
      rm -f "$PID_FILE"
    fi
    ;;

  *)
    echo "Usage: $0 {start|status [--json]|wait|log|stop}"
    exit 1
    ;;
esac
