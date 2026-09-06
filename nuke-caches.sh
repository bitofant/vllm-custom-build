#!/usr/bin/env bash
# nuke-caches.sh — stop vLLM, empty the rebuildable caches, bring it back up.
#
# Scope is deliberately narrow. Only caches that cost TIME to rebuild are nuked:
#
#   ~/.cache/vllm      torch.compile AOT graphs + FlashInfer autotune configs.
#                      Bind-mounted by ~/scripts/vllm.sh; content-keyed, nothing
#                      ever evicts it. Cost of losing it: one cold boot.
#   docker build cache BuildKit layers. Cost: the next image build runs cold.
#
# NEVER touched (see CACHE-CLEANUP.md for the reasoning):
#   ~/.cache/huggingface   191 GB of MODEL WEIGHTS, not cache. 20-25 GB per
#                          re-download. Nuking this is never the right move here.
#   stopped containers     `docker container prune` would delete the very
#                          container this script just stopped and is about to
#                          restart. Do not add it.
#   vllm-custom:latest     pinned by vllm.sh; --images keeps it + one rollback.
#
# Usage: ./nuke-caches.sh [--dry-run] [--images] [--yes]
#   --dry-run  print what would happen, change nothing
#   --images   also remove old vllm-custom images (keeps latest + 1 rollback,
#              skips any image pinned by a container)
#   --yes      skip the confirmation prompt

set -uo pipefail

VLLM_SH=~/scripts/vllm.sh
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY=0; DO_IMAGES=0; ASSUME_YES=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY=1 ;;
    --images)  DO_IMAGES=1 ;;
    --yes|-y)  ASSUME_YES=1 ;;
    *) echo "Unknown option: $arg"; sed -n '2,25p' "$0"; exit 2 ;;
  esac
done

run() {  # echo + execute, or just echo under --dry-run
  if [ "$DRY" = 1 ]; then echo "  [dry-run] $*"; else echo "  + $*"; "$@"; fi
}

# --- guard: never prune build cache out from under a running build -----------
if [ -f "$REPO_DIR/build.pid" ] && kill -0 "$(cat "$REPO_DIR/build.pid")" 2>/dev/null; then
  echo "ERROR: a build is running (build.pid $(cat "$REPO_DIR/build.pid"))."
  echo "Pruning the build cache now would corrupt it. Wait: ./build-async.sh wait"
  exit 1
fi

# --- what is running right now? ----------------------------------------------
RUNNING=$(docker ps --filter "name=vllm_" --format '{{.Names}}')
RESTORE=""
if [ -n "$RUNNING" ]; then
  if [ "$(echo "$RUNNING" | wc -l)" -gt 1 ]; then
    echo "WARNING: multiple vLLM containers running:"
    echo "$RUNNING" | sed 's/^/    /'
    echo "Only one can hold the GPU + port 8000; will restore the first."
  fi
  RESTORE=$(echo "$RUNNING" | head -1)
fi

# Map container name -> vllm.sh config token, so the restart reuses the exact
# config. vllm.sh is sourced (its tail guards against dispatch on source).
CFG=""
if [ -n "$RESTORE" ]; then
  # shellcheck source=/dev/null
  source "$VLLM_SH" >/dev/null 2>&1
  CFG=$(rec_config_for_container "$RESTORE") || {
    echo "WARNING: '$RESTORE' is not in rec_config_for_container; will use"
    echo "         'docker start' instead of vllm.sh (no health wait)."
    CFG=""
  }
fi

# --- plan --------------------------------------------------------------------
VLLM_CACHE_SZ=$(du -sh ~/.cache/vllm 2>/dev/null | cut -f1)
BUILD_CACHE_SZ=$(docker builder du 2>/dev/null | awk '/^Total:/{print $2}')
echo "Plan:"
echo "  stop     : ${RESTORE:-<nothing running>}"
echo "  nuke     : ~/.cache/vllm (${VLLM_CACHE_SZ:-0})"
echo "  nuke     : docker build cache (${BUILD_CACHE_SZ:-unknown})"
[ "$DO_IMAGES" = 1 ] && echo "  nuke     : old vllm-custom images (keeping latest + 1 rollback)"
echo "  keep     : ~/.cache/huggingface (model weights), all containers"
if [ -n "$RESTORE" ]; then
  echo "  restart  : $RESTORE${CFG:+ (via vllm.sh $CFG)}"
  echo "             expect a COLD boot — for config 1 that is normally one OOM"
  echo "             crash + auto-restart, ~220 s to healthy instead of ~120 s."
fi
echo

if [ "$DRY" = 0 ] && [ "$ASSUME_YES" = 0 ]; then
  read -rp "Proceed? [y/N]: " reply
  case "$reply" in y|Y|yes|YES) ;; *) echo "Aborted."; exit 0 ;; esac
fi

# --- 1. stop ------------------------------------------------------------------
if [ -n "$RUNNING" ]; then
  echo "[1/4] stopping vLLM"
  run bash "$VLLM_SH" stop
else
  echo "[1/4] nothing running, skipping stop"
fi

# --- 2. vLLM startup caches ---------------------------------------------------
# Empty the CONTENTS, keep the directory: it is a live bind-mount target, and
# removing the dir itself makes the next `docker run` mount a root-owned stub.
echo "[2/4] emptying ~/.cache/vllm"
if [ -d ~/.cache/vllm ]; then
  run bash -c 'rm -rf ~/.cache/vllm/{*,.[!.]*} 2>/dev/null; true'
else
  run mkdir -p ~/.cache/vllm
fi

# --- 3. docker build cache (+ optional old images) ----------------------------
echo "[3/4] pruning docker build cache"
run docker builder prune -af

if [ "$DO_IMAGES" = 1 ]; then
  echo "      pruning old vllm-custom images"
  KEEP=$(docker image inspect vllm-custom:latest --format '{{.Id}}' 2>/dev/null)
  # Second-newest distinct image ID is the rollback; keep it too.
  ROLLBACK=$(docker images vllm-custom --format '{{.CreatedAt}}\t{{.ID}}' |
             sort -r | awk '{print $NF}' | awk '!seen[$0]++' | sed -n '2p')
  ROLLBACK_FULL=$(docker image inspect "$ROLLBACK" --format '{{.Id}}' 2>/dev/null)
  docker images vllm-custom --format '{{.ID}}' | sort -u | while read -r id; do
    full=$(docker image inspect "$id" --format '{{.Id}}' 2>/dev/null)
    [ "$full" = "$KEEP" ] && continue
    [ -n "$ROLLBACK_FULL" ] && [ "$full" = "$ROLLBACK_FULL" ] && continue
    # A stopped container pinning this image makes rmi fail; that is correct.
    # Never -f past it — that strands containers on a dead image.
    if [ "$DRY" = 1 ]; then
      echo "  [dry-run] docker rmi $id  $(docker image inspect "$id" --format '{{.RepoTags}}')"
    else
      docker rmi "$id" >/dev/null 2>&1 && echo "  + removed $id" \
        || echo "  - kept $id (pinned by a container)"
    fi
  done
fi

# --- 4. restart ---------------------------------------------------------------
if [ -n "$RESTORE" ]; then
  echo "[4/4] restarting $RESTORE (cold caches — this will be slow)"
  if [ -n "$CFG" ]; then
    run bash "$VLLM_SH" "$CFG"
  else
    run docker start "$RESTORE"
  fi
else
  echo "[4/4] nothing to restart"
fi

echo
echo "Done. Disk now:"
df -h / | tail -1
docker system df 2>/dev/null | sed 's/^/  /'
