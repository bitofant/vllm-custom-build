#!/bin/bash

################################################################################
# vLLM Vendored-Source Updater + Build Launcher
#
# Automates the mechanical part of the `/latest` workflow:
#   1. Fast-forward ./vllm/ to the tip of upstream main (never forces).
#   2. Report the new version and how many commits were pulled.
#   3. Kick off a detached async build via ./build-async.sh start.
#
# It deliberately does NOT wait on the build (~60 min) and does NOT attempt
# to fix build failures — those are tuned Dockerfile patches that need a human
# (or agent) to re-validate against new upstream. See CLAUDE.md.
#
# Usage:
#   ./update.sh              # update + start build
#   ./update.sh --no-build   # update only, don't kick off a build
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VLLM_DIR="$SCRIPT_DIR/vllm"
BRANCH="main"

NO_BUILD=0
[[ "${1:-}" == "--no-build" ]] && NO_BUILD=1

cd "$VLLM_DIR"

# Refuse to clobber local work: the vendored tree must be clean.
if [[ -n "$(git status --porcelain)" ]]; then
    echo "ERROR: $VLLM_DIR has uncommitted changes — refusing to update." >&2
    echo "       vllm/ is a vendored upstream clone; patch via the top-level Dockerfile instead." >&2
    git status --short >&2
    exit 1
fi

OLD_DESCRIBE="$(git describe --tags)"

echo ">>> Fetching upstream..."
git fetch origin

# Ensure we're on the tracked branch before fast-forwarding.
if [[ "$(git rev-parse --abbrev-ref HEAD)" != "$BRANCH" ]]; then
    git checkout "$BRANCH"
fi

echo ">>> Fast-forwarding $BRANCH (--ff-only; will abort on divergence)..."
if ! git pull --ff-only origin "$BRANCH"; then
    echo "ERROR: --ff-only pull failed (local commits or non-fast-forward)." >&2
    echo "       Resolve manually; not forcing." >&2
    exit 1
fi

NEW_DESCRIBE="$(git describe --tags)"
NEW_SHA="$(git rev-parse --short HEAD)"
COUNT="$(git rev-list --count "${OLD_DESCRIBE}..HEAD" 2>/dev/null || echo '?')"

echo
echo ">>> Updated:  $OLD_DESCRIBE  ->  $NEW_DESCRIBE ($NEW_SHA)"
echo ">>> Pulled:   $COUNT commit(s)"
git log -1 --format='>>> Tip:      %ci %s'
echo

if [[ "$NO_BUILD" == "1" ]]; then
    echo ">>> --no-build: skipping build. Start it with: ./build-async.sh start"
    exit 0
fi

echo ">>> Kicking off async build..."
cd "$SCRIPT_DIR"
./build-async.sh start
sleep 2
./build-async.sh status --json

echo
echo ">>> Build running (detached). It does NOT block here (~60 min)."
echo ">>> Follow: ./build-async.sh log   |   Wait: ./build-async.sh wait"
