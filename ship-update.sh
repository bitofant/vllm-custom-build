#!/bin/bash

################################################################################
# Ship a vLLM submodule bump
#
# Run this AFTER ./update.sh (and after the build you care about), when the
# working tree holds the new vllm/ pointer + build.number/build.history churn.
#
# It does, in order:
#   1. branch off main               (joran/bump-vllm-submodule-commit)
#   2. git add -A
#   3. commit "bump vllm submodule to commit <sha>"
#   4. push to origin
#   5. gh pr create
#   6. gh pr merge (merge commit, deletes remote branch)
#   7. back to an up-to-date main, local bump branch deleted
#
# Every git/gh command is echoed as "+ <command>" before it runs.
#
# Usage:
#   ./ship-update.sh
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

BRANCH="joran/bump-vllm-submodule-commit"
BASE="main"

# Seconds to wait after opening the PR before merging it (see "merge" below).
MERGE_DELAY="${MERGE_DELAY:-3}"

# Echo a command, then run it.
run() {
    echo "+ $*"
    "$@"
}

die() { echo "ERROR: $*" >&2; exit 1; }

########################  preflight  ###########################################

command -v gh >/dev/null || die "gh CLI not found (needed to open/merge the PR)."
gh auth status >/dev/null 2>&1 || die "gh is not authenticated — run: gh auth login"

[[ -n "$(git status --porcelain)" ]] \
    || die "nothing to commit — did you run ./update.sh yet?"

CURRENT_BRANCH="$(git rev-parse --abbrev-ref HEAD)"
[[ "$CURRENT_BRANCH" == "$BASE" ]] \
    || die "expected to start from '$BASE', but you're on '$CURRENT_BRANCH'."

# Old pointer = the gitlink recorded in the last commit; new pointer = what's
# checked out in vllm/ right now. Grab both before we commit anything.
OLD_SHA="$(git rev-parse "HEAD:vllm")"
NEW_SHA="$(git -C vllm rev-parse HEAD)"
[[ "$OLD_SHA" != "$NEW_SHA" ]] \
    || die "vllm/ still points at $(git -C vllm rev-parse --short HEAD) — no submodule bump to ship."

NEW_SHORT="$(git -C vllm rev-parse --short HEAD)"
OLD_DESCRIBE="$(git -C vllm describe --tags "$OLD_SHA" 2>/dev/null || echo "$OLD_SHA")"
NEW_DESCRIBE="$(git -C vllm describe --tags "$NEW_SHA" 2>/dev/null || echo "$NEW_SHA")"

echo ">>> vllm submodule: $OLD_DESCRIBE  ->  $NEW_DESCRIBE"
echo ">>> Files to be committed:"
git status --short
echo

########################  branch + commit  #####################################

# -B so a leftover branch from an earlier run is just reset onto main;
# uncommitted changes ride along untouched.
run git checkout -B "$BRANCH"

run git add -A
run git commit -m "bump vllm submodule to commit $NEW_SHORT"

########################  push + PR  ###########################################

# --force-with-lease: this branch is a disposable per-bump scratch branch, so a
# stale remote copy from an aborted run should be overwritten, not merged with.
run git push -u --force-with-lease origin "$BRANCH"

PR_TITLE="bump vllm submodule to commit $NEW_SHORT"
PR_BODY="Bump \`vllm/\` submodule to \`$NEW_DESCRIBE\` (was \`$OLD_DESCRIBE\`)."

echo "+ gh pr create --base $BASE --head $BRANCH --title \"$PR_TITLE\" --body \"$PR_BODY\""
PR_URL="$(gh pr create --base "$BASE" --head "$BRANCH" --title "$PR_TITLE" --body "$PR_BODY")"
echo ">>> PR: $PR_URL"

########################  merge  ###############################################

# GitHub computes a PR's mergeability asynchronously after creation; merging
# too fast can be rejected while that state is still "unknown". Give it a
# moment to settle.
run sleep "$MERGE_DELAY"

run gh pr merge "$PR_URL" --merge --delete-branch

########################  back to main  ########################################

# Fast-forward local $BASE *while it is not checked out*, then switch to it.
# This keeps the working tree from ever passing through the pre-update state:
# checking out a stale $BASE first (and pulling afterwards) would rewind
# build.number/build.history/Dockerfile/... to their pre-bump contents for as
# long as the pull took — and leave them there for good if the pull failed.
# `fetch <remote> <ref>:<ref>` refuses a non-fast-forward on its own, so this
# is as strict as the `pull --ff-only` it replaces.
run git fetch origin "$BASE:$BASE"
run git checkout "$BASE"

# --delete-branch above removes the remote branch, but the local one survives
# whenever gh couldn't delete it — which is the normal case here, since it's
# still checked out at merge time.
if git show-ref --quiet "refs/heads/$BRANCH"; then
    run git branch -D "$BRANCH"
else
    echo ">>> Local branch $BRANCH already gone."
fi

# Drop the now-dangling origin/$BRANCH remote-tracking ref.
run git fetch --prune origin

echo
echo ">>> Done. On $BASE at $(git rev-parse --short HEAD) — merged $PR_URL"
