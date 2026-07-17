---
description: Update ./vllm/ to the latest upstream commit and kick off an async rebuild
---

Update the vendored vLLM source to the latest upstream commit and start a fresh build.

## Steps

1. **Update `./vllm/` to the latest commit.** This is a vendored upstream clone (do **not** edit files inside it directly — all patches live in the top-level `Dockerfile`).
   - `cd vllm`
   - Fetch and fast-forward to the tip of the upstream default branch:
     - `git fetch origin`
     - `git checkout main` (the tracked branch) if not already on it
     - `git pull --ff-only origin main`
   - If the pull fails (e.g. local commits or a non-fast-forward), stop and report the situation to the user rather than forcing it. Do not discard local changes without asking.
   - Report the new `git describe` / short SHA and how many commits were pulled in.

2. **Kick off an async rebuild.** From the repo root (`/home/joran/src/vllm`):
   - `./build-async.sh start`
   - Confirm it started (check `./build-async.sh status --json` shows it running).

3. **Stop here.** Do **not** block on `./build-async.sh wait` or tail the full log — the build takes ~60 minutes.

## After kicking off the build

Tell the user the build is running, and that you're happy to check whether it succeeded — and to investigate and fix any failures — once it finishes. Remind them they can ask you to poll `./build-async.sh status` / `wait`, or they can check the live log themselves with `./build-async.sh log`.
