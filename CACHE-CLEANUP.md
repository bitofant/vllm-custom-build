# Cache Cleanup

Reclaiming disk on this box. Written for a coding agent — every command is verified, every
"never" is a real failure mode. Read the whole section before running anything in it.

Five independent stores. They are listed in **safety order**: 1 is pure disk-for-time, 5 can
cost a 25 GB re-download. Work down the list and stop when you have enough space.

## Baseline (measured 2026-09-05, root fs 91% full — 172 G free of 1.8 T)

| # | Store | Size | Reclaimable | Measure with |
|---|-------|------|-------------|--------------|
| 1 | Docker build cache | 209.9 GB | 120.9 GB private | `docker builder du \| tail -4` |
| 2 | Stopped containers | 2.4 GB | 2.1 GB | `docker ps -a` |
| 3 | `vllm-custom` images | 152.7 GB | 34.3 GB | `docker images vllm-custom` |
| 4 | `~/.cache/vllm` | 200 MB | all | `du -sh ~/.cache/vllm` |
| 5 | `~/.cache/huggingface` | 191 GB | ~66 GB unreferenced | `du -sh ~/.cache/huggingface/hub/*` |

If these numbers have drifted a lot, re-measure before acting — do not trust the table.

---

## 1. Docker build cache — biggest win, lowest risk

Costs only rebuild time (~60 min, per `CLAUDE.md`). Never touches images or containers.

Docker here is **29.6.1**, where `docker builder prune` is an alias for `docker buildx prune`.
The size flags are `--max-used-space` / `--min-free-space` / `--reserved-space`.
`--keep-storage` is the old BuildKit flag: it still works but is **deprecated and silently
remapped to `--reserved-space`**, which has different semantics (space *kept*, not a cap).
Use the current flags; do not assume a `--keep-storage` invocation is a no-op.

```bash
docker builder du | tail -4                          # measure
docker builder prune -f --filter until=168h          # drop >1 week old (preferred)
docker builder prune -f --max-used-space 50GB        # cap the store
docker builder prune -af                             # nuke everything
```

**Never prune while a build is running** — it will corrupt the in-flight build. Check first:

```bash
./build-async.sh status --json    # or: test -f build.pid && echo BUILD RUNNING
```

## 2. Stopped containers

Small on their own, but they **pin images**, which blocks §3. Check before pruning: this box
runs non-vLLM stacks (`owui`, `openclaw-*`, `webserver_nginx`, comfyui) whose stopped
containers may still be wanted.

```bash
docker ps -a --format '{{.Names}}\t{{.Image}}\t{{.Status}}'
docker rm vllm_qwen_3_8_bf16kv vllm_nemotron35_lightning_30b_a3b    # targeted (preferred)
docker container prune -f                                          # ALL stopped, all stacks
```

Removing a stopped `vllm_*` container is cheap: `vllm.sh <config>` recreates it, and both the
HF weights (§5) and the vLLM startup caches (§4) are bind-mounted, so nothing re-downloads
and nothing recompiles.

## 3. Old `vllm-custom` images

Each successful build produces **three tags pointing at one image ID**: `b<N>`,
`<git-describe>`, and `latest` (which moves). Removing one tag only untags — the image is
freed when its last tag goes. Prefer removing by ID.

Keep `latest` **plus one rollback** (at baseline: b21 = latest, b20 = rollback).

List everything safe to consider, latest excluded automatically:

```bash
KEEP=$(docker image inspect vllm-custom:latest --format '{{.Id}}')
docker images vllm-custom --format '{{.ID}}' | sort -u | while read id; do
  [ "$(docker image inspect "$id" --format '{{.Id}}')" = "$KEEP" ] ||
    echo "$id $(docker image inspect "$id" --format '{{.RepoTags}}')"
done
```

Then `docker rmi <id>` each one you chose. If `rmi` fails with a container reference, a
stopped container pins it — remove that container (§2) and retry.

- **Never** `docker rmi` the ID behind `vllm-custom:latest`. `~/scripts/vllm.sh` pins that tag
  (`DOCKER_IMAGE="vllm-custom:latest"`); killing it breaks every config.
- **Never** use `docker rmi -f` to push past a container pin. Remove the container instead;
  `-f` leaves containers referencing a dead image (baseline already has two such strays
  pointing at the deleted `416704aa67b3`).
- Do not edit `build.number` / `build.history` to match. They are an append-only record of
  builds, not of images on disk.

## 4. `~/.cache/vllm` — vLLM startup caches

Bind-mounted into every container by `~/scripts/vllm.sh`. Holds `torch_compile_cache/`
(~200 MB of AOT graphs), `flashinfer_autotune_cache/` (tuned fp4 GEMM configs), `modelinfos/`.
Content-keyed by vLLM version + compilation config, so entries accumulate per build and
**nothing ever evicts them**.

```bash
du -sh ~/.cache/vllm/*
rm -rf ~/.cache/vllm/*        # full reset — note the /*
```

- **Keep the directory itself.** `rm -rf ~/.cache/vllm` (no `/*`) destroys a live mount point;
  do it only with no vLLM container running, and `mkdir -p` it again immediately.
- Cost of a reset: the next boot is cold. For `vllm.sh` config `1` that currently means one
  OOM crash + `unless-stopped` auto-restart before it comes up (~220 s instead of ~119 s) —
  expected, documented in the `vllm.sh` config-1 comment block.
- Targeted alternative: drop stale flashinfer versions only. Compare
  `ls ~/.cache/vllm/flashinfer_autotune_cache/` against the pin in `Dockerfile`
  (`grep flashinfer Dockerfile`) and delete the version dirs that no longer match.

## 5. HuggingFace hub — largest store, most care

191 GB. Layout is `hub/models--<org>--<name>/{blobs,refs,snapshots}`; `blobs/` holds the bytes.
There is **no `hf` or `huggingface-cli` on PATH**, so removal is `rm -rf` of the whole
`models--*` directory. Deleting individual blobs leaves dangling snapshot symlinks — don't.

Build the referenced set first (`org/name` maps to `models--org--name`):

```bash
grep -oP 'MODEL_ID="\K[^"]+' ~/scripts/vllm.sh          # target models
grep -oP '"model":"\K[^"]+' ~/scripts/vllm.sh           # MTP drafters
du -sh ~/.cache/huggingface/hub/* | sort -rh            # what is on disk
```

**Absence from `vllm.sh` does not mean unused.** Other stacks pull their own models and are
invisible to that grep — `sentence-transformers/all-MiniLM-L6-v2` (888 MB) belongs to
`owui`/`openclaw`, not vLLM. Re-downloads are 20–25 GB each.

**Confirm with the user before deleting any model.** Do not treat a cross-reference miss as
authorization.

Unreferenced by `vllm.sh` at baseline (~66 GB, candidates only):
`QuantTrio--Qwen3.5-35B-A3B-AWQ` (24 G), `QuantTrio--Qwen3.6-27B-AWQ` (21 G),
`nvidia--Qwen3.6-27B-NVFP4` (21 G).

Never touch `~/.cache/huggingface/token` or `stored_tokens` — those are credentials, not cache.

---

## `./nuke-caches.sh` — the scripted path

Covers §1 and §4, which are the two stores that are pure disk-for-time: stop vLLM, empty
`~/.cache/vllm` + the build cache, restart the same config. Prefer it over doing those by hand.

```bash
./nuke-caches.sh --dry-run     # print the plan, change nothing
./nuke-caches.sh               # prompts, then does it
./nuke-caches.sh --images      # also drop old vllm-custom images (keeps latest + 1 rollback)
```

It refuses to run while a build is in progress, never touches §5 (model weights), and never
prunes containers — `docker container prune` would delete the container it just stopped and is
about to restart. §3 is opt-in via `--images`; §5 stays manual and user-confirmed.

## Safe sweep

No confirmation needed, no model or image loss. Reclaims ~120 GB at baseline.

```bash
test -f build.pid && { echo "build running — abort"; exit 1; }
docker builder prune -f --filter until=168h
docker container prune -f      # review `docker ps -a` first if other stacks matter
docker system df               # verify
```

Anything beyond this (§3 image removal, §4 reset, §5 model deletion) is a judgement call —
state what you intend to remove and how much it reclaims, then ask.
