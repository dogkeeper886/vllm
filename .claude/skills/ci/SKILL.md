---
name: ci
description: Run K80-specific CI workflows for vLLM on the self-hosted runner. Use when triggering a remote build or smoke test, or deciding between local vs CI.
---

# CI

Trigger build and smoke-test workflows via GitHub Actions on the self-hosted
runner (labels `self-hosted`, `vllm`) that has K80 GPU access.

## When to use
- You don't have K80 hardware on the current machine
- Validating a branch/PR before merging
- Reproducing a build in a clean container environment

## Workflows
| File | Purpose |
|---|---|
| `.github/workflows/k80-build.yml` | Docker build of `vllm37-local` image |
| `.github/workflows/k80-runtime.yml` | Start container, poll `/v1/models`, smoke-test `/v1/completions` |
| `.github/workflows/k80-pipeline.yml` | `k80-build` then `k80-runtime` |

All run on `runs-on: [self-hosted, vllm]`.

## Trigger

GitHub UI → Actions tab → select workflow → **Run workflow**.

CLI:
```bash
gh workflow run k80-pipeline.yml -f tp_size=1 -f no_cache=false
gh workflow run k80-build.yml -f no_cache=true
gh workflow run k80-runtime.yml -f tp_size=1 -f model=TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

## Safety defaults (do not change casually)
- `tp_size: 1` — TP=4 previously halted the system (power-related). TP=2 is
  known-working but only with verified PSU setup. TP=4 is unvalidated.
- `NCCL_P2P_DISABLE=1` — required for any TP>1 on K80 (PCIe only). Baked
  into `docker/k80/docker-compose.yml`.
- Containers are torn down with `if: always()`; logs uploaded as artifact
  `k80-runtime-logs` for post-mortem.

## Local vs CI
- K80 present → `cd docker/k80 && make build-local && make run` is fastest
- K80 absent → trigger the CI pipeline
- Builder image rebuild (~120 min) is only triggered via the
  `rebuild_builder: true` input on `k80-build.yml`; otherwise it pulls
  `dogkeeper886/vllm37-builder:latest` on first CI run.

## Related
- `/k80-build` — build locally when you do have K80 hardware
- `docker/k80/README.md` — runtime configuration (model, TP size, etc.)
