# vLLM for Tesla K80 (sm_37) + CUDA 11.4

This is a fork of vLLM targeting legacy Tesla K80 GPUs (compute capability 3.7)
on CUDA 11.4. All K80-specific work lives under `docker/k80/`.

## Build and run: Docker only

**Do not attempt to build vLLM directly on the host.** The build requires
PyTorch 2.0.1 compiled against CUDA 11.4 with sm_37, GCC 10, CMake 4, and
Python 3.10 — pinned together in the builder image. Host builds will fail
or waste hours.

All builds and runs go through `docker/k80/`:

```bash
cd docker/k80

# Build from local source (requires vllm37-builder already present)
make build-local

# Start server
make run && make logs

# Stop
make stop
```

If `vllm37-builder` is missing, either `make build-builder` (~120 min) or
pull the prebuilt one:

```bash
docker pull dogkeeper886/vllm37-builder:latest
docker tag dogkeeper886/vllm37-builder:latest vllm37-builder:latest
```

Full details: `docker/k80/README.md`.

## CI / CD

GitHub Actions workflows (manual trigger, self-hosted runner with K80 access):

| Workflow | Purpose |
|---|---|
| `k80-build.yml` | Docker build of `vllm37-local` image |
| `k80-runtime.yml` | Start container, health check, `/v1/completions` smoke test |
| `k80-pipeline.yml` | Full: build → runtime smoke |

All run on `runs-on: [self-hosted, vllm]`. CI defaults to `TP_SIZE=1` for
safety (see *Safety rules* below).

## Safety rules — read before running workloads on hardware

- **Do not run `nvidia-smi -pl`** to change power limits. Has caused system
  halts on this hardware. Not a CI concern, but never script it.
- **TP=4 is power-risky.** 2x K80 on a shared CPU EPS connector can exceed
  the rail rating under full load. CI defaults to TP=1; use TP=2 manually
  after verifying PSU setup. TP=4 is unvalidated.
- **`NCCL_P2P_DISABLE=1`** is required for any TP>1 run (K80 is PCIe-only,
  no NVLink). Already baked into `docker/k80/docker-compose.yml`.

## Hardware constraints (sm_37)

- No FP16 compute, no BF16, no quantization (GPTQ/AWQ/BnB need sm_53+).
- No flash-attn, no xformers, no CUDA graphs (need sm_70+).
- FP32 only → max ~7B TP=4, ~2B single die.
- Runtime env (set in compose): `VLLM_USE_V1=0`,
  `VLLM_ATTENTION_BACKEND=TORCH_SDPA`, `TORCHDYNAMO_DISABLE=1`.

## Repo conventions

- K80-specific files live under `docker/k80/` and `requirements/cuda_k80.txt`.
  Upstream vLLM files should remain unmodified unless a K80 fix requires it.
- Commit prefix `[Hardware]` for K80-specific patches (see `git log`).
- Workflow files for K80 are prefixed `k80-` so they don't collide with
  upstream vLLM workflows (`lint-and-deploy.yaml`, `publish.yml`, etc.).
