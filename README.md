# vllm37 — vLLM for Tesla K80 (sm_37) + CUDA 11.4

A personal fork of [vLLM](https://github.com/vllm-project/vllm) that runs on
legacy Tesla K80 GPUs (compute capability 3.7) with CUDA 11.4. Everything
ships as Docker images — the host never needs the build toolchain.

> **Status:** experimental. For upstream vLLM on modern GPUs, see
> [vllm-project/vllm](https://github.com/vllm-project/vllm).

## Why this exists

Upstream vLLM dropped sm_37 support several releases ago. This fork restores
it by pinning the toolchain to what K80 actually supports:

- CUDA 11.4
- GCC 10
- PyTorch 2.0.1
- Python 3.10

The v1 engine and most modern kernels require sm_70+, so this fork stays on
the v0 engine with PyTorch SDPA attention and no CUDA graphs.

## Hardware target

- Tesla K80 (sm_3.7, GK210B) — the only hardware this fork is tested on
  (2× K80 = 4 GPU dies, 12 GiB each)
- sm_3.5 Kepler Teslas (K20, K20X, K40) — would need
  `TORCH_CUDA_ARCH_LIST="3.5"` in the builder Dockerfile and are untested

## Quick start

Prebuilt builder image on Docker Hub:

```bash
docker pull dogkeeper886/vllm37-builder:latest
```

Build and run from this checkout:

```bash
cd docker/k80

# Tag the prebuilt builder so the Makefile finds it (saves ~120 min)
docker tag dogkeeper886/vllm37-builder:latest vllm37-builder:latest

# Build the runtime image from the current checkout (~10 min, ~7 GB)
make build-local
```

> **Warning:** the default `docker/k80/.env` sets `TP_SIZE=4`. TP=4 on 2× K80
> sharing a single CPU EPS rail can exceed the rail's spec and previously
> halted the system. Start with `TP_SIZE=1` (or `TP_SIZE=2` if you've
> verified your PSU) before running `make run`.

```bash
# Recommended: set a safe TP size first
echo "TP_SIZE=1" >> .env

# Start the vLLM server (defaults to TinyLlama 1.1B)
make run && make logs

# Test — uses the MODEL from .env
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "prompt": "Hello", "max_tokens": 50}'
```

Full configuration: [`docker/k80/README.md`](docker/k80/README.md).

## Tested configurations (2× K80)

| Config | Model | Status | Notes |
|---|---|---|---|
| TP=1 | TinyLlama 1.1B | Works | 4.1 GiB weights, ~58× concurrency |
| TP=2 | TinyLlama 1.1B | Works | ~164× concurrency, NCCL over SHM |
| TP=4 | TinyLlama 1.1B | Power-risky | 2× K80 on shared CPU EPS rail can exceed spec |

FP32 only. K80 has no native FP16, no BF16, and no compute capability for
the quantization schemes in upstream vLLM (GPTQ, AWQ, BnB all need sm_53+).

## Known limitations

- FP32 only
- Eager mode only — no `torch.compile`, no CUDA graphs
- No flash-attn, no xformers, no CUTLASS kernels
- Engine v1 disabled, v0 only
- Practical model size: ~7B parameters with TP=4, ~2B on a single die

## CI

Three manual-trigger GitHub Actions workflows run on a self-hosted K80 runner
(see [`CLAUDE.md`](CLAUDE.md)):

- `k80-build.yml` — Docker build of the runtime image
- `k80-runtime.yml` — Container + `/v1/completions` smoke test
- `k80-pipeline.yml` — Chains build → runtime

## Contributing

PRs are welcome but the scope is narrow: if a change isn't required to keep
K80 working, please send it upstream to
[vllm-project/vllm](https://github.com/vllm-project/vllm) instead.

## Upstream

Based on [vllm-project/vllm](https://github.com/vllm-project/vllm). Not
affiliated with or endorsed by the vLLM project or PyTorch Foundation.

If you use vLLM in your research, please cite the upstream paper:
[Efficient Memory Management for Large Language Model Serving with
PagedAttention](https://arxiv.org/abs/2309.06180).

## License

Apache 2.0, inherited from upstream vLLM. See [`LICENSE`](LICENSE).
