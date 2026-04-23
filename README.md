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

- Tesla K80 (compute capability 3.7) — tested on 2× K80 cards (4 dies)
- Other sm_37 cards (Tesla M60, GRID K520) — untested but should work

## Quick start

Prebuilt builder image on Docker Hub:

```bash
docker pull dogkeeper886/vllm37-builder:latest
```

Build and run from source:

```bash
cd docker/k80

# Tag the prebuilt builder so the Makefile finds it (saves ~120 min)
docker tag dogkeeper886/vllm37-builder:latest vllm37-builder:latest

# Build the runtime image from the current checkout (~10 min)
make build-local

# Start the vLLM server (defaults: TinyLlama 1.1B, TP=4)
make run && make logs

# Test
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
