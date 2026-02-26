# vLLM for Tesla K80 (sm_37) + CUDA 11.4

Two-stage Docker build system for running vLLM on Tesla K80 GPUs
(compute capability 3.7). Follows the `dogkeeper886/ollama37` build pattern:
Rocky Linux 8, CUDA 11.4, GCC 10, CMake 4.

## Hardware Target

- 2x Tesla K80 cards (4 GPU dies, 12GB each, 48GB total)
- Tensor parallelism across all 4 dies
- Models up to ~4B parameters in FP32

## Quick Start

```bash
cd docker/k80/

# 1. Build builder image (~120 min first time)
make build-builder

# 2a. Build runtime from GitHub
make build-runtime

# 2b. OR build runtime from local source (for development)
make build-local

# 3. Run
make run
make logs

# 4. Test
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "prompt": "Hello", "max_tokens": 50}'
```

## Configuration

Edit `.env` to adjust build and runtime settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `JOBS` | 4 | Parallel build jobs |
| `VLLM_REPO` | dogkeeper886/vllm | GitHub repo for runtime build |
| `VLLM_BRANCH` | main | Git branch to clone |
| `MODEL` | TinyLlama-1.1B | Model to serve |
| `TP_SIZE` | 4 | Tensor parallel size |
| `DTYPE` | float32 | Only float32 on K80 |

## Architecture

```
docker/k80/
├── builder/
│   └── Dockerfile          # CUDA 11.4 + GCC 10 + CMake 4 + Python 3.10 + PyTorch 2.0.1
├── runtime/
│   ├── Dockerfile          # Clone from GitHub, build vLLM
│   └── Dockerfile.local    # Copy local source, build vLLM
├── Makefile                # Build orchestration
├── docker-compose.yml      # Runtime with GPU access
├── .env                    # Configuration
└── README.md               # This file
```

## Key Build Flags

- `TORCH_CUDA_ARCH_LIST="3.7"` - Target K80 compute capability
- `VLLM_BUILD_LEGACY_CUDA=1` - Skip CUTLASS-dependent kernels (require sm_70+)
- `VLLM_USE_V1=0` - Use V0 engine (V1 requires PyTorch 2.5+)
- `VLLM_ATTENTION_BACKEND=TORCH_SDPA` - Use PyTorch SDPA (no xformers/flash-attn)

## Limitations

- FP32 only (K80 lacks native FP16 compute, no BF16)
- No CUDA graphs (requires sm_70+)
- No flash attention or xformers
- No CUTLASS quantization kernels
- Eager mode only (no torch.compile)
