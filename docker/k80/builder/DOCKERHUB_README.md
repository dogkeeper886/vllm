# vllm37-builder

Build environment for compiling [vLLM](https://github.com/vllm-project/vllm) with **Tesla K80** (CUDA compute capability 3.7) support.

NVIDIA dropped Kepler (sm_37) support in CUDA 12+, so this image pins an older toolchain that still targets it.

## What's Inside

| Component | Version | Why |
|-----------|---------|-----|
| Base OS | Rocky Linux 8 | Long-term support, RHEL-compatible |
| CUDA Toolkit | 11.4.4 | Last toolkit line supporting sm_37 |
| cuDNN | 8.x | Required for vLLM CUDA kernels |
| GCC | 10.5.0 | Max version allowed by CUDA 11.4 |
| CMake | 4.0.1 | Modern CMake for build configuration |
| Python | 3.10.16 | Required by vLLM |
| PyTorch | 2.0.1 (from source) | Built with `TORCH_CUDA_ARCH_LIST="3.7"` |
| NumPy | 1.26.4 | PyTorch/vLLM dependency |

## Usage

```bash
docker pull dogkeeper886/vllm37-builder:latest

docker run -it dogkeeper886/vllm37-builder:latest bash
```

### Build vLLM inside the container

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
```

## Build Args

The image can be rebuilt with custom versions:

```bash
docker build \
  --build-arg CUDA_VERSION=11.4.4 \
  --build-arg GCC_VERSION=10.5.0 \
  --build-arg CMAKE_VERSION=4.0.1 \
  --build-arg PYTHON_VERSION=3.10.16 \
  --build-arg PYTORCH_VERSION=v2.0.1 \
  --build-arg JOBS=4 \
  -t vllm37-builder \
  -f docker/k80/builder/Dockerfile .
```

> **Note:** The default `JOBS=4` keeps memory usage reasonable. Increase if your build machine has enough RAM (~2 GB per job for GCC/PyTorch).

## Why This Exists

The Tesla K80 (Kepler, compute 3.7) is still widely available on the used market and in older cloud instances. CUDA 12+ dropped Kepler support, so running modern ML frameworks on K80 requires carefully pinned toolchain versions. This image provides that ready-made environment.

## Related

- [dogkeeper886/ollama37](https://hub.docker.com/r/dogkeeper886/ollama37) — Ollama built with K80 support
