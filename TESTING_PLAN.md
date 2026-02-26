# Tesla K80 (sm_37) + CUDA 11.4 Testing Plan

Branch: `k80-sm37-cuda11.4-support`
Commit: `13c413738`

---

## Phase 1: Build

### 1.1 Build builder image (~120 min first time)

```bash
cd docker/k80/
make build-builder
```

Verify:
```bash
docker run --rm vllm37-builder python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.version.cuda}')"
```
Expected: `PyTorch 2.0.1, CUDA: 11.4`

### 1.2 Build runtime image from local source (~15 min)

```bash
make build-local
```

Verify build succeeded:
```bash
docker run --rm vllm37-local python -c "import vllm; print(f'vLLM loaded')"
```

### 1.3 Alternative: build runtime from GitHub

```bash
# Edit .env to point VLLM_REPO and VLLM_BRANCH to your fork
make build
```

---

## Phase 2: Smoke Test

### 2.1 Start server with docker compose

```bash
cd docker/k80/
docker compose up -d
docker compose logs -f
```

Wait for: `Uvicorn running on http://0.0.0.0:8000`

### 2.2 API completions test

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "prompt": "Hello, world!",
    "max_tokens": 50
  }'
```

Expected: JSON response with generated text.

### 2.3 Chat completions test

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 50
  }'
```

### 2.4 Stop server

```bash
docker compose down
```

---

## Phase 3: Tensor Parallelism Tests

### 3.1 TP=1 (single GPU die)

```bash
docker run --rm --runtime=nvidia --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm37-local \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dtype float32 --enforce-eager \
  --tensor-parallel-size 1 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85
```

Then test with curl (same as 2.2).

### 3.2 TP=2 (two GPU dies)

```bash
docker run --rm --runtime=nvidia --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm37-local \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dtype float32 --enforce-eager \
  --tensor-parallel-size 2 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85
```

### 3.3 TP=4 (all four GPU dies)

```bash
docker run --rm --runtime=nvidia --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm37-local \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --dtype float32 --enforce-eager \
  --tensor-parallel-size 4 \
  --max-model-len 2048 \
  --gpu-memory-utilization 0.85
```

---

## Phase 4: GPU Utilization Verification

While a server is running:

```bash
# From host
nvidia-smi

# Or inside container
docker exec <container_id> nvidia-smi
```

Check:
- [ ] All expected GPU dies show memory usage
- [ ] vLLM worker processes visible on each die
- [ ] No GPU errors

---

## Phase 5: Log Verification

Check server startup logs for these expected messages:

- [ ] `"Kepler GPU detected (sm < 70): forcing eager mode and disabling CUDA graphs."`
- [ ] `"Using Torch SDPA attention backend for Kepler GPU (sm < 60)."`
- [ ] No `FlashAttention` or `xformers` import errors
- [ ] `VLLM_USE_V1=0` (V0 engine active)
- [ ] Model loads successfully in float32

---

## Phase 6: Stress / Longer Generation

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "prompt": "Once upon a time in a land far away, there lived a",
    "max_tokens": 500,
    "temperature": 0.7
  }'
```

Check:
- [ ] No CUDA errors during longer generation
- [ ] Output is coherent (not garbage)
- [ ] Server stays up after multiple requests

---

## Troubleshooting

### Build fails: CUDA kernel compile error
- Check the build log for the exact `.cu` file and line
- Likely cause: FP16 intrinsic we missed in a kernel file other than dtype_float16.cuh
- Fix: add `#if __CUDA_ARCH__ >= 530` guard with FP32 fallback

### Build fails: PyTorch 2.0 API incompatibility in torch_bindings.cpp
- Check error for the exact function/macro that's missing
- Likely cause: op registration API changed between PyTorch 2.0 and 2.7
- Fix: add `#if TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR < 4` guards

### Runtime: shared memory exceeded
- Reduce block size: `--block-size 8`
- K80 has 48KB shared memory per SM

### Runtime: OOM
- Lower `--gpu-memory-utilization 0.7`
- Lower `--max-model-len 1024` or `512`
- Use smaller model

### Runtime: NCCL errors with TP > 1
- Add `NCCL_DEBUG=INFO` env var to see details
- Try `NCCL_P2P_DISABLE=1` if P2P fails over PCIe
- K80 uses PCIe (no NVLink), so NCCL should still work but slower

---

## Files Changed (for reference)

| File | Change |
|------|--------|
| `CMakeLists.txt` | sm_37 arch, VLLM_BUILD_LEGACY_CUDA option |
| `csrc/attention/dtype_float16.cuh` | FP16 intrinsic FP32 fallbacks |
| `csrc/torch_bindings.cpp` | Guard CUTLASS ops with VLLM_BUILD_LEGACY_CUDA |
| `setup.py` | Skip flash-attn extensions in legacy mode |
| `vllm/platforms/cuda.py` | K80 routing (V0, eager, NCCL, SDPA) |
| `vllm/attention/backends/torch_sdpa.py` | NEW - TorchSDPA V0 attention backend |
| `requirements/cuda_k80.txt` | NEW - K80-compatible deps |
| `docker/k80/builder/Dockerfile` | NEW - builder image |
| `docker/k80/runtime/Dockerfile` | NEW - runtime (GitHub clone) |
| `docker/k80/runtime/Dockerfile.local` | NEW - runtime (local source) |
| `docker/k80/Makefile` | NEW - build orchestration |
| `docker/k80/docker-compose.yml` | NEW - runtime compose |
| `docker/k80/.env` | NEW - configuration |
