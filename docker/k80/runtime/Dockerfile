# vLLM K80 Runtime Image (GitHub clone)
# Clones vLLM fork from GitHub and builds against the builder image.
#
# Requires: vllm37-builder image (built from docker/k80/builder/Dockerfile)
#
# Build:
#   docker build -t vllm37 \
#     --build-arg VLLM_REPO=https://github.com/<your-user>/vllm.git \
#     --build-arg VLLM_BRANCH=main \
#     -f docker/k80/runtime/Dockerfile .

ARG BUILDER_IMAGE=vllm37-builder
FROM ${BUILDER_IMAGE}

ARG VLLM_REPO=https://github.com/dogkeeper886/vllm.git
ARG VLLM_BRANCH=main
ARG MAX_JOBS=4

# ============================================================================
# Clone and build vLLM
# ============================================================================
RUN cd /usr/local/src && \
    git clone --depth 1 --branch ${VLLM_BRANCH} ${VLLM_REPO} vllm37

WORKDIR /usr/local/src/vllm37

# Install Python dependencies (use K80-specific requirements)
RUN pip install --no-cache-dir -r requirements/cuda_k80.txt 2>/dev/null || \
    pip install --no-cache-dir -r requirements/common.txt

# Build and install vLLM with legacy CUDA flags
RUN CUDA_HOME=/usr/local/cuda-11.4 \
    TORCH_CUDA_ARCH_LIST="3.7" \
    VLLM_BUILD_LEGACY_CUDA=1 \
    MAX_JOBS=${MAX_JOBS} \
    pip install -e . --no-build-isolation

# ============================================================================
# Runtime configuration
# ============================================================================
ENV VLLM_USE_V1=0
ENV CUDA_HOME=/usr/local/cuda-11.4
ENV VLLM_ATTENTION_BACKEND=TORCH_SDPA

EXPOSE 8000

ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", \
     "--dtype", "float32", \
     "--enforce-eager", \
     "--tensor-parallel-size", "4", \
     "--max-model-len", "2048", \
     "--gpu-memory-utilization", "0.85"]
