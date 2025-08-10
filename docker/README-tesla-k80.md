# Tesla K80 Docker Setup for vLLM

This directory contains Docker configuration for running vLLM on Tesla K80 GPUs with compute capability 3.7.

## Prerequisites

### Host System Requirements
- **NVIDIA Driver**: Version 470.x (last driver supporting Tesla K80)
  - Driver 470 supports CUDA 11.0 through 11.4 (CUDA 11.5+ not supported)
- **Docker**: Version 20.10 or later
- **NVIDIA Container Toolkit**: Latest version
- **Tesla K80** with 12GB VRAM per GPU (dual-GPU card)
- **Host OS**: Linux distribution compatible with driver 470

### Driver Installation
```bash
# Check current driver version
nvidia-smi

# For Ubuntu/Debian - install driver 470
sudo apt update
sudo apt install nvidia-driver-470
sudo reboot

# For Rocky Linux/RHEL - install driver 470
sudo dnf install nvidia-driver:470xx nvidia-driver-cuda:470xx
sudo reboot
```

### NVIDIA Container Toolkit Installation
```bash
# Ubuntu/Debian (Modern method - Ubuntu 20.04+)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker

# Ubuntu/Debian (Legacy method - Ubuntu 18.04 and older)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker

# Rocky Linux/RHEL
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-docker.repo
sudo dnf install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Build Instructions

### Option 1: Docker Compose (Recommended)
```bash
# Build and run vLLM server
docker-compose -f docker/docker-compose.k80.yml up --build vllm-k80

# Or run in development mode
docker-compose -f docker/docker-compose.k80.yml up --build vllm-k80-dev
```

### Option 2: Manual Docker Build
```bash
# Build the image
docker build -f docker/Dockerfile.tesla-k80 -t vllm:tesla-k80 .

# Run the container
docker run --runtime=nvidia --gpus all \
  -v $(pwd):/workspace/vllm \
  -p 8000:8000 \
  -e VLLM_ATTENTION_BACKEND=XFORMERS \
  -e VLLM_DISABLE_QUANTIZATION=1 \
  vllm:tesla-k80
```

## Testing the Installation

### Verify GPU Detection
```bash
# Test NVIDIA runtime
docker run --rm --gpus all vllm:tesla-k80 nvidia-smi

# Test CUDA availability
docker run --rm --gpus all vllm:tesla-k80 python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'Device count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    print(f'Compute capability: {torch.cuda.get_device_capability(0)}')
"
```

### Test vLLM Import
```bash
docker run --rm --gpus all vllm:tesla-k80 python -c "
import vllm
print('vLLM imported successfully')
from vllm.platforms import current_platform
print(f'Platform: {current_platform}')
print(f'Device capability: {current_platform.get_device_capability()}')
"
```

### Run Simple Inference
```bash
# Start server
docker-compose -f docker/docker-compose.k80.yml up vllm-k80

# Test inference (in another terminal)
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "microsoft/DialoGPT-small",
    "prompt": "Hello, how are you?",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

## Configuration Options

### Environment Variables
- `VLLM_ATTENTION_BACKEND=XFORMERS` - Force XFormers (required for K80)
- `VLLM_DISABLE_QUANTIZATION=1` - Disable quantization features
- `VLLM_WORKER_USE_RAY=0` - Disable Ray for single GPU
- `CUDA_VISIBLE_DEVICES=0` - Specify GPU device

### Model Recommendations for Tesla K80
Due to K80's limitations, use smaller models:

**Recommended Models:**
- `microsoft/DialoGPT-small` (117M parameters)
- `gpt2` (124M parameters)
- `distilgpt2` (82M parameters)
- `facebook/opt-350m` (350M parameters)

**Avoid These Models:**
- Large models (>1B parameters) may cause OOM
- Models requiring quantization
- Models with attention optimizations for newer GPUs

## Troubleshooting

### Common Issues

#### "CUDA driver version is insufficient"
```bash
# Check driver version
nvidia-smi
# Should show driver 470.x

# Verify container can access GPU
docker run --rm --gpus all nvidia/cuda:11.4.3-base-rockylinux8 nvidia-smi
```

#### "No CUDA-capable device is detected"
```bash
# Check NVIDIA container runtime
docker info | grep nvidia
# Should show nvidia runtime

# Test GPU access
docker run --rm --gpus all vllm:tesla-k80 nvidia-smi
```

#### "RuntimeError: CUDA out of memory"
Reduce model size or memory usage:
```bash
# Use smaller model
--model microsoft/DialoGPT-small

# Reduce memory utilization
--gpu-memory-utilization 0.6

# Reduce max sequence length
--max-model-len 512
```

#### Build fails with "unsupported GPU architecture"
Ensure TORCH_CUDA_ARCH_LIST is set correctly:
```bash
# Check the Dockerfile has:
ENV TORCH_CUDA_ARCH_LIST="3.7"
ENV CUDA_ARCH_LIST="3.7"

# If building manually, set environment variables:
export TORCH_CUDA_ARCH_LIST="3.7"
export CUDA_ARCH_LIST="3.7"
```

### Performance Expectations
- **Speed**: 1-3 tokens/second for small models
- **Memory**: Uses ~8-10GB of K80's 12GB VRAM
- **Latency**: Higher than modern GPUs due to older architecture
- **Throughput**: Suitable for development/testing, not production

## Limitations

### Unsupported Features
- FlashAttention-2 (requires compute capability ≥8.0) 
- FlashAttention-1 (requires compute capability ≥7.0)
- FP8/INT8/FP4 quantization (BitBlas ≥7.0, Marlin ≥8.0)
- Tensor parallelism across multiple GPUs
- Advanced MoE models with modern kernels
- Speculative decoding
- CUTLASS kernels (require newer architectures)

### Supported Features
- Basic text generation
- XFormers attention
- FP16 and FP32 precision
- Single GPU inference
- OpenAI-compatible API

## Important Notes

### Tesla K80 Architecture Details
- **Dual-GPU Design**: Tesla K80 contains two GK210 GPUs on one card
- **Compute Capability**: 3.7 (Kepler architecture, released 2014)
- **Memory**: 12GB GDDR5 per GPU (24GB total, but vLLM uses single GPU)
- **CUDA Support**: Maximum CUDA 11.4 with driver 470.x

### Known Limitations
- **Legacy Support**: Tesla K80 is at minimum supported compute capability
- **Performance**: Significantly slower than modern GPUs (5-10x slower)
- **Power Consumption**: High power draw (~300W) for relatively low performance
- **Driver End-of-Life**: Driver 470 is EOL, no future updates

### Development vs Production
This Tesla K80 support is intended for:
- ✅ **Development and testing**
- ✅ **Learning and experimentation** 
- ✅ **Legacy hardware utilization**
- ❌ **Production deployments** (use modern GPUs)

## Support

For issues specific to Tesla K80 support, please:
1. Check this README for common solutions
2. Verify driver version 470.x is installed
3. Ensure using recommended model sizes
4. Include GPU info (`nvidia-smi`) in bug reports
5. Note that Tesla K80 is legacy hardware with limited support