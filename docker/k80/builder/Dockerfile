# vLLM K80 Builder Image
# Base image with CUDA 11.4, GCC 10, CMake 4, Python 3.10, and PyTorch 2.0.1
# Follows the dogkeeper886/ollama37 two-stage build pattern on Rocky Linux 8.
#
# Build:
#   docker build -t vllm37-builder -f docker/k80/builder/Dockerfile .
#
# Expected build time: ~120+ minutes first time (GCC ~60m, Python ~15m, PyTorch ~30m)

FROM rockylinux/rockylinux:8

ARG CUDA_VERSION=11.4.4
ARG GCC_VERSION=10.5.0
ARG CMAKE_VERSION=4.0.1
ARG PYTHON_VERSION=3.10.16
ARG PYTORCH_VERSION=v2.0.1
ARG NUMPY_VERSION=1.26.4
ARG JOBS=4

# ============================================================================
# System packages
# ============================================================================
RUN dnf -y install \
        dnf-plugins-core \
        epel-release && \
    dnf config-manager --set-enabled powertools && \
    dnf -y install \
        bzip2 bzip2-devel \
        diffutils \
        file \
        findutils \
        gdbm-devel \
        git \
        libffi-devel \
        lz4-devel \
        make \
        ncurses-devel \
        openssl-devel \
        patch \
        pciutils \
        readline-devel \
        sqlite-devel \
        tar \
        tk-devel \
        unzip \
        wget \
        which \
        xz xz-devel \
        zlib-devel \
    && dnf clean all

# ============================================================================
# CUDA 11.4 Toolkit
# ============================================================================
RUN dnf config-manager --add-repo \
        https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo && \
    dnf -y install \
        cuda-toolkit-11-4 \
        libcudnn8-devel \
    && dnf clean all

ENV CUDA_HOME=/usr/local/cuda-11.4
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# ============================================================================
# GCC 10 from source (Rocky 8 ships GCC 8 which is too old for PyTorch 2.0)
# ============================================================================
RUN cd /tmp && \
    wget -q https://ftp.gnu.org/gnu/gcc/gcc-${GCC_VERSION}/gcc-${GCC_VERSION}.tar.xz && \
    tar xf gcc-${GCC_VERSION}.tar.xz && \
    cd gcc-${GCC_VERSION} && \
    ./contrib/download_prerequisites && \
    mkdir build && cd build && \
    ../configure \
        --prefix=/usr/local \
        --enable-languages=c,c++ \
        --disable-multilib \
        --disable-bootstrap && \
    make -j${JOBS} && \
    make install && \
    cd /tmp && rm -rf gcc-${GCC_VERSION}*

ENV CC=/usr/local/bin/gcc
ENV CXX=/usr/local/bin/g++
ENV LD_LIBRARY_PATH=/usr/local/lib64:${LD_LIBRARY_PATH}

# Update linker cache for new GCC
RUN echo "/usr/local/lib64" > /etc/ld.so.conf.d/local-lib64.conf && ldconfig

# ============================================================================
# CMake 4 from source
# ============================================================================
RUN cd /tmp && \
    wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz && \
    tar xf cmake-${CMAKE_VERSION}.tar.gz && \
    cd cmake-${CMAKE_VERSION} && \
    ./bootstrap --prefix=/usr/local --parallel=${JOBS} && \
    make -j${JOBS} && \
    make install && \
    cd /tmp && rm -rf cmake-${CMAKE_VERSION}*

# ============================================================================
# Python 3.10 from source (Rocky 8 only has 3.8)
# ============================================================================
RUN cd /tmp && \
    wget -q https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz && \
    tar xf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure \
        --prefix=/usr/local \
        --enable-shared \
        --enable-optimizations \
        --with-lto \
        LDFLAGS="-Wl,-rpath=/usr/local/lib" && \
    make -j${JOBS} && \
    make install && \
    cd /tmp && rm -rf Python-${PYTHON_VERSION}*

RUN ln -sf /usr/local/bin/python3.10 /usr/local/bin/python3 && \
    ln -sf /usr/local/bin/python3 /usr/local/bin/python && \
    ln -sf /usr/local/bin/pip3.10 /usr/local/bin/pip3 && \
    ln -sf /usr/local/bin/pip3 /usr/local/bin/pip

# ============================================================================
# Python build dependencies
# ============================================================================
RUN pip install --no-cache-dir \
    "numpy==${NUMPY_VERSION}" \
    packaging \
    setuptools \
    wheel \
    ninja \
    pyyaml \
    typing_extensions \
    cffi \
    future \
    six \
    requests \
    dataclasses \
    filelock \
    jinja2 \
    networkx \
    sympy

# ============================================================================
# PyTorch 2.0.1 from source (with CUDA 11.4 + sm_37)
# ============================================================================
RUN cd /usr/local/src && \
    git clone --depth 1 --branch ${PYTORCH_VERSION} --recursive \
        https://github.com/pytorch/pytorch.git && \
    cd pytorch && \
    pip install --no-cache-dir -r requirements.txt && \
    CUDA_HOME=${CUDA_HOME} \
    TORCH_CUDA_ARCH_LIST="3.7" \
    CMAKE_PREFIX_PATH=/usr/local \
    CMAKE_POLICY_VERSION_MINIMUM=3.5 \
    USE_CUDA=1 \
    USE_CUDNN=1 \
    USE_NCCL=1 \
    USE_DISTRIBUTED=1 \
    USE_MKLDNN=0 \
    BUILD_TEST=0 \
    MAX_JOBS=${JOBS} \
    python setup.py install && \
    cd /usr/local/src && rm -rf pytorch

# Verify PyTorch installation
RUN python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# ============================================================================
# Final state: builder image ready for vLLM compilation
# ============================================================================
WORKDIR /usr/local/src
