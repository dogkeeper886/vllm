#!/usr/bin/env bash
#
# Build the Story #28 sm_37 CUTLASS SGEMM reproducer.
#
# Designed to run inside the K80 builder image (vllm37-builder:latest), which
# already has CUDA 11.4 + GCC 10 + CMake 4 + Python 3.10. The script:
#
#   1. Clones CUTLASS at the FetchContent-pinned tag (v4.0.0 per
#      vllm/CMakeLists.txt:302) into a scratch directory.
#   2. Applies sm37-trait.patch (Story #25, merged via PR #54).
#   3. Compiles sgemm_sm37.cu with nvcc, targeting sm_37.
#   4. Prints the resulting binary path + size for the caller.
#
# The runtime execution is intentionally NOT done here — see the
# k80-cutlass-repro CI workflow for the orchestration.
#
# Idempotent: re-running re-applies the patch via `git apply -3` so an
# already-patched tree doesn't trip up subsequent builds.

set -euo pipefail

REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_DIR="${REPRO_DIR}/../cutlass-patches"
WORK_DIR="${WORK_DIR:-/tmp/cutlass-sm37-build}"
CUTLASS_TAG="${CUTLASS_TAG:-v4.0.0}"
CUTLASS_REPO="${CUTLASS_REPO:-https://github.com/NVIDIA/cutlass.git}"
NVCC="${NVCC:-nvcc}"
TARGET_ARCH="${TARGET_ARCH:-sm_37}"

echo "=== Build env ==="
echo "REPRO_DIR     = ${REPRO_DIR}"
echo "PATCH_DIR     = ${PATCH_DIR}"
echo "WORK_DIR      = ${WORK_DIR}"
echo "CUTLASS_TAG   = ${CUTLASS_TAG}"
echo "TARGET_ARCH   = ${TARGET_ARCH}"
echo "NVCC          = ${NVCC}"
"${NVCC}" --version | head -4 || true
echo ""

echo "=== Step 1: prepare CUTLASS source at ${CUTLASS_TAG} ==="
if [ -d "${WORK_DIR}/cutlass" ] && [ -d "${WORK_DIR}/cutlass/.git" ]; then
    echo "found existing CUTLASS clone at ${WORK_DIR}/cutlass — reusing"
    git -C "${WORK_DIR}/cutlass" fetch --depth 1 origin "${CUTLASS_TAG}"
    git -C "${WORK_DIR}/cutlass" reset --hard "FETCH_HEAD"
else
    rm -rf "${WORK_DIR}/cutlass"
    mkdir -p "${WORK_DIR}"
    git clone --depth 1 --branch "${CUTLASS_TAG}" "${CUTLASS_REPO}" "${WORK_DIR}/cutlass"
fi
echo ""

echo "=== Step 2: apply sm_37 patches ==="
# Use `git apply -3` for 3-way merge — handles already-applied AND minor
# context drift if upstream CUTLASS adds nearby content. README §
# "Idempotency" documents the rationale.
for patch_file in "${PATCH_DIR}"/*.patch; do
    [ -f "${patch_file}" ] || continue
    patch_name="$(basename "${patch_file}")"
    echo "applying ${patch_name}..."
    git -C "${WORK_DIR}/cutlass" apply -3 "${patch_file}" \
        || git -C "${WORK_DIR}/cutlass" apply --check --reverse "${patch_file}" \
        || { echo "patch ${patch_name} failed and is not already applied"; exit 1; }
done
echo "verifying Sm37 struct present:"
grep -A1 "^struct Sm37" "${WORK_DIR}/cutlass/include/cutlass/arch/arch.h" \
    || { echo "ERROR: Sm37 struct not present after patching"; exit 1; }
echo ""

echo "=== Step 3: compile sgemm_sm37.cu for ${TARGET_ARCH} ==="
mkdir -p "${WORK_DIR}/build"
"${NVCC}" \
    -std=c++17 \
    -O3 \
    -arch="${TARGET_ARCH}" \
    -I "${WORK_DIR}/cutlass/include" \
    -I "${WORK_DIR}/cutlass/tools/util/include" \
    -Xcompiler -Wall \
    -DCUTLASS_NVCC_ARCHS="37" \
    "${REPRO_DIR}/sgemm_sm37.cu" \
    -o "${WORK_DIR}/build/sgemm_sm37"

ls -la "${WORK_DIR}/build/sgemm_sm37"
echo ""

echo "=== Build complete ==="
echo "binary: ${WORK_DIR}/build/sgemm_sm37"
echo "to run: ${WORK_DIR}/build/sgemm_sm37"
