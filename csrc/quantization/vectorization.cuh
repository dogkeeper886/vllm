#pragma once
/**
 * __device__ datatypes vectorized by 4
 */

// Include both AMD and NVIDIA fp8 types to avoid circular import.
// These headers were added in PyTorch 2.1+; skip for legacy builds (e.g. 2.0.x).
#ifndef VLLM_BUILD_LEGACY_CUDA
#include <c10/util/Float8_e4m3fnuz.h>
#include <c10/util/Float8_e4m3fn.h>
#endif

namespace vllm {

// Vectorization containers
template <typename scalar_t, size_t vec_size>
struct __align__(vec_size * sizeof(scalar_t)) vec_n_t {
  scalar_t val[vec_size];
};

template <typename quant_type_t, size_t vec_size>
struct __align__(vec_size * sizeof(quant_type_t)) q8_n_t {
  static_assert(std::is_same_v<quant_type_t, int8_t>
#ifndef VLLM_BUILD_LEGACY_CUDA
                || std::is_same_v<quant_type_t, c10::Float8_e4m3fn>
                || std::is_same_v<quant_type_t, c10::Float8_e4m3fnuz>
#endif
  );
  quant_type_t val[vec_size];
};

template <typename scalar_t>
using vec4_t = vec_n_t<scalar_t, 4>;
template <typename quant_type_t>
using q8x4_t = q8_n_t<quant_type_t, 4>;

}  // namespace vllm
