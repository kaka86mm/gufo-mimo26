#include "src/models/mimo2/kernels/rocm/dflash_v1_kernels.hpp"

#include <cmath>

namespace gufo::models::mimo2::rocm {

__global__ void DFlashV1AddKernel(float* dst, const float* src,
                                  std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    dst[i] += src[i];
  }
}

void DFlashV1AddInPlace(float* dst, const float* src, std::size_t count,
                        hipStream_t stream) {
  if (count == 0) {
    return;
  }
  const auto blocks = (count + 255) / 256;
  hipLaunchKernelGGL(DFlashV1AddKernel, dim3(blocks), dim3(256), 0, stream, dst,
                     src, count);
}

__global__ void DFlashV1MaskSubstituteKernel(float* hidden,
                                             const std::int32_t* tokens,
                                             std::uint32_t hidden_size,
                                             std::int32_t mask_token_id,
                                             const float* mask_embedding) {
  const std::uint32_t row = blockIdx.x;
  if (tokens[row] != mask_token_id) {
    return;
  }
  float* const dst = hidden + static_cast<std::size_t>(row) * hidden_size;
  for (std::uint32_t i = threadIdx.x; i < hidden_size; i += blockDim.x) {
    dst[i] = mask_embedding[i];
  }
}

void DFlashV1MaskSubstitute(float* hidden, const std::int32_t* tokens,
                            std::uint32_t rows, std::uint32_t hidden_size,
                            std::int32_t mask_token_id,
                            const float* mask_embedding, hipStream_t stream) {
  if (rows == 0 || hidden_size == 0 || mask_embedding == nullptr) {
    return;
  }
  hipLaunchKernelGGL(DFlashV1MaskSubstituteKernel, dim3(rows), dim3(256), 0,
                     stream, hidden, tokens, hidden_size, mask_token_id,
                     mask_embedding);
}

/// One block per (row, slice); 256 threads cooperate on the slice's sum of
/// squares and then scale it in place.
__global__ void DFlashV1SliceRmsNormKernel(float* x, std::uint32_t slice_width,
                                           std::uint32_t slices, float eps) {
  const std::uint32_t pair = blockIdx.x;
  const std::uint32_t row = pair / slices;
  const std::uint32_t slice = pair % slices;
  float* const base =
      x + (static_cast<std::size_t>(row) * slices + slice) * slice_width;
  float sum = 0.0F;
  for (std::uint32_t i = threadIdx.x; i < slice_width; i += blockDim.x) {
    const float value = base[i];
    sum += value * value;
  }
  __shared__ float partial[256];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (std::uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  const float inv =
      1.0F / sqrtf(partial[0] / static_cast<float>(slice_width) + eps);
  for (std::uint32_t i = threadIdx.x; i < slice_width; i += blockDim.x) {
    base[i] *= inv;
  }
}

void DFlashV1SliceRmsNorm(float* x, std::uint32_t rows,
                          std::uint32_t slice_width, std::uint32_t slices,
                          float eps, hipStream_t stream) {
  if (rows == 0 || slices == 0 || slice_width == 0) {
    return;
  }
  hipLaunchKernelGGL(DFlashV1SliceRmsNormKernel,
                     dim3(rows * slices), dim3(256), 0, stream, x, slice_width,
                     slices, eps);
}

/// One warp per (block row, query head). The warp cooperatively reads each key
/// row (lane-strided), reduces the score with shuffle butterflies and keeps a
/// per-lane slice of the value accumulator. Online softmax with the sink as an
/// extra denominator term.
__global__ void DFlashV1AttentionKernel(
    const float* __restrict__ q, const float* __restrict__ hist_k,
    const float* __restrict__ hist_v, const float* __restrict__ block_k,
    const float* __restrict__ block_v, const float* __restrict__ sinks,
    float* __restrict__ out, const std::uint32_t* position,
    std::uint32_t block_count, std::uint32_t window, std::uint32_t q_heads,
    std::uint32_t kv_heads, std::uint32_t head_dim, float scale,
    std::uint32_t history_capacity, bool causal_block) {
  // The committed length is read from device memory so one captured graph can
  // replay at any position.
  const std::uint32_t history_length = *position;
  const std::uint32_t warp = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t pair = blockIdx.x * (blockDim.x >> 5) + warp;
  const std::uint32_t t = pair / q_heads;
  const std::uint32_t h = pair % q_heads;
  if (t >= block_count || h >= q_heads) {
    return;
  }
  const std::uint32_t group = q_heads / kv_heads;
  const std::uint32_t kv = h / group;
  const std::uint32_t kper = head_dim / 32u;  // head_dim % 32 == 0
  const std::size_t kv_row = static_cast<std::size_t>(kv_heads) * head_dim;

  const float* const q_row =
      q + (static_cast<std::size_t>(t) * q_heads + h) * head_dim;
  float acc[8];
#pragma unroll
  for (std::uint32_t j = 0; j < 8; ++j) {
    acc[j] = 0.0F;
  }
  float m = -INFINITY;
  float l = 0.0F;

  const auto attend = [&](const float* k_row, const float* v_row) {
    float partial = 0.0F;
#pragma unroll
    for (std::uint32_t j = 0; j < 8; ++j) {
      if (j < kper) {
        const std::uint32_t i = j * 32u + lane;
        partial += q_row[i] * k_row[i];
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      partial += __shfl_xor(partial, offset);
    }
    const float score = partial * scale;
    const float m_new = fmaxf(m, score);
    const float rescale = m == -INFINITY ? 0.0F : __expf(m - m_new);
    const float weight = __expf(score - m_new);
    l = l * rescale + weight;
#pragma unroll
    for (std::uint32_t j = 0; j < 8; ++j) {
      if (j < kper) {
        const std::uint32_t i = j * 32u + lane;
        acc[j] = acc[j] * rescale + weight * v_row[i];
      }
    }
    m = m_new;
  };

  // The whole block shares one history window anchored at the context end,
  // exactly like the v1 reference's `swa_lo = committed - window`.
  const std::uint32_t visible =
      history_length > window ? history_length - window : 0u;
  for (std::uint32_t pos = visible; pos < history_length; ++pos) {
    const std::uint32_t row = pos % history_capacity;
    attend(hist_k + static_cast<std::size_t>(row) * kv_row +
               static_cast<std::size_t>(kv) * head_dim,
           hist_v + static_cast<std::size_t>(row) * kv_row +
               static_cast<std::size_t>(kv) * head_dim);
  }
  // Sliding layers keep the block causal (row t sees rows `<= t`); full
  // layers expose every block row.
  const std::uint32_t block_rows = causal_block ? t + 1u : block_count;
  for (std::uint32_t row = 0; row < block_rows; ++row) {
    attend(block_k + static_cast<std::size_t>(row) * kv_row +
               static_cast<std::size_t>(kv) * head_dim,
           block_v + static_cast<std::size_t>(row) * kv_row +
               static_cast<std::size_t>(kv) * head_dim);
  }
  if (sinks != nullptr) {
    const float sink = sinks[h];
    const float m_new = fmaxf(m, sink);
    const float rescale = m == -INFINITY ? 0.0F : __expf(m - m_new);
    l = l * rescale + __expf(sink - m_new);
#pragma unroll
    for (std::uint32_t j = 0; j < 8; ++j) {
      acc[j] *= rescale;
    }
    m = m_new;
  }
  const float inv = l > 0.0F ? 1.0F / l : 0.0F;
  float* const out_row =
      out + (static_cast<std::size_t>(t) * q_heads + h) * head_dim;
#pragma unroll
  for (std::uint32_t j = 0; j < 8; ++j) {
    if (j < kper) {
      out_row[j * 32u + lane] = acc[j] * inv;
    }
  }
}

void DFlashV1BlockAttention(const float* q, const float* hist_k,
                            const float* hist_v, const float* block_k,
                            const float* block_v, const float* sinks,
                            float* out, const std::uint32_t* position,
                            std::uint32_t block_count, std::uint32_t window,
                            std::uint32_t q_heads, std::uint32_t kv_heads,
                            std::uint32_t head_dim, float scale,
                            std::uint32_t history_capacity, bool causal_block,
                            hipStream_t stream) {
  if (block_count == 0 || q_heads == 0 || kv_heads == 0 || head_dim == 0 ||
      head_dim % 32u != 0u || head_dim > 256u || head_dim / 32u > 8u ||
      history_capacity == 0 || position == nullptr) {
    return;
  }
  const std::uint32_t warps_per_block = 8;
  const std::uint32_t pairs = block_count * q_heads;
  const auto blocks = (pairs + warps_per_block - 1) / warps_per_block;
  hipLaunchKernelGGL(DFlashV1AttentionKernel, dim3(blocks),
                     dim3(warps_per_block * 32), 0, stream, q, hist_k, hist_v,
                     block_k, block_v, sinks, out, position, block_count,
                     window, q_heads, kv_heads, head_dim, scale,
                     history_capacity, causal_block);
}

}  // namespace gufo::models::mimo2::rocm