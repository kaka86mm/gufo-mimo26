#include "src/models/mimo2/kernels/rocm/kernels.hpp"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <hipcub/block/block_radix_sort.hpp>
#include <stdexcept>
#include <type_traits>

#include "src/models/mimo2/mtp_sampling.hpp"

// HIP kernels follow the layouts and operator formulas in reference.cpp.

namespace gufo::models::mimo2::rocm {
namespace {


// Bring-up debug: sync after the named launch and report the first kernel
// whose stream carries an illegal access. Guarded by GUFO_SYNC_TRACE=1.
static bool kSyncTrace = [] {
  const char* env = getenv("GUFO_SYNC_TRACE");
  return env != nullptr && env[0] == '1';
}();
static void SyncTrace(const char* name, hipStream_t stream) {
  if (!kSyncTrace) {
    return;
  }
  hipStreamSynchronize(stream);
  const hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "SYNCTRACE FAULT kernel=%s err=%d (%s)\n", name,
            static_cast<int>(err), hipGetErrorString(err));
    fflush(stderr);
    abort();
  }
}

constexpr unsigned kThreads = 256;

__device__ __forceinline__ float SigmoidF(float x) {
  return 1.0f / (1.0f + __expf(-x));
}
__device__ __forceinline__ float SoftplusF(float x) {
  return x > 20.0f ? x : log1pf(__expf(x));
}
__device__ __forceinline__ float SiluF(float x) {
  return x * SigmoidF(x);
}
__device__ __forceinline__ float Bf16ToF32(std::uint16_t h) {
  return __uint_as_float(static_cast<std::uint32_t>(h) << 16);
}

/// Reads element i of a Q8_0 / F32 / BF16 / F16 row.
__device__ __forceinline__ float RowElement(const void* row, WeightType type,
                                            std::uint32_t i) {
  switch (type) {
    case WeightType::kF32:
      return static_cast<const float*>(row)[i];
    case WeightType::kBF16:
      return Bf16ToF32(static_cast<const std::uint16_t*>(row)[i]);
    case WeightType::kF16:
      return __half2float(static_cast<const __half*>(row)[i]);
    case WeightType::kQ8_0: {
      const auto* blk = static_cast<const std::uint8_t*>(row) + (i / 32) * 34;
      const __half d = *reinterpret_cast<const __half*>(blk);
      const auto q = static_cast<const std::int8_t*>(
          static_cast<const void*>(blk + 2))[i % 32];
      return __half2float(d) * static_cast<float>(q);
    }
  }
  return 0.0f;
}

__device__ __forceinline__ std::size_t RowBytes(WeightType type,
                                                std::uint32_t k) {
  switch (type) {
    case WeightType::kF32:
      return static_cast<std::size_t>(k) * 4;
    case WeightType::kBF16:
    case WeightType::kF16:
      return static_cast<std::size_t>(k) * 2;
    case WeightType::kQ8_0:
      return static_cast<std::size_t>(k / 32) * 34;
  }
  return 0;
}

/// Sum over one wave; every lane receives the total. The wave-per-row
/// kernels split rows across lanes (gfx1151 runs wave32).
__device__ __forceinline__ float WaveSum(float v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor(v, offset);
  }
  return v;
}

/// Block-wide sum over kThreads threads; every thread receives the total.
__device__ float BlockSum(float v, float* shared) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_xor(v, offset);
  }
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  __syncthreads();
  if (lane == 0) {
    shared[warp] = v;
  }
  __syncthreads();
  float total = 0.0f;
  for (int w = 0; w < static_cast<int>(blockDim.x / warpSize); ++w) {
    total += shared[w];
  }
  return total;
}

__device__ float BlockMax(float v, float* shared) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_xor(v, offset));
  }
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  __syncthreads();
  if (lane == 0) {
    shared[warp] = v;
  }
  __syncthreads();
  float total = -INFINITY;
  for (int w = 0; w < static_cast<int>(blockDim.x / warpSize); ++w) {
    total = fmaxf(total, shared[w]);
  }
  return total;
}

__global__ void EmbedKernel(const void* table, WeightType type,
                            const std::int32_t* tokens, float* res,
                            std::uint32_t hidden, std::uint32_t streams) {
  const std::uint32_t t = blockIdx.x;
  const auto* row = static_cast<const std::uint8_t*>(table) +
                    RowBytes(type, hidden) * tokens[t];
  for (std::uint32_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = RowElement(row, type, i);
    for (std::uint32_t s = 0; s < streams; ++s) {
      res[(static_cast<std::size_t>(t) * streams + s) * hidden + i] = v;
    }
  }
}

__global__ void RmsNormKernel(const float* x, const float* gamma, float* out,
                              std::uint32_t group_dim, std::uint32_t groups,
                              float eps) {
  __shared__ float shared[32];
  const std::size_t row = blockIdx.x;
  const float* src = x + row * group_dim;
  float* dst = out + row * group_dim;
  float ss = 0.0f;
  for (std::uint32_t i = threadIdx.x; i < group_dim; i += blockDim.x) {
    ss += src[i] * src[i];
  }
  ss = BlockSum(ss, shared);
  const float scale = rsqrtf(ss / static_cast<float>(group_dim) + eps);
  // A grouped norm shares one gamma row across the groups of a token.
  const float* g =
      gamma == nullptr ? nullptr : gamma + (row % groups) * group_dim;
  for (std::uint32_t i = threadIdx.x; i < group_dim; i += blockDim.x) {
    dst[i] = src[i] * scale * (g != nullptr ? g[i] : 1.0f);
  }
}

/// Four adjacent hidden lanes per thread. This keeps the stream arithmetic in
/// registers while cutting the number of blocks and inject partials by almost
/// four for the model's 2,560-wide, four-stream rows.
__device__ __forceinline__ float4 Load4(const float* p) {
  return *reinterpret_cast<const float4*>(p);
}
__device__ __forceinline__ float4 Load4(const __half* p) {
  const auto packed = *reinterpret_cast<const __half2*>(p);
  const auto packed_hi = *reinterpret_cast<const __half2*>(p + 2);
  const float2 lo = __half22float2(packed);
  const float2 hi = __half22float2(packed_hi);
  return float4{lo.x, lo.y, hi.x, hi.y};
}

/// The W8A8 GEMM's activation tiles: 16 tokens x one 32-wide K block,
/// codes in fragment order (two 256-byte halves of 16 tokens x 16 codes)
/// followed by the sixteen per-token scales.
constexpr std::size_t kQ8ActTileTokens = 16;
constexpr std::size_t kQ8ActTileBytes = 576;
constexpr std::size_t kQ8ActScaleOffset = 512;
__device__ __forceinline__ std::int8_t* Q8ActTile(void* base,
                                                  std::size_t num_blocks,
                                                  std::size_t tt,
                                                  std::size_t kb) {
  return static_cast<std::int8_t*>(base) +
         (((tt * num_blocks) + kb) * kQ8ActTileBytes);
}

__global__ void SiluScaleKernel(float* x, float scale, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    x[i] = SiluF(x[i] * scale);
  }
}

/// out = silu(gate) * up as F16, four elements per thread.
__global__ void SwigluHalfKernel(const float* __restrict__ gate,
                                 const float* __restrict__ up,
                                 __half* __restrict__ out, std::size_t count) {
  const std::size_t i =
      (blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x) * 4;
  if (i + 4 > count) {
    for (std::size_t j = i; j < count; ++j) {
      out[j] = __float2half(SiluF(gate[j]) * up[j]);
    }
    return;
  }
  const float4 g = *reinterpret_cast<const float4*>(gate + i);
  const float4 u = *reinterpret_cast<const float4*>(up + i);
  const __half2 lo = __floats2half2_rn(SiluF(g.x) * u.x, SiluF(g.y) * u.y);
  const __half2 hi = __floats2half2_rn(SiluF(g.z) * u.z, SiluF(g.w) * u.w);
  *reinterpret_cast<__half2*>(out + i) = lo;
  *reinterpret_cast<__half2*>(out + i + 2) = hi;
}

/// silu(gate) * up over rows of `k` elements, written only as the W8A8
/// tiled Q8 layout: four elements per lane, eight lanes per K block.
__global__ void SwigluQ8Kernel(const float* gate, const float* up, void* out_q8,
                               std::size_t n_rows, std::size_t k) {
  const std::size_t chunk =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  const std::size_t chunks_per_row = k / 4;
  const std::size_t row = chunk / chunks_per_row;
  const std::size_t i = (chunk % chunks_per_row) * 4;
  // Rows past the batch still join the wave reduction below.
  const bool live = row < n_rows;
  float4 v{0.0F, 0.0F, 0.0F, 0.0F};
  if (live) {
    const std::size_t idx = (row * k) + i;
    const float4 g = *reinterpret_cast<const float4*>(gate + idx);
    const float4 u = *reinterpret_cast<const float4*>(up + idx);
    v = float4{SiluF(g.x) * u.x, SiluF(g.y) * u.y, SiluF(g.z) * u.z,
               SiluF(g.w) * u.w};
  }
  float max_abs =
      fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
  for (int off = 4; off > 0; off >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
  }
  if (!live) {
    return;
  }
  const float d = max_abs / 127.0F;
  const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
  const auto q0 = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(v.x * id))));
  const auto q1 = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(v.y * id))));
  const auto q2 = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(v.z * id))));
  const auto q3 = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(v.w * id))));
  const std::size_t tl = row % kQ8ActTileTokens;
  const std::uint32_t pos = static_cast<std::uint32_t>(i % 32);
  std::int8_t* tile = Q8ActTile(out_q8, k / 32, row / kQ8ActTileTokens, i / 32);
  *reinterpret_cast<std::uint32_t*>(tile + ((pos >> 4u) * 256) + (tl * 16) +
                                    (pos & 15u)) =
      q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
  if (pos == 0) {
    *reinterpret_cast<float*>(tile + kQ8ActScaleOffset + (tl * sizeof(float))) =
        d;
  }
}

__global__ void SwigluKernel(float* gate, const float* up, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    gate[i] = SiluF(gate[i]) * up[i];
  }
}

__global__ void SigmoidMulKernel(float* x, const float* g, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    x[i] *= SigmoidF(g[i]);
  }
}

template<typename T>
__global__ void NarrowKernel(const float* x, T* out, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    out[i] = T(x[i]);
  }
}

/// One wave per weight row, kSmallGemmRows rows per block: the row is read
/// once (four consecutive elements per lane per step) while up to eight
/// tokens accumulate in registers, then a wave reduction per token.
constexpr unsigned kSmallGemmRows = 4;
template<WeightType type, unsigned tokens, bool grouped = false>
__global__ void SmallGemmKernel(const void* w, const float* x, float* out,
                                std::uint32_t m, std::uint32_t k) {
  if constexpr (grouped) {
    x += std::size_t{blockIdx.y} * tokens * k;
    out += std::size_t{blockIdx.y} * tokens * m;
  }
  const std::uint32_t lane = threadIdx.x % warpSize;
  const std::uint32_t row =
      blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
  if (row >= m) {
    return;
  }
  const auto* wrow =
      static_cast<const std::uint8_t*>(w) + RowBytes(type, k) * row;
  float acc[tokens] = {};
  // Q8_0 rows are 34-byte blocks, so a row start is only 2-byte aligned: the
  // four codes a lane owns are fetched as two aligned halfwords rather than
  // four bytes, and the activations as one float4. The block scale is shared
  // by eight lanes and read once. Only a row of whole blocks is eligible.
  bool vectorized = false;
  if constexpr (type == WeightType::kQ8_0) {
    if ((k & 31u) == 0u) {
      for (std::uint32_t i0 = lane * 4; i0 < k; i0 += warpSize * 4) {
        const auto* block_bytes = reinterpret_cast<const std::uint8_t*>(
            wrow + static_cast<std::size_t>(i0 >> 5) * 34);
        const float scale =
            __half2float(*reinterpret_cast<const __half*>(block_bytes));
        const std::uint32_t offset = 2u + (i0 & 31u);
        const std::uint32_t lo = *reinterpret_cast<const std::uint16_t*>(
            block_bytes + offset);
        const std::uint32_t hi = *reinterpret_cast<const std::uint16_t*>(
            block_bytes + offset + 2);
        const std::uint32_t packed = lo | (hi << 16);
#pragma unroll
        for (unsigned j = 0; j < tokens; ++j) {
          const float4 xv = *reinterpret_cast<const float4*>(
              x + static_cast<std::size_t>(j) * k + i0);
          float dot = 0.0f;
#pragma unroll
          for (unsigned r = 0; r < 4; ++r) {
            const std::int8_t code =
                static_cast<std::int8_t>((packed >> (8 * r)) & 0xFF);
            dot += static_cast<float>(code) * (&xv.x)[r];
          }
          acc[j] += scale * dot;
        }
      }
      vectorized = true;
    }
  }
  if (!vectorized) {
    for (std::uint32_t i0 = lane * 4; i0 < k; i0 += warpSize * 4) {
      float wv[4];
#pragma unroll
      for (unsigned r = 0; r < 4; ++r) {
        wv[r] = i0 + r < k ? RowElement(wrow, type, i0 + r) : 0.0f;
      }
#pragma unroll
      for (unsigned j = 0; j < tokens; ++j) {
        const float* xr = x + static_cast<std::size_t>(j) * k + i0;
        float dot = 0.0f;
#pragma unroll
        for (unsigned r = 0; r < 4; ++r) {
          dot += wv[r] * (i0 + r < k ? xr[r] : 0.0f);
        }
        acc[j] += dot;
      }
    }
  }
#pragma unroll
  for (unsigned j = 0; j < tokens; ++j) {
    const float total = WaveSum(acc[j]);
    if (lane == 0) {
      out[static_cast<std::size_t>(j) * m + row] = total;
    }
  }
}

__global__ void RopeKernel(float* x, std::uint32_t heads, std::uint32_t d,
                           std::uint32_t rotary_dim,
                           const std::uint32_t* start_pos, float theta,
                           const qwen::vision::DeviceRope* rope,
                           const std::uint32_t* row_pos) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t half = rotary_dim / 2;
  // row_pos, when given, carries the absolute position of every row of a
  // fused batch whose rows belong to different sessions.
  const std::uint32_t pos = row_pos != nullptr ? row_pos[t] : (*start_pos + t);
  for (std::uint32_t idx = threadIdx.x; idx < heads * half; idx += blockDim.x) {
    const std::uint32_t h = idx / half;
    const std::uint32_t i = idx % half;
    float* v = x + (static_cast<std::size_t>(t) * heads + h) * d;
    const float freq = powf(
        theta, -2.0f * static_cast<float>(i) / static_cast<float>(rotary_dim));
    float s = 0.0f;
    float c = 0.0f;
    sincosf(qwen::vision::RopePosition(rope, pos, i) * freq, &s, &c);
    const float a = v[i];
    const float b = v[i + half];
    v[i] = a * c - b * s;
    v[i + half] = a * s + b * c;
  }
}

__global__ void StoreKvKernel(const float* src, __half* cache,
                              std::int8_t* cache_q, __half* scales,
                              std::uint32_t row_dim, std::uint32_t ring_rows,
                              const std::uint32_t* start_pos) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t pos = *start_pos + t;
  // Sliding layers keep a ring larger than window + batch: row = pos &
  // (ring_rows - 1). A batch can never overwrite rows that earlier queries
  // in the same batch still attend to.
  const std::uint32_t row =
      ring_rows != 0 ? (pos & (ring_rows - 1)) : pos;
  for (std::uint32_t i = threadIdx.x; i < row_dim; i += blockDim.x) {
    cache[static_cast<std::size_t>(row) * row_dim + i] =
        __float2half(src[static_cast<std::size_t>(t) * row_dim + i]);
  }
  if (cache_q == nullptr || row_dim % kKvQuantBlock != 0) {
    return;
  }
  // int8 twin: one warp per 32-value block, so the absmax is a warp
  // reduction and the block's bytes plus its scale leave as one store each.
  const std::uint32_t warp = threadIdx.x >> 5u;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warps = blockDim.x >> 5u;
  const std::uint32_t blocks = row_dim / kKvQuantBlock;
  for (std::uint32_t block = warp; block < blocks; block += warps) {
    const float value = src[static_cast<std::size_t>(t) * row_dim +
                            block * kKvQuantBlock + lane];
    float best = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      best = fmaxf(best, __shfl_xor(best, offset));
    }
    const float scale = best / 127.0f;
    const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    const std::size_t out =
        static_cast<std::size_t>(row) * row_dim + block * kKvQuantBlock;
    cache_q[out + lane] = static_cast<std::int8_t>(__float2int_rn(value * inv));
    if (lane == 0) {
      scales[static_cast<std::size_t>(row) * blocks + block] =
          __float2half(scale);
    }
  }
}

// Slot-table twin of StoreKvKernel: the cache and ring live in kv[slot]
// (`which` 0 picks k_cache, 1 v_cache — one kernel serves both stores); a
// null cache is a cancelled row and stores nothing.
__global__ void StoreKvSlotKernel(const float* src, const SlotKv* kv,
                                  std::uint32_t slot, std::uint32_t which,
                                  std::uint32_t row_dim,
                                  const std::uint32_t* start_pos,
                                  const std::uint32_t* row_slot,
                                  const std::uint32_t* row_pos) {
  const std::uint32_t t = blockIdx.x;
  const SlotKv s = kv[row_slot != nullptr ? row_slot[t] : slot];
  __half* cache = which == 0 ? s.k_cache : s.v_cache;
  if (cache == nullptr) {
    return;
  }
  std::int8_t* cache_q = which == 0 ? s.k_cache_q : s.v_cache_q;
  __half* scales = which == 0 ? s.k_scales : s.v_scales;
  const std::uint32_t pos = row_pos != nullptr ? row_pos[t] : (*start_pos + t);
  const std::uint32_t row =
      s.ring_rows != 0 ? (pos & (s.ring_rows - 1)) : pos;
  for (std::uint32_t i = threadIdx.x; i < row_dim; i += blockDim.x) {
    cache[static_cast<std::size_t>(row) * row_dim + i] =
        __float2half(src[static_cast<std::size_t>(t) * row_dim + i]);
  }
  if (cache_q == nullptr || scales == nullptr ||
      row_dim % kKvQuantBlock != 0) {
    return;
  }
  const std::uint32_t warp = threadIdx.x >> 5u;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warps = blockDim.x >> 5u;
  const std::uint32_t blocks = row_dim / kKvQuantBlock;
  for (std::uint32_t block = warp; block < blocks; block += warps) {
    const float value = src[static_cast<std::size_t>(t) * row_dim +
                            block * kKvQuantBlock + lane];
    float best = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      best = fmaxf(best, __shfl_xor(best, offset));
    }
    const float scale = best / 127.0f;
    const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    const std::size_t out =
        static_cast<std::size_t>(row) * row_dim + block * kKvQuantBlock;
    cache_q[out + lane] = static_cast<std::int8_t>(__float2int_rn(value * inv));
    if (lane == 0) {
      scales[static_cast<std::size_t>(row) * blocks + block] =
          __float2half(scale);
    }
  }
}

// F16 WMMA fragments (wave32): sixteen halves per lane, eight F32
// accumulators per lane.
using v16h = __attribute__((__vector_size__(16 * sizeof(_Float16)))) _Float16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;

__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

/// dst[t] = row < 0 ? alt[t] : base[(row + t)]: the draft block's hidden
/// input, a kept trunk row or its own carried residual.
__global__ void MtpHiddenKernel(const float* base, const float* alt,
                                const std::int32_t* row, float* dst,
                                std::uint32_t width) {
  const std::uint32_t t = blockIdx.x;
  const float* src = *row < 0
                         ? alt + static_cast<std::size_t>(t) * width
                         : base + (static_cast<std::size_t>(*row) + t) * width;
  for (std::uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
    dst[static_cast<std::size_t>(t) * width + i] = src[i];
  }
}

// Slot-table twin: base/alt from ptrs[slot], the row from ctrl[slot]. A null
// base (cancelled slot) writes nothing.
__global__ void MtpHiddenSlotKernel(const SlotPtrs* ptrs,
                                    const SlotControl* ctrl,
                                    std::uint32_t slot, float* dst,
                                    std::uint32_t width) {
  const float* base = ptrs[slot].target_hidden;
  if (base == nullptr) {
    return;
  }
  const float* alt = ptrs[slot].mtp_h;
  const std::int32_t row = ctrl[slot].hidden_row;
  const std::uint32_t t = blockIdx.x;
  const float* src = row < 0
                         ? alt + static_cast<std::size_t>(t) * width
                         : base + (static_cast<std::size_t>(row) + t) * width;
  for (std::uint32_t i = threadIdx.x; i < width; i += blockDim.x) {
    dst[static_cast<std::size_t>(t) * width + i] = src[i];
  }
}

// The slot-indirected device copy: reads its destination from a table so a
// replayed graph writes each session's private rows (hipMemcpy nodes cannot
// indirect). A null destination is a cancelled slot and writes nothing.
__global__ void CopyRowsSlotKernel(const float* src, float* const* dstp,
                                   std::uint32_t count) {
  float* dst = *dstp;
  if (dst == nullptr) {
    return;
  }
  for (std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += gridDim.x * blockDim.x) {
    dst[i] = src[i];
  }
}

__global__ void AddRowsBroadcastKernel(const float* addend, float* residual,
                                       std::uint32_t hidden,
                                       std::uint32_t streams) {
  const std::uint32_t t = blockIdx.x;
  for (std::uint32_t i = threadIdx.x; i < streams * hidden; i += blockDim.x) {
    residual[static_cast<std::size_t>(t) * streams * hidden + i] +=
        addend[static_cast<std::size_t>(t) * hidden + i % hidden];
  }
}

/// The captured-exchange form: the addend lives in the receive window of the
/// replaying exchange, recovered from the pinned slot like the staged send and
/// the replayed host function do.
__global__ void AddRowsBroadcastSlottedKernel(
    const float* __restrict__ peer, float* residual, std::uint32_t hidden,
    std::uint32_t streams, const std::uint64_t* __restrict__ slot) {
  const std::uint64_t value =
      __hip_atomic_load(slot, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
  const float* addend =
      peer + ExchangeWindowOffset(value - 1) / sizeof(float);
  const std::uint32_t t = blockIdx.x;
  for (std::uint32_t i = threadIdx.x; i < streams * hidden; i += blockDim.x) {
    residual[static_cast<std::size_t>(t) * streams * hidden + i] +=
        addend[static_cast<std::size_t>(t) * hidden + i % hidden];
  }
}

// Every thread's copies reach the system scope before its block is counted;
// the last block to finish publishes `value`, so the host that reads it also
// sees the whole partial.
__global__ void StagePartialKernel(const float* __restrict__ partial,
                                   float* __restrict__ send, std::size_t count,
                                   std::uint32_t* arrivals,
                                   std::uint64_t* ready, std::uint64_t value) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const std::size_t first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t vectors = count / 4;
  for (std::size_t i = first; i < vectors; i += stride) {
    reinterpret_cast<float4*>(send)[i] =
        reinterpret_cast<const float4*>(partial)[i];
  }
  for (std::size_t i = vectors * 4 + first; i < count; i += stride) {
    send[i] = partial[i];
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0 && __hip_atomic_fetch_add(arrivals, 1u, __ATOMIC_ACQ_REL,
                                                 __HIP_MEMORY_SCOPE_SYSTEM) +
                                  1 ==
                              gridDim.x) {
    __hip_atomic_store(arrivals, 0u, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_SYSTEM);
    __hip_atomic_store(ready, value, __ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

/// Half-payload twins of the exchange kernels: the wire carries halves while
/// the local partial and the received sum stay f32. At cohort row counts the
/// payload is the bulk of an all-reduce (24 rows x 4096 x 4B = 393KB against a
/// ~0.1ms round trip), so halving it is the lever; the sum precision does not
/// move because the peer's half widens before the add.
__global__ void StagePartialHalfKernel(const float* __restrict__ partial,
                                       __half* __restrict__ send,
                                       std::size_t count,
                                       std::uint32_t* arrivals,
                                       std::uint64_t* ready,
                                       std::uint64_t value) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const std::size_t first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (std::size_t i = first; i < count; i += stride) {
    send[i] = __float2half_rn(partial[i]);
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0 && __hip_atomic_fetch_add(arrivals, 1u, __ATOMIC_ACQ_REL,
                                                 __HIP_MEMORY_SCOPE_SYSTEM) +
                                  1 ==
                              gridDim.x) {
    __hip_atomic_store(arrivals, 0u, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_SYSTEM);
    __hip_atomic_store(ready, value, __ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__global__ void AddRowsBroadcastHalfKernel(const __half* __restrict__ addend,
                                           float* residual,
                                           std::uint32_t hidden,
                                           std::uint32_t streams) {
  // The local operand narrows to fp16 before the add: the wire carries
  // fp16(r_peer), and rounding only one side would leave the two ranks with
  // different sums (r_local + fp16(r_peer) here, r_peer + fp16(r_local)
  // there). That asymmetry diverges greedy choices, which desynchronizes the
  // speculative chain lengths, which then shows up as a collective size
  // mismatch and poisons the transport. Both ranks must compute
  // fp16(r_local) + fp16(r_peer).
  const std::uint32_t t = blockIdx.x;
  for (std::uint32_t i = threadIdx.x; i < streams * hidden; i += blockDim.x) {
    float* slot = &residual[static_cast<std::size_t>(t) * streams * hidden + i];
    *slot = __half2float(__float2half_rn(*slot)) +
            __half2float(
                addend[static_cast<std::size_t>(t) * hidden + i % hidden]);
  }
}

__global__ void WaitValueKernel(const std::uint64_t* flag, std::uint64_t value,
                                std::uint64_t ticks, std::uint32_t* timed_out) {
  const std::uint64_t start = wall_clock64();
  while (__hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) <
         value) {
    if (wall_clock64() - start > ticks) {
      __hip_atomic_store(timed_out, 1u, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_SYSTEM);
      return;
    }
    __builtin_amdgcn_s_sleep(1);
  }
}

/// Captured-exchange variants: the target value is read from a pinned,
/// host-mapped slot each execution, so a graph replay advances to the next
/// exchange id without touching kernel parameters. `send` is the send
/// buffer's payload base; the window comes from the slot, matching the
/// replayed host function.
__global__ void StagePartialSlottedKernel(
    const float* __restrict__ partial, float* __restrict__ send,
    std::size_t count, std::uint32_t* arrivals, std::uint64_t* ready,
    const std::uint64_t* __restrict__ slot) {
  const std::uint64_t value =
      __hip_atomic_load(slot, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
  send += ExchangeWindowOffset(value - 1) / sizeof(float);
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const std::size_t first =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t vectors = count / 4;
  for (std::size_t i = first; i < vectors; i += stride) {
    reinterpret_cast<float4*>(send)[i] =
        reinterpret_cast<const float4*>(partial)[i];
  }
  for (std::size_t i = vectors * 4 + first; i < count; i += stride) {
    send[i] = partial[i];
  }
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0 && __hip_atomic_fetch_add(arrivals, 1u, __ATOMIC_ACQ_REL,
                                                 __HIP_MEMORY_SCOPE_SYSTEM) +
                                  1 ==
                              gridDim.x) {
    __hip_atomic_store(arrivals, 0u, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_SYSTEM);
    __hip_atomic_store(ready, value, __ATOMIC_RELEASE,
                       __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

__global__ void WaitValueSlottedKernel(const std::uint64_t* flag,
                                       const std::uint64_t* __restrict__ slot,
                                       std::uint64_t ticks,
                                       std::uint32_t* timed_out) {
  const std::uint64_t value =
      __hip_atomic_load(slot, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
  const std::uint64_t start = wall_clock64();
  while (__hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) <
         value) {
    if (wall_clock64() - start > ticks) {
      __hip_atomic_store(timed_out, 1u, __ATOMIC_RELAXED,
                         __HIP_MEMORY_SCOPE_SYSTEM);
      return;
    }
    __builtin_amdgcn_s_sleep(1);
  }
}

__device__ __forceinline__ ArgmaxCandidate BetterCandidate(ArgmaxCandidate a,
                                                           ArgmaxCandidate b) {
  return b.value > a.value || (b.value == a.value && b.index < a.index) ? b : a;
}

__device__ ArgmaxCandidate ArgmaxBlock(ArgmaxCandidate best,
                                       ArgmaxCandidate* shared) {
  const unsigned lane = threadIdx.x & 31u;
  const unsigned wave = threadIdx.x >> 5u;
  for (int offset = 16; offset > 0; offset >>= 1) {
    best = BetterCandidate(
        best, {__shfl_xor(best.value, offset), __shfl_xor(best.index, offset)});
  }
  if (lane == 0) {
    shared[wave] = best;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (unsigned w = 1; w < kThreads / 32; ++w) {
      best = BetterCandidate(best, shared[w]);
    }
  }
  return best;
}

__global__ void ArgmaxPartialKernel(const float* logits,
                                    ArgmaxCandidate* partial,
                                    std::uint32_t vocab) {
  __shared__ ArgmaxCandidate shared[kThreads / 32];
  const std::uint32_t t = blockIdx.y;
  const float* src = logits + static_cast<std::size_t>(t) * vocab;
  ArgmaxCandidate best{-INFINITY, INT32_MAX};
  for (std::uint32_t i = blockIdx.x * kThreads + threadIdx.x; i < vocab;
       i += kArgmaxParts * kThreads) {
    best = BetterCandidate(best, {src[i], static_cast<std::int32_t>(i)});
  }
  best = ArgmaxBlock(best, shared);
  if (threadIdx.x == 0) {
    partial[t * kArgmaxParts + blockIdx.x] = best;
  }
}

__global__ void ArgmaxFinishKernel(const float* logits,
                                   const ArgmaxCandidate* partial,
                                   std::int32_t* out, std::uint32_t vocab) {
  __shared__ ArgmaxCandidate shared[kThreads / 32];
  const std::uint32_t t = blockIdx.x;
  ArgmaxCandidate best{-INFINITY, INT32_MAX};
  if (threadIdx.x < kArgmaxParts) {
    best = partial[t * kArgmaxParts + threadIdx.x];
  }
  best = ArgmaxBlock(best, shared);
  if (threadIdx.x == 0) {
    // std::max_element keeps element zero when it is NaN; later NaNs
    // never replace a finite candidate.
    out[t] =
        isnan(logits[static_cast<std::size_t>(t) * vocab]) ? 0 : best.index;
  }
}

constexpr unsigned kMtpCandidateTile = 1024;

__global__ void GatherArgmaxCandidatesKernel(const float* logits,
                                             const std::uint32_t* ids,
                                             ArgmaxCandidate* out,
                                             std::uint32_t rows,
                                             std::uint32_t vocab) {
  const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) {
    const auto id = ids[row];
    out[row] = id < vocab
                   ? ArgmaxCandidate{logits[std::size_t(row) * vocab + id],
                                     static_cast<std::int32_t>(id)}
                   : ArgmaxCandidate{NAN, -1};
  }
}

// A token excluded from its tile's top Keep cannot enter the global top Keep.
// Reduce tiles repeatedly, retaining the original order of equal scores.
// Only token IDs need to survive between passes; the original logit buffer
// supplies scores and preserves the sign of zero in the final output.
template<unsigned Keep>
__global__ void MtpCandidateTileKernel(const float* logits,
                                       const std::uint32_t* input,
                                       std::uint32_t* output, float* scores,
                                       std::uint32_t size,
                                       std::uint32_t vocab) {
  using Sort = hipcub::BlockRadixSort<float, kThreads, 4, std::uint32_t>;
  __shared__ Sort::TempStorage scratch;
  float keys[4];
  std::uint32_t ids[4];
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    const std::size_t i =
        std::size_t(blockIdx.x) * kMtpCandidateTile + threadIdx.x * 4 + j;
    const auto id = i < size ? (input != nullptr ? input[i] : i) : UINT32_MAX;
    ids[j] = static_cast<std::uint32_t>(id);
    const float value = id < vocab ? logits[id] : -INFINITY;
    keys[j] = isfinite(value) ? (value == 0.0F ? 0.0F : value) : -INFINITY;
  }
  Sort(scratch).SortDescending(keys, ids);
#pragma unroll
  for (unsigned j = 0; j < 4; ++j) {
    const unsigned rank = threadIdx.x * 4 + j;
    if (rank < min(Keep, size)) {
      output[blockIdx.x * Keep + rank] = ids[j];
      if (scores != nullptr) {
        const float value = ids[j] < vocab ? logits[ids[j]] : -INFINITY;
        scores[rank] = isfinite(value) ? value : -INFINITY;
      }
    }
  }
}

__global__ void CopyKernel(const float* src, float* dst, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    dst[i] = src[i];
  }
}

inline unsigned Blocks(std::size_t count) {
  return static_cast<unsigned>((count + kThreads - 1) / kThreads);
}

// W8A8 int8 WMMA GEMM over the GGUF Q8_0 weights, ported from the Qwen 27B
// route (src/models/qwen/hip/kernels/prefill_quant_gemm.hip,
// opt-c163-blocked-w8a8 with the opt-c179 addressing). Weights stay in their
// row-major 34-byte block_q8_0 layout; activations are quantized per 32-wide
// block into WMMA B-fragment order:
//
//   tile(tt, kb) = [b0: 16 tokens x 16 bytes]   offset   0
//                  [b1: 16 tokens x 16 bytes]   offset 256
//                  [16 fp32 token scales    ]   offset 512
//
// with tt = token / 16 and kb the 32-element K block, 576 bytes per tile.
struct Q8_0Block {
  __half d;
  std::int8_t qs[32];
};
static_assert(sizeof(Q8_0Block) == 34, "block_q8_0 must be 34 bytes");

using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;

__device__ __forceinline__ int32x8_t WmmaI8(int32x4_t a, int32x4_t b,
                                            int32x8_t c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, true);
}

/// Four elements per lane, eight lanes per 32-wide K block, four blocks per
/// wave: per-block absmax scale, four codes stored as one word straight
/// into the fragment order the GEMM stages from.
__global__ void QuantizeQ8TiledVec4Kernel(const float* __restrict__ x,
                                          void* __restrict__ y,
                                          std::size_t batch, std::size_t k) {
  const std::size_t num_blocks = k / 32;
  const std::size_t b_idx =
      (blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x) >> 3u;
  const std::uint32_t lane8 = threadIdx.x & 7u;
  if (b_idx >= batch * num_blocks) {
    return;
  }
  const std::size_t tok = b_idx / num_blocks;
  const std::size_t blk = b_idx % num_blocks;
  const float4 v = *reinterpret_cast<const float4*>(x + (tok * k) + (blk * 32) +
                                                    (lane8 * 4));
  float max_abs =
      fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
  for (int off = 4; off > 0; off >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
  }
  const float d = max_abs / 127.0F;
  const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
  const auto q0 = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(v.x * id))));
  const auto q1 = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(v.y * id))));
  const auto q2 = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(v.z * id))));
  const auto q3 = static_cast<std::uint32_t>(
      static_cast<std::uint8_t>(static_cast<std::int8_t>(roundf(v.w * id))));
  const std::size_t tt = tok / kQ8ActTileTokens;
  const std::size_t tl = tok % kQ8ActTileTokens;
  std::int8_t* tile = Q8ActTile(y, num_blocks, tt, blk);
  const std::uint32_t pos = lane8 * 4;  // 0, 4, ..., 28
  *reinterpret_cast<std::uint32_t*>(tile + ((pos >> 4u) * 256) + (tl * 16) +
                                    (pos & 15u)) =
      q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
  if (lane8 == 0) {
    *reinterpret_cast<float*>(tile + kQ8ActScaleOffset + (tl * sizeof(float))) =
        d;
  }
}

/// One wave per (token, 32-wide K block): per-block absmax scale, codes
/// written straight into the fragment order the GEMM stages from.
__global__ void QuantizeQ8TiledKernel(const float* __restrict__ x,
                                      void* __restrict__ y, std::size_t batch,
                                      std::size_t k) {
  const std::size_t num_blocks = k / 32;
  const std::size_t b_idx =
      (blockIdx.x * (blockDim.x >> 5u)) + (threadIdx.x >> 5u);
  const std::size_t lane_id = threadIdx.x & 31u;
  if (b_idx >= batch * num_blocks) {
    return;
  }
  const std::size_t tok = b_idx / num_blocks;
  const std::size_t blk = b_idx % num_blocks;
  const float val = x[(tok * k) + (blk * 32) + lane_id];
  float max_abs = fabsf(val);
  for (int off = 16; off > 0; off >>= 1) {
    max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
  }
  const float d = max_abs / 127.0F;
  const float id = (d != 0.0F) ? (1.0F / d) : 0.0F;
  const auto q = static_cast<std::int8_t>(roundf(val * id));
  const std::size_t tt = tok / kQ8ActTileTokens;
  const std::size_t tl = tok % kQ8ActTileTokens;
  std::int8_t* tile = Q8ActTile(y, num_blocks, tt, blk);
  const std::size_t half = lane_id >> 4u;
  const std::size_t pos = lane_id & 15u;
  tile[(half * 256) + (tl * 16) + pos] = q;
  if (lane_id == 0) {
    *reinterpret_cast<float*>(tile + kQ8ActScaleOffset + (tl * sizeof(float))) =
        d;
  }
}

// Preserve the separate scale/SiLU kernel's F32 intermediates before narrowing.
// Without these compiler boundaries, fused F16 output can round differently.
__device__ __forceinline__ float HcScaledSilu(float value) {
  float scaled = value * 0.25F;
  asm volatile("" : "+v"(scaled));
  float result = SiluF(scaled);
  asm volatile("" : "+v"(result));
  return result;
}

/// Two-dimensionally blocked W8A8 WMMA GEMM: block = BM rows x BN tokens, BK
/// 32-element K blocks per LDS stage, waves = WM row groups x WN token groups.
/// Out-of-range rows and K blocks are clamped and their scale zeroed, so
/// they contribute exactly zero without divergence. y is [batch][m].
template<int BM, int BN, int BK, int WM, int WN, bool kHcDown = false>
__launch_bounds__(256) __global__ void W8A8BlockedWmmaGEMMKernel(
    const void* __restrict__ w, const void* __restrict__ x_blocks,
    std::conditional_t<kHcDown, __half, float>* __restrict__ y,
    std::size_t batch, std::size_t m, std::size_t k) {
  static_assert(WM * WN == 8, "256 threads is 8 waves");
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
  static_assert(BN / 16 <= 8,
                "the activation stage assigns one wave per token subtile");
  constexpr int kRowTiles = BM / 16;
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = kRowTiles / WM;
  constexpr int kWaveTokTiles = kTokTiles / WN;

  // One LDS plane: codes and scales of both operands; the epilogue reuses
  // the first 16 KB to transpose 16 x 32 result tiles per wave.
  constexpr int kABytes = BK * kRowTiles * 32 * 16;
  constexpr int kDwBytes = BK * kRowTiles * 16 * 4;
  constexpr int kBBytes = BK * kTokTiles * 32 * 16;
  constexpr int kDxBytes = BK * kTokTiles * 16 * 4;
  constexpr int kLdsBytes = kABytes + kDwBytes + kBBytes + kDxBytes;
  static_assert(kLdsBytes >= 8 * 512 * 4, "epilogue transposes 16 KB");
  static_assert(kWaveRowTiles % 2 == 0, "the epilogue pairs row tiles");
  __shared__ __attribute__((aligned(16))) std::uint8_t lds[kLdsBytes];
  auto* s_a = reinterpret_cast<int32x4_t(*)[kRowTiles][32]>(lds);
  auto* s_dw = reinterpret_cast<float (*)[kRowTiles][16]>(lds + kABytes);
  auto* s_b =
      reinterpret_cast<int32x4_t(*)[kTokTiles][32]>(lds + kABytes + kDwBytes);
  auto* s_dx = reinterpret_cast<float (*)[kTokTiles][16]>(lds + kABytes +
                                                          kDwBytes + kBBytes);

  const auto* w_blocks = static_cast<const Q8_0Block*>(w);
  const std::size_t num_blocks = k / 32;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int wave_row = wave_id / WN;
  const int wave_tok = wave_id % WN;

  const std::size_t r_block = static_cast<std::size_t>(blockIdx.y) * BM;
  const std::size_t t_block = static_cast<std::size_t>(blockIdx.x) * BN;
  const std::size_t tt_block = t_block / kQ8ActTileTokens;

  float acc[kWaveRowTiles][kWaveTokTiles][8];
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        acc[i][j][l] = 0.0F;
      }
    }
  }

  // Staging registers for the next K stage, fetched one stage ahead so the
  // scattered 34-byte block reads overlap the WMMA work.
  constexpr int kPrefetch = (BM * BK) / 256;
  int32x4_t r_q0[kPrefetch];
  int32x4_t r_q1[kPrefetch];
  float r_dw[kPrefetch];
  int32x4_t r_b[BK];
  float r_dx[BK];
  const int b_tile = wave_id;  // one wave per token subtile

  const int num_kb = static_cast<int>(num_blocks);
  const int m_i = static_cast<int>(m);
  const Q8_0Block* w_row[kPrefetch];
  float row_live[kPrefetch];
#pragma unroll
  for (int p = 0; p < kPrefetch; ++p) {
    const int r = static_cast<int>(r_block) + (((p * 256) + tid) / BK);
    const int r_clamped = (r < m_i) ? r : (m_i - 1);
    w_row[p] = w_blocks + (static_cast<std::size_t>(r_clamped) * num_blocks);
    row_live[p] = (r < m_i) ? 1.0F : 0.0F;
  }
  const auto* b_base = static_cast<const std::int8_t*>(x_blocks) +
                       ((tt_block + static_cast<std::size_t>(b_tile)) *
                        num_blocks * kQ8ActTileBytes);
  const bool b_live = b_tile < kTokTiles;

  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      const int kb = kb0 + (((p * 256) + tid) % BK);
      const int kb_clamped = (kb < num_kb) ? kb : (num_kb - 1);
      const Q8_0Block& blk = w_row[p][kb_clamped];
      __builtin_memcpy(&r_q0[p], blk.qs + 0, 16);
      __builtin_memcpy(&r_q1[p], blk.qs + 16, 16);
      r_dw[p] = __half2float(blk.d) * ((kb < num_kb) ? row_live[p] : 0.0F);
    }
    if (b_live) {
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        const int kb = kb0 + i;
        const int kb_clamped = (kb < num_kb) ? kb : (num_kb - 1);
        const auto* tile =
            b_base + (static_cast<std::size_t>(kb_clamped) * kQ8ActTileBytes);
        r_b[i] = reinterpret_cast<const int32x4_t*>(tile)[lane_id];
        r_dx[i] = ((kb < num_kb) ? 1.0F : 0.0F) *
                  reinterpret_cast<const float*>(
                      tile + kQ8ActScaleOffset)[lane_id & 15];
      }
    }
  };

  const auto commit_stage = [&]() {
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      const int idx = (p * 256) + tid;
      const int rr = idx / BK;
      const int kk = idx % BK;
      const int rs = rr / 16;
      const int rl = rr % 16;
      s_a[kk][rs][rl] = r_q0[p];
      s_a[kk][rs][16 + rl] = r_q1[p];
      s_dw[kk][rs][(rl % 2 == 0) ? (rl / 2) : (8 + (rl / 2))] = r_dw[p];
    }
    if (b_live) {
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        s_b[i][b_tile][lane_id] = r_b[i];
        if (lane_id < 16) {
          s_dx[i][b_tile][lane_id] = r_dx[i];
        }
      }
    }
  };

  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch_stage(kb0 + BK);
    }

#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      int32x4_t a0[kWaveRowTiles];
      int32x4_t a1[kWaveRowTiles];
      float dw[kWaveRowTiles][8];
#pragma unroll
      for (int i = 0; i < kWaveRowTiles; ++i) {
        const int rs = (wave_row * kWaveRowTiles) + i;
        a0[i] = s_a[kb][rs][sub_lane];
        a1[i] = s_a[kb][rs][16 + sub_lane];
        const float4 lo =
            *reinterpret_cast<const float4*>(&s_dw[kb][rs][half_id * 8]);
        const float4 up =
            *reinterpret_cast<const float4*>(&s_dw[kb][rs][(half_id * 8) + 4]);
        dw[i][0] = lo.x;
        dw[i][1] = lo.y;
        dw[i][2] = lo.z;
        dw[i][3] = lo.w;
        dw[i][4] = up.x;
        dw[i][5] = up.y;
        dw[i][6] = up.z;
        dw[i][7] = up.w;
      }
      // Token-tile operands are read one tile at a time to keep the live
      // register set small enough for the BK=2 stage.
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j) {
        const int ts = (wave_tok * kWaveTokTiles) + j;
        const int32x4_t b0 = s_b[kb][ts][sub_lane];
        const int32x4_t b1 = s_b[kb][ts][16 + sub_lane];
        const float dx = s_dx[kb][ts][sub_lane];
#pragma unroll
        for (int i = 0; i < kWaveRowTiles; ++i) {
          int32x8_t c = {0, 0, 0, 0, 0, 0, 0, 0};
          c = WmmaI8(a0[i], b0, c);
          c = WmmaI8(a1[i], b1, c);
#pragma unroll
          for (int l = 0; l < 8; ++l) {
            acc[i][j][l] += (dw[i][l] * dx) * static_cast<float>(c[l]);
          }
        }
      }
#if __clang_major__ >= 23
      // LLVM 23 hoists the next K block's operands above this one's WMMAs,
      // which spills registers on gfx1151. ROCm 7.2.3 (LLVM 22) does not, and
      // the fence costs it about 3%.
      __builtin_amdgcn_sched_barrier(0);
#endif
    }
    __syncthreads();
  }

  // Transpose the result through LDS, two row tiles at a time, so every
  // global store covers 32 consecutive rows of one token: a full 128-byte
  // line (half lines cost a read-modify-write on the fabric).
  __syncthreads();
  constexpr unsigned kOutputStride = kHcDown ? 36 : 32;
  static_assert(8 * 16 * kOutputStride * sizeof(float) <= sizeof(lds));
  float* tile_scratch =
      reinterpret_cast<float*>(lds) + wave_id * 16 * kOutputStride;
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; i += 2) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
      // scratch[token][row] over 16 tokens x 32 rows.
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        tile_scratch[(sub_lane * kOutputStride) + (2 * l) + half_id] =
            acc[i][j][l];
        tile_scratch[(sub_lane * kOutputStride) + 16 + (2 * l) + half_id] =
            acc[i + 1][j][l];
      }
      __builtin_amdgcn_wave_barrier();
      const std::size_t r0 =
          r_block +
          static_cast<std::size_t>((((wave_row * kWaveRowTiles) + i) * 16));
      const std::size_t t0 =
          t_block +
          static_cast<std::size_t>((((wave_tok * kWaveTokTiles) + j) * 16));
      // Lane pair (2p, 2p + 1) stores token p's 32 rows as eight float4.
      const int tok_l = lane_id >> 1;
      const int row_l = (lane_id & 1) * 16;
      const std::size_t tok = t0 + static_cast<std::size_t>(tok_l);
      const auto* src = reinterpret_cast<const float4*>(
          tile_scratch + (tok_l * kOutputStride) + row_l);
      if (tok < batch && r0 + 32 <= m) {
        if constexpr (kHcDown) {
          auto* dst = y + tok * m + r0 + static_cast<std::size_t>(row_l);
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const float4 v = src[q];
            *reinterpret_cast<__half2*>(dst + 4 * q) =
                __floats2half2_rn(HcScaledSilu(v.x), HcScaledSilu(v.y));
            *reinterpret_cast<__half2*>(dst + 4 * q + 2) =
                __floats2half2_rn(HcScaledSilu(v.z), HcScaledSilu(v.w));
          }
        } else {
          auto* dst = reinterpret_cast<float4*>(
              y + (tok * m) + r0 + static_cast<std::size_t>(row_l));
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            dst[q] = src[q];
          }
        }
      } else if (tok < batch) {
#pragma unroll
        for (int q = 0; q < 16; ++q) {
          const std::size_t r = r0 + static_cast<std::size_t>(row_l + q);
          if (r < m) {
            const float value =
                tile_scratch[(tok_l * kOutputStride) + row_l + q];
            if constexpr (kHcDown)
              y[(tok * m) + r] = __float2half_rn(HcScaledSilu(value));
            else
              y[(tok * m) + r] = value;
          }
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

// Routed expert GEMMs: assignment rows are compacted by expert with every
// bucket padded to a 16-row tile (`pad_bounds`), so a token tile never
// straddles experts; `rows_out` maps a compact row to its (token, slot)
// output row or -1 for padding.
constexpr std::size_t kRoutedTileTokens = 16;

struct Q4KBlock {
  __half d;
  __half dmin;
  std::uint8_t scales[12];
  std::uint8_t qs[128];
};
static_assert(sizeof(Q4KBlock) == 144, "block_q4_K must be 144 bytes");

struct Q5_1Block {
  __half d;
  __half m;
  std::uint32_t qh;
  std::uint8_t qs[16];
};
static_assert(sizeof(Q5_1Block) == 24, "block_q5_1 must be 24 bytes");

/// Byte `i` of the 16-byte block header (d, dmin, scales[12]).
__device__ __forceinline__ std::uint32_t HeaderByte(const uint4& h,
                                                    std::uint32_t i) {
  const std::uint32_t word = i < 4 ? h.x : i < 8 ? h.y : i < 12 ? h.z : h.w;
  return (word >> (8U * (i & 3U))) & 0xFFU;
}

// Routed F16 WMMA expert GEMM. The int8 kernel above pays a float epilogue
// and an activation-sum correction every K block because the per-32 scales
// of both operands sit outside the integer dot product; here the weights are
// dequantized to F16 right after the LDS read (each wave decodes only its own
// sixteen rows) and the activations are F16 rows, so the matrix core
// accumulates the whole K extent in F32 with no per-block work. The codes
// stay packed in LDS (4 bits for Q4_K, 4 + 1 for Q5_1), which keeps a
// two-K-block stage at 11-12 KB and five blocks resident per WGP.
//
// A code becomes a half through a byte permute into the mantissa of 1024.0
// (0x6400 | q = 1024 + q exactly for q < 32, the half's unit being 1 there),
// a packed subtract of 1024 (exact), then one packed FMA:
//
//     w = q * scale + bias
//
// with (scale, bias) = (d * sc, -dmin * mn) for Q4_K and (d, m) for Q5_1,
// staged per (row, K block) as a half2.
//
// grid (m / BM, tiles): `tiles[y]` packs the expert in the low 16 bits and
// the token macro tile index in the high 16, so no block is launched for an
// empty tile; the row blocks of one tile are consecutive in dispatch order so
// they share the tile's gathered activations through L2. Block (x, y)
// computes rows x*BM.. of the expert against its
// compact rows [pad_bounds[e] + j*BN, +BN) and scatters them to
// out[rows_out[c]][row] (F32, or F16 with the SwiGLU applied when `out_half`
// is given: the up projection then writes the down projection's input).
constexpr std::uint32_t kHalfMagic = 0x64646464U;  // 1024.0 high bytes

/// block_q5_K: the Q4_K header, 32 high-bit bytes (bit s of byte j is the
/// fifth bit of element j of K block s), then the Q4_K nibble layout.
constexpr std::size_t kQ5KBlockBytes = 176;

template<WeightType kType>
__device__ __forceinline__ std::size_t RoutedF16RowBytes(std::size_t k) {
  return kType == WeightType::kQ4_K   ? (k / 256) * sizeof(Q4KBlock)
         : kType == WeightType::kQ5_K ? (k / 256) * kQ5KBlockBytes
         : kType == WeightType::kQ5_1 ? (k / 32) * sizeof(Q5_1Block)
         : kType == WeightType::kMXFP4 ? (k / 32) * 17
                                      : (k / 32) * sizeof(Q8_0Block);
}

/// MXFP4 packs 32 elements per block: one E8M0 exponent byte then 16 bytes
/// of E2M1 nibbles. The nibble's F16 bit pattern is (n >> 1) + 13 in the
/// exponent with the mantissa bit in place; zero stays zero (handled by
/// the table below rather than the formula).
constexpr std::uint16_t kMxFp4E2M1Half[16] = {
    0x0000, 0x3800, 0x3C00, 0x3E00, 0x4000, 0x4200, 0x4400, 0x4500,
    0x8000, 0xB800, 0xBC00, 0xBE00, 0xC000, 0xC200, 0xC400, 0xC500};

/// Bit `s` of each of the four bytes of `w`, packed into bits 0-3.
__device__ __forceinline__ std::uint32_t GatherBit(std::uint32_t w, int s) {
  // 0x01020408 moves byte b's bit to bit 24 + b.
  return (((w >> s) & 0x01010101U) * 0x01020408U) >> 24U;
}

/// Four packed 5-bit codes: `nib` holds the 4-bit parts one per byte, `bits`
/// bits j..j+3 of the Q5_1 high-bit word, spread to bit 4 of each byte.
__device__ __forceinline__ std::uint32_t SpreadHighBits(std::uint32_t bits) {
  // 0x00204081 = 1 + 2^7 + 2^14 + 2^21: bit b of `bits` lands at 8b, every
  // cross term falls off the 0x01010101 mask.
  return (__umul24(bits, 0x00204081U) & 0x01010101U) << 4U;
}

/// Four halves from four code bytes: 1024 + q as F16, minus `magic` (1024,
/// or 1152 for a signed byte carried as q + 128), then the affine.
__device__ __forceinline__ void CodesToHalves(std::uint32_t codes,
                                              __half2 magic, __half2 scale2,
                                              __half2 bias2, __half2& lo,
                                              __half2& hi) {
  const std::uint32_t p0 =
      __builtin_amdgcn_perm(codes, kHalfMagic, 0x01050004U);
  const std::uint32_t p1 =
      __builtin_amdgcn_perm(codes, kHalfMagic, 0x03070206U);
  lo = __hfma2(__hadd2(__builtin_bit_cast(__half2, p0), magic), scale2, bias2);
  hi = __hfma2(__hadd2(__builtin_bit_cast(__half2, p1), magic), scale2, bias2);
}

template<WeightType kType, int BM, int BN, int BK, bool kPair = false>
__launch_bounds__(256) __global__
    void RoutedF16GEMMKernel(const void* __restrict__ w,
                             const __half* __restrict__ x,
                             const std::int32_t* __restrict__ tiles,
                             const std::int32_t* __restrict__ pad_bounds,
                             const std::int32_t* __restrict__ rows_in,
                             const std::int32_t* __restrict__ rows_out,
                             const float* __restrict__ swiglu_gate,
                             float* __restrict__ out,
                             __half* __restrict__ out_half, std::size_t m,
                             std::size_t k, const void* __restrict__ w_up,
                             int dbg) {
  static_assert(BM == 128 || BM == 256, "eight waves, 16-row tiles");
  static_assert(!kPair || BM == 128);
  static_assert(BN % 16 == 0 && BN / 16 <= 8);
  static_assert(BK == 2, "one stage is one 32-byte Q4_K nibble group");
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = BM / 128;  // 16-row tiles per wave
  constexpr bool kQ5 = kType == WeightType::kQ5_1;
  constexpr bool kQ5K = kType == WeightType::kQ5_K;
  constexpr bool kQ8 = kType == WeightType::kQ8_0;
  constexpr bool kMX = kType == WeightType::kMXFP4;
  constexpr bool kKQuant = kType == WeightType::kQ4_K || kQ5K;
  // 16-byte code chunks per row and stage: Q4_K's nibble pair and Q5_1's
  // two nibble blocks are two, Q8_0's two byte blocks are four.
  constexpr int kChunks = kQ8 ? 2 * BK : BK;

  // LDS plan (bytes): the code plane holds BM rows x kChunks 16-byte chunks
  // with the chunks of nearby rows permuted so a fragment read (one row per
  // lane) covers all bank groups; the activation plane is
  // [kb][16-element quarter][token][16 B] so a fragment read is 256
  // contiguous bytes; the epilogue reuses it all.
  constexpr int kCodeBytes = BM * kChunks * 16;
  constexpr int kHighBytes = (kQ5 || kQ5K) ? BK * BM * 4 : 0;
  constexpr int kScaleBytes = BK * BM * 4;
  // One slot of padding per activation quarter plane: the eight chunks of
  // a token then land on eight bank groups when they are written.
  constexpr int kActStride = BN + 1;
  constexpr int kActBytes = BK * 4 * kActStride * 16;
  // The epilogue transposes one 16x16 tile per wave through the same
  // bytes (8 KB), which the narrow tile's stages do not reach.
  constexpr int kStageBytes = kCodeBytes + kHighBytes + kScaleBytes + kActBytes;
  constexpr int kLdsBytes = kStageBytes > 8 * 1024 ? kStageBytes : 8 * 1024;
  __shared__ __attribute__((aligned(16))) std::uint8_t lds[kLdsBytes];
  // MXFP4 decode table (E2M1 nibble -> F16), shared for gather decode;
  // parked after the stage planes, whose size the epilogue also rounds.
  // One __half per entry: the gather reads through a uint16_t view with the
  // raw nibble as the index, so a __half2 table would hand back entry
  // nibble/2 (both negatives-as-positives and squashed magnitudes).
  __shared__ __half s_mx_lut[16];
  auto* s_codes = reinterpret_cast<uint4*>(lds);
  auto* s_high = reinterpret_cast<std::uint32_t*>(lds + kCodeBytes);
  auto* s_scale =
      reinterpret_cast<std::uint32_t*>(lds + kCodeBytes + kHighBytes);
  auto* s_act =
      reinterpret_cast<uint4*>(lds + kCodeBytes + kHighBytes + kScaleBytes);

  if (threadIdx.x < 16) {
    s_mx_lut[threadIdx.x] =
        __builtin_bit_cast(__half, kMxFp4E2M1Half[threadIdx.x]);
  }
  __syncthreads();
  const std::int32_t tile = tiles[blockIdx.y];
  const int expert = tile & 0xFFFF;
  const int t_local = (tile >> 16) * BN;
  const int bucket_begin = pad_bounds[expert];
  const int bucket_rows = pad_bounds[expert + 1] - bucket_begin;
  const int live_tok_tiles =
      std::min(kTokTiles, (bucket_rows - t_local + 15) / 16);
  const int num_kb = static_cast<int>(k / 32);
  const int m_i = static_cast<int>(m);
  const std::size_t row_bytes = RoutedF16RowBytes<kType>(k);
  const auto* w_expert = static_cast<const std::uint8_t*>(w) +
                         static_cast<std::size_t>(expert) * m * row_bytes;

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  constexpr int kRows = kPair ? BM / 2 : BM;
  const int r_block = static_cast<int>(blockIdx.x) * kRows;

  // Weight fetch: unit u of a thread is (row = tid / 2 + 128 u, chunk c =
  // tid % 2). Q4_K: the two 16-byte halves of one 32-byte nibble group (two
  // K blocks, low and high nibbles); Q5_1: one 24-byte K block each.
  const int f_c = tid & 1;
  const std::uint8_t* f_ptr[kWaveRowTiles];
  bool f_live[kWaveRowTiles];
  uint4 f_header[kWaveRowTiles];
#pragma unroll
  for (int u = 0; u < kWaveRowTiles; ++u) {
    const int r =
        r_block + (kPair ? (tid >> 1) % kRows : (tid >> 1) + (u * 128));
    f_live[u] = r < m_i;
    const std::uint8_t* weights = w_expert;
    if constexpr (kPair) {
      if ((tid >> 1) >= kRows) {
        weights = static_cast<const std::uint8_t*>(w_up) +
                  static_cast<std::size_t>(expert) * m * row_bytes;
      }
    }
    f_ptr[u] = weights +
               static_cast<std::size_t>(f_live[u] ? r : (m_i - 1)) * row_bytes;
    f_header[u] = make_uint4(0u, 0u, 0u, 0u);
  }
  // The next stage's weights and activations, fetched one stage ahead. The
  // (scale, bias) pair is derived from the raw header word only when the
  // stage is committed, so nothing waits on the loads before the compute.
  uint4 f_codes[kWaveRowTiles];
  // Paired gate/up tiles reuse the full quantized block over four stages.
  // Keep its remaining codes in registers alongside the cached header.
  uint4 code_cache[kWaveRowTiles][4];
  uint4 f_codes_hi[kWaveRowTiles];  ///< Q8_0: the block's second 16 codes
  uint4 f_qh[kWaveRowTiles][2];     ///< Q5_K: the superblock's high bits
  std::uint32_t f_high[kWaveRowTiles];
  std::uint32_t f_dm[kWaveRowTiles];  ///< Q5_1: d | m; Q8_0: d
  int f_sb32[kWaveRowTiles];          ///< Q4_K: the K block in its superblock
  constexpr int kActFetch = BN <= 64 ? 2 : 4;
  uint4 a_data[kActFetch];

  // Activation fetch: BN tokens x (BK * 64) bytes per stage in 16-byte
  // chunks, eight per token; each thread fetches consecutive 256-chunk
  // strides, up to four for a 128-token tile.
  constexpr int kActChunks = BN * BK * 4;
  static_assert(kActChunks <= kActFetch * 256);
  const __half* a_src[kActFetch];
  int a_slot[kActFetch];
#pragma unroll
  for (int i = 0; i < kActFetch; ++i) {
    const int chunk = tid + (i * 256);
    const int t = chunk / (BK * 4);
    const int sub = chunk % (BK * 4);
    const int c_row = t_local + t;
    const std::int32_t src = (chunk < kActChunks && c_row < bucket_rows)
                                 ? rows_in[bucket_begin + c_row]
                                 : -1;
    a_src[i] = src >= 0 ? x + (static_cast<std::size_t>(src) * k) + (sub * 8)
                        : nullptr;
    // s_act[(kb * 4 + quarter) * kActStride + t]
    a_slot[i] = chunk < kActChunks ? (sub * kActStride) + t : -1;
  }

  const auto swizzle = [](int row, int c) {
    return (row * kChunks) +
           (c ^ (kChunks == 4 ? ((row >> 1) & 3) : ((row >> 2) & 1)));
  };

  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int u = 0; u < kWaveRowTiles; ++u) {
      if constexpr (kQ5) {
        const int kb = kb0 + f_c;
        const auto* words = reinterpret_cast<const uint2*>(f_ptr[u]) + (kb * 3);
        const uint2 w0 = words[0];
        const uint2 w1 = words[1];
        const uint2 w2 = words[2];
        f_codes[u] = make_uint4(w1.x, w1.y, w2.x, w2.y);
        f_high[u] = w0.y;
        f_dm[u] = w0.x;
      } else if constexpr (kQ8) {
        // block_q8_0 is 34 bytes, so the code loads are 2-byte aligned.
        const auto* blk = f_ptr[u] + ((kb0 + f_c) * 34);
        f_dm[u] = *reinterpret_cast<const std::uint16_t*>(blk);
        __builtin_memcpy(&f_codes[u], blk + 2, 16);
        __builtin_memcpy(&f_codes_hi[u], blk + 18, 16);
      } else if constexpr (kMX) {
        // MXFP4: 17-byte blocks (E8M0 byte, then 16 nibble bytes); the
        // scale rides in f_sb32, the nibbles in f_codes. The 16-byte code
        // span starts at an odd address (17n + 1), so the load must be
        // scalar bytes — a vector load there is undefined on this ISA
        // (and faults).
        const auto* blk = f_ptr[u] + ((kb0 + f_c) * 17);
        f_sb32[u] = blk[0];
        f_codes[u].x = static_cast<std::uint32_t>(blk[1]) |
                       (static_cast<std::uint32_t>(blk[2]) << 8) |
                       (static_cast<std::uint32_t>(blk[3]) << 16) |
                       (static_cast<std::uint32_t>(blk[4]) << 24);
        f_codes[u].y = static_cast<std::uint32_t>(blk[5]) |
                       (static_cast<std::uint32_t>(blk[6]) << 8) |
                       (static_cast<std::uint32_t>(blk[7]) << 16) |
                       (static_cast<std::uint32_t>(blk[8]) << 24);
        f_codes[u].z = static_cast<std::uint32_t>(blk[9]) |
                       (static_cast<std::uint32_t>(blk[10]) << 8) |
                       (static_cast<std::uint32_t>(blk[11]) << 16) |
                       (static_cast<std::uint32_t>(blk[12]) << 24);
        f_codes[u].w = static_cast<std::uint32_t>(blk[13]) |
                       (static_cast<std::uint32_t>(blk[14]) << 8) |
                       (static_cast<std::uint32_t>(blk[15]) << 16) |
                       (static_cast<std::uint32_t>(blk[16]) << 24);
      } else {
        constexpr int kBlockChunks = kQ5K ? 11 : 9;
        constexpr int kCodeChunk = kQ5K ? 3 : 1;
        const int block = kb0 / 8;
        const auto* blk =
            reinterpret_cast<const uint4*>(f_ptr[u]) + (block * kBlockChunks);
        // The K sweep enters a new superblock every eight Q8-sized blocks.
        if (kb0 % 8 == 0) {
          f_header[u] = blk[0];
          if constexpr (kPair) {
#pragma unroll
            for (int group = 0; group < 4; ++group)
              code_cache[u][group] = blk[kCodeChunk + group * 2 + f_c];
          }
          if constexpr (kQ5K) {
            f_qh[u][0] = blk[1];
            f_qh[u][1] = blk[2];
          }
        }
        const int sb32 = (kb0 % 8) + f_c;
        if constexpr (kPair) {
          const int group = sb32 / 2;
          f_codes[u] = group == 0   ? code_cache[u][0]
                       : group == 1 ? code_cache[u][1]
                       : group == 2 ? code_cache[u][2]
                                    : code_cache[u][3];
        } else {
          f_codes[u] = blk[kCodeChunk + (sb32 / 2) * 2 + f_c];
        }
        f_sb32[u] = sb32;
        if constexpr (kQ5K) {
          // Bit sb32 of the 32 high-bit bytes, packed as the Q5_1 word.
          const std::uint32_t qh[8] = {f_qh[u][0].x, f_qh[u][0].y, f_qh[u][0].z,
                                       f_qh[u][0].w, f_qh[u][1].x, f_qh[u][1].y,
                                       f_qh[u][1].z, f_qh[u][1].w};
          std::uint32_t high = 0;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            high |= GatherBit(qh[i], sb32) << (4 * i);
          }
          f_high[u] = high;
        }
      }
    }
#pragma unroll
    for (int i = 0; i < kActFetch; ++i) {
      a_data[i] = a_src[i] != nullptr
                      ? *reinterpret_cast<const uint4*>(a_src[i] + (kb0 * 32))
                      : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  const auto commit_stage = [&]() {
#pragma unroll
    for (int u = 0; u < kWaveRowTiles; ++u) {
      const int row = (tid >> 1) + (u * 128);
      std::uint32_t scale_bias = 0;
      if constexpr (kQ8) {
        s_codes[swizzle(row, 2 * f_c)] = f_codes[u];
        s_codes[swizzle(row, (2 * f_c) + 1)] = f_codes_hi[u];
        scale_bias = f_live[u] ? f_dm[u] : 0U;  // half2 (d, 0)
      } else {
        s_codes[swizzle(row, f_c)] = f_codes[u];
      }
      if constexpr (kMX) {
        // E8M0 exponent to an F16 scale of 2^(e - 127); the bias lane
        // stays zero (MXFP4 has no zero point). Extreme exponents (beyond
        // the F16 exponent range) saturate — the artifact's blocks sit far
        // inside it.
        const int e = f_live[u] ? f_sb32[u] : 127;
        const int f16_exp = e - 112;  // 2^(e-127) as an F16 bit exponent.
        const std::uint16_t sc =
            f16_exp > 0 ? static_cast<std::uint16_t>(f16_exp << 10) : 0;
        scale_bias = static_cast<std::uint32_t>(sc);  // half2 (scale, 0)
      }
      if constexpr (kQ5K) {
        s_high[(f_c * BM) + row] = f_high[u];
      }
      if constexpr (kQ8) {
      } else if constexpr (kQ5) {
        s_high[(f_c * BM) + row] = f_high[u];
        const __half2 dm = __builtin_bit_cast(__half2, f_dm[u]);
        const float d = f_live[u] ? __low2float(dm) : 0.0F;
        const float mn = f_live[u] ? __high2float(dm) : 0.0F;
        scale_bias =
            __builtin_bit_cast(std::uint32_t, __floats2half2_rn(d, mn));
      } else if constexpr (!kMX) {
        const int sb32 = f_sb32[u];
        std::uint32_t sc = 0;
        std::uint32_t mn = 0;
        if (sb32 < 4) {
          sc = HeaderByte(f_header[u], 4 + sb32) & 0x3FU;
          mn = HeaderByte(f_header[u], 8 + sb32) & 0x3FU;
        } else {
          sc = (HeaderByte(f_header[u], 8 + sb32) & 0x0FU) |
               ((HeaderByte(f_header[u], sb32) >> 6U) << 4U);
          mn = (HeaderByte(f_header[u], 8 + sb32) >> 4U) |
               ((HeaderByte(f_header[u], 4 + sb32) >> 6U) << 4U);
        }
        const __half2 dm = __builtin_bit_cast(__half2, f_header[u].x);
        const float scale =
            f_live[u] ? __low2float(dm) * static_cast<float>(sc) : 0.0F;
        const float offset =
            f_live[u] ? __high2float(dm) * static_cast<float>(mn) : 0.0F;
        scale_bias = __builtin_bit_cast(std::uint32_t,
                                        __floats2half2_rn(scale, -offset));
      }
      s_scale[(f_c * BM) + row] = scale_bias;
    }
#pragma unroll
    for (int i = 0; i < kActFetch; ++i) {
      if (a_slot[i] >= 0) {
        s_act[a_slot[i]] = a_data[i];
      }
    }
  };

  v8f acc[kWaveRowTiles][kTokTiles];
#pragma unroll
  for (int u = 0; u < kWaveRowTiles; ++u) {
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
      acc[u][j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    }
  }

  const __half2 magic =
      __floats2half2_rn(kQ8 ? -1152.0F : -1024.0F, kQ8 ? -1152.0F : -1024.0F);
  const auto compute_stage = [&]() {
    uint4 raw[kWaveRowTiles][BK];
    if constexpr (!kQ8) {
#pragma unroll
      for (int u = 0; u < kWaveRowTiles; ++u) {
        const int row = (wave_id * 16) + (u * 128) + sub_lane;
#pragma unroll
        for (int c = 0; c < BK; ++c) {
          raw[u][c] = s_codes[swizzle(row, c)];
        }
      }
    }
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      v16h a_lo[kWaveRowTiles];
      v16h a_hi[kWaveRowTiles];
#pragma unroll
      for (int u = 0; u < kWaveRowTiles; ++u) {
        const int row = (wave_id * 16) + (u * 128) + sub_lane;
        const __half2 sb =
            __builtin_bit_cast(__half2, s_scale[(kb * BM) + row]);
        const __half2 scale2 = __low2half2(sb);
        const __half2 bias2 = __high2half2(sb);
        std::uint32_t nib[8];
        if constexpr (kMX) {
          // MXFP4 packs in the Q4_0 order the vendored vec_dot decodes:
          // elements 0-15 are the low nibbles of the 16 code bytes,
          // elements 16-31 the high nibbles (v.x pairs with activation
          // ints 0-3, v.y with 4-7). nib[i] = elements 4i..4i+3.
          const std::uint32_t w0 = raw[u][kb].x;
          const std::uint32_t w1 = raw[u][kb].y;
          const std::uint32_t w2 = raw[u][kb].z;
          const std::uint32_t w3 = raw[u][kb].w;
          nib[0] = w0 & 0x0F0F0F0FU;
          nib[1] = w1 & 0x0F0F0F0FU;
          nib[2] = w2 & 0x0F0F0F0FU;
          nib[3] = w3 & 0x0F0F0F0FU;
          nib[4] = (w0 >> 4) & 0x0F0F0F0FU;
          nib[5] = (w1 >> 4) & 0x0F0F0F0FU;
          nib[6] = (w2 >> 4) & 0x0F0F0F0FU;
          nib[7] = (w3 >> 4) & 0x0F0F0F0FU;
        } else if constexpr (kQ8) {
          // Q8_0: the block's 32 signed bytes are chunks 2 kb and 2 kb + 1;
          // flipping the sign bit carries q + 128, which the 1152 magic
          // takes back out.
          const uint4 c0 = s_codes[swizzle(row, 2 * kb)];
          const uint4 c1 = s_codes[swizzle(row, (2 * kb) + 1)];
          const std::uint32_t words[8] = {c0.x, c0.y, c0.z, c0.w,
                                          c1.x, c1.y, c1.z, c1.w};
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            nib[i] = words[i] ^ 0x80808080U;
          }
        } else if constexpr (kQ5) {
          // Q5_1: K block kb's 16 bytes are chunk kb; elements 0-15 take
          // the low nibbles, 16-31 the high, plus bit j of the high-bit
          // word.
          const uint4 r = raw[u][kb];
          const std::uint32_t high = s_high[(kb * BM) + row];
          const std::uint32_t words[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
          for (int i = 0; i < 4; ++i) {
            nib[i] = (words[i] & 0x0F0F0F0FU) |
                     SpreadHighBits((high >> (4 * i)) & 0xFU);
            nib[4 + i] = ((words[i] >> 4U) & 0x0F0F0F0FU) |
                         SpreadHighBits((high >> (16 + 4 * i)) & 0xFU);
          }
        } else {
          // Q4_K / Q5_K / MXFP4: elements 0-15 of K block kb0 + kb are the
          // low (kb = 0) or high (kb = 1) nibbles of chunk 0, elements
          // 16-31 of chunk 1; Q5_K adds bit j of the staged high-bit word.
          // MXFP4 packs its E2M1 nibbles in the same Q4_0 order.
          const unsigned shift = 4U * static_cast<unsigned>(kb);
          const std::uint32_t words[8] = {raw[u][0].x, raw[u][0].y, raw[u][0].z,
                                          raw[u][0].w, raw[u][1].x, raw[u][1].y,
                                          raw[u][1].z, raw[u][1].w};
          const std::uint32_t high = kQ5K ? s_high[(kb * BM) + row] : 0U;
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            nib[i] = (words[i] >> shift) & 0x0F0F0F0FU;
            if constexpr (kQ5K) {
              nib[i] |= SpreadHighBits((high >> (4 * i)) & 0xFU);
            }
          }
        }
        __half2 h[16];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          if constexpr (kMX) {
            // Four E2M1 nibbles per word: gather F16 values from the
            // shared table and apply the block scale (no zero point).
            // nib[i] holds four byte-sized indices (each already 0-15
            // after the nibble mask); one whole byte selects the table
            // entry, not half of one.
            const auto* lut =
                reinterpret_cast<const std::uint16_t*>(s_mx_lut);
            const std::uint32_t w = nib[i];
            const std::uint16_t v0 = lut[(w >> 0) & 0xFFU];
            const std::uint16_t v1 = lut[(w >> 8) & 0xFFU];
            const std::uint16_t v2 = lut[(w >> 16) & 0xFFU];
            const std::uint16_t v3 = lut[(w >> 24) & 0xFFU];
            h[2 * i] = __hmul2(__builtin_bit_cast(
                                     __half2,
                                     static_cast<std::uint32_t>(v0) |
                                         (static_cast<std::uint32_t>(v1) << 16)),
                               scale2);
            h[2 * i + 1] = __hmul2(__builtin_bit_cast(
                                       __half2,
                                       static_cast<std::uint32_t>(v2) |
                                           (static_cast<std::uint32_t>(v3) << 16)),
                                 scale2);
          } else {
            CodesToHalves(nib[i], magic, scale2, bias2, h[2 * i],
                          h[2 * i + 1]);
          }
          if constexpr (kMX) {
            if (dbg && kb == 0 && i == 0 && blockIdx.x == 0 &&
                blockIdx.y == 0 && lane_id == 0 && wave_id == 0) {
              printf("MXDBG sb=%08x scale2=%04x%04x nib0=%08x lut=%04x "
                     "h0=%f\n",
                     __builtin_bit_cast(std::uint32_t, sb),
                     __builtin_bit_cast(std::uint16_t, __low2half(scale2)),
                     __builtin_bit_cast(std::uint16_t, __high2half(scale2)),
                     nib[0],
                     kMxFp4E2M1Half[nib[0] & 0xF],
                     __half2float(__low2half(h[0])));
            }
          }
        }
        __builtin_memcpy(&a_lo[u], &h[0], 32);
        __builtin_memcpy(&a_hi[u], &h[8], 32);
      }
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
        if constexpr (kPair || ((kQ5 || kQ8 || kMX) && BN >= 48)) {
          // Keep one token tile's LDS fragments live at a time. Hoisting
          // all eight tiles spills registers and defeats the wider tile's
          // reuse of each weight decode. This is a compiler barrier only.
          // MX needs it most: without it the compiler hoists all three
          // token tiles' B fragments and the accumulator lives at ~180
          // VGPRs, capping residency at four blocks/SM; the barrier lands
          // it under 160 and five blocks fit (1.49x on the production
          // shape).
          asm volatile("" ::: "memory");
        }
        // A short expert bucket has no output in the remaining token
        // tiles, so omit their WMMA work.
        if constexpr ((kPair || ((kQ5 || kQ8 || kMX) && BN >= 48)) && kTokTiles > 1) {
          if (j >= live_tok_tiles)
            continue;
        }
        const uint4* frag =
            s_act + ((kb * 4) * kActStride) + (j * 16) + sub_lane;
        uint4 b[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          b[q] = frag[q * kActStride];
        }
        v16h b_lo;
        v16h b_hi;
        __builtin_memcpy(&b_lo, &b[0], 32);
        __builtin_memcpy(&b_hi, &b[2], 32);
#pragma unroll
        for (int u = 0; u < kWaveRowTiles; ++u) {
          acc[u][j] = Wmma(a_lo[u], b_lo, acc[u][j]);
          acc[u][j] = Wmma(a_hi[u], b_hi, acc[u][j]);
        }
      }
    }
  };

  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch_stage(kb0 + BK);
    }
    compute_stage();
    __syncthreads();
  }

  if constexpr (kPair) {
    // Four waves compute gate rows and four compute the matching up rows.
    // Pair them in the existing LDS allocation, keeping the K accumulation
    // order and avoiding the gate's F32 write/read between projections.
    // Two padding floats keep the accumulator scatter off repeated banks.
    constexpr unsigned stride = 18;
    constexpr unsigned plane = 16 * stride;
    static_assert(8 * plane * sizeof(float) <= kLdsBytes);
    float* base = reinterpret_cast<float*>(lds);
    float* scratch = base + wave_id * plane;
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        scratch[sub_lane * stride + 2 * l + half_id] = acc[0][j][l];
      }
      __syncthreads();
#pragma unroll
      for (int unit = 0; unit < 2; ++unit) {
        const int flat = (unit * 256 + tid) * 2;
        const int t = t_local + j * 16 + flat / kRows;
        const int r = flat % kRows;
        if (t < bucket_rows && r_block + r < m_i) {
          const std::int32_t dst = rows_out[bucket_begin + t];
          if (dst >= 0) {
            const int idx = (r / 16) * plane + (flat / kRows) * stride + r % 16;
            // Preserve the separate projection epilogue's F32 evaluation
            // order before narrowing. Fast-math can otherwise regroup the
            // products and change an F16 rounding tie.
            __half values[2];
#pragma unroll
            for (int v = 0; v < 2; ++v) {
              float product = base[idx + v + 4 * plane] * base[idx + v];
              asm volatile("" : "+v"(product));
              float value = product * SigmoidF(base[idx + v]);
              asm volatile("" : "+v"(value));
              if constexpr (kMX) {
                // Same up-only F16 overflow park as the separate epilogue:
                // one 2^-4 here meets the single 2^4 of MxUpscaleF16.
                value *= 0.0625F;
              }
              values[v] = __float2half(value);
            }
            const auto offset = static_cast<std::size_t>(dst) * m + r_block + r;
            if (m % 2 == 0 && r_block + r + 1 < m_i) {
              *reinterpret_cast<__half2*>(out_half + offset) =
                  __halves2half2(values[0], values[1]);
            } else {
              out_half[offset] = values[0];
              if (r_block + r + 1 < m_i)
                out_half[offset + 1] = values[1];
            }
          }
        }
      }
      __syncthreads();
    }
    return;
  }

  // One wave writes a complete 128-byte line of F16 output. Padding the
  // shared row by two floats also makes the accumulator scatter conflict-free.
  // Narrow buckets keep the lighter wave-local epilogue below.
  if constexpr (BN >= 48) {
    static_assert(kLdsBytes >= 16 * (BM + 2) * sizeof(float));
    if (out_half != nullptr) {
      constexpr unsigned stride = BM + 2;
      float* scratch = reinterpret_cast<float*>(lds);
#pragma unroll
      for (int j = 0; j < kTokTiles; ++j) {
#pragma unroll
        for (int u = 0; u < kWaveRowTiles; ++u) {
#pragma unroll
          for (int l = 0; l < 8; ++l)
            scratch[sub_lane * stride + wave_id * 16 + u * 128 + 2 * l +
                    half_id] = acc[u][j][l];
        }
        __syncthreads();
#pragma unroll
        for (int round = 0; round < 16 * BM / (256 * 2); ++round) {
          const unsigned flat = (round * 256 + tid) * 2;
          const unsigned tr = flat / BM, row = flat % BM;
          const unsigned t = t_local + j * 16 + tr, r = r_block + row;
          if (t < unsigned(bucket_rows) && r < m) {
            const int dst = rows_out[bucket_begin + t];
            if (dst >= 0) {
              float2 v =
                  *reinterpret_cast<const float2*>(scratch + tr * stride + row);
              if constexpr (kType == WeightType::kMXFP4) {
                // Strong rows of the MX up pass overflow F16 before the
                // down GEMM reads them; park them two bins down and
                // restore after the down projection. Only the up pass
                // (swiglu_gate set) needs the park: the gate's F32 rows
                // feed a nonlinear silu, and the down pass consumes
                // already-parked inputs, so one park plus the single
                // 2^4 of MxUpscaleF16 restores the chain exactly.
                if (swiglu_gate != nullptr) {
                  v.x *= 0.0625F;
                  v.y *= 0.0625F;
                }
              }
              const size_t o = size_t(dst) * m + r;
              if (swiglu_gate != nullptr) {
                v.x *= SiluF(swiglu_gate[o]);
                if (r + 1 < m)
                  v.y *= SiluF(swiglu_gate[o + 1]);
              }
              if (m % 2 == 0 && r + 1 < m)
                *reinterpret_cast<__half2*>(out_half + o) =
                    __floats2half2_rn(v.x, v.y);
              else {
                out_half[o] = __float2half(v.x);
                if (r + 1 < m)
                  out_half[o + 1] = __float2half(v.y);
              }
            }
          }
        }
        __syncthreads();
      }
      return;
    }
  }
  // Transpose each 16x16 tile through LDS, then scatter the 16 rows of each
  // token to its output row.
  float* tile_scratch = reinterpret_cast<float*>(lds) + (wave_id * 256);
#pragma unroll
  for (int u = 0; u < kWaveRowTiles; ++u) {
    const int r0 = r_block + (wave_id * 16) + (u * 128);
#pragma unroll
    for (int j = 0; j < kTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        tile_scratch[(sub_lane * 16) + (2 * l) + half_id] = acc[u][j][l];
      }
      // The wave barrier converges lanes but carries no memory ordering:
      // fence the LDS scatter so every lane's read below sees this
      // iteration's values, not the next iteration's overwrite of the
      // reused scratch (measured free against the barrier-only variant).
      __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");
      __builtin_amdgcn_wave_barrier();
      const int t0 = t_local + (j * 16);
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        const int flat = (s * 32) + lane_id;
        const int t = t0 + (flat >> 4);
        const int r = r0 + (flat & 15);
        if (t < bucket_rows && r < m_i) {
          const std::int32_t dst = rows_out[bucket_begin + t];
          if (dst >= 0) {
            const std::size_t o = (static_cast<std::size_t>(dst) * m) +
                                  static_cast<std::size_t>(r);
            float v = tile_scratch[flat];
            if constexpr (kType == WeightType::kMXFP4) {
              // Same up-only park as the wide epilogue above: F32 gate
              // rows and the down pass (already-parked inputs) stay
              // unscaled so one MxUpscaleF16 restores everything.
              if (out_half != nullptr && swiglu_gate != nullptr) {
                v *= 0.0625F;
              }
            }
            if (dbg && out != nullptr && blockIdx.x == 0 && blockIdx.y == 0 &&
                s == 0 && flat == 0) {
              printf("MXEP dst=%d r=%u v=%.5f\n", dst, r, v);
            }
            if (out_half != nullptr) {
              out_half[o] = __float2half(
                  swiglu_gate != nullptr ? v * SiluF(swiglu_gate[o]) : v);
            } else {
              out[o] = v;
            }
          }
        }
      }
      __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");
      __builtin_amdgcn_wave_barrier();
    }
  }
}

/// Compacts the routed assignments by expert with 16-row padded buckets.
/// One block: exclusive scan of the padded counts into pad_bounds[0..E].
__global__ void RoutedPadBoundsKernel(const std::uint32_t* __restrict__ counts,
                                      std::int32_t* __restrict__ pad_bounds,
                                      std::int32_t* __restrict__ cursors,
                                      std::uint32_t n_experts) {
  __shared__ std::int32_t padded[1024];
  for (std::uint32_t e = threadIdx.x; e < n_experts; e += blockDim.x) {
    padded[e] = static_cast<std::int32_t>((counts[e] + 15u) / 16u * 16u);
    cursors[e] = 0;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    std::int32_t running = 0;
    for (std::uint32_t e = 0; e < n_experts; ++e) {
      pad_bounds[e] = running;
      running += padded[e];
    }
    pad_bounds[n_experts] = running;
  }
}

/// rows_token[c] / rows_slot[c] for every routed (token, slot); the order
/// inside a bucket is whatever the atomics produce, which changes nothing:
/// every output row is computed from its own inputs only.
__global__ void RoutedScatterKernel(const std::int32_t* __restrict__ ids,
                                    const std::int32_t* __restrict__ pad_bounds,
                                    std::int32_t* __restrict__ cursors,
                                    std::int32_t* __restrict__ rows_token,
                                    std::int32_t* __restrict__ rows_slot,
                                    std::uint32_t slots, std::uint32_t k) {
  const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= slots) {
    return;
  }
  const std::int32_t e = ids[slot];
  if (e < 0) {
    return;
  }
  const std::int32_t c = pad_bounds[e] + atomicAdd(&cursors[e], 1);
  rows_token[c] = static_cast<std::int32_t>(slot / k);
  rows_slot[c] = static_cast<std::int32_t>(slot);
}

}  // namespace

void EmbedTokens(const void* table, WeightType type, const std::int32_t* tokens,
                 float* res, std::uint32_t n_tokens, std::uint32_t hidden,
                 std::uint32_t streams, hipStream_t stream) {
  hipLaunchKernelGGL(EmbedKernel, dim3(n_tokens), dim3(kThreads), 0, stream,
                     table, type, tokens, res, hidden, streams);
  SyncTrace("EmbedTokens", stream);
}

void RmsNormRows(const float* x, const float* gamma, float* out,
                 std::uint32_t n_rows, std::uint32_t dim, std::uint32_t groups,
                 float eps, hipStream_t stream) {
  // Every (row, group) pair is one block; the kernel recovers the group
  // from the block index to pick its gamma slice.
  if (kSyncTrace) {
    float probe = 0.0f;
    const hipError_t pe =
        gamma != nullptr
            ? hipMemcpy(&probe, gamma, sizeof(probe), hipMemcpyDeviceToHost)
            : hipSuccess;
    fprintf(stderr,
            "SYNCTRACE call RmsNormRows x=%p gamma=%p out=%p rows=%u dim=%u "
            "groups=%u gamma_probe=%d val=%.4f\n",
            static_cast<const void*>(x), static_cast<const void*>(gamma),
            static_cast<void*>(out), n_rows, dim, groups, static_cast<int>(pe),
            probe);
    fflush(stderr);
  }
  hipLaunchKernelGGL(RmsNormKernel, dim3(n_rows * groups), dim3(kThreads), 0,
                     stream, x, gamma, out, dim / groups, groups, eps);
  SyncTrace("RmsNormRows", stream);
}

void SiluScale(float* x, float scale, std::size_t count, hipStream_t stream) {
  hipLaunchKernelGGL(SiluScaleKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, x, scale, count);
}

void Swiglu(float* gate, const float* up, std::size_t count,
            hipStream_t stream) {
  hipLaunchKernelGGL(SwigluKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, gate, up, count);
}

void SwigluHalf(const float* gate, const float* up, __half* out,
                std::size_t count, hipStream_t stream) {
  hipLaunchKernelGGL(SwigluHalfKernel, dim3(Blocks((count + 3) / 4)),
                     dim3(kThreads), 0, stream, gate, up, out, count);
}

bool SwigluQ8Tiled(const float* gate, const float* up, void* out_q8,
                   std::size_t n_rows, std::size_t k, hipStream_t stream) {
  if (k % 32 != 0) {
    return false;
  }
  const std::size_t chunks = n_rows * (k / 4);
  hipLaunchKernelGGL(SwigluQ8Kernel, dim3(Blocks(chunks)), dim3(kThreads), 0,
                     stream, gate, up, out_q8, n_rows, k);
  return true;
}

__global__ void ScaleKernel(float* x, float scale, std::size_t count) {
  const std::size_t i =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    x[i] *= scale;
  }
}

void ScaleInPlace(float* x, float scale, std::size_t count,
                  hipStream_t stream) {
  if (count == 0) {
    return;
  }
  const std::size_t blocks = (count + 255) / 256;
  hipLaunchKernelGGL(ScaleKernel, dim3(static_cast<unsigned>(blocks)),
                     dim3(256), 0, stream, x, scale, count);
  SyncTrace("ScaleInPlace", stream);
}

void SigmoidMul(float* x, const float* g, std::size_t count,
                hipStream_t stream) {
  hipLaunchKernelGGL(SigmoidMulKernel, dim3(Blocks(count)), dim3(kThreads), 0,
                     stream, x, g, count);
}

void NarrowActivations(const float* x, void* out, bool bf16, std::size_t count,
                       hipStream_t stream) {
  if (bf16) {
    hipLaunchKernelGGL(NarrowKernel<hip_bfloat16>, dim3(Blocks(count)),
                       dim3(kThreads), 0, stream, x,
                       static_cast<hip_bfloat16*>(out), count);
  } else {
    hipLaunchKernelGGL(NarrowKernel<__half>, dim3(Blocks(count)),
                       dim3(kThreads), 0, stream, x, static_cast<__half*>(out),
                       count);
  }
}

std::size_t Q8TiledBytes(std::size_t batch, std::size_t k) {
  // The GEMM stages whole 128-token macro tiles, so the buffer is sized to
  // the batch rounded up to one (the padding tiles are never read as
  // results, only staged).
  constexpr std::size_t kMacroTokens = 128;
  const std::size_t padded =
      (batch + kMacroTokens - 1) / kMacroTokens * kMacroTokens;
  return (padded / kQ8ActTileTokens) * (k / 32) * kQ8ActTileBytes;
}

void QuantizeQ8Tiled(const float* x, void* out, std::size_t batch,
                     std::size_t k, hipStream_t stream) {
  const std::size_t blocks = batch * (k / 32);
  if (batch * k >= 4096) {
    const std::size_t per_block = kThreads / 8;
    hipLaunchKernelGGL(QuantizeQ8TiledVec4Kernel,
                       dim3((blocks + per_block - 1) / per_block),
                       dim3(kThreads), 0, stream, x, out, batch, k);
    return;
  }
  const std::size_t waves = kThreads / 32;
  hipLaunchKernelGGL(QuantizeQ8TiledKernel, dim3((blocks + waves - 1) / waves),
                     dim3(kThreads), 0, stream, x, out, batch, k);
}

bool W8A8Gemm(const void* w, const void* x_tiled, float* out, std::size_t batch,
              std::size_t m, std::size_t k, hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0 || k % 32 != 0) {
    return false;
  }
  // Qwen's wave64 matrix kernel with four row groups improves the model's
  // large output projections while preserving every K32 accumulator update.
  if (batch >= 1024 && m == 2560 && k == 6144) {
    W8A8GemmWave64(w, x_tiled, out, batch, m, k, stream);
    return true;
  }
  // A 128-token macro tile is the throughput configuration; short chunks
  // would leave most of it idle and take the 64-token variant. A narrow
  // projection (the 320-row mixer down over K = 10240) gets 64-row tiles so
  // it still fills the device.
  constexpr int kBM = 128;
  if (m <= 512 && batch >= 96) {
    constexpr int kBN = 128;
    constexpr int kNarrowBM = 64;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kNarrowBM - 1) / kNarrowBM));
    hipLaunchKernelGGL((W8A8BlockedWmmaGEMMKernel<kNarrowBM, kBN, 4, 2, 4>),
                       grid, dim3(kThreads), 0, stream, w, x_tiled, out, batch,
                       m, k);
  } else if (batch >= 96) {
    constexpr int kBN = 128;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kBM - 1) / kBM));
    hipLaunchKernelGGL((W8A8BlockedWmmaGEMMKernel<kBM, kBN, 2, 4, 2>), grid,
                       dim3(kThreads), 0, stream, w, x_tiled, out, batch, m, k);
  } else {
    constexpr int kBN = 64;
    if (m <= 2048) {
      // A tall-narrow decode projection launches only eight-to-sixteen
      // 128-row blocks — well under this device's wave capacity. Halving
      // the row block doubles the blocks and measures 1.55x at m 1024
      // (batch 1-16) and 1.07x at 2048, bit-identical.
      const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                      static_cast<unsigned int>((m + 63) / 64));
      hipLaunchKernelGGL((W8A8BlockedWmmaGEMMKernel<64, kBN, 4, 2, 4>),
                         grid, dim3(kThreads), 0, stream, w, x_tiled, out,
                         batch, m, k);
    } else {
      const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                      static_cast<unsigned int>((m + kBM - 1) / kBM));
      hipLaunchKernelGGL((W8A8BlockedWmmaGEMMKernel<kBM, kBN, 4, 4, 2>), grid,
                         dim3(kThreads), 0, stream, w, x_tiled, out, batch, m,
                         k);
    }
  }
  return true;
  SyncTrace("W8A8Gemm", stream);
}

std::size_t RoutedCompactRows(std::size_t slots, std::size_t n_experts) {
  return slots + (n_experts * (kRoutedTileTokens - 1));
}

/// Router logits through the matrix core: out[t][e] = x[t] . w[e]. One wave
/// owns a 16-token x 16-expert tile; the A fragment is the token row, the B
/// fragment the expert row, both straight F16 global reads (the weight stays
/// resident in L2 across the token blocks of one layer). The C fragment
/// lands as (token = 2e + half_id, expert = sub_lane) — the probed wave32
/// map. Replaces the F32 row-per-wave SmallGemm, whose activation re-reads
/// (every weight row sweeps the whole batch) made the router L2-bound;
/// measured 13.4x on the production shape with F16-input dot accuracy the
/// top-8 gap distribution tolerates (zero selection flips over a full
/// prompt).
__global__ __launch_bounds__(256) void RouterGemmWmmaKernel(
    const __half* __restrict__ w_half, const __half* __restrict__ x_half,
    float* __restrict__ out, std::uint32_t n_tokens, std::uint32_t m,
    std::uint32_t k) {
  const std::uint32_t w = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31;
  const std::uint32_t sub_lane = lane & 15;
  const std::uint32_t half_id = lane >> 4;
  const std::uint32_t t0 = blockIdx.x * 16;
  const std::uint32_t e0 = blockIdx.y * 128 + w * 16;
  if (e0 >= m) {
    return;
  }
  const std::uint32_t t_live =
      t0 + sub_lane < n_tokens ? t0 + sub_lane : n_tokens - 1;
  const __half2* xr = reinterpret_cast<const __half2*>(
      x_half + static_cast<std::size_t>(t_live) * k);
  const __half2* wr = reinterpret_cast<const __half2*>(
      w_half + static_cast<std::size_t>(e0 + sub_lane) * k);
  v8f acc{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll 4
  for (std::uint32_t d16 = 0; d16 < k / 16; ++d16) {
    v16h a, b;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      reinterpret_cast<__half2*>(&a)[i] = xr[d16 * 8 + i];
      reinterpret_cast<__half2*>(&b)[i] = wr[d16 * 8 + i];
    }
    acc = Wmma(a, b, acc);
  }
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const std::uint32_t t = t0 + (2 * e) + half_id;
    const std::uint32_t ex = e0 + sub_lane;
    if (t < n_tokens && ex < m) {
      out[static_cast<std::size_t>(t) * m + ex] = acc[e];
    }
  }
}

void RouterGemmWmma(const __half* w_half, const __half* x_half, float* out,
                    std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
                    hipStream_t stream) {
  hipLaunchKernelGGL(RouterGemmWmmaKernel,
                     dim3((n_tokens + 15) / 16, (m + 127) / 128), dim3(256),
                     0, stream, w_half, x_half, out, n_tokens, m, k);
  SyncTrace("RouterGemmWmma", stream);
}

/// Split-K form of `RouterGemmWmmaKernel`: the same 16-token x 16-expert tile,
/// but each block owns one k slice and writes its partial product to
/// `partials[blockIdx.z]`. A decode batch launches only two blocks (16 warps)
/// in the tiled form, far too few to cover the 2 MB weight sweep's DRAM
/// latency — the production trace prices it at 74.8us/layer against a ~9us
/// bandwidth floor. Slicing k raises the in-flight byte count without reading
/// a weight byte twice; the caller adds the slices back in a fixed order, so
/// both ranks still see bit-identical logits.
__global__ __launch_bounds__(256) void RouterGemmWmmaSplitKernel(
    const __half* __restrict__ w_half, const __half* __restrict__ x_half,
    float* __restrict__ partials, std::uint32_t n_tokens, std::uint32_t m,
    std::uint32_t k, std::uint32_t slice) {
  const std::uint32_t w = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31;
  const std::uint32_t sub_lane = lane & 15;
  const std::uint32_t half_id = lane >> 4;
  const std::uint32_t t0 = blockIdx.x * 16;
  const std::uint32_t e0 = blockIdx.y * 128 + w * 16;
  if (e0 >= m) {
    return;
  }
  const std::uint32_t k0 = blockIdx.z * slice;
  const std::uint32_t t_live =
      t0 + sub_lane < n_tokens ? t0 + sub_lane : n_tokens - 1;
  const __half2* xr = reinterpret_cast<const __half2*>(
      x_half + static_cast<std::size_t>(t_live) * k + k0);
  const __half2* wr = reinterpret_cast<const __half2*>(
      w_half + static_cast<std::size_t>(e0 + sub_lane) * k + k0);
  v8f acc{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll 4
  for (std::uint32_t d16 = 0; d16 < slice / 16; ++d16) {
    v16h a, b;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      reinterpret_cast<__half2*>(&a)[i] = xr[d16 * 8 + i];
      reinterpret_cast<__half2*>(&b)[i] = wr[d16 * 8 + i];
    }
    acc = Wmma(a, b, acc);
  }
  float* dst = partials + static_cast<std::size_t>(blockIdx.z) *
                                static_cast<std::size_t>(n_tokens) *
                                static_cast<std::size_t>(m);
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const std::uint32_t t = t0 + (2 * e) + half_id;
    const std::uint32_t ex = e0 + sub_lane;
    if (t < n_tokens && ex < m) {
      dst[static_cast<std::size_t>(t) * m + ex] = acc[e];
    }
  }
}

/// Adds the split-K partials back: out[i] = sum_z partials[z * count + i].
/// Ascending z is what makes the sum order — and therefore the router's
/// expert selection — identical on both ranks.
__global__ void RouterGemmReduceKernel(const float* __restrict__ partials,
                                       float* __restrict__ out,
                                       std::uint32_t count,
                                       std::uint32_t splits) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  float sum = 0.0F;
  for (std::uint32_t z = 0; z < splits; ++z) {
    sum += partials[static_cast<std::size_t>(z) * count + i];
  }
  out[i] = sum;
}

void RouterGemmWmmaSplit(const __half* w_half, const __half* x_half, float* out,
                         float* partials, std::uint32_t n_tokens,
                         std::uint32_t m, std::uint32_t k,
                         std::uint32_t splits, hipStream_t stream) {
  const std::uint32_t slice = k / splits;
  hipLaunchKernelGGL(RouterGemmWmmaSplitKernel,
                     dim3((n_tokens + 15) / 16, (m + 127) / 128, splits),
                     dim3(256), 0, stream, w_half, x_half, partials, n_tokens,
                     m, k, slice);
  const std::uint32_t count = n_tokens * m;
  hipLaunchKernelGGL(RouterGemmReduceKernel, dim3((count + 255) / 256),
                     dim3(256), 0, stream, partials, out, count, splits);
  SyncTrace("RouterGemmWmmaSplit", stream);
}

/// Narrow-batch router: out[t][e] = x[t] . w[e] with one warp per expert row.
/// The WMMA kernel above tiles tokens x experts, which for a decode row means
/// 16 waves sweeping 256 d16 chunks apiece — latency bound at ~170us/layer.
/// Here 32 blocks x 8 warps read the 2MB of weights once at full width: the
/// expert row stays in registers across the token loop and every token row is
/// accumulated against it (a verify batch's token block is 8 x 8KB, so those
/// reads come from L2). The row walk is k/4 uint2 units, 8 bytes per lane per
/// step, covering the whole row — a k/8 bound leaves half of it unread.
/// F16 inputs and the per-lane accumulation order match the WMMA path, so the
/// two agree within the F16-input gate the top-8 gap tolerates.
constexpr std::uint32_t kRouterMaxTokens = 8;

__global__ __launch_bounds__(256) void RouterGemmNarrowKernel(
    const __half* __restrict__ w_half, const __half* __restrict__ x_half,
    float* __restrict__ out, std::uint32_t n_tokens, std::uint32_t m,
    std::uint32_t k) {
  const std::uint32_t warp = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31;
  const std::uint32_t expert = blockIdx.x * (blockDim.x >> 5) + warp;
  if (expert >= m) {
    return;
  }
  const uint2* wr = reinterpret_cast<const uint2*>(
      w_half + static_cast<std::size_t>(expert) * k);
  const std::uint32_t units = k / 4;
  float acc[kRouterMaxTokens];
#pragma unroll
  for (std::uint32_t t = 0; t < kRouterMaxTokens; ++t) {
    acc[t] = 0.0F;
  }
  for (std::uint32_t i = lane; i < units; i += 32) {
    const uint2 wv = wr[i];
    const __half2* wp = reinterpret_cast<const __half2*>(&wv);
    const float2 w0 = __half22float2(wp[0]);
    const float2 w1 = __half22float2(wp[1]);
    for (std::uint32_t t = 0; t < n_tokens; ++t) {
      const uint2* xr = reinterpret_cast<const uint2*>(
          x_half + static_cast<std::size_t>(t) * k);
      const uint2 xv = xr[i];
      const __half2* xp = reinterpret_cast<const __half2*>(&xv);
      const float2 x0 = __half22float2(xp[0]);
      const float2 x1 = __half22float2(xp[1]);
      acc[t] = fmaf(w0.x, x0.x, acc[t]);
      acc[t] = fmaf(w0.y, x0.y, acc[t]);
      acc[t] = fmaf(w1.x, x1.x, acc[t]);
      acc[t] = fmaf(w1.y, x1.y, acc[t]);
    }
  }
#pragma unroll
  for (std::uint32_t t = 0; t < kRouterMaxTokens; ++t) {
    if (t >= n_tokens) {
      break;
    }
    float v = acc[t];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      v += __shfl_xor(v, off);
    }
    if (lane == 0) {
      out[static_cast<std::size_t>(t) * m + expert] = v;
    }
  }
}

void RouterGemmNarrow(const __half* w_half, const __half* x_half, float* out,
                      std::uint32_t n_tokens, std::uint32_t m,
                      std::uint32_t k, hipStream_t stream) {
  hipLaunchKernelGGL(RouterGemmNarrowKernel, dim3((m + 7) / 8), dim3(256), 0,
                     stream, w_half, x_half, out, n_tokens, m, k);
  SyncTrace("RouterGemmNarrow", stream);
}

void RoutedCompact(const std::int32_t* ids, const std::uint32_t* counts,
                   std::int32_t* pad_bounds, std::int32_t* cursors,
                   std::int32_t* rows_token, std::int32_t* rows_slot,
                   std::uint32_t n_tokens, std::uint32_t k,
                   std::uint32_t n_experts, hipStream_t stream) {
  const std::size_t slots = static_cast<std::size_t>(n_tokens) * k;
  const std::size_t rows = RoutedCompactRows(slots, n_experts);
  (void)hipMemsetAsync(rows_token, 0xFF, rows * sizeof(std::int32_t), stream);
  (void)hipMemsetAsync(rows_slot, 0xFF, rows * sizeof(std::int32_t), stream);
  hipLaunchKernelGGL(RoutedPadBoundsKernel, dim3(1), dim3(1024), 0, stream,
                     counts, pad_bounds, cursors, n_experts);
  hipLaunchKernelGGL(RoutedScatterKernel, dim3(Blocks(slots)), dim3(kThreads),
                     0, stream, ids, pad_bounds, cursors, rows_token, rows_slot,
                     static_cast<std::uint32_t>(slots), k);
}

template<int BN>
bool LaunchRoutedF16(const void* w, WeightType type, const __half* x,
                     const std::int32_t* tiles, std::uint32_t n_tiles,
                     const std::int32_t* pad_bounds,
                     const std::int32_t* rows_in, const std::int32_t* rows_out,
                     const float* swiglu_gate, float* out, __half* out_half,
                     std::size_t m, std::size_t k, hipStream_t stream) {
  constexpr int kBM = 128;
  constexpr int kBK = 2;
  const dim3 grid(static_cast<unsigned int>((m + kBM - 1) / kBM), n_tiles);
  switch (type) {
    case WeightType::kQ4_K:
      if constexpr (BN > 48) {
        return false;
      } else {
        hipLaunchKernelGGL(
            (RoutedF16GEMMKernel<WeightType::kQ4_K, kBM, BN, kBK>), grid,
            dim3(kThreads), 0, stream, w, x, tiles, pad_bounds, rows_in,
            rows_out, swiglu_gate, out, out_half, m, k, nullptr, 0);
        return true;
      }
    case WeightType::kQ5_1:
      hipLaunchKernelGGL((RoutedF16GEMMKernel<WeightType::kQ5_1, kBM, BN, kBK>),
                         grid, dim3(kThreads), 0, stream, w, x, tiles,
                         pad_bounds, rows_in, rows_out, swiglu_gate, out,
                         out_half, m, k, nullptr, 0);
      return true;
    case WeightType::kQ8_0:
      hipLaunchKernelGGL((RoutedF16GEMMKernel<WeightType::kQ8_0, kBM, BN, kBK>),
                         grid, dim3(kThreads), 0, stream, w, x, tiles,
                         pad_bounds, rows_in, rows_out, swiglu_gate, out,
                         out_half, m, k, nullptr, 0);
      return true;
    case WeightType::kQ5_K:
      if constexpr (BN > 48) {
        return false;
      } else {
        hipLaunchKernelGGL(
            (RoutedF16GEMMKernel<WeightType::kQ5_K, kBM, BN, kBK>), grid,
            dim3(kThreads), 0, stream, w, x, tiles, pad_bounds, rows_in,
            rows_out, swiglu_gate, out, out_half, m, k, nullptr, 0);
        return true;
      }
    case WeightType::kMXFP4: {
      static const bool dbg = getenv("GUFO_MX_PRINT") != nullptr;
      hipLaunchKernelGGL(
          (RoutedF16GEMMKernel<WeightType::kMXFP4, kBM, BN, kBK>), grid,
          dim3(kThreads), 0, stream, w, x, tiles, pad_bounds, rows_in,
          rows_out, swiglu_gate, out, out_half, m, k, nullptr,
          dbg ? 1 : 0);
      return true;
    }
    default:
      return false;
  }
}

bool RoutedF16Gemm(const void* w, WeightType type, const __half* x,
                   const std::int32_t* tiles, std::uint32_t n_tiles,
                   std::uint32_t tile_rows, const std::int32_t* pad_bounds,
                   const std::int32_t* rows_in, const std::int32_t* rows_out,
                   const float* swiglu_gate, float* out, __half* out_half,
                   std::size_t m, std::size_t k, hipStream_t stream) {
  const std::size_t block_elems =
      type == WeightType::kMXFP4 ? 32
      : (type == WeightType::kQ4_K || type == WeightType::kQ5_K) ? 256
                                                                  : 64;
  if (m == 0 || k == 0 || k % block_elems != 0 || n_tiles == 0 ||
      (out_half == nullptr) == (out == nullptr)) {
    fprintf(stderr, "RF16 reject: type=%d m=%zu k=%zu be=%zu tiles=%u out=%p "
                    "outh=%p\n", static_cast<int>(type), m, k, block_elems,
            n_tiles, static_cast<void*>(out), static_cast<void*>(out_half));
    fflush(stderr);
    return false;
  }
  if (tile_rows != 16 && tile_rows != 48 && tile_rows != 64) {
    fprintf(stderr, "RF16 bad tile_rows=%u\n", tile_rows);
    fflush(stderr);
    return false;
  }
  switch (tile_rows) {
    case 16:
      return LaunchRoutedF16<16>(w, type, x, tiles, n_tiles, pad_bounds,
                                 rows_in, rows_out, swiglu_gate, out, out_half,
                                 m, k, stream);
    case 48:
      return LaunchRoutedF16<48>(w, type, x, tiles, n_tiles, pad_bounds,
                                 rows_in, rows_out, swiglu_gate, out, out_half,
                                 m, k, stream);
    case 64:
      return LaunchRoutedF16<64>(w, type, x, tiles, n_tiles, pad_bounds,
                                 rows_in, rows_out, swiglu_gate, out, out_half,
                                 m, k, stream);
    default:
      return false;
  }
}

template<int BN>
bool LaunchRoutedGatedF16(const void* gate, const void* up, WeightType type,
                          const __half* x, const std::int32_t* tiles,
                          std::uint32_t n_tiles, const std::int32_t* pad_bounds,
                          const std::int32_t* rows_in,
                          const std::int32_t* rows_out, __half* out,
                          std::size_t m, std::size_t k, hipStream_t stream) {
  if (m == 0 || k == 0 || k % 256 != 0 || n_tiles == 0 || out == nullptr) {
    return false;
  }
  const dim3 grid(static_cast<unsigned int>((m + 63) / 64), n_tiles);
  switch (type) {
    case WeightType::kQ4_K:
      hipLaunchKernelGGL(
          (RoutedF16GEMMKernel<WeightType::kQ4_K, 128, BN, 2, true>), grid,
          dim3(kThreads), 0, stream, gate, x, tiles, pad_bounds, rows_in,
          rows_out, nullptr, nullptr, out, m, k, up, 0);
      return true;
    case WeightType::kQ5_K:
      hipLaunchKernelGGL(
          (RoutedF16GEMMKernel<WeightType::kQ5_K, 128, BN, 2, true>), grid,
          dim3(kThreads), 0, stream, gate, x, tiles, pad_bounds, rows_in,
          rows_out, nullptr, nullptr, out, m, k, up, 0);
      return true;
    case WeightType::kMXFP4:
      // The 17-byte MX block needs no superblock code cache, so the pair
      // form (four waves per tensor, fused SwiGLU epilogue) drops in
      // directly; it reads the staged activations once and keeps the gate
      // partials out of global memory.
      hipLaunchKernelGGL(
          (RoutedF16GEMMKernel<WeightType::kMXFP4, 128, BN, 2, true>), grid,
          dim3(kThreads), 0, stream, gate, x, tiles, pad_bounds, rows_in,
          rows_out, nullptr, nullptr, out, m, k, up, 0);
      return true;
    default:
      return false;
  }
}

bool RoutedGatedF16Gemm(const void* gate, const void* up, WeightType type,
                        const __half* x, const std::int32_t* tiles,
                        std::uint32_t n_tiles, std::uint32_t tile_rows,
                        const std::int32_t* pad_bounds,
                        const std::int32_t* rows_in,
                        const std::int32_t* rows_out, __half* out,
                        std::size_t m, std::size_t k, hipStream_t stream) {
  if (tile_rows == 128) {
    return LaunchRoutedGatedF16<128>(gate, up, type, x, tiles, n_tiles,
                                     pad_bounds, rows_in, rows_out, out, m, k,
                                     stream);
  }
  return tile_rows == 64 &&
         LaunchRoutedGatedF16<64>(gate, up, type, x, tiles, n_tiles, pad_bounds,
                                  rows_in, rows_out, out, m, k, stream);
}

// Keep the separate four-tap convolution's F32 rounding order when its
// inputs come from the projection's LDS tile.
__device__ __forceinline__ float SsmConv4Value(float4 w, float x0, float x1,
                                               float x2, float x3) {
  float acc = __fmaf_rn(w.x, x0, __fmul_rn(w.y, x1));
  acc = __fmaf_rn(w.z, x2, acc);
  acc = __fmaf_rn(w.w, x3, acc);
  return SiluF(acc);
}

constexpr unsigned kSsmProjectionTileTokens = 32;

// Optional output layout for the stacked QKV projection. The projection's
// 256-row tile covers one query, gate, key or value head.
struct AttentionProjectionOutput {
  const float* q_gamma;
  const float* k_gamma;
  float* query;
  float* gate;
  __half* keys;
  __half* values;
  const std::uint32_t* position;
  float theta;
  float eps;
  const qwen::vision::DeviceRope* rope;
  /// Query and KV heads of the projection: all of them, or a TP rank's.
  unsigned heads;
  unsigned kv_heads;
};

/// Dense F16 WMMA GEMM over Q8_0 or F16 weights: block = BM rows x BN tokens,
/// BK 32-element K blocks per LDS stage, waves = WM row groups x WN token
/// groups. The codes are dequantized to F16 once per stage as they are
/// committed to LDS (magic-number F16 construction, exact for a Q8_0 code),
/// and the activations are F16 rows [batch][k], so the matrix cores
/// accumulate in F32 with no per-block scaling. F16 weights skip decoding
/// and use the same ordered K16 products. y is [batch][m].
template<int BM, int BN, int BK, int WM, int WN, int kRowGroup = 1,
         bool kHcMix = false, bool kSsmConv = false, bool kAttention = false,
         bool kHalfWeights = false>
__launch_bounds__(256) __global__ void DenseF16GEMMKernel(
    const void* __restrict__ w, const __half* __restrict__ x,
    float* __restrict__ y, std::size_t batch, std::size_t m, std::size_t k,
    const __half* xn = nullptr, __half* mixed_half = nullptr,
    void* mixed_q8 = nullptr, const float* conv_w = nullptr,
    float* conv_out = nullptr, AttentionProjectionOutput attention = {},
    std::uint32_t conv_channels = 0) {
  static_assert(WM * WN == 8, "256 threads is 8 waves");
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
  constexpr int kRowTiles = BM / 16;
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = kRowTiles / WM;
  constexpr int kWaveTokTiles = kTokTiles / WN;
  // Weight and activation K blocks staged per thread; a unit past the
  // stage's block count is idle.
  constexpr int kAUnits = BM * BK;
  constexpr int kBUnits = BN * BK;
  constexpr int kAPer = (kAUnits + 255) / 256;
  constexpr int kBPer = (kBUnits + 255) / 256;

  // One K block of one row or token is four 16-byte chunks; the chunks of
  // nearby rows are permuted so a fragment read (one row per lane, 64-byte
  // stride) covers all bank groups.
  constexpr int kLdsChunks = BK * (BM + BN) * 4;
  static_assert(kLdsChunks * 16 >= 8 * 512 * 4, "epilogue transposes 16 KB");
  constexpr int kTransposeChunks = 8 * 16 * 36 * sizeof(float) / sizeof(uint4);
  constexpr int kLdsStorage =
      kLdsChunks < kTransposeChunks ? kTransposeChunks : kLdsChunks;
  __shared__ __attribute__((aligned(16))) uint4 s_lds[kLdsStorage];
  auto* s_a = reinterpret_cast<uint4(*)[BM][4]>(s_lds);
  auto* s_b = reinterpret_cast<uint4(*)[BN][4]>(s_lds + (BK * BM * 4));
  const auto swizzle = [](int row, int c) { return c ^ ((row >> 1) & 3); };

  const int num_kb = static_cast<int>(k / 32);
  const int m_i = static_cast<int>(m);
  const auto* w_bytes = static_cast<const std::uint8_t*>(w);

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int wave_row = wave_id / WN;
  const int wave_tok = wave_id % WN;
  // Group row tiles for the narrow HC up projection. The tile grid's row
  // count is divisible by kRowGroup; every dot product keeps its K order.
  const unsigned row_group = blockIdx.y / kRowGroup;
  const unsigned within = (blockIdx.y % kRowGroup) * gridDim.x + blockIdx.x;
  const int r_block = (row_group * kRowGroup + within % kRowGroup) * BM;
  const int t_block = (within / kRowGroup) * BN;

  // Weight fetch unit p of a thread: row (p * 256 + tid) / BK, K block
  // (p * 256 + tid) % BK of the stage; rows past m read the last row with
  // a zero scale.
  const std::uint8_t* a_ptr[kAPer];
  bool a_live[kAPer];
#pragma unroll
  for (int p = 0; p < kAPer; ++p) {
    const int idx = (p * 256) + tid;
    const int r = r_block + (idx / BK);
    a_live[p] = idx < kAUnits && r < m_i;
    // Interleave the four gate streams within the tile, without repacking
    // weights. Each group of four rows produces one mixed hidden element.
    const int weight_row = kHcMix ? (r % 4) * (m_i / 4) + r / 4 : r;
    a_ptr[p] = w_bytes +
               static_cast<std::size_t>(a_live[p] ? weight_row : (m_i - 1)) *
                   static_cast<std::size_t>(num_kb) * (kHalfWeights ? 64 : 34);
  }
  const __half* b_ptr[kBPer];
#pragma unroll
  for (int p = 0; p < kBPer; ++p) {
    const int idx = (p * 256) + tid;
    const int t = t_block + (idx / BK);
    b_ptr[p] = idx < kBUnits && t < static_cast<int>(batch)
                   ? x + (static_cast<std::size_t>(t) * k)
                   : nullptr;
  }
  uint4 a_codes[kAPer][2];
  uint4 a_half[kHalfWeights ? kAPer : 1][4];
  std::uint32_t a_d[kAPer];
  uint4 b_data[kBPer][4];

  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int p = 0; p < kAPer; ++p) {
      const int kb = kb0 + (((p * 256) + tid) % BK);
      const bool live = a_live[p] && kb < num_kb;
      if constexpr (kHalfWeights) {
        const auto* src = reinterpret_cast<const uint4*>(
            a_ptr[p] + static_cast<std::size_t>(kb) * 64);
#pragma unroll
        for (int c = 0; c < 4; ++c)
          a_half[p][c] = live ? src[c] : make_uint4(0u, 0u, 0u, 0u);
      } else {
        const std::uint8_t* blk =
            a_ptr[p] +
            (static_cast<std::size_t>(live ? kb : (num_kb - 1)) * 34);
        a_d[p] = live ? *reinterpret_cast<const std::uint16_t*>(blk) : 0U;
        __builtin_memcpy(&a_codes[p][0], blk + 2, 16);
        __builtin_memcpy(&a_codes[p][1], blk + 18, 16);
      }
    }
#pragma unroll
    for (int p = 0; p < kBPer; ++p) {
      const int kb = kb0 + (((p * 256) + tid) % BK);
      if (b_ptr[p] != nullptr && kb < num_kb) {
        const auto* src = reinterpret_cast<const uint4*>(b_ptr[p] + (kb * 32));
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          b_data[p][c] = src[c];
        }
      } else {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          b_data[p][c] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
    }
  };

  const __half2 magic = __floats2half2_rn(-1152.0F, -1152.0F);
  const __half2 zero2 = __floats2half2_rn(0.0F, 0.0F);
  const auto commit_stage = [&]() {
#pragma unroll
    for (int p = 0; p < kAPer; ++p) {
      const int idx = (p * 256) + tid;
      if (idx >= kAUnits) {
        break;
      }
      const int row = idx / BK;
      const int kk = idx % BK;
      if constexpr (kHalfWeights) {
#pragma unroll
        for (int c = 0; c < 4; ++c)
          s_a[kk][row][swizzle(row, c)] = a_half[p][c];
      } else {
        // Q8_0 codes are signed; flipping the sign bit carries q + 128, which
        // the 1152 magic takes back out.
        const std::uint32_t words[8] = {
            a_codes[p][0].x, a_codes[p][0].y, a_codes[p][0].z, a_codes[p][0].w,
            a_codes[p][1].x, a_codes[p][1].y, a_codes[p][1].z, a_codes[p][1].w};
        const __half2 scale2 = __half2half2(
            __builtin_bit_cast(__half, static_cast<std::uint16_t>(a_d[p])));
        __half2 h[16];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          CodesToHalves(words[i] ^ 0x80808080U, magic, scale2, zero2, h[2 * i],
                        h[2 * i + 1]);
        }
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          uint4 v;
          __builtin_memcpy(&v, &h[4 * c], 16);
          s_a[kk][row][swizzle(row, c)] = v;
        }
      }
    }
#pragma unroll
    for (int p = 0; p < kBPer; ++p) {
      const int idx = (p * 256) + tid;
      if (idx >= kBUnits) {
        break;
      }
      const int t = idx / BK;
      const int kk = idx % BK;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        s_b[kk][t][swizzle(t, c)] = b_data[p][c];
      }
    }
  };

  v8f acc[kWaveRowTiles][kWaveTokTiles];
  // Separate K16 chains reduce FP32 accumulation error for the sensitive
  // unquantized router/gate projections, with the same order in every chunk.
  v8f acc_high[kHalfWeights ? kWaveRowTiles : 1]
              [kHalfWeights ? kWaveTokTiles : 1]{};
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
      acc[i][j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    }
  }

  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) {
      fetch_stage(kb0 + BK);
    }
    // Keeps the next stage's loads ahead of the matrix work: scheduled
    // freely, the compiler sinks them below it and the commit waits on
    // the full memory latency.
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      v16h a_lo[kWaveRowTiles];
      v16h a_hi[kWaveRowTiles];
#pragma unroll
      for (int i = 0; i < kWaveRowTiles; ++i) {
        const int row = (((wave_row * kWaveRowTiles) + i) * 16) + sub_lane;
        uint4 c[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          c[q] = s_a[kb][row][swizzle(row, q)];
        }
        __builtin_memcpy(&a_lo[i], &c[0], 32);
        __builtin_memcpy(&a_hi[i], &c[2], 32);
      }
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j) {
        const int t = (((wave_tok * kWaveTokTiles) + j) * 16) + sub_lane;
        uint4 c[4];
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          c[q] = s_b[kb][t][swizzle(t, q)];
        }
        v16h b_lo;
        v16h b_hi;
        __builtin_memcpy(&b_lo, &c[0], 32);
        __builtin_memcpy(&b_hi, &c[2], 32);
#pragma unroll
        for (int i = 0; i < kWaveRowTiles; ++i) {
          acc[i][j] = Wmma(a_lo[i], b_lo, acc[i][j]);
          if constexpr (kHalfWeights)
            acc_high[i][j] = Wmma(a_hi[i], b_hi, acc_high[i][j]);
          else
            acc[i][j] = Wmma(a_hi[i], b_hi, acc[i][j]);
        }
      }
    }
    __syncthreads();
  }

  if constexpr (kHalfWeights) {
#pragma unroll
    for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j)
        acc[i][j] += acc_high[i][j];
    }
  }

  if constexpr (kAttention) {
    static_assert(BM == 256 && BN == 128 && BK == 2 && WM == 8 && WN == 1);
    static_assert(!kHcMix && !kSsmConv && kRowGroup == 1);
    constexpr unsigned stride = 36, dim = 256;
    const unsigned width = attention.heads * dim;
    const unsigned kvwidth = attention.kv_heads * dim;
    float* scratch = reinterpret_cast<float*>(s_lds);
    float* tile = scratch + wave_id * 16 * stride;
    // Row tiles: each query head's query then gate, the key heads, the value
    // heads.
    const unsigned projection_head = r_block / 256;
    const unsigned qg_tiles = 2 * attention.heads;
    const bool query = projection_head < qg_tiles && (projection_head % 2) == 0;
    const bool key = projection_head >= qg_tiles &&
                     projection_head < qg_tiles + attention.kv_heads;
    const bool gate = projection_head < qg_tiles && (projection_head % 2) == 1;
    const unsigned head = projection_head < qg_tiles ? projection_head / 2
                          : key
                              ? projection_head - qg_tiles
                              : projection_head - qg_tiles - attention.kv_heads;
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        tile[sub_lane * stride + 2 * l + half_id] = acc[0][j][l];
        tile[sub_lane * stride + 16 + 2 * l + half_id] = acc[1][j][l];
      }
      __syncthreads();
#pragma unroll
      for (unsigned phase = 0; phase < 2; ++phase) {
        // One wave handles a token, emulating its original eight-wave norm.
        const unsigned tok_local = wave_id * 2 + phase;
        const std::size_t tok = t_block + j * 16 + tok_local;
        if (tok < batch) {
          float v[8];
#pragma unroll
          for (unsigned c = 0; c < 8; ++c)
            v[c] = scratch[(c * 16 + tok_local) * stride + lane_id];
          if (query || key) {
            float total = 0.0F;
#pragma unroll
            for (unsigned c = 0; c < 8; ++c) {
              // Match the separate norm: round each square and each wave sum
              // before accumulating the eight partials in their original order.
              float sq = v[c] * v[c];
              asm volatile("" : "+v"(sq));
              float ss = WaveSum(sq);
              asm volatile("" : "+v"(ss));
              total += ss;
              asm volatile("" : "+v"(total));
            }
            const float scale = rsqrtf(total / 256.0F + attention.eps);
            const float* gamma = query ? attention.q_gamma : attention.k_gamma;
#pragma unroll
            for (unsigned c = 0; c < 8; ++c) {
              v[c] = v[c] * scale;
              asm volatile("" : "+v"(v[c]));
              v[c] = v[c] * gamma[c * 32 + lane_id];
              asm volatile("" : "+v"(v[c]));
            }
            const float freq = powf(
                attention.theta, -2.0F * static_cast<float>(lane_id) / 64.0F);
            float sn = 0.0F, cs = 0.0F;
            sincosf(qwen::vision::RopePosition(
                        attention.rope, *attention.position + tok, lane_id) *
                        freq,
                    &sn, &cs);
            const float lo = __fmaf_rn(v[0], cs, -__fmul_rn(v[1], sn));
            const float hi = __fmaf_rn(v[0], sn, __fmul_rn(v[1], cs));
            v[0] = lo;
            v[1] = hi;
          }
#pragma unroll
          for (unsigned c = 0; c < 8; ++c) {
            const unsigned col = c * 32 + lane_id;
            if (query)
              attention.query[tok * width + head * dim + col] = v[c];
            else if (gate)
              attention.gate[tok * width + head * dim + col] = v[c];
            else if (key)
              attention.keys[std::size_t(*attention.position + tok) * kvwidth +
                             head * dim + col] = __float2half_rn(v[c]);
            else
              attention
                  .values[std::size_t(*attention.position + tok) * kvwidth +
                          head * dim + col] = __float2half_rn(v[c]);
          }
        }
      }
      __syncthreads();
    }
    return;
  }

  if constexpr (kHcMix) {
    static_assert(BM == 256 && BN == 128 && BK == 1 && WM == 4 && WN == 2);
    constexpr unsigned kHiddenTile = BM / 4;
    constexpr unsigned kStreamStride = kHiddenTile + 1;
    constexpr unsigned kPlane = 16 * kStreamStride;
    static_assert(4 * kPlane * sizeof(float) <= sizeof(s_lds));
    float* gates = reinterpret_cast<float*>(s_lds);
    const std::size_t hidden = m / 4;
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
#pragma unroll
      for (int token_group = 0; token_group < WN; ++token_group) {
        if (wave_tok == token_group) {
#pragma unroll
          for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
            for (int l = 0; l < 8; ++l) {
              const unsigned row =
                  (wave_row * kWaveRowTiles + i) * 16 + 2 * l + half_id;
              gates[(row % 4) * kPlane + sub_lane * kStreamStride + row / 4] =
                  acc[i][j][l];
            }
          }
        }
        __syncthreads();
        // A half-wave writes all 64 hidden values of one token. The stream
        // planes are padded to avoid the gate transpose's LDS bank conflicts.
        const unsigned token_in_tile = tid / (kHiddenTile / 4);
        const unsigned h_local = (tid % (kHiddenTile / 4)) * 4;
        const std::size_t token =
            t_block + (token_group * kWaveTokTiles + j) * 16 + token_in_tile;
        const std::size_t h = r_block / 4 + h_local;
        if (token < batch) {
          float4 value{0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
          for (unsigned stream = 0; stream < 4; ++stream) {
            const float* g = gates + stream * kPlane +
                             token_in_tile * kStreamStride + h_local;
            const float4 v = Load4(xn + token * m + stream * hidden + h);
            // Preserve the separate mixer's F32 FMA rounding.
            value.x = __fmaf_rn(v.x, SigmoidF(g[0]), value.x);
            value.y = __fmaf_rn(v.y, SigmoidF(g[1]), value.y);
            value.z = __fmaf_rn(v.z, SigmoidF(g[2]), value.z);
            value.w = __fmaf_rn(v.w, SigmoidF(g[3]), value.w);
          }
          value.x *= 0.25F;
          value.y *= 0.25F;
          value.z *= 0.25F;
          value.w *= 0.25F;
          *reinterpret_cast<float4*>(y + token * hidden + h) = value;
          if (mixed_half != nullptr) {
            *reinterpret_cast<__half2*>(mixed_half + token * hidden + h) =
                __floats2half2_rn(value.x, value.y);
            *reinterpret_cast<__half2*>(mixed_half + token * hidden + h + 2) =
                __floats2half2_rn(value.z, value.w);
          }
          if (mixed_q8 != nullptr) {
            float max_abs = fmaxf(fmaxf(fabsf(value.x), fabsf(value.y)),
                                  fmaxf(fabsf(value.z), fabsf(value.w)));
#pragma unroll
            for (int off = 4; off > 0; off >>= 1) {
              max_abs = fmaxf(max_abs, __shfl_xor(max_abs, off));
            }
            const float d = max_abs / 127.0F;
            const float id = d != 0.0F ? 1.0F / d : 0.0F;
            const auto q0 = static_cast<unsigned>(static_cast<unsigned char>(
                static_cast<signed char>(roundf(value.x * id))));
            const auto q1 = static_cast<unsigned>(static_cast<unsigned char>(
                static_cast<signed char>(roundf(value.y * id))));
            const auto q2 = static_cast<unsigned>(static_cast<unsigned char>(
                static_cast<signed char>(roundf(value.z * id))));
            const auto q3 = static_cast<unsigned>(static_cast<unsigned char>(
                static_cast<signed char>(roundf(value.w * id))));
            const unsigned pos = h % 32;
            auto* tile = Q8ActTile(mixed_q8, hidden / 32,
                                   token / kQ8ActTileTokens, h / 32);
            const std::size_t tl = token % kQ8ActTileTokens;
            *reinterpret_cast<unsigned*>(tile + (pos >> 4) * 256 + tl * 16 +
                                         (pos & 15)) =
                q0 | q1 << 8 | q2 << 16 | q3 << 24;
            if (pos == 0) {
              *reinterpret_cast<float*>(tile + kQ8ActScaleOffset +
                                        tl * sizeof(float)) = d;
            }
          }
        }
        __syncthreads();
      }
    }
    return;
  }

  // Transpose the result through LDS, two row tiles at a time, so every
  // global store covers 32 consecutive rows of one token: a full 128-byte
  // line (half lines cost a read-modify-write on the fabric).
  // Four padding floats reduce scatter bank conflicts and retain float4
  // alignment for both output stores and the fused convolution's reads.
  constexpr unsigned kOutputStride = 36;
  // Pair token tiles for SSM: half as many convolutions cross a tile edge.
  // The larger transpose still fits the projection's existing LDS allocation.
  constexpr unsigned kOutputTokens = kSsmConv ? kSsmProjectionTileTokens : 16;
  constexpr unsigned kOutputGroups = kOutputTokens / 16;
  static_assert(8 * kOutputTokens * kOutputStride * sizeof(float) <=
                sizeof(s_lds));
  static_assert(kWaveRowTiles % 2 == 0, "the epilogue pairs row tiles");
  float* tile_scratch =
      reinterpret_cast<float*>(s_lds) + wave_id * kOutputTokens * kOutputStride;
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; i += 2) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; j += kOutputGroups) {
#pragma unroll
      for (unsigned group = 0; group < kOutputGroups; ++group) {
        // scratch[token][row], with 32 output rows per token.
#pragma unroll
        for (int l = 0; l < 8; ++l) {
          tile_scratch[((sub_lane + group * 16) * kOutputStride) + (2 * l) +
                       half_id] = acc[i][j + group][l];
          tile_scratch[((sub_lane + group * 16) * kOutputStride) + 16 +
                       (2 * l) + half_id] = acc[i + 1][j + group][l];
        }
      }
      __builtin_amdgcn_wave_barrier();
      const std::size_t r0 =
          static_cast<std::size_t>(r_block) +
          static_cast<std::size_t>((((wave_row * kWaveRowTiles) + i) * 16));
      const std::size_t t0 =
          static_cast<std::size_t>(t_block) +
          static_cast<std::size_t>((((wave_tok * kWaveTokTiles) + j) * 16));
#pragma unroll
      for (unsigned group = 0; group < kOutputGroups; ++group) {
        // Lane pair (2p, 2p + 1) stores token p's 32 rows as eight float4.
        const int tok_l = (lane_id >> 1) + group * 16;
        const int row_l = (lane_id & 1) * 16;
        const std::size_t tok = t0 + static_cast<std::size_t>(tok_l);
        const auto* src = reinterpret_cast<const float4*>(
            tile_scratch + (tok_l * kOutputStride) + row_l);
        if constexpr (kSsmConv) {
          static_assert(BM == 256 && BN == 128 && BK == 2 && WM == 8 &&
                        WN == 1);
          static_assert(!kHcMix && kRowGroup == 1);
          // The convolved q|k|v rows precede the gate rows.
          const std::uint32_t channels = conv_channels;
          if (tok < batch) {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
              const unsigned row = r0 + row_l + v * 4;
              if (row >= m)
                continue;
              const float4 current = src[v];
              if (row >= channels || tok_l < 3 || tok_l >= kOutputTokens - 3 ||
                  tok + 3 >= batch) {
                *reinterpret_cast<float4*>(y + tok * m + row) = current;
              }
              if (row < channels && tok_l >= 3) {
                const float4 x0 = *reinterpret_cast<const float4*>(
                    tile_scratch + (tok_l - 3) * kOutputStride + row_l + v * 4);
                const float4 x1 = *reinterpret_cast<const float4*>(
                    tile_scratch + (tok_l - 2) * kOutputStride + row_l + v * 4);
                const float4 x2 = *reinterpret_cast<const float4*>(
                    tile_scratch + (tok_l - 1) * kOutputStride + row_l + v * 4);
                const float4 w0 =
                    *reinterpret_cast<const float4*>(conv_w + (row + 0) * 4);
                const float4 w1 =
                    *reinterpret_cast<const float4*>(conv_w + (row + 1) * 4);
                const float4 w2 =
                    *reinterpret_cast<const float4*>(conv_w + (row + 2) * 4);
                const float4 w3 =
                    *reinterpret_cast<const float4*>(conv_w + (row + 3) * 4);
                const float4 value{
                    SsmConv4Value(w0, x0.x, x1.x, x2.x, current.x),
                    SsmConv4Value(w1, x0.y, x1.y, x2.y, current.y),
                    SsmConv4Value(w2, x0.z, x1.z, x2.z, current.z),
                    SsmConv4Value(w3, x0.w, x1.w, x2.w, current.w)};
                *reinterpret_cast<float4*>(conv_out + tok * channels + row) =
                    value;
              }
            }
          }
        } else {
          if (tok < batch && r0 + 32 <= m && ((tok * m) % 4 == 0)) {
            auto* dst = reinterpret_cast<float4*>(
                y + (tok * m) + r0 + static_cast<std::size_t>(row_l));
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              dst[q] = src[q];
            }
          } else if (tok < batch) {
#pragma unroll
            for (int q = 0; q < 16; ++q) {
              const std::size_t r = r0 + static_cast<std::size_t>(row_l + q);
              if (r < m) {
                y[(tok * m) + r] =
                    tile_scratch[(tok_l * kOutputStride) + row_l + q];
              }
            }
          }
        }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

bool UnquantizedF16Gemm(const void* w, const __half* x, float* out,
                        std::size_t batch, std::size_t m, std::size_t k,
                        hipStream_t stream) {
  if (m == 0 || batch == 0 || k == 0 || k % 32 != 0)
    return false;
  hipLaunchKernelGGL(
      (DenseF16GEMMKernel<64, 64, 2, 2, 4, 1, false, false, false, true>),
      dim3((batch + 63) / 64, (m + 63) / 64), dim3(kThreads), 0, stream, w, x,
      out, batch, m, k);
  return true;
}

bool DenseF16Gemm(const void* w, const __half* x, float* out, std::size_t batch,
                  std::size_t m, std::size_t k, hipStream_t stream) {
  if (m == 0 || k == 0 || batch == 0 || k % 32 != 0) {
    return false;
  }
  // The same tile plan as the W8A8 route: a 128-token macro tile for wide
  // batches, 64-row tiles for narrow projections, 64 tokens below 96.
  constexpr int kBM = 128;
  if (m <= 512 && batch >= 96) {
    constexpr int kBN = 128;
    constexpr int kNarrowBM = 64;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kNarrowBM - 1) / kNarrowBM));
    hipLaunchKernelGGL((DenseF16GEMMKernel<kNarrowBM, kBN, 2, 2, 4>), grid,
                       dim3(kThreads), 0, stream, w, x, out, batch, m, k);
  } else if (batch >= 96) {
    // 64 x 64 wave tiles: half the LDS fragment bytes per matrix product
    // of the 32 x 64 tile (the F16 fragments are twice the int8 ones).
    constexpr int kWideBM = 256;
    constexpr int kBN = 128;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kWideBM - 1) / kWideBM));
    if (m == 10240 && k == 320 && batch >= 1024) {
      hipLaunchKernelGGL((DenseF16GEMMKernel<kWideBM, kBN, 1, 4, 2, 8>), grid,
                         dim3(kThreads), 0, stream, w, x, out, batch, m, k);
    } else if (batch >= 1024 && (((m == 16384 || m == 13312) && k == 2560) ||
                                 (m == 2560 && k == 6144))) {
      // Eight row groups reuse each weight fragment across all token tiles
      // and keep fewer weight fragments live. K accumulation is unchanged.
      hipLaunchKernelGGL((DenseF16GEMMKernel<kWideBM, kBN, 2, 8, 1>), grid,
                         dim3(kThreads), 0, stream, w, x, out, batch, m, k);
    } else {
      hipLaunchKernelGGL((DenseF16GEMMKernel<kWideBM, kBN, 1, 4, 2>), grid,
                         dim3(kThreads), 0, stream, w, x, out, batch, m, k);
    }
  } else {
    constexpr int kBN = 64;
    const dim3 grid(static_cast<unsigned int>((batch + kBN - 1) / kBN),
                    static_cast<unsigned int>((m + kBM - 1) / kBM));
    hipLaunchKernelGGL((DenseF16GEMMKernel<kBM, kBN, 4, 4, 2>), grid,
                       dim3(kThreads), 0, stream, w, x, out, batch, m, k);
  }
  return true;
  SyncTrace("DenseF16Gemm", stream);
}

template<WeightType type, unsigned tokens>
void LaunchSmallGemm(const void* w, const float* x, float* out, std::uint32_t m,
                     std::uint32_t k, hipStream_t stream) {
  hipLaunchKernelGGL((SmallGemmKernel<type, tokens>),
                     dim3((m + kSmallGemmRows - 1) / kSmallGemmRows),
                     dim3(kSmallGemmRows * 32), 0, stream, w, x, out, m, k);
}

template<WeightType type>
void SmallGemmForType(const void* w, const float* x, float* out,
                      std::uint32_t tokens, std::uint32_t m, std::uint32_t k,
                      hipStream_t stream) {
  if (tokens > 8) {
    const auto groups = tokens / 8;
    hipLaunchKernelGGL((SmallGemmKernel<type, 8, true>),
                       dim3((m + kSmallGemmRows - 1) / kSmallGemmRows, groups),
                       dim3(kSmallGemmRows * 32), 0, stream, w, x, out, m, k);
    const auto consumed = groups * 8;
    x += std::size_t{consumed} * k;
    out += std::size_t{consumed} * m;
    tokens -= consumed;
    if (tokens == 0)
      return;
  }
  switch (tokens) {
    case 1:
      return LaunchSmallGemm<type, 1>(w, x, out, m, k, stream);
    case 2:
      return LaunchSmallGemm<type, 2>(w, x, out, m, k, stream);
    case 3:
      return LaunchSmallGemm<type, 3>(w, x, out, m, k, stream);
    case 4:
      return LaunchSmallGemm<type, 4>(w, x, out, m, k, stream);
    case 5:
      return LaunchSmallGemm<type, 5>(w, x, out, m, k, stream);
    case 6:
      return LaunchSmallGemm<type, 6>(w, x, out, m, k, stream);
    case 7:
      return LaunchSmallGemm<type, 7>(w, x, out, m, k, stream);
    case 8:
      return LaunchSmallGemm<type, 8>(w, x, out, m, k, stream);
    default:
      throw std::logic_error("small projection requires 1-8 token rows");
  }
}

void SmallGemm(const void* w, WeightType type, const float* x, float* out,
               std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
               hipStream_t stream) {
  switch (type) {
    case WeightType::kF32:
      return SmallGemmForType<WeightType::kF32>(w, x, out, n_tokens, m, k,
                                                stream);
    case WeightType::kBF16:
      return SmallGemmForType<WeightType::kBF16>(w, x, out, n_tokens, m, k,
                                                 stream);
    case WeightType::kF16:
      return SmallGemmForType<WeightType::kF16>(w, x, out, n_tokens, m, k,
                                                stream);
    case WeightType::kQ8_0:
      // The row decoder handles Q8_0 codes; MiMo's quantized narrow
      // projections ride the same grouped path.
      return SmallGemmForType<WeightType::kQ8_0>(w, x, out, n_tokens, m, k,
                                                 stream);
    default:
      throw std::logic_error("unsupported small projection format");
  }
  SyncTrace("SmallGemm", stream);
}

void Rope(float* x, std::uint32_t n_tokens, std::uint32_t heads,
          std::uint32_t d, std::uint32_t rotary_dim,
          const std::uint32_t* start_pos, float theta, hipStream_t stream,
          const std::uint32_t* row_pos) {
  hipLaunchKernelGGL(RopeKernel, dim3(n_tokens), dim3(kThreads), 0, stream, x,
                     heads, d, rotary_dim, start_pos, theta, nullptr, row_pos);
}

/// Visible-key count from which a decode row reads the int8 K/V twin: below
/// it the fp16 rows cost the same bytes with less arithmetic, and a cohort
/// mixes deep and shallow rows, so every launch carries the threshold
/// (GUFO_MIMO_KV_MIN_KEYS overrides, 0 forces the twin everywhere).
std::uint32_t KvInt8MinKeysHost() {
  static const std::uint32_t value = [] {
    const char* env = getenv("GUFO_MIMO_KV_MIN_KEYS");
    const long v = env != nullptr ? std::strtol(env, nullptr, 10) : -1;
    return v < 0 ? 4096U : static_cast<std::uint32_t>(v);
  }();
  return value;
}

void StoreKv(const float* src, __half* cache, std::uint32_t n_tokens,
             std::uint32_t row_dim, std::uint32_t ring_rows,
             const std::uint32_t* start_pos, hipStream_t stream,
             std::int8_t* cache_q, __half* scales) {
  hipLaunchKernelGGL(StoreKvKernel, dim3(n_tokens), dim3(kThreads), 0, stream,
                     src, cache, cache_q, scales, row_dim, ring_rows,
                     start_pos);
}

void StoreKvSlot(const float* src, const SlotKv* kv, std::uint32_t slot,
                 bool v_side, std::uint32_t n_tokens, std::uint32_t row_dim,
                 const std::uint32_t* start_pos, hipStream_t stream,
                 const std::uint32_t* row_slot,
                 const std::uint32_t* row_pos) {
  hipLaunchKernelGGL(StoreKvSlotKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, src, kv, slot, v_side ? 1u : 0u, row_dim,
                     start_pos, row_slot, row_pos);
}

__global__ void ExpertCountsKernel(const std::int32_t* ids,
                                   std::uint32_t* counts, std::size_t slots) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < slots && ids[i] >= 0) {
    atomicAdd(counts + ids[i], 1u);
  }
}

void ExpertCounts(const std::int32_t* ids, std::uint32_t* counts,
                  std::uint32_t n_tokens, std::uint32_t n_experts,
                  std::uint32_t k, hipStream_t stream) {
  (void)hipMemsetAsync(counts, 0, n_experts * sizeof(std::uint32_t), stream);
  const std::size_t slots = static_cast<std::size_t>(n_tokens) * k;
  hipLaunchKernelGGL(ExpertCountsKernel, dim3(Blocks(slots)), dim3(kThreads), 0,
                     stream, ids, counts, slots);
  SyncTrace("ExpertCounts", stream);
}

void MtpHidden(const float* base, const float* alt, const std::int32_t* row,
               float* dst, std::uint32_t n_tokens, std::uint32_t width,
               hipStream_t stream) {
  hipLaunchKernelGGL(MtpHiddenKernel, dim3(n_tokens), dim3(kThreads), 0, stream,
                     base, alt, row, dst, width);
}

void MtpHiddenSlot(const SlotPtrs* ptrs, const SlotControl* ctrl,
                   std::uint32_t slot, float* dst, std::uint32_t n_tokens,
                   std::uint32_t width, hipStream_t stream) {
  hipLaunchKernelGGL(MtpHiddenSlotKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, ptrs, ctrl, slot, dst, width);
}

void CopyRowsSlot(const float* src, float* const* dstp, std::uint32_t rows,
                  std::uint32_t width, hipStream_t stream) {
  const std::uint32_t count = rows * width;
  hipLaunchKernelGGL(CopyRowsSlotKernel, dim3(Blocks(count)), dim3(kThreads),
                     0, stream, src, dstp, count);
}

void AddRowsBroadcast(const float* addend, float* residual,
                      std::uint32_t n_tokens, std::uint32_t hidden,
                      std::uint32_t streams, hipStream_t stream) {
  hipLaunchKernelGGL(AddRowsBroadcastKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, addend, residual, hidden, streams);
}

void StagePartialHalf(const float* partial, __half* send, std::size_t count,
                      std::uint32_t* arrivals, std::uint64_t* ready,
                      std::uint64_t value, hipStream_t stream) {
  const std::size_t blocks = (count + kThreads - 1) / kThreads;
  hipLaunchKernelGGL(StagePartialHalfKernel, dim3(blocks), dim3(kThreads), 0,
                     stream, partial, send, count, arrivals, ready, value);
}

void AddRowsBroadcastHalf(const __half* addend, float* residual,
                          std::uint32_t n_tokens, std::uint32_t hidden,
                          std::uint32_t streams, hipStream_t stream) {
  hipLaunchKernelGGL(AddRowsBroadcastHalfKernel, dim3(n_tokens),
                     dim3(kThreads), 0, stream, addend, residual, hidden,
                     streams);
}

void StagePartial(const float* partial, float* send, std::size_t count,
                  std::uint32_t* arrivals, std::uint64_t* ready,
                  std::uint64_t value, hipStream_t stream) {
  const std::size_t threads = std::max<std::size_t>(1, count / 4);
  const auto blocks = static_cast<unsigned>(
      std::min<std::size_t>(1024, (threads + kThreads - 1) / kThreads));
  hipLaunchKernelGGL(StagePartialKernel, dim3(blocks), dim3(kThreads), 0,
                     stream, partial, send, count, arrivals, ready, value);
}

void StagePartialSlotted(const float* partial, float* send, std::size_t count,
                         std::uint32_t* arrivals, std::uint64_t* ready,
                         const std::uint64_t* slot, hipStream_t stream) {
  const std::size_t blocks =
      (count / 4 + kThreads - 1) / kThreads + (count % 4 ? 1 : 0);
  hipLaunchKernelGGL(StagePartialSlottedKernel,
                     dim3(static_cast<unsigned>(std::max<std::size_t>(blocks, 1))),
                     dim3(kThreads), 0, stream, partial, send, count, arrivals,
                     ready, slot);
}

void WaitValueSlotted(const std::uint64_t* flag, const std::uint64_t* slot,
                      std::uint64_t ticks, std::uint32_t* timed_out,
                      hipStream_t stream) {
  hipLaunchKernelGGL(WaitValueSlottedKernel, dim3(1), dim3(1), 0, stream, flag,
                     slot, ticks, timed_out);
}

void AddRowsBroadcastSlotted(const float* peer, float* residual,
                             std::uint32_t n_tokens, std::uint32_t hidden,
                             std::uint32_t streams, const std::uint64_t* slot,
                             hipStream_t stream) {
  hipLaunchKernelGGL(AddRowsBroadcastSlottedKernel, dim3(n_tokens),
                     dim3(kThreads), 0, stream, peer, residual, hidden,
                     streams, slot);
}

void WaitValue(const std::uint64_t* flag, std::uint64_t value,
               std::uint64_t ticks, std::uint32_t* timed_out,
               hipStream_t stream) {
  hipLaunchKernelGGL(WaitValueKernel, dim3(1), dim3(1), 0, stream, flag, value,
                     ticks, timed_out);
}

void Argmax(const float* logits, ArgmaxCandidate* scratch, std::int32_t* out,
            std::uint32_t n_tokens, std::uint32_t vocab, hipStream_t stream) {
  hipLaunchKernelGGL(ArgmaxPartialKernel, dim3(kArgmaxParts, n_tokens),
                     dim3(kThreads), 0, stream, logits, scratch, vocab);
  hipLaunchKernelGGL(ArgmaxFinishKernel, dim3(n_tokens), dim3(kThreads), 0,
                     stream, logits, scratch, out, vocab);
}

void GatherArgmaxCandidates(const float* logits, const std::uint32_t* ids,
                            ArgmaxCandidate* out, std::uint32_t rows,
                            std::uint32_t vocab, hipStream_t stream) {
  hipLaunchKernelGGL(GatherArgmaxCandidatesKernel, dim3(Blocks(rows)),
                     dim3(kThreads), 0, stream, logits, ids, out, rows, vocab);
}

template<unsigned Keep>
void SelectMtpCandidates(const float* logits, std::uint32_t* ids,
                         std::uint32_t* scratch_ids, float* scores,
                         std::uint32_t vocab, hipStream_t stream) {
  if (logits == nullptr || ids == nullptr || scratch_ids == nullptr ||
      vocab == 0) {
    throw std::invalid_argument("invalid MTP candidate selection");
  }
  const auto tiles = [](std::uint32_t n) {
    return 1U + (n - 1U) / kMtpCandidateTile;
  };
  unsigned passes = 1;
  for (auto size = vocab; size > kMtpCandidateTile; size = tiles(size) * Keep) {
    ++passes;
  }
  // Choose the first buffer so the final pass always lands in `ids`.
  auto* destination = passes % 2 != 0 ? ids : scratch_ids;
  const std::uint32_t* source = nullptr;
  auto size = vocab;
  for (;;) {
    const auto blocks = tiles(size);
    hipLaunchKernelGGL((MtpCandidateTileKernel<Keep>), dim3(blocks),
                       dim3(kThreads), 0, stream, logits, source, destination,
                       blocks == 1 ? scores : nullptr, size, vocab);
    if (blocks == 1) {
      break;
    }
    size = blocks * Keep;
    source = destination;
    destination = destination == ids ? scratch_ids : ids;
  }
}

std::uint32_t MtpCandidateWorkspaceSize(std::uint32_t vocab) {
  const auto tiles = (vocab + 1023) / 1024;
  return (tiles > 2 ? tiles : 2) * kMtpCandidates;
}

void MtpTopCandidates(const float* logits, std::uint32_t* ids,
                      std::uint32_t* scratch_ids, float* scores,
                      std::uint32_t vocab, hipStream_t stream) {
  if (scores == nullptr)
    throw std::invalid_argument("invalid MTP candidate scores");
  SelectMtpCandidates<kMtpCandidates>(logits, ids, scratch_ids, scores, vocab,
                                      stream);
}

// ---------------------------------------------------------------------------
// Attention: asymmetric heads (query/key 192 wide, value 128), per-layer
// sliding window, optional per-head attention sinks. One block per (head,
// token, split); a wave covers one key per tile step, six key lanes and four
// value lanes per thread.
// ---------------------------------------------------------------------------

/// Value-context lanes per thread: the value head width over the wave.
constexpr std::uint32_t kMimoVLanes = 4;
/// Key lanes per thread: the key head width over the wave (192 = 32 * 6).
constexpr std::uint32_t kMimoKLanes = 6;
/// Waves per CTA: the tile step partitions the 256 keys over 8 waves, so the
/// per-wave accumulators must be reduced before one wave writes the result.
constexpr std::uint32_t kMimoBlockWaves = kThreads / 32;

/// Simulated int8 store precision for the K/V caches (GUFO_ATTN_KV_SIM_BITS):
/// rounds a half to its top `bits` mantissa bits, which is the precision a
/// per-channel int8 quantizer leaves (7 bits ~= 0.78% relative error). It is
/// a read-side probe for judging the quality cost of a quantized cache
/// before building one; 0 (or >= 10) leaves the value untouched.
__device__ inline float SimKvHalf(unsigned short raw, std::uint32_t bits) {
  if (bits == 0 || bits >= 10) {
    return __half2float(__ushort_as_half(raw));
  }
  const std::uint32_t drop = 10u - bits;
  const std::uint32_t mask = (1u << drop) - 1u;
  const unsigned short rounded = static_cast<unsigned short>(
      (static_cast<std::uint32_t>(raw) + (1u << (drop - 1u))) & ~mask);
  return __half2float(__ushort_as_half(rounded));
}

__global__ void MimoAttentionKernel(
    const float* q, const __half* k_cache_in, const __half* v_cache_in,
    const float* sinks, float* out, float* partials, std::uint32_t splits,
    const std::uint32_t* start_pos, std::uint32_t heads,
    std::uint32_t kv_heads, std::uint32_t d_qk, std::uint32_t d_v,
    std::uint32_t window, std::uint32_t ring_rows_in, const SlotKv* slot_kv,
    std::uint32_t slot, const std::uint32_t* row_slot,
    const std::uint32_t* row_pos, std::uint32_t packed_ok,
    std::uint32_t batch_keys, std::uint32_t kv_sim_bits,
    const std::int8_t* k_cache_q_in, const std::int8_t* v_cache_q_in,
    const __half* k_scales_in, const __half* v_scales_in,
    std::uint32_t kv_min_keys) {
  // Slot indirection: with a table the caches and ring come from kv[slot] at
  // run time (replay-safe); a null cache is a cancelled row, which the eager
  // path skipped before a memset of the context. A fused row batch carries
  // one slot per row (its rows belong to different sessions). The int8 twins
  // ride along: present, the K/V reads halve; null keeps the fp16 path.
  const std::uint32_t my_slot =
      row_slot != nullptr ? row_slot[blockIdx.y] : slot;
  const __half* k_cache =
      slot_kv != nullptr ? slot_kv[my_slot].k_cache : k_cache_in;
  const __half* v_cache =
      slot_kv != nullptr ? slot_kv[my_slot].v_cache : v_cache_in;
  const std::int8_t* k_cache_q = slot_kv != nullptr
                                     ? slot_kv[my_slot].k_cache_q
                                     : k_cache_q_in;
  const std::int8_t* v_cache_q = slot_kv != nullptr
                                     ? slot_kv[my_slot].v_cache_q
                                     : v_cache_q_in;
  const __half* k_scales =
      slot_kv != nullptr ? slot_kv[my_slot].k_scales : k_scales_in;
  const __half* v_scales =
      slot_kv != nullptr ? slot_kv[my_slot].v_scales : v_scales_in;
  const std::uint32_t ring_rows =
      slot_kv != nullptr ? slot_kv[my_slot].ring_rows : ring_rows_in;
  constexpr std::uint32_t kTile = 256;
  // Sentinel for empty key lanes; a plain kTile collided with the legal
  // absolute position 256 on global layers (deep context dropped one key).
  constexpr std::uint32_t kEmptyKey = 0xFFFFFFFFu;
  const std::uint32_t split = blockIdx.z;
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  if (slot_kv != nullptr && k_cache == nullptr) {
    float* o = out + (static_cast<std::size_t>(t) * heads + h) * d_v;
    for (std::uint32_t i = threadIdx.x; i < d_v; i += blockDim.x) {
      o[i] = 0.0f;
    }
    return;
  }
  const std::uint32_t pos = row_pos != nullptr ? row_pos[t] : (*start_pos + t);
  const std::uint32_t kvh = h / (heads / kv_heads);
  __shared__ float qs[192];
  __shared__ float p[kTile];
  __shared__ float shared[32];
  __shared__ std::uint32_t keys[kTile];
  // Each wave accumulates the V mix of its own 32 key slots, so the writeback
  // below first sums the waves' partials through shared memory. Without the
  // reduction every wave covered the whole d_v row with divergent values (its
  // own slice's mix, zeros for empty waves), which raced per store.
  __shared__ float vaccs[kMimoBlockWaves][32][kMimoVLanes];
  const std::uint32_t i = threadIdx.x;
  const std::uint32_t lane = i & 31u;
  const std::uint32_t wave = i >> 5u;
  const float scale = rsqrtf(static_cast<float>(d_qk));
  const std::size_t k_stride = static_cast<std::size_t>(kv_heads) * d_qk;
  const std::size_t v_stride = static_cast<std::size_t>(kv_heads) * d_v;
  const __half* k_head = k_cache + kvh * d_qk;
  const __half* v_head = v_cache + kvh * d_v;
  // Sliding layers attend the trailing `window` positions only.
  const std::uint32_t vis_begin =
      window != 0 && pos + 1 > window ? pos + 1 - window : 0;
  const std::uint32_t n_vis = pos + 1 - vis_begin;
#pragma unroll
  for (std::uint32_t j = 0; j < kMimoKLanes; ++j) {
    const std::uint32_t idx = (lane * kMimoKLanes) + j;
    qs[idx] = idx < d_qk
                  ? q[(static_cast<std::size_t>(t) * heads + h) * d_qk + idx]
                  : 0.0f;
  }
  __syncthreads();
  float m = -INFINITY;
  float l = 0.0f;
  float acc[kMimoVLanes] = {0.0f, 0.0f, 0.0f, 0.0f};

  // Six consecutive key halves per lane when the row is wave-wide.
  // The packed form takes two halves per 4-byte integer load and unpacks
  // through bit ops: lane*12 and lane*8 byte offsets stay 4-byte aligned, and
  // integer loads sidestep the packed half2 vector path this toolchain
  // miscompiles (harness-proven: [5,5,5,5] read back as [0,5,5,0]). The
  // scalar form stays for head widths the packing does not divide, and
  // GUFO_ATTN_SCALAR forces it for A/B.
  const auto unpack2 = [&](unsigned int v, float* out) {
    out[0] = SimKvHalf(static_cast<unsigned short>(v), kv_sim_bits);
    out[1] = SimKvHalf(static_cast<unsigned short>(v >> 16u), kv_sim_bits);
  };
  const bool packed_k = packed_ok != 0 && kMimoKLanes % 2 == 0 &&
                        d_qk == 32u * kMimoKLanes;
  const bool packed_v = packed_ok != 0 && kMimoVLanes % 2 == 0 &&
                        d_v == 32u * kMimoVLanes;
  // int8 twins: six bytes per lane cover the same six values as the fp16
  // form, so the loads only shrink. A lane's six values can straddle one
  // 32-value block boundary, which is why the scale is picked per value.
  // int8 rows only pay off once the key range is long: below the threshold
  // the fp16 rows cost the same bytes but less arithmetic, so the choice is
  // per row (a cohort mixes deep and shallow rows).
  const bool kv_int8_row = k_cache_q != nullptr && v_cache_q != nullptr &&
                           k_scales != nullptr && v_scales != nullptr &&
                           n_vis >= kv_min_keys;
  const std::int8_t* k_head_q =
      kv_int8_row ? k_cache_q + kvh * d_qk : nullptr;
  const std::int8_t* v_head_q =
      kv_int8_row ? v_cache_q + kvh * d_v : nullptr;
  // A row's scales are [kv_heads][d / kKvQuantBlock]; the head's slice starts
  // at kvh * (d / kKvQuantBlock), like the cache rows themselves.
  const __half* k_scale_head =
      k_scales != nullptr ? k_scales + kvh * (d_qk / kKvQuantBlock) : nullptr;
  const __half* v_scale_head =
      v_scales != nullptr ? v_scales + kvh * (d_v / kKvQuantBlock) : nullptr;
  const auto load_k6 = [&](std::size_t key_pos, float* k6) {
    if (k_head_q != nullptr) {
      const std::int8_t* row_q =
          k_head_q + key_pos * k_stride + (lane * kMimoKLanes);
      const __half* scales = k_scale_head + key_pos * (k_stride / kKvQuantBlock);
      const std::uint32_t gap = (lane * kMimoKLanes) % kKvQuantBlock;
      const std::uint32_t first = (lane * kMimoKLanes) / kKvQuantBlock;
      const float s0 = __half2float(scales[first]);
      const float s1 = gap + kMimoKLanes > kKvQuantBlock
                           ? __half2float(scales[first + 1])
                           : s0;
      // Three 2-byte loads cover the lane's six bytes (6*lane is even).
      const unsigned short* pq =
          reinterpret_cast<const unsigned short*>(row_q);
      const unsigned short w0 = pq[0];
      const unsigned short w1 = pq[1];
      const unsigned short w2 = pq[2];
      const std::int8_t q[6] = {
          static_cast<std::int8_t>(w0 & 0xFFu),
          static_cast<std::int8_t>(w0 >> 8u),
          static_cast<std::int8_t>(w1 & 0xFFu),
          static_cast<std::int8_t>(w1 >> 8u),
          static_cast<std::int8_t>(w2 & 0xFFu),
          static_cast<std::int8_t>(w2 >> 8u)};
#pragma unroll
      for (std::uint32_t j = 0; j < kMimoKLanes; ++j) {
        const std::uint32_t idx = (lane * kMimoKLanes) + j;
        k6[j] = idx < d_qk
                    ? static_cast<float>(q[j]) *
                          (gap + j < kKvQuantBlock ? s0 : s1)
                    : 0.0f;
      }
      return;
    }
    const __half* row = k_head + key_pos * k_stride + (lane * kMimoKLanes);
    if (packed_k) {
      const unsigned int* pk =
          reinterpret_cast<const unsigned int*>(row);
#pragma unroll
      for (std::uint32_t j = 0; j < kMimoKLanes / 2; ++j) {
        unpack2(pk[j], k6 + 2 * j);
      }
      return;
    }
#pragma unroll
    for (std::uint32_t j = 0; j < kMimoKLanes; ++j) {
      const std::uint32_t idx = (lane * kMimoKLanes) + j;
      k6[j] = idx < d_qk
                  ? SimKvHalf(__half_as_ushort(row[j]), kv_sim_bits)
                  : 0.0f;
    }
  };
  // Scalar halves only: the packed uint2 load drops alternating halves on
  // this toolchain (harness-proven: [5,5,5,5] read back as [0,5,5,0]), and
  // the vector path measured no win to justify the risk.
  const auto load_v4 = [&](std::size_t key_pos, float* v4) {
    if (v_head_q != nullptr) {
      const std::int8_t* row_q =
          v_head_q + key_pos * v_stride + (lane * kMimoVLanes);
      // Four values per lane divide 32, so one scale covers the lane.
      const float s = __half2float(
          v_scale_head[key_pos * (v_stride / kKvQuantBlock) +
                       (lane * kMimoVLanes) / kKvQuantBlock]);
      const unsigned int packed_q =
          *reinterpret_cast<const unsigned int*>(row_q);
#pragma unroll
      for (std::uint32_t j = 0; j < kMimoVLanes; ++j) {
        const std::uint32_t idx = (lane * kMimoVLanes) + j;
        // 4*lane is 4-aligned, so the four bytes sit in one word.
        const auto byte = static_cast<std::int8_t>(packed_q >> (8u * j));
        v4[j] = idx < d_v ? static_cast<float>(byte) * s : 0.0f;
      }
      return;
    }
    const __half* row = v_head + key_pos * v_stride + (lane * kMimoVLanes);
    if (packed_v) {
      const unsigned int* pv =
          reinterpret_cast<const unsigned int*>(row);
#pragma unroll
      for (std::uint32_t j = 0; j < kMimoVLanes / 2; ++j) {
        unpack2(pv[j], v4 + 2 * j);
      }
      return;
    }
#pragma unroll
    for (std::uint32_t j = 0; j < kMimoVLanes; ++j) {
      const std::uint32_t idx = (lane * kMimoVLanes) + j;
      v4[j] = idx < d_v ? SimKvHalf(__half_as_ushort(row[j]), kv_sim_bits)
                        : 0.0f;
    }
  };

  const auto tile_step = [&]() {
    float qv[kMimoKLanes];
#pragma unroll
    for (std::uint32_t j = 0; j < kMimoKLanes; ++j) {
      qv[j] = qs[(lane * kMimoKLanes) + j];
    }
    float s_mine = -INFINITY;
    // The per-key WaveSum drains the load pipeline after every key, and at
    // depth that capped a block near 2.6GB/s of key/value traffic. Grouping
    // four keys issues sixteen loads before the first consume, which is the
    // same arithmetic with four times the memory-level parallelism
    // (GUFO_ATTN_BATCH_KEYS=0 keeps the serial form for A/B).
    constexpr std::uint32_t kGroup = 4;
    if (batch_keys != 0) {
#pragma unroll
      for (std::uint32_t kk0 = 0; kk0 < 32; kk0 += kGroup) {
        std::uint32_t js[kGroup];
        float kbuf[kGroup][kMimoKLanes];
#pragma unroll
        for (std::uint32_t g = 0; g < kGroup; ++g) {
          js[g] = keys[(wave * 32) + kk0 + g];
          if (js[g] != kEmptyKey) {
            load_k6(js[g], kbuf[g]);
          }
        }
#pragma unroll
        for (std::uint32_t g = 0; g < kGroup; ++g) {
          float dot = 0.0f;
          if (js[g] != kEmptyKey) {
#pragma unroll
            for (std::uint32_t x = 0; x < kMimoKLanes; ++x) {
              dot += qv[x] * kbuf[g][x];
            }
          }
          dot = WaveSum(dot);
          if (lane == kk0 + g) {
            s_mine = js[g] != kEmptyKey ? dot * scale : -INFINITY;
          }
        }
      }
    } else {
      for (std::uint32_t kk = 0; kk < 32; ++kk) {
        const std::uint32_t slot = (wave * 32) + kk;
        const std::uint32_t j = keys[slot];
        float dot = 0.0f;
        if (j != kEmptyKey) {
          float k6[kMimoKLanes];
          load_k6(j, k6);
#pragma unroll
          for (std::uint32_t x = 0; x < kMimoKLanes; ++x) {
            dot += qv[x] * k6[x];
          }
        }
        dot = WaveSum(dot);
        if (lane == kk) {
          s_mine = j != kEmptyKey ? dot * scale : -INFINITY;
        }
      }
    }
    const float tile_max = BlockMax(s_mine, shared);
    const float m_new = fmaxf(m, tile_max);
    const float rescale = m == -INFINITY ? 0.0f : __expf(m - m_new);
    const float pj = s_mine == -INFINITY ? 0.0f : __expf(s_mine - m_new);
    p[i] = pj;
    const float tile_sum = BlockSum(pj, shared);
    l = l * rescale + tile_sum;
#pragma unroll
    for (std::uint32_t x = 0; x < kMimoVLanes; ++x) {
      acc[x] *= rescale;
    }
    m = m_new;
    __syncthreads();
    if (batch_keys != 0) {
#pragma unroll
      for (std::uint32_t kk0 = 0; kk0 < 32; kk0 += kGroup) {
        float ws[kGroup];
        float vbuf[kGroup][kMimoVLanes];
#pragma unroll
        for (std::uint32_t g = 0; g < kGroup; ++g) {
          ws[g] = p[(wave * 32) + kk0 + g];
          if (ws[g] != 0.0f) {
            load_v4(keys[(wave * 32) + kk0 + g], vbuf[g]);
          }
        }
#pragma unroll
        for (std::uint32_t g = 0; g < kGroup; ++g) {
          if (ws[g] != 0.0f) {
#pragma unroll
            for (std::uint32_t x = 0; x < kMimoVLanes; ++x) {
              acc[x] += ws[g] * vbuf[g][x];
            }
          }
        }
      }
    } else {
      for (std::uint32_t kk = 0; kk < 32; ++kk) {
        const std::uint32_t slot = (wave * 32) + kk;
        const float w = p[slot];
        if (w != 0.0f) {
          float v4[kMimoVLanes];
          load_v4(keys[slot], v4);
#pragma unroll
          for (std::uint32_t x = 0; x < kMimoVLanes; ++x) {
            acc[x] += w * v4[x];
          }
        }
      }
    }
    __syncthreads();
  };

  for (std::uint32_t tile = split * kTile; tile < n_vis;
       tile += kTile * splits) {
    const std::uint32_t base = tile + i;
    // kEmptyKey marks an empty lane. Sliding layers store ring-mapped rows
    // (position & (ring_rows - 1); the ring exceeds window + max batch, so
    // a prefill batch never clobbers rows its own earlier queries read).
    // Global layers store the absolute position, where 0xFFFFFFFF is
    // unreachable for any context the allocator admits.
    keys[i] = base < n_vis
                  ? (ring_rows != 0 ? ((vis_begin + base) & (ring_rows - 1))
                                    : vis_begin + base)
                  : kEmptyKey;
    __syncthreads();
    tile_step();
  }

  // An attention sink contributes one extra denominator term (gpt-oss style).
  // TODO(port): cross-check the exact sink scaling against llama.cpp once an
  // artifact carrying attn_sinks is at hand; absent sinks are the common case.
  // A split range folds it into split 0 only: the merge's log-sum-exp weights
  // that partial against the others, so every split folding it would count it
  // `splits` times. (Production never combines the two: sink-carrying layers
  // are the sliding ones, and those never split.)
  if (sinks != nullptr && (splits == 1 || split == 0)) {
    const float sink = sinks[h];
    const float m_new = fmaxf(m, sink);
    const float rescale = m == -INFINITY ? 0.0f : __expf(m - m_new);
    l = l * rescale + __expf(sink - m_new);
    m = m_new;
#pragma unroll
    for (std::uint32_t x = 0; x < kMimoVLanes; ++x) {
      acc[x] *= rescale;
    }
  }

  // Reduce the per-wave accumulators: every wave holds a disjoint slice of
  // the numerator, and one wave writes the whole row.
  __syncthreads();
#pragma unroll
  for (std::uint32_t x = 0; x < kMimoVLanes; ++x) {
    vaccs[wave][lane][x] = acc[x];
  }
  __syncthreads();

  if (splits == 1) {
    if (wave == 0) {
      const float inv = 1.0f / l;
#pragma unroll
      for (std::uint32_t x = 0; x < kMimoVLanes; ++x) {
        float total = 0.0f;
#pragma unroll
        for (std::uint32_t w = 0; w < kMimoBlockWaves; ++w) {
          total += vaccs[w][lane][x];
        }
        const std::uint32_t idx = (lane * kMimoVLanes) + x;
        if (idx < d_v) {
          out[(static_cast<std::size_t>(t) * heads + h) * d_v + idx] =
              total * inv;
        }
      }
    }
    return;
  }
  if (wave == 0) {
    float* part =
        partials +
        ((static_cast<std::size_t>(t) * heads + h) * splits + split) *
            (kMimoVLanes * 32 + 2);
#pragma unroll
    for (std::uint32_t x = 0; x < kMimoVLanes; ++x) {
      float total = 0.0f;
#pragma unroll
      for (std::uint32_t w = 0; w < kMimoBlockWaves; ++w) {
        total += vaccs[w][lane][x];
      }
      part[(lane * kMimoVLanes) + x] = total;
    }
    if (i == 0) {
      part[kMimoVLanes * 32] = m;
      part[kMimoVLanes * 32 + 1] = l;
    }
  }
}

/// Keys staged per round by the shared-KV decode kernel: 20 KiB of fp16 K/V,
/// so two blocks fit the 64 KiB LDS budget.
constexpr std::uint32_t kMimoSharedStage = 32;

/// Ceiling on the shared-KV split count: the merge cost grows linearly with
/// it and the executor's partials scratch is sized for the same cap.
constexpr std::uint32_t kAttnSplitsCap = 32;

/// Shared-KV decode attention for the global layers (window == 0).
///
/// The per-head kernel hands every (head, row, split) block its own copy of
/// the key range, so the heads / kv_heads query heads of one kv head stream
/// the same rows -- 16x the unique KV on the production shape, which the 65K
/// profile showed as the dominant cost of the global layers. Here one block
/// owns a whole kv head instead: it stages each key row through LDS once and
/// every warp consumes it for kHeadsPerWarp query heads, so the block streams
/// the key range once instead of once per head. The partials record is
/// MimoAttentionMergeKernel's layout and the key ranges are a contiguous
/// partition of [0, n_vis), so the two forms are interchangeable behind the
/// merge.
///
/// Grid is (kv_heads, rows, splits); the split count is what carries the
/// parallelism, since the block count no longer multiplies by the head count.
/// Sliding-window layers take the same form with `window` set: their range is
/// short (128 keys), so one unsplit block per kv head walks it and the bytes
/// are 1/(heads/kv_heads) of what the per-head form streams. Sinks fold into
/// the denominator exactly as in the per-head kernel.
template <std::uint32_t kHeadsPerWarp>
__global__ void MimoAttentionSharedKernel(
    const float* q, const __half* k_cache_in, const __half* v_cache_in,
    const float* sinks, float* out, float* partials, std::uint32_t splits,
    const std::uint32_t* start_pos, std::uint32_t heads,
    std::uint32_t kv_heads, std::uint32_t window, std::uint32_t ring_rows_in,
    const SlotKv* slot_kv, std::uint32_t slot,
    const std::uint32_t* row_slot, const std::uint32_t* row_pos) {
  // The host gates on this shape (production attention geometry); anything else
  // keeps the per-head kernel.
  constexpr std::uint32_t kDqk = 32u * kMimoKLanes;
  constexpr std::uint32_t kDv = 32u * kMimoVLanes;
  constexpr std::uint32_t kWarps = kMimoBlockWaves;
  constexpr std::uint32_t kStage = kMimoSharedStage;
  constexpr std::uint32_t kEmptyKey = 0xFFFFFFFFu;
  static_assert(kStage == 32u,
                "the lane-owns-key softmax maps lane k to key k");

  const std::uint32_t kvh = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  const std::uint32_t split = blockIdx.z;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warp = threadIdx.x >> 5u;
  const std::uint32_t my_slot = row_slot != nullptr ? row_slot[t] : slot;
  const __half* k_cache =
      slot_kv != nullptr ? slot_kv[my_slot].k_cache : k_cache_in;
  const __half* v_cache =
      slot_kv != nullptr ? slot_kv[my_slot].v_cache : v_cache_in;
  const std::uint32_t ring_rows =
      slot_kv != nullptr ? slot_kv[my_slot].ring_rows : ring_rows_in;
  const std::uint32_t head0 = kvh * (kWarps * kHeadsPerWarp);
  // A cancelled row (its session was retired by a replay) reads as zeros,
  // like the per-head kernel.
  if (slot_kv != nullptr && k_cache == nullptr) {
#pragma unroll
    for (std::uint32_t hh = 0; hh < kHeadsPerWarp; ++hh) {
      float* o = out + (static_cast<std::size_t>(t) * heads + head0 + warp +
                        hh * kWarps) *
                           kDv;
      for (std::uint32_t i = threadIdx.x; i < kDv; i += blockDim.x) {
        o[i] = 0.0f;
      }
    }
    return;
  }
  const std::uint32_t pos = row_pos != nullptr ? row_pos[t] : (*start_pos + t);
  // Sliding-window layers see the trailing `window` positions (the per-head
  // kernel's rule); the row index still walks the ring.
  const std::uint32_t lo =
      (window != 0 && pos + 1 > window) ? pos + 1 - window : 0;
  const std::uint32_t n_vis = pos + 1 - lo;
  const float scale = rsqrtf(static_cast<float>(kDqk));

  __shared__ __align__(16) __half sk[kStage][kDqk];
  __shared__ __align__(16) __half sv[kStage][kDv];
  __shared__ float sp[kWarps][kHeadsPerWarp][kStage];
  __shared__ std::uint32_t keys[kStage];

  // Six halves per lane per head, the same cover the per-head kernel uses.
  float qv[kHeadsPerWarp][kMimoKLanes];
#pragma unroll
  for (std::uint32_t hh = 0; hh < kHeadsPerWarp; ++hh) {
    const float* qh = q + (static_cast<std::size_t>(t) * heads + head0 + warp +
                           hh * kWarps) *
                              kDqk;
#pragma unroll
    for (std::uint32_t j = 0; j < kMimoKLanes; ++j) {
      qv[hh][j] = qh[(lane * kMimoKLanes) + j];
    }
  }
  float m[kHeadsPerWarp];
  float l[kHeadsPerWarp];
  float acc[kHeadsPerWarp][kMimoVLanes];
#pragma unroll
  for (std::uint32_t hh = 0; hh < kHeadsPerWarp; ++hh) {
    m[hh] = -INFINITY;
    l[hh] = 0.0f;
#pragma unroll
    for (std::uint32_t x = 0; x < kMimoVLanes; ++x) {
      acc[hh][x] = 0.0f;
    }
  }
  // Contiguous partition of the visible range: the merge combines whatever
  // records the splits write, so the boundaries are free.
  const std::uint32_t per = (n_vis + splits - 1u) / splits;
  const std::uint32_t base_begin = split * per < n_vis ? split * per : n_vis;
  const std::uint32_t base_end =
      base_begin + per < n_vis ? base_begin + per : n_vis;
  const __half* k_head = k_cache + kvh * kDqk;
  const __half* v_head = v_cache + kvh * kDv;
  const std::size_t k_stride = static_cast<std::size_t>(kv_heads) * kDqk;
  const std::size_t v_stride = static_cast<std::size_t>(kv_heads) * kDv;

  for (std::uint32_t base = base_begin; base < base_end; base += kStage) {
    // Stage the round: key slots, then the K and V rows. The empty lane of a
    // short last round is zero-filled so its stale LDS cannot poison a V
    // accumulator through the 0 * NaN path.
    if (threadIdx.x < kStage) {
      keys[threadIdx.x] = base + threadIdx.x < base_end ? 1u : kEmptyKey;
    }
    for (std::uint32_t g = threadIdx.x; g < kStage * (kDqk / 8u);
         g += blockDim.x) {
      const std::uint32_t kk = g / (kDqk / 8u);
      const std::uint32_t seg = g % (kDqk / 8u);
      const std::uint32_t off = base + kk;
      uint4* dst = reinterpret_cast<uint4*>(&sk[kk][seg * 8u]);
      if (off < base_end) {
        const std::uint32_t abs = lo + off;
        const std::uint32_t j =
            ring_rows != 0 ? (abs & (ring_rows - 1u)) : abs;
        *dst = *reinterpret_cast<const uint4*>(k_head + j * k_stride +
                                               seg * 8u);
      } else {
        *dst = make_uint4(0u, 0u, 0u, 0u);
      }
    }
    for (std::uint32_t g = threadIdx.x; g < kStage * (kDv / 8u);
         g += blockDim.x) {
      const std::uint32_t kk = g / (kDv / 8u);
      const std::uint32_t seg = g % (kDv / 8u);
      const std::uint32_t off = base + kk;
      uint4* dst = reinterpret_cast<uint4*>(&sv[kk][seg * 8u]);
      if (off < base_end) {
        const std::uint32_t abs = lo + off;
        const std::uint32_t j =
            ring_rows != 0 ? (abs & (ring_rows - 1u)) : abs;
        *dst = *reinterpret_cast<const uint4*>(v_head + j * v_stride +
                                               seg * 8u);
      } else {
        *dst = make_uint4(0u, 0u, 0u, 0u);
      }
    }
    __syncthreads();
    // Scores: the warp sweeps the round one key at a time and each staged
    // row is read once for every head the warp owns. Lane kk ends holding
    // key kk's score.
    float s[kHeadsPerWarp];
#pragma unroll 4
    for (std::uint32_t kk = 0; kk < kStage; ++kk) {
      float dot[kHeadsPerWarp];
#pragma unroll
      for (std::uint32_t hh = 0; hh < kHeadsPerWarp; ++hh) {
        dot[hh] = 0.0f;
      }
      if (keys[kk] != kEmptyKey) {
        const __half* krow = &sk[kk][lane * kMimoKLanes];
#pragma unroll
        for (std::uint32_t x = 0; x < kMimoKLanes; x += 2) {
          // Two halves per 32-bit LDS load; a lane stride of twelve bytes
          // stays four-byte aligned.
          const unsigned int packed =
              *reinterpret_cast<const unsigned int*>(krow + x);
          const __half2 halves = __halves2half2(
              __ushort_as_half(static_cast<unsigned short>(packed)),
              __ushort_as_half(static_cast<unsigned short>(packed >> 16u)));
          const float2 pair = __half22float2(halves);
#pragma unroll
          for (std::uint32_t hh = 0; hh < kHeadsPerWarp; ++hh) {
            dot[hh] += qv[hh][x] * pair.x;
            dot[hh] += qv[hh][x + 1] * pair.y;
          }
        }
      }
#pragma unroll
      for (std::uint32_t hh = 0; hh < kHeadsPerWarp; ++hh) {
        dot[hh] = WaveSum(dot[hh]);
        if (lane == kk) {
          s[hh] = keys[kk] != kEmptyKey ? dot[hh] * scale : -INFINITY;
        }
      }
    }
    // Online softmax over the round, per head, warp-local.
#pragma unroll
    for (std::uint32_t hh = 0; hh < kHeadsPerWarp; ++hh) {
      float tile_max = s[hh];
      for (int off = 16; off > 0; off >>= 1) {
        tile_max = fmaxf(tile_max, __shfl_xor(tile_max, off));
      }
      const float m_new = fmaxf(m[hh], tile_max);
      const float rescale =
          m[hh] == -INFINITY ? 0.0f : __expf(m[hh] - m_new);
      const float pj = s[hh] == -INFINITY ? 0.0f : __expf(s[hh] - m_new);
      sp[warp][hh][lane] = pj;
      float tile_sum = pj;
      for (int off = 16; off > 0; off >>= 1) {
        tile_sum += __shfl_xor(tile_sum, off);
      }
      l[hh] = l[hh] * rescale + tile_sum;
#pragma unroll
      for (std::uint32_t x = 0; x < kMimoVLanes; ++x) {
        acc[hh][x] *= rescale;
      }
      m[hh] = m_new;
    }
    __syncwarp();
    // V mix: a lane covers four V dims for every key of the round, and each
    // staged value feeds every head's accumulator.
#pragma unroll 8
    for (std::uint32_t kk = 0; kk < kStage; ++kk) {
      float w[kHeadsPerWarp];
#pragma unroll
      for (std::uint32_t hh = 0; hh < kHeadsPerWarp; ++hh) {
        w[hh] = sp[warp][hh][kk];
      }
      const __half* vrow = &sv[kk][lane * kMimoVLanes];
#pragma unroll
      for (std::uint32_t x = 0; x < kMimoVLanes; x += 2) {
        const unsigned int packed =
            *reinterpret_cast<const unsigned int*>(vrow + x);
        const __half2 halves = __halves2half2(
            __ushort_as_half(static_cast<unsigned short>(packed)),
            __ushort_as_half(static_cast<unsigned short>(packed >> 16u)));
        const float2 pair = __half22float2(halves);
#pragma unroll
        for (std::uint32_t hh = 0; hh < kHeadsPerWarp; ++hh) {
          acc[hh][x] += w[hh] * pair.x;
          acc[hh][x + 1] += w[hh] * pair.y;
        }
      }
    }
    __syncthreads();
  }
  // One split writes the row directly; several hand the merge kernel the
  // same record layout the per-head kernel writes.
  if (sinks != nullptr && (splits == 1 || split == 0)) {
    // An attention sink contributes one extra denominator term per head, the
    // same fold the per-head kernel applies (the sliding layers carry them),
    // and by the same rule it lands in one split only.
#pragma unroll
    for (std::uint32_t hh = 0; hh < kHeadsPerWarp; ++hh) {
      const float sink = sinks[head0 + warp + hh * kWarps];
      const float m_new = fmaxf(m[hh], sink);
      const float rescale =
          m[hh] == -INFINITY ? 0.0f : __expf(m[hh] - m_new);
      l[hh] = l[hh] * rescale + __expf(sink - m_new);
      m[hh] = m_new;
#pragma unroll
      for (std::uint32_t x = 0; x < kMimoVLanes; ++x) {
        acc[hh][x] *= rescale;
      }
    }
  }
#pragma unroll
  for (std::uint32_t hh = 0; hh < kHeadsPerWarp; ++hh) {
    const std::uint32_t h = head0 + warp + hh * kWarps;
    if (splits == 1) {
      const float inv = 1.0f / l[hh];
      float* o = out + (static_cast<std::size_t>(t) * heads + h) * kDv;
#pragma unroll
      for (std::uint32_t x = 0; x < kMimoVLanes; ++x) {
        o[(lane * kMimoVLanes) + x] = acc[hh][x] * inv;
      }
    } else {
      float* part =
          partials +
          ((static_cast<std::size_t>(t) * heads + h) * splits + split) *
              (kMimoVLanes * 32 + 2);
#pragma unroll
      for (std::uint32_t x = 0; x < kMimoVLanes; ++x) {
        part[(lane * kMimoVLanes) + x] = acc[hh][x];
      }
      if (lane == 0) {
        part[kMimoVLanes * 32] = m[hh];
        part[kMimoVLanes * 32 + 1] = l[hh];
      }
    }
  }
}

// Prefill batch attention: sixteen queries per block, K/V mini-tiles staged
// through LDS with uint4 loads, the QK^T product on the matrix core (waves
// 0-3 each own sixteen keys of the tile; every wave sweeps the full
// sixteen-query tile through twelve dim segments), then a two-queries-per-wave
// softmax and PV mix laid out lane-owns-output-dim. The scalar lane-owns-key
// form this replaces was instruction-issue bound at matrix widths; the WMMA
// form measures 1.5x (global) to 1.5x (sliding) of it at n=663 against the
// CPU golden reference (F16 query staging, F32 softmax, six shapes green).
constexpr std::uint32_t kFbMQ = 16;   ///< queries per block (2 per wave in PV)
constexpr std::uint32_t kFbKT = 64;    ///< keys per staged mini-tile
/// Row strides are padded to 16-byte multiples so every staged access is
/// provably aligned: a 32-byte fragment load and a 16-byte staging store both
/// lower to single LDS instructions only then. At 196/202/130 halves the
/// disassembly showed no 128-bit LDS access at all -- the operand reads came
/// out as 4-byte pieces, which is what saturates the address pipeline. The
/// strides stay 4 mod 32 words (100 words here) so the 16 rows one wave reads
/// land on distinct banks.
constexpr std::uint32_t kFbQS = 200;  ///< F16 halves per staged query row
constexpr std::uint32_t kFbKS = 200;  ///< F16 halves per staged K row
constexpr std::uint32_t kFbVS = 136;  ///< F16 halves per staged V row

/// Prefill tile: 32 queries per block, 16 staged keys. Rows divide both the KV
/// stream (a block walks its whole visible context) and the block count; the
/// sweep over the 58.7K prefill says the block count is what pays -- 64 rows
/// cost 210.4 s and 16 rows 235.7 s against 180.9 s at 32. The key tile sets
/// LDS and with it the resident blocks per CU: 16 keys is 27 KiB (two blocks,
/// best), 32 keys is 41 KiB (one, +2.0 s) and 16x16 is 19 KiB (three, +4.9 s).
/// 32 rows with the 64-key tile would need 71 KiB and does not fit at all.
constexpr std::uint32_t kFbMRows = 32;    ///< prefill queries per block
constexpr std::uint32_t kFbKPrefill = 32; ///< keys per staged tile, 32-key arm
/// Widest prefill tile: 64 queries share one 32-key staged tile. Depth costs
/// bytes proportional to (rows per block)^-1 -- every block walks its whole
/// visible context -- so doubling the rows halves the KV stream. Measured: the
/// 64-row/16-key shape (43 KiB) ran 23% *slower* than the 32-row baseline --
/// halving the staged key tile doubles the per-key iterations, and that
/// overhead outweighs the byte saving. 64 rows with the 32-key tile is
/// 61.25 KiB, which does fit the 64 KiB budget.
constexpr std::uint32_t kFbMRows64 = 64;    ///< widest prefill queries
constexpr std::uint32_t kFbKPrefill64 = 32; ///< widest prefill staged keys
/// Prefill tile shape, as rows x staged keys per block. Rows per block divide
/// both the KV stream (a block walks its whole visible context) and the block
/// count; measurement says the block count wins -- 64 rows measured 210.4 s on
/// the 58.7K prefill against 183.5 s at 32 rows. The key tile sets the block's
/// LDS footprint, and with it how many blocks a CU can hold: 32x32 is 41 KiB
/// (one block per CU), 32x16 is 27 KiB and 16x16 is 19 KiB (two and three).
/// GUFO_ATTN_TILE=RxK picks one of the shapes below.
static std::uint32_t WideTileRows() {
  static const std::uint32_t rows = [] {
    const char* env = getenv("GUFO_ATTN_TILE");
    const long v = env != nullptr ? std::strtol(env, nullptr, 10) : -1;
    // 64 rows is the measured default now that the PV/LDS work shifted the
    // balance toward fewer KV reads: 147.1 s against 151.7 s at 32 rows on the
    // 58.7K prefill, and the kernel harness ranks it first too. GUFO_ATTN_TILE
    // =32x.. or =16x.. brings the older shapes back for comparison.
    return (v == 16 || v == 32) ? static_cast<std::uint32_t>(v) : kFbMRows64;
  }();
  return rows;
}

/// True when the 32-row prefill tile should stage 16 keys rather than 32.
/// That is the measured default: 27 KiB of LDS holds two blocks per CU, where
/// the 32-key tile (41 KiB) holds one, and the second resident block is worth
/// 2.0 s on the 58.7K prefill (180.9 s against 182.9 s at one block). Three
/// resident blocks is past the optimum -- 16x16 (19 KiB) costs 4.9 s.
/// GUFO_ATTN_TILE=32x32 is the opt-out comparison arm.
static bool WideTileKeys16() {
  static const bool narrow = [] {
    const char* env = getenv("GUFO_ATTN_TILE");
    if (env == nullptr) { return true; }
    const char* x = strchr(env, 'x');
    return x != nullptr && std::strtol(x + 1, nullptr, 10) == 16;
  }();
  return narrow;
}
/// Wide prefill tile: `kRows` queries (16 or 32) share one staged key tile of
/// `kKeys` keys. The KV stream is what makes prefill cost grow with depth --
/// every block walks its whole visible context -- so the row count is what
/// buys depth: 32 rows halve the blocks (and therefore the KV reads) per head,
/// and the wider query block also puts all eight waves on the QK^T pass
/// instead of four. 32 rows do not fit the 64 KiB LDS budget with a 64-key
/// tile (71 KiB), so the wider shape stages 32 keys instead (42 KiB); the
/// 16-row shape keeps the 64-key tile.
/// Per-shape block size. The register file is the binding constraint: at 8
/// warps the 64-row tile needs 248 VGPRs/thread (~= the whole 64K-lane file),
/// so one block per CU is all that fits and half the CU's warp slots stay
/// empty. 16 warps halve the per-wave row count (4 rows/wave), which brings the
/// per-thread state under 128 VGPRs -- same footprint, twice the warps to hide
/// the LDS/DRAM latency that bounds this kernel. The 16-row tiles keep 8 warps:
/// they have only one 16-row query fragment, so more warps would idle.
/// (Keyed on kRows alone on purpose: __launch_bounds__ is a macro, so a comma
/// inside its argument would split the argument list.)
template <std::uint32_t kRows>
struct WideAttnThreads {
  static constexpr std::uint32_t value = kRows >= 32 ? 512 : 256;
};

template <std::uint32_t kRows, std::uint32_t kKeys>
__global__ __launch_bounds__(WideAttnThreads<kRows>::value) void MimoBatchAttentionWmmaKernel(
    const float* __restrict__ q, const __half* __restrict__ k_cache_in,
    const __half* __restrict__ v_cache_in, const float* __restrict__ sinks,
    float* __restrict__ out, const std::uint32_t* __restrict__ start_pos,
    std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t d_qk,
    std::uint32_t d_v, std::uint32_t window, std::uint32_t ring_rows_in,
    std::uint32_t n_tokens, const SlotKv* slot_kv, std::uint32_t slot,
    float* partials, std::uint32_t splits) {
  static_assert(kRows % 16 == 0, "rows must tile into WMMA 16-row fragments");
  static_assert(kKeys % 16 == 0, "keys must tile into WMMA 16-key segments");
  constexpr std::uint32_t kThreads = WideAttnThreads<kRows>::value;
  constexpr std::uint32_t kRowsPerWave = kRows / (kThreads / 32);
  constexpr std::uint32_t kQuarters = kKeys / 16;
  constexpr std::uint32_t kDv = 32u * kMimoVLanes;  // 128 output dims
  constexpr std::uint32_t kSlots = (kRows / 16) * kQuarters;
  // Key-range split index. A narrow batch (one 16-row tile per head) has no
  // parallelism in grid.y at depth, so the caller splits the key range and
  // merges the per-row partials (the layout the one-query kernel writes).
  const std::uint32_t split = blockIdx.z;
  const __half* k_cache =
      slot_kv != nullptr ? slot_kv[slot].k_cache : k_cache_in;
  const __half* v_cache =
      slot_kv != nullptr ? slot_kv[slot].v_cache : v_cache_in;
  const std::uint32_t ring_rows =
      slot_kv != nullptr ? slot_kv[slot].ring_rows : ring_rows_in;
  __shared__ alignas(16) __half s_q[kRows][kFbQS];
  __shared__ alignas(16) __half s_k[kKeys][kFbKS];
  // V transposed to [dim][key]: the WMMA B operand's lane index is the n
  // dimension (here d_v) and its 16 values span k (the keys) -- see the PV.
  __shared__ alignas(16) __half s_v[kDv][kKeys + 8];
  // Row-major fp16, padded to a 16-byte row stride: the PV's A fragment wants
  // one row's 16 key-values in one 32-byte LDS load per lane, which is how the
  // QK^T loads Q. Zero-filled past ntile (see the softmax store).
  __shared__ alignas(16) __half s_p[kRows][kKeys + 8];
  __shared__ float s_scores[kRows][kKeys];
  // Per-row online-softmax rescale factor, applied to the PV's C fragments,
  // and the final 1/l used by the epilogue (both written per tile / once).
  __shared__ float s_rescale[kRows];
  __shared__ float s_final[kRows];
  // Tile id at which some row last rebased its accumulator. The PV reads it to
  // decide whether any rescale is needed at all for the current tile.
  __shared__ std::uint32_t s_rebase_tile;
  // p is stored as fp16, so the reference may drift this far before a rebase
  // is forced: exp(8) = 2981 leaves ample headroom under fp16's 65504.
  constexpr float kDriftCap = 8.0f;
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t w = threadIdx.x >> 5;
  const std::uint32_t lane = threadIdx.x & 31;
  const std::uint32_t sub_lane = lane & 15;
  const std::uint32_t half_id = lane >> 4;
  const std::uint32_t kvh = h / (heads / kv_heads);
  const std::uint32_t t0 = blockIdx.y * kRows;
  if (slot_kv != nullptr && k_cache == nullptr) {
    // Cancelled row: the eager path memset its context and skipped attention.
    for (std::uint32_t idx = threadIdx.x; idx < kRows * d_v;
         idx += kThreads) {
      const std::uint32_t tt = t0 + idx / d_v;
      if (tt < n_tokens) {
        out[(static_cast<std::size_t>(tt) * heads + h) * d_v + idx % d_v] =
            0.0f;
      }
    }
    return;
  }
  const std::uint32_t q0 = w * kRowsPerWave;  // this wave's first query row
  const float scale = rsqrtf(static_cast<float>(d_qk));
  const std::size_t k_stride = static_cast<std::size_t>(kv_heads) * d_qk;
  const std::size_t v_stride = static_cast<std::size_t>(kv_heads) * d_v;
  const __half* k_head = k_cache + static_cast<std::size_t>(kvh) * d_qk;
  const __half* v_head = v_cache + static_cast<std::size_t>(kvh) * d_v;

  for (std::uint32_t idx = threadIdx.x; idx < kRows * 192;
       idx += kThreads) {
    const std::uint32_t mi = idx / 192;
    const std::uint32_t d = idx % 192;
    const std::uint32_t tt = t0 + mi;
    s_q[mi][d] = __float2half(
        tt < n_tokens && d < d_qk
            ? q[(static_cast<std::size_t>(tt) * heads + h) * d_qk + d]
            : 0.0f);
  }
  __syncthreads();

  const std::uint32_t t_last_raw = t0 + kRows - 1;
  const std::uint32_t t_last =
      t_last_raw < n_tokens ? t_last_raw : (n_tokens != 0 ? n_tokens - 1 : 0);
  const std::uint32_t pos_first = *start_pos + t0;
  const std::uint32_t pos_last = *start_pos + t_last;
  const std::uint32_t lo =
      (window != 0 && pos_first + 1 > window) ? pos_first + 1 - window : 0;
  const std::uint32_t n_keys_block = pos_last >= lo ? pos_last - lo + 1 : 0;

  bool livev[kRowsPerWave];
  std::uint32_t posv[kRowsPerWave];
  for (std::uint32_t b = 0; b < kRowsPerWave; ++b) {
    const std::uint32_t t = t0 + q0 + b;
    livev[b] = t < n_tokens;
    const std::uint32_t tc = livev[b] ? t : (n_tokens != 0 ? n_tokens - 1 : 0);
    posv[b] = *start_pos + tc;
  }

  float l[kRowsPerWave];
  for (std::uint32_t b = 0; b < kRowsPerWave; ++b) {
    l[b] = 0.0f;
  }
  // PV fragment ownership: (kRows/16) row fragments x (128/16) d_v fragments,
  // split evenly across the block's warps (2 pairs per warp at 64 rows and 16
  // warps, 1 at 32/16 rows). Each pair accumulates its own C fragment across
  // the whole key range.
  constexpr std::uint32_t kDvFrags = kDv / 16;
  constexpr std::uint32_t kPairs = (kRows / 16) * kDvFrags;
  constexpr std::uint32_t kWarps = kThreads / 32;
  static_assert(kPairs % kWarps == 0, "fragment pairs must divide across warps");
  constexpr std::uint32_t kPairsPerWarp = kPairs / kWarps;
  const std::uint32_t pair0 = (threadIdx.x >> 5) * kPairsPerWarp;
  float m_ref[kRowsPerWave];
  for (std::uint32_t b = 0; b < kRowsPerWave; ++b) {
    m_ref[b] = -INFINITY;
  }
  if (threadIdx.x == 0) {
    s_rebase_tile = ~0u;
  }
  v8f c_acc[kPairsPerWarp];
#pragma unroll
  for (std::uint32_t p = 0; p < kPairsPerWarp; ++p) {
    c_acc[p] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
  }

  // The staging loop is what stalls this kernel: a thread's LDS store waits on
  // its own global load, and at depth that is a DRAM round trip per tile. So
  // the next tile's K/V vectors are loaded into registers before the current
  // tile is computed and committed at the top of the next iteration -- the
  // loads then overlap the QK^T, softmax and PV of the tile already staged.
  // Shared memory is untouched, so the two resident blocks per CU stay.
  constexpr bool kPrefetch = kKeys <= 32;
  constexpr std::uint32_t kKSlots =
      kPrefetch ? (kKeys * 24 + kThreads - 1) / kThreads : 1;
  // V's gather is a 2-byte transpose, so its item count is dims x key pairs.
  constexpr std::uint32_t kVSlots =
      kPrefetch ? (kDv * kKeys / 2 + kThreads - 1) / kThreads : 1;
  uint4 k_pre[kKSlots];
  __half2 v_pre[kVSlots];
  const auto issue = [&](std::uint32_t b, std::uint32_t n) {
    if constexpr (!kPrefetch) {
      return;
    }
#pragma unroll
    for (std::uint32_t s = 0; s < kKSlots; ++s) {
      const std::uint32_t idx = threadIdx.x + s * kThreads;
      if (idx < n * 24) {
        const std::uint32_t kj = idx / 24;
        const std::uint32_t u = idx % 24;
        const std::uint32_t key_pos = lo + b + kj;
        const std::uint32_t row =
            ring_rows ? (key_pos & (ring_rows - 1)) : key_pos;
        k_pre[s] = *reinterpret_cast<const uint4*>(
            k_head + static_cast<std::size_t>(row) * k_stride + u * 8);
      }
    }
#pragma unroll
    for (std::uint32_t s = 0; s < kVSlots; ++s) {
      const std::uint32_t idx = threadIdx.x + s * kThreads;
      const std::uint32_t d = idx / (kKeys / 2);
      const std::uint32_t kp = idx % (kKeys / 2);
      if (d < kDv && kp * 2 < n) {
        // One dword covers two adjacent keys of one dim row; a key at or past
        // the tile end carries p = 0, so filling 0 is as good as skipping it.
        const std::uint32_t key_pos0 = lo + b + kp * 2;
        const std::uint32_t row0 =
            ring_rows ? (key_pos0 & (ring_rows - 1)) : key_pos0;
        const __half lo_h =
            v_head[static_cast<std::size_t>(row0) * v_stride + d];
        __half hi_h = __half(0);
        if (kp * 2 + 1 < n) {
          const std::uint32_t key_pos1 = key_pos0 + 1;
          const std::uint32_t row1 =
              ring_rows ? (key_pos1 & (ring_rows - 1)) : key_pos1;
          hi_h = v_head[static_cast<std::size_t>(row1) * v_stride + d];
        }
        v_pre[s] = __halves2half2(lo_h, hi_h);
      }
    }
  };
  const auto commit = [&](std::uint32_t b, std::uint32_t n) {
    if constexpr (!kPrefetch) {
      for (std::uint32_t idx = threadIdx.x; idx < n * 24; idx += kThreads) {
        const std::uint32_t kj = idx / 24;
        const std::uint32_t u = idx % 24;
        const std::uint32_t key_pos = lo + b + kj;
        const std::uint32_t row =
            ring_rows ? (key_pos & (ring_rows - 1)) : key_pos;
        *reinterpret_cast<uint4*>(&s_k[kj][u * 8]) =
            *reinterpret_cast<const uint4*>(
                k_head + static_cast<std::size_t>(row) * k_stride + u * 8);
      }
      for (std::uint32_t idx = threadIdx.x; idx < kDv * kKeys / 2;
           idx += kThreads) {
        const std::uint32_t d = idx / (kKeys / 2);
        const std::uint32_t kp = idx % (kKeys / 2);
        if (d >= kDv || kp * 2 >= n) {
          continue;
        }
        const std::uint32_t key_pos0 = lo + b + kp * 2;
        const std::uint32_t row0 =
            ring_rows ? (key_pos0 & (ring_rows - 1)) : key_pos0;
        const __half lo_h =
            v_head[static_cast<std::size_t>(row0) * v_stride + d];
        __half hi_h = __half(0);
        if (kp * 2 + 1 < n) {
          const std::uint32_t key_pos1 = key_pos0 + 1;
          const std::uint32_t row1 =
              ring_rows ? (key_pos1 & (ring_rows - 1)) : key_pos1;
          hi_h = v_head[static_cast<std::size_t>(row1) * v_stride + d];
        }
        *reinterpret_cast<__half2*>(&s_v[d][kp * 2]) = __halves2half2(lo_h, hi_h);
      }
      return;
    }
#pragma unroll
    for (std::uint32_t s = 0; s < kKSlots; ++s) {
      const std::uint32_t idx = threadIdx.x + s * kThreads;
      if (idx < n * 24) {
        const std::uint32_t kj = idx / 24;
        const std::uint32_t u = idx % 24;
        *reinterpret_cast<uint4*>(&s_k[kj][u * 8]) = k_pre[s];
      }
    }
#pragma unroll
    for (std::uint32_t s = 0; s < kVSlots; ++s) {
      const std::uint32_t idx = threadIdx.x + s * kThreads;
      const std::uint32_t d = idx / (kKeys / 2);
      const std::uint32_t kp = idx % (kKeys / 2);
      if (d < kDv && kp * 2 < n) {
        *reinterpret_cast<__half2*>(&s_v[d][kp * 2]) = v_pre[s];
      }
    }
  };

  const std::uint32_t base_first = split * kKeys;
  if (base_first < n_keys_block) {
    issue(base_first, std::min(kKeys, n_keys_block - base_first));
  }
  for (std::uint32_t base = base_first; base < n_keys_block;
       base += kKeys * splits) {
    const std::uint32_t ntile = std::min(kKeys, n_keys_block - base);
    const std::uint32_t tile_id = base;
    // The first barrier retires the previous iteration's PV reads before the
    // commit overwrites those rows; the second publishes the staged tile.
    __syncthreads();
    commit(base, ntile);
    __syncthreads();
    const std::uint32_t next_base = base + kKeys * splits;
    if (next_base < n_keys_block) {
      issue(next_base, std::min(kKeys, n_keys_block - next_base));
    }

    // QK^T on the matrix core: (rows/16) row fragments x (keys/16) key
    // quarters tile the eight waves; every wave computes a full 16-query
    // fragment over 12 dim segments. The C fragment lands as (query =
    // sub_lane, key = kj0 + 2e + half_id) - the same lane map the RoutedF16
    // narrow epilogue transposes through.
    if (w < kSlots) {
      const std::uint32_t frag = w / kQuarters;
      const std::uint32_t kj0 = (w % kQuarters) * 16;
      v8f sc{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
#pragma unroll
      for (int dseg = 0; dseg < 12; ++dseg) {
        // One 32-byte load per operand; the padded strides make the address
        // provably 16-byte aligned, so this is two LDS instructions rather
        // than eight.
        v16h a, b;
        __builtin_memcpy(&a, &s_q[(frag * 16) + sub_lane][dseg * 16], 32);
        __builtin_memcpy(&b, &s_k[kj0 + sub_lane][dseg * 16], 32);
        sc = Wmma(a, b, sc);
      }
      // acc[e] is C[2e + half_id][sub_lane]: query row 2e+half_id, key column
      // kj0 + sub_lane (probed on hardware; the RoutedF16 epilogue uses the
      // same map transposed).
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const std::uint32_t row = (frag * 16) + (2 * e) + half_id;
        const std::uint32_t tq = t0 + row;
        const std::uint32_t kk = kj0 + sub_lane;
        float s = -INFINITY;
        if (kk < ntile && tq < n_tokens) {
          const std::uint32_t posq = *start_pos + tq;
          const std::uint32_t visq =
              (window != 0 && posq + 1 > window) ? posq + 1 - window : 0;
          const std::uint32_t key_pos = lo + base + kk;
          s = (key_pos >= visq && key_pos <= posq) ? sc[e] * scale : -INFINITY;
        }
        s_scores[row][kk] = s;
      }
    }
    __syncthreads();

    // Softmax: one warp covers TWO rows at once, 16 lanes each. Lane L of half
    // h owns row (2*pass + h)'s keys {(L&15)*kPerLane .. +kPerLane}, with
    // kPerLane = kKeys/16 (1, 2 or 4). Both rows reduce in the same four
    // shuffle steps -- the xor masks 8,4,2,1 stay inside a 16-lane half -- which
    // halves the per-row reduction work, and each lane stores its p values as
    // one vector.
    constexpr std::uint32_t kPerLane = kKeys / 16;
    static_assert(kRowsPerWave % 2 == 0, "softmax pairs rows");
    const std::uint32_t row_half = lane >> 4;
    const std::uint32_t lane16 = lane & 15;
#pragma unroll
    for (std::uint32_t pass = 0; pass < kRowsPerWave / 2; ++pass) {
      const std::uint32_t b = 2 * pass + row_half;
      const std::uint32_t r = q0 + b;
      const bool valid = livev[b];
      float v[kPerLane];
#pragma unroll
      for (std::uint32_t i = 0; i < kPerLane; ++i) {
        const std::uint32_t key = lane16 * kPerLane + i;
        v[i] = (valid && key < ntile) ? s_scores[r][key] : -INFINITY;
      }
      // p = exp(score - m_ref) is the accumulator's own scale, so no tile max
      // is needed while the reference is fresh. A stale reference would push p
      // past fp16, so crossing the drift cap turns this tile into a rebase:
      // the row max is then recovered with the shuffle reduction the fast path
      // no longer pays for. Testing the raw score keeps the check at two
      // compares, and the vote makes the branch warp-uniform so its shuffles
      // are legal. Every row's first tile has m_ref = -inf and lands here.
      if (lane16 == 0) {
        s_rescale[r] = 1.0f;  // the slow path below overwrites it on a rebase
      }
      const bool stale = (m_ref[b] == -INFINITY || v[0] - m_ref[b] > kDriftCap) ||
                         (kPerLane > 1 && v[1] - m_ref[b] > kDriftCap) ||
                         (kPerLane > 2 && v[2] - m_ref[b] > kDriftCap) ||
                         (kPerLane > 3 && v[3] - m_ref[b] > kDriftCap);
      if (__any_sync(0xffffffffull, stale)) {
        float lm = v[0];
#pragma unroll
        for (std::uint32_t i = 1; i < kPerLane; ++i) {
          lm = fmaxf(lm, v[i]);
        }
#pragma unroll
        for (int off = 8; off > 0; off >>= 1) {
          lm = fmaxf(lm, __shfl_xor(lm, off));
        }
        const bool seeding = (m_ref[b] == -INFINITY);
        const float rb = seeding ? 1.0f : __expf(m_ref[b] - lm);
        if (!seeding) {
          l[b] *= rb;
          s_rescale[r] = rb;
        }
        m_ref[b] = lm;
        s_rebase_tile = tile_id;
      }
      float p[kPerLane];
      float tile_sum = 0.0f;
#pragma unroll
      for (std::uint32_t i = 0; i < kPerLane; ++i) {
        p[i] = (v[i] == -INFINITY ? 0.0f : __expf(v[i] - m_ref[b]));
        tile_sum += p[i];
      }
#pragma unroll
      for (int off = 8; off > 0; off >>= 1) {
        tile_sum += __shfl_xor(tile_sum, off);
      }
      l[b] += tile_sum;
      // Row-major fp16, zero past ntile: the PV's A fragment reads a whole
      // 16-key step regardless of the tile tail, and p = 0 makes the stale V
      // slots there harmless.
      if constexpr (kPerLane == 2) {
        *reinterpret_cast<__half2*>(&s_p[r][lane16 * 2]) =
            __halves2half2(__float2half(p[0]), __float2half(p[1]));
      } else if constexpr (kPerLane == 4) {
        *reinterpret_cast<__half2*>(&s_p[r][lane16 * 4]) =
            __halves2half2(__float2half(p[0]), __float2half(p[1]));
        *reinterpret_cast<__half2*>(&s_p[r][lane16 * 4 + 2]) =
            __halves2half2(__float2half(p[2]), __float2half(p[3]));
      } else {
        s_p[r][lane16] = __float2half(p[0]);
      }
      // s_rescale is 1.0 except on rebase tiles, where the slow path above
      // already published the factor.
    }
    __syncthreads();

    // PV: lane owns dims lane*4..lane*4+4 of each of this wave's rows. The key
    // loop is outside the row loop on purpose: the V vector is 256 B per key
    // and re-reading it per row made this the kernel's largest LDS stream
    // (1 KB per key per wave for 256 B of data). Loaded once, applied to all of
    // the wave's rows from registers.
    // Four keys in flight: all the loads for the group are issued before any
    // arithmetic, so one key's LDS latency overlaps the others' FMAs. The loop
    // is latency-bound (halving the KV bytes moves it 2.5%, but every LDS
    // instruction removed lands), and one key at a time exposed a full round
    // trip per key.
    // PV on the matrix core: 4 WMMA per warp per tile instead of ~700 scalar
    // instructions. A is row-major fp16 P (lane = row, 16 keys), B is the
    // transposed V (lane = dim, 16 keys); C element e is (row rf*16 + 2e +
    // half_id, dim df*16 + sub_lane) and the rescale below applies the rows'
    // online-softmax factors.
#pragma unroll
    for (std::uint32_t p = 0; p < kPairsPerWarp; ++p) {
      const std::uint32_t pidx = pair0 + p;
      const std::uint32_t rf = pidx / kDvFrags;
      const std::uint32_t df = pidx % kDvFrags;
      // One stamp read covers the whole warp: the row factors are only
      // non-unity on a tile where some row rebased.
      if (s_rebase_tile == tile_id) {
        float f[8];
#pragma unroll
        for (std::uint32_t e = 0; e < 8; ++e) {
          f[e] = s_rescale[rf * 16 + 2 * e + half_id];
        }
#pragma unroll
        for (std::uint32_t e = 0; e < 8; ++e) {
          c_acc[p][e] *= f[e];
        }
      }
      for (std::uint32_t k0 = 0; k0 < ntile; k0 += 16) {
        v16h a, b;
        __builtin_memcpy(&a, &s_p[rf * 16 + sub_lane][k0], 32);
        __builtin_memcpy(&b, &s_v[df * 16 + sub_lane][k0], 32);
        c_acc[p] = Wmma(a, b, c_acc[p]);
      }
    }
    // No trailing barrier: the next iteration opens with one, and the epilogue
    // below only touches registers and global memory.
  }

  __syncthreads();  // the last tile's s_v reads are done; s_rescale is reusable
  // Same two-rows-per-warp ownership: lane 0 of each half writes its row.
#pragma unroll
  for (std::uint32_t pass = 0; pass < kRowsPerWave / 2; ++pass) {
    const std::uint32_t b = 2 * pass + (lane >> 4);
    const std::uint32_t r = q0 + b;
    if ((lane & 15) != 0) {
      continue;
    }
    if (!livev[b]) {
      s_rescale[r] = 0.0f;
      s_final[r] = 0.0f;
      continue;
    }
    // Both the sink term and the partials live at the row's reference scale, so
    // no accumulator rescale is needed here either.
    if (sinks != nullptr) {
      l[b] += __expf(sinks[h] - m_ref[b]);
    }
    s_rescale[r] = 1.0f;
    s_final[r] = 1.0f / l[b];
    if (splits > 1) {  // this lane is the single writer for its row
      // The merge kernel wants [d_v partials][m][l] per (row, head, split); m
      // must be the accumulator's reference, not the running max.
      float* part =
          partials +
          ((static_cast<std::size_t>(t0 + r) * heads + h) * splits + split) *
              (kMimoVLanes * 32 + 2);
      part[kMimoVLanes * 32] = m_ref[b];
      part[kMimoVLanes * 32 + 1] = l[b];
    }
  }
  __syncthreads();

#pragma unroll
  for (std::uint32_t p = 0; p < kPairsPerWarp; ++p) {
    const std::uint32_t pidx = pair0 + p;
    const std::uint32_t rf = pidx / kDvFrags;
    const std::uint32_t df = pidx % kDvFrags;
    // Merge layout first: [d_v partials][m][l] per (row, head, split), the same
    // one MimoAttentionMergeKernel consumes for the one-query form. The partial
    // index is just the dim index, which is all that kernel reads.
    if (splits > 1) {
#pragma unroll
      for (std::uint32_t e = 0; e < 8; ++e) {
        const std::uint32_t r = rf * 16 + 2 * e + half_id;
        const std::uint32_t dim = df * 16 + sub_lane;
        if (dim >= kDv) {
          continue;
        }
        float* part =
            partials +
            ((static_cast<std::size_t>(t0 + r) * heads + h) * splits + split) *
                (kMimoVLanes * 32 + 2);
        part[dim] = c_acc[p][e] * s_rescale[r];  // 1.0 unless the row rebased
      }
      continue;
    }
#pragma unroll
    for (std::uint32_t e = 0; e < 8; ++e) {
      const std::uint32_t r = rf * 16 + 2 * e + half_id;
      const std::uint32_t t = t0 + r;
      const std::uint32_t dim = df * 16 + sub_lane;
      if (t < n_tokens && dim < d_v) {
        out[(static_cast<std::size_t>(t) * heads + h) * d_v + dim] =
            c_acc[p][e] * s_rescale[r] * s_final[r];
      }
    }
  }
}

/// Launch wrapper: hipLaunchKernelGGL is a macro, so a comma inside a template
/// argument list would split its own arguments; the native <<<>>> form used
/// here keeps the two tile shapes readable at the call sites.
template <std::uint32_t kRows, std::uint32_t kKeys>
inline void LaunchWideAttention(
    dim3 grid, const float* q, const __half* k_cache, const __half* v_cache,
    const float* sinks, float* out, const std::uint32_t* start_pos,
    std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t d_qk,
    std::uint32_t d_v, std::uint32_t window, std::uint32_t ring_rows,
    std::uint32_t n_tokens, const SlotKv* slot_kv, std::uint32_t slot,
    float* partials, std::uint32_t splits, hipStream_t stream) {
  MimoBatchAttentionWmmaKernel<kRows, kKeys>
      <<<grid, dim3(WideAttnThreads<kRows>::value), 0, stream>>>(
          q, k_cache, v_cache, sinks, out, start_pos, heads, kv_heads, d_qk,
          d_v, window, ring_rows, n_tokens, slot_kv, slot, partials, splits);
}

__global__ void MimoAttentionMergeKernel(const float* partials, float* out,
                                         std::uint32_t splits,
                                         std::uint32_t n_tokens,
                                         std::uint32_t heads,
                                         std::uint32_t d_v) {
  constexpr std::uint32_t kVLanes = kMimoVLanes;
  const std::uint32_t h = blockIdx.x;
  const std::uint32_t t = blockIdx.y;
  const std::uint32_t lane = threadIdx.x & 31u;
  // The lanes of one wave cover the row; the other waves would only rewrite
  // the same values.
  if ((threadIdx.x >> 5u) != 0) {
    return;
  }
  float acc[kVLanes] = {0.0f, 0.0f, 0.0f, 0.0f};
  float m = -INFINITY;
  float l = 0.0f;
  for (std::uint32_t z = 0; z < splits; ++z) {
    const float* part =
        partials +
        ((static_cast<std::size_t>(t) * heads + h) * splits + z) *
            (kVLanes * 32 + 2);
    const float mz = part[kVLanes * 32];
    const float lz = part[kVLanes * 32 + 1];
    const float m_new = fmaxf(m, mz);
    const float rz = m == -INFINITY ? 0.0f : __expf(m - m_new);
    const float rzp = mz == -INFINITY ? 0.0f : __expf(mz - m_new);
    l = l * rz + lz * rzp;
#pragma unroll
    for (std::uint32_t x = 0; x < kVLanes; ++x) {
      acc[x] = acc[x] * rz + part[(lane * kVLanes) + x] * rzp;
    }
    m = m_new;
  }
  const float inv = 1.0f / l;
#pragma unroll
  for (std::uint32_t x = 0; x < kVLanes; ++x) {
    const std::uint32_t idx = (lane * kVLanes) + x;
    if (idx < d_v) {
      out[(static_cast<std::size_t>(t) * heads + h) * d_v + idx] = acc[x] * inv;
    }
  }
}

/// Escape hatch for the shared-KV decode form: GUFO_ATTN_SHARED=0 keeps the
/// per-head kernel, so the two can be A/B'd from one binary.
static bool SharedAttentionEnabled() {
  static const bool enabled = [] {
    const char* env = getenv("GUFO_ATTN_SHARED");
    return env == nullptr || std::strtol(env, nullptr, 10) != 0;
  }();
  return enabled;
}

/// Shapes the shared-KV kernel covers: the production attention geometry in
/// fp16, global and sliding-window layers alike (the window slices the range
/// and the ring maps the rows). The int8 twin has no shared form.
static bool SharedAttentionShape(std::uint32_t window, const float* sinks,
                                 std::uint32_t heads, std::uint32_t kv_heads,
                                 std::uint32_t d_qk, std::uint32_t d_v,
                                 const std::int8_t* k_cache_q) {
  (void)window;
  (void)sinks;
  if (k_cache_q != nullptr) {
    return false;
  }
  if (d_qk != 32u * kMimoKLanes || d_v != 32u * kMimoVLanes ||
      kv_heads == 0u || heads % kv_heads != 0u) {
    return false;
  }
  const std::uint32_t group = heads / kv_heads;
  return group == kMimoBlockWaves * 2u || group == kMimoBlockWaves;
}

/// Launch the shared form plus its merge. The split count is sized here from
/// the visible range (one split per eight keys, capped at the merge's
/// budget): the block count of this form is kv_heads x rows x splits, so the
/// splits carry the parallelism alone, and the probe put the cap at or near
/// the optimum from 1K keys up (65K: 32 splits 0.83ms per layer against
/// 2.13ms at 8 and 7.17ms for the deployed one-split per-head form). A
/// sliding layer's range is bounded by its window, and without a partials
/// budget the form runs unsplit: kv_heads x rows blocks each walking at most
/// `window` keys, which still streams 1/(heads/kv_heads) of the per-head
/// bytes (128-key windows measured 132us per layer per-head at six rows).
static void LaunchSharedAttention(
    const float* q, const __half* k_cache, const __half* v_cache,
    const float* sinks, float* out, float* partials, std::uint32_t n_tokens,
    const std::uint32_t* start_pos, const std::uint32_t* row_pos,
    std::uint32_t heads, std::uint32_t kv_heads, std::uint32_t d_v,
    std::uint32_t window, std::uint32_t visible_keys, std::uint32_t ring_rows,
    hipStream_t stream, const SlotKv* slot_kv, std::uint32_t slot,
    const std::uint32_t* row_slot) {
  static const std::uint32_t forced = [] {
    const char* env = getenv("GUFO_ATTN_SHARED_SPLITS");
    const long v = env != nullptr ? std::strtol(env, nullptr, 10) : -1;
    return static_cast<std::uint32_t>(v > 0 ? v : 0);
  }();
  const std::uint32_t range =
      window != 0 && visible_keys > window ? window : visible_keys;
  const std::uint32_t want = forced != 0 ? forced : (range + 7u) / 8u;
  const std::uint32_t z =
      partials == nullptr
          ? 1u
          : (want < 1u ? 1u
                       : (want > kAttnSplitsCap ? kAttnSplitsCap : want));
  const std::uint32_t group = heads / kv_heads;
  const dim3 grid(kv_heads, n_tokens, z);
  const dim3 block(kThreads);
  if (group == kMimoBlockWaves) {
    hipLaunchKernelGGL(MimoAttentionSharedKernel<1>, grid, block, 0, stream, q,
                       k_cache, v_cache, sinks, out, partials, z, start_pos,
                       heads, kv_heads, window, ring_rows, slot_kv, slot,
                       row_slot, row_pos);
  } else {
    hipLaunchKernelGGL(MimoAttentionSharedKernel<2>, grid, block, 0, stream, q,
                       k_cache, v_cache, sinks, out, partials, z, start_pos,
                       heads, kv_heads, window, ring_rows, slot_kv, slot,
                       row_slot, row_pos);
  }
  if (z > 1) {
    hipLaunchKernelGGL(MimoAttentionMergeKernel, dim3(heads, n_tokens),
                       dim3(kThreads), 0, stream, partials, out, z, n_tokens,
                       heads, d_v);
    SyncTrace("Attention", stream);
  }
}

/// Row count from which a narrow batch takes the 16-row WMMA tile instead of
/// the one-query kernel. The packed two-halves loads made the one-query form
/// 1.76x faster (35K: 1.12 vs 1.97ms per layer), which moved the crossover:
/// at 35K a two-row WMMA tile costs 5.3ms against 2 x 1.12ms per-row. The
/// wide tile only wins once its 16-row amortisation covers that gap, and it
/// stays the only option for >= 16 rows.
static std::uint32_t WideAttentionMin() {
  static const std::uint32_t rows = [] {
    const char* env = getenv("GUFO_ATTN_WIDE_MIN");
    const long v = env != nullptr ? std::strtol(env, nullptr, 10) : -1;
    return static_cast<std::uint32_t>(v >= 0 ? v : kFbMQ);
  }();
  return rows;
}

/// Grouped key loads in the one-query kernel (see the K pass comment).
/// Measured neutral at 35K (1.606 vs 1.603ms per layer, n=1), so it stays
/// off by default; GUFO_ATTN_BATCH_KEYS=1 turns it on for experiments.
static std::uint32_t BatchKeyLoads() {
  static const std::uint32_t batched = [] {
    const char* env = getenv("GUFO_ATTN_BATCH_KEYS");
    if (env == nullptr) {
      return 0u;
    }
    return std::strtol(env, nullptr, 10) != 0 ? 1u : 0u;
  }();
  return batched;
}

/// Packed two-halves-per-load K/V reads in the one-query kernel; 0 forces the
/// scalar form for A/B (GUFO_ATTN_SCALAR=1).
static std::uint32_t PackedKvLoads() {
  static const std::uint32_t packed = getenv("GUFO_ATTN_SCALAR") == nullptr
                                          ? 1u
                                          : 0u;
  return packed;
}

/// Quality probe: rounds the K/V reads to `bits` mantissa bits, standing in
/// for a per-channel int8 cache (7 bits) without changing the storage. 0
/// leaves the kernel bit-identical to the unquantized build.
static std::uint32_t KvSimBits() {
  static const std::uint32_t bits = [] {
    const char* env = getenv("GUFO_ATTN_KV_SIM_BITS");
    const long v = env != nullptr ? std::strtol(env, nullptr, 10) : 0;
    return static_cast<std::uint32_t>(v > 0 && v < 10 ? v : 0);
  }();
  return bits;
}

void Attention(const float* q, const __half* k_cache, const __half* v_cache,
               const float* sinks, float* out, float* partials,
               std::uint32_t splits, std::uint32_t wide_splits,
               std::uint32_t n_tokens,
               const std::uint32_t* start_pos, std::uint32_t heads,
               std::uint32_t kv_heads, std::uint32_t d_qk, std::uint32_t d_v,
               std::uint32_t window, std::uint32_t ring_rows,
               std::uint32_t visible_keys, hipStream_t stream,
               const std::int8_t* k_cache_q,
               const std::int8_t* v_cache_q, const __half* k_scales,
               const __half* v_scales) {
  // v1 keeps one split per row; the partials path stays for prefill tuning.
  const std::uint32_t z = partials != nullptr ? splits : 1;
  // A block stages each key/value tile once, so a cohort's rows read their
  // context once instead of once per row. Prefill takes the 32-row shape: the
  // blocks per head are what re-reads the context at depth, so halving them
  // halves the KV stream (one 2048-row chunk at 61K measured 15.3s).
  if (z == 1 && n_tokens >= WideAttentionMin()) {
    const std::uint32_t tile = WideTileRows();
    if (tile == kFbMRows64 && n_tokens >= kFbMRows64) {
      LaunchWideAttention<kFbMRows64, kFbKPrefill64>(
          dim3(heads, (n_tokens + kFbMRows64 - 1) / kFbMRows64), q, k_cache,
          v_cache, sinks, out, start_pos, heads, kv_heads, d_qk, d_v, window,
          ring_rows, n_tokens, nullptr, 0u, nullptr, 1u, stream);
      return;
    }
    if (tile == kFbMQ && WideTileKeys16()) {
      LaunchWideAttention<kFbMQ, kFbMQ>(
          dim3(heads, (n_tokens + kFbMQ - 1) / kFbMQ), q, k_cache, v_cache,
          sinks, out, start_pos, heads, kv_heads, d_qk, d_v, window, ring_rows,
          n_tokens, nullptr, 0u, nullptr, 1u, stream);
      return;
    }
    if (tile == kFbMQ) {
      LaunchWideAttention<kFbMQ, kFbKT>(
          dim3(heads, (n_tokens + kFbMQ - 1) / kFbMQ), q, k_cache, v_cache,
          sinks, out, start_pos, heads, kv_heads, d_qk, d_v, window, ring_rows,
          n_tokens, nullptr, 0u, nullptr, 1u, stream);
      return;
    }
    if (WideTileKeys16() && n_tokens >= kFbMRows) {
      LaunchWideAttention<kFbMRows, kFbMQ>(
          dim3(heads, (n_tokens + kFbMRows - 1) / kFbMRows), q, k_cache,
          v_cache, sinks, out, start_pos, heads, kv_heads, d_qk, d_v, window,
          ring_rows, n_tokens, nullptr, 0u, nullptr, 1u, stream);
      return;
    }
    if (n_tokens >= kFbMRows) {
      LaunchWideAttention<kFbMRows, kFbKPrefill>(
          dim3(heads, (n_tokens + kFbMRows - 1) / kFbMRows), q, k_cache, v_cache,
          sinks, out, start_pos, heads, kv_heads, d_qk, d_v, window, ring_rows,
          n_tokens, nullptr, 0u, nullptr, 1u, stream);
      return;
    }
    LaunchWideAttention<kFbMQ, kFbKT>(
        dim3(heads, (n_tokens + kFbMQ - 1) / kFbMQ), q, k_cache, v_cache, sinks,
        out, start_pos, heads, kv_heads, d_qk, d_v, window, ring_rows, n_tokens,
        nullptr, 0u, nullptr, 1u, stream);
    return;
  }
  if (wide_splits > 1 && n_tokens >= 2 && n_tokens <= kFbMQ) {
    // Narrow batch at depth: split the key range over the one 16-row tile.
    // Below the depth gate the caller passes one split and the wide branch is
    // skipped entirely, so shallow contexts keep the single-block form.
    LaunchWideAttention<kFbMQ, kFbKT>(
        dim3(heads, 1, wide_splits), q, k_cache, v_cache, sinks, out, start_pos,
        heads, kv_heads, d_qk, d_v, window, ring_rows, n_tokens, nullptr, 0u,
        partials, wide_splits, stream);
    hipLaunchKernelGGL(MimoAttentionMergeKernel, dim3(heads, n_tokens),
                       dim3(kThreads), 0, stream, partials, out, wide_splits,
                       n_tokens, heads, d_v);
    SyncTrace("Attention", stream);
    return;
  }
  // Shared-KV decode form: one block per kv head stages the key rows once
  // instead of once per head, which is where the depth profile put the global
  // layers' cost. Global layers ride the split path (`partials`, the same
  // depth gate the per-head split form uses); sliding layers take the form
  // unsplit at any depth, since a window's range is short and each row would
  // otherwise stream it once per head.
  if (SharedAttentionEnabled() && (partials != nullptr || window != 0) &&
      SharedAttentionShape(window, sinks, heads, kv_heads, d_qk, d_v,
                           k_cache_q)) {
    LaunchSharedAttention(q, k_cache, v_cache, sinks, out, partials, n_tokens,
                          start_pos, nullptr, heads, kv_heads, d_v, window,
                          visible_keys, ring_rows, stream, nullptr, 0u,
                          nullptr);
    return;
  }
  hipLaunchKernelGGL(
      MimoAttentionKernel, dim3(heads, n_tokens, z), dim3(kThreads), 0, stream,
      q, k_cache, v_cache, sinks, out, partials, z, start_pos, heads, kv_heads,
      d_qk, d_v, window, ring_rows, nullptr, 0u, nullptr, nullptr,
      PackedKvLoads(), BatchKeyLoads(), KvSimBits(), k_cache_q, v_cache_q,
      k_scales, v_scales, KvInt8MinKeysHost());
  if (z > 1) {
    hipLaunchKernelGGL(MimoAttentionMergeKernel, dim3(heads, n_tokens),
                       dim3(kThreads), 0, stream, partials, out, z, n_tokens,
                       heads, d_v);
    SyncTrace("Attention", stream);
}
}

void AttentionSlot(const float* q, const SlotKv* kv, std::uint32_t slot,
                   const float* sinks, float* out, float* partials,
                   std::uint32_t splits, std::uint32_t n_tokens,
                   const std::uint32_t* start_pos, std::uint32_t heads,
                   std::uint32_t kv_heads, std::uint32_t d_qk,
                   std::uint32_t d_v, std::uint32_t window,
                   std::uint32_t visible_keys, hipStream_t stream) {
  const std::uint32_t tile = WideTileRows();
  if (tile == kFbMRows64 && n_tokens >= kFbMRows64) {
    LaunchWideAttention<kFbMRows64, kFbKPrefill64>(
        dim3(heads, (n_tokens + kFbMRows64 - 1) / kFbMRows64), q, nullptr,
        nullptr, sinks, out, start_pos, heads, kv_heads, d_qk, d_v, window, 0u,
        n_tokens, kv, slot, nullptr, 1u, stream);
    return;
  }
  if (tile == kFbMQ && WideTileKeys16()) {
    LaunchWideAttention<kFbMQ, kFbMQ>(
        dim3(heads, (n_tokens + kFbMQ - 1) / kFbMQ), q, nullptr, nullptr, sinks,
        out, start_pos, heads, kv_heads, d_qk, d_v, window, 0u, n_tokens, kv,
        slot, nullptr, 1u, stream);
    return;
  }
  if (tile == kFbMQ) {
    LaunchWideAttention<kFbMQ, kFbKT>(
        dim3(heads, (n_tokens + kFbMQ - 1) / kFbMQ), q, nullptr, nullptr, sinks,
        out, start_pos, heads, kv_heads, d_qk, d_v, window, 0u, n_tokens, kv,
        slot, nullptr, 1u, stream);
    return;
  }
  if (WideTileKeys16() && n_tokens >= kFbMRows) {
    LaunchWideAttention<kFbMRows, kFbMQ>(
        dim3(heads, (n_tokens + kFbMRows - 1) / kFbMRows), q, nullptr, nullptr,
        sinks, out, start_pos, heads, kv_heads, d_qk, d_v, window, 0u, n_tokens,
        kv, slot, nullptr, 1u, stream);
    return;
  }
  if (n_tokens >= kFbMRows) {
    LaunchWideAttention<kFbMRows, kFbKPrefill>(
        dim3(heads, (n_tokens + kFbMRows - 1) / kFbMRows), q, nullptr, nullptr,
        sinks, out, start_pos, heads, kv_heads, d_qk, d_v, window, 0u, n_tokens,
        kv, slot, nullptr, 1u, stream);
    return;
  }
  if (n_tokens >= 16) {
    LaunchWideAttention<kFbMQ, kFbKT>(
        dim3(heads, (n_tokens + kFbMQ - 1) / kFbMQ), q, nullptr, nullptr, sinks,
        out, start_pos, heads, kv_heads, d_qk, d_v, window, 0u, n_tokens, kv,
        slot, nullptr, 1u, stream);
    return;
  }
  // A single-token decode at depth gets its parallelism from the split count
  // alone, so the shared-KV form is the one that scales: the per-head form
  // would launch one block per head for the whole range. Sliding layers take
  // it unsplit at any depth (short range, per-head bytes are the cost).
  const std::uint32_t z = partials != nullptr ? splits : 1u;
  if (SharedAttentionEnabled() && (partials != nullptr || window != 0) &&
      SharedAttentionShape(window, sinks, heads, kv_heads, d_qk, d_v,
                           nullptr)) {
    LaunchSharedAttention(q, nullptr, nullptr, sinks, out, partials, n_tokens,
                          start_pos, nullptr, heads, kv_heads, d_v, window,
                          visible_keys, 0u, stream, kv, slot, nullptr);
    return;
  }
  hipLaunchKernelGGL(
      MimoAttentionKernel, dim3(heads, n_tokens, z), dim3(kThreads), 0, stream,
      q, nullptr, nullptr, sinks, out, partials, z, start_pos, heads, kv_heads,
      d_qk, d_v, window, 0u, kv, slot, nullptr, nullptr, PackedKvLoads(),
      BatchKeyLoads(), KvSimBits(), nullptr, nullptr, nullptr, nullptr,
      KvInt8MinKeysHost());
  if (z > 1) {
    hipLaunchKernelGGL(MimoAttentionMergeKernel, dim3(heads, n_tokens),
                       dim3(kThreads), 0, stream, partials, out, z, n_tokens,
                       heads, d_v);
    SyncTrace("Attention", stream);
  }
}

void AttentionRows(const float* q, const SlotKv* kv,
                   const std::uint32_t* row_slot,
                   const std::uint32_t* row_pos, const float* sinks,
                   float* out, float* partials, std::uint32_t splits,
                   std::uint32_t rows, std::uint32_t heads,
                   std::uint32_t kv_heads, std::uint32_t d_qk,
                   std::uint32_t d_v, std::uint32_t window,
                   std::uint32_t visible_keys, hipStream_t stream) {
  // One launch for a whole cohort step: every row reads its own session's
  // caches at its own position, so six two-row launches become one. Each row
  // walks its own visible range, so `splits` shortens a deep row's serial key
  // loop without telling the shallow rows which range to cover.
  const std::uint32_t z = partials != nullptr ? splits : 1u;
  if (SharedAttentionEnabled() && (partials != nullptr || window != 0) &&
      SharedAttentionShape(window, sinks, heads, kv_heads, d_qk, d_v,
                           nullptr)) {
    LaunchSharedAttention(q, nullptr, nullptr, sinks, out, partials, rows,
                          nullptr, row_pos, heads, kv_heads, d_v, window,
                          visible_keys, 0u, stream, kv, 0u, row_slot);
    return;
  }
  hipLaunchKernelGGL(
      MimoAttentionKernel, dim3(heads, rows, z), dim3(kThreads), 0, stream, q,
      nullptr, nullptr, sinks, out, partials, z, nullptr, heads, kv_heads,
      d_qk, d_v, window, 0u, kv, 0u, row_slot, row_pos, PackedKvLoads(),
      BatchKeyLoads(), KvSimBits(), nullptr, nullptr, nullptr, nullptr,
      KvInt8MinKeysHost());
  if (z > 1) {
    hipLaunchKernelGGL(MimoAttentionMergeKernel, dim3(heads, rows),
                       dim3(kThreads), 0, stream, partials, out, z, rows, heads,
                       d_v);
    SyncTrace("Attention", stream);
  }
}

void PrepareQkv(const float* q, const float* k, const float* v, float* q_roped,
                __half* k_cache, __half* v_cache, std::uint32_t n_tokens,
                std::uint32_t heads, std::uint32_t kv_heads,
                std::uint32_t d_qk, std::uint32_t d_v,
                std::uint32_t rotary_dim, std::uint32_t ring_rows,
                const std::uint32_t* start_pos, float theta,
                hipStream_t stream, std::int8_t* k_cache_q,
                std::int8_t* v_cache_q, __half* k_scales, __half* v_scales) {
  // The query rows are roped in place; callers pass q_roped == q.
  Rope(const_cast<float*>(q), n_tokens, heads, d_qk, rotary_dim, start_pos,
       theta, stream);
  if (q_roped != nullptr && q_roped != q) {
    hipMemcpyAsync(q_roped, q,
                   static_cast<std::size_t>(n_tokens) * heads * d_qk *
                       sizeof(float),
                   hipMemcpyDeviceToDevice, stream);
  }
  Rope(const_cast<float*>(k), n_tokens, kv_heads, d_qk, rotary_dim, start_pos,
       theta, stream);
  StoreKv(k, k_cache, n_tokens, kv_heads * d_qk, ring_rows, start_pos, stream,
          k_cache_q, k_scales);
  StoreKv(v, v_cache, n_tokens, kv_heads * d_v, ring_rows, start_pos, stream,
          v_cache_q, v_scales);
  SyncTrace("PrepareQkv", stream);
}

void PrepareQkvSlot(const float* q, const float* k, const float* v,
                    const SlotKv* kv, std::uint32_t slot,
                    std::uint32_t n_tokens, std::uint32_t heads,
                    std::uint32_t kv_heads, std::uint32_t d_qk,
                    std::uint32_t d_v, std::uint32_t rotary_dim,
                    const std::uint32_t* start_pos, float theta,
                    hipStream_t stream, const std::uint32_t* row_slot,
                    const std::uint32_t* row_pos) {
  // A fused row batch (row_slot/row_pos set) ropes and stores every row into
  // its own session's cache at its own position; without them the batch is a
  // single session's, whose rows run consecutively from *start_pos.
  Rope(const_cast<float*>(q), n_tokens, heads, d_qk, rotary_dim, start_pos,
       theta, stream, row_pos);
  Rope(const_cast<float*>(k), n_tokens, kv_heads, d_qk, rotary_dim, start_pos,
       theta, stream, row_pos);
  StoreKvSlot(k, kv, slot, /*v_side=*/false, n_tokens, kv_heads * d_qk,
              start_pos, stream, row_slot, row_pos);
  StoreKvSlot(v, kv, slot, /*v_side=*/true, n_tokens, kv_heads * d_v,
              start_pos, stream, row_slot, row_pos);
  SyncTrace("PrepareQkv", stream);
}

// ---------------------------------------------------------------------------
// Sigmoid router: unbiased probabilities are the expert weights, the biased
// scores pick the top-k (DeepSeek-V3 semantics, llama.cpp SIGMOID gating).
// ---------------------------------------------------------------------------

/// Warp top-k: one warp per token, eight experts per lane, k rounds of a warp
/// max-reduce. The serial form below scans 256 shared scores per pick from a
/// single thread (2048 shared reads per token); at a six-row batch cycle that
/// was 35us/layer of pure latency. The reduce's tie-break keeps the lowest
/// expert id (the serial scan walks ascending with a strict >), so ids and
/// normalized weights come out bit-identical — the probe checks that,
/// duplicate logits included.
__global__ __launch_bounds__(32) void RouterSigmoidWarpKernel(
    const float* __restrict__ logits, std::uint32_t stride,
    const float* __restrict__ bias, std::int32_t* __restrict__ ids,
    float* __restrict__ weights, std::uint32_t n_experts, std::uint32_t k) {
  const std::uint32_t lane = threadIdx.x & 31;
  const std::uint32_t t = blockIdx.x;
  float sc[8];
  float pr[8];
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const std::uint32_t e = lane * 8 + j;
    const bool live = e < n_experts;
    const float lg =
        live ? logits[static_cast<std::size_t>(t) * stride + e] : -INFINITY;
    pr[j] = live ? 1.0F / (1.0F + __expf(-lg)) : 0.0F;
    sc[j] = live ? (bias != nullptr ? pr[j] + bias[e] : pr[j]) : -INFINITY;
  }
  const std::uint32_t picks = k < 8 ? k : 8;
  float total = 0.0F;
  for (std::uint32_t s = 0; s < picks; ++s) {
    float best = -INFINITY;
    int best_j = 0;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (sc[j] > best) {
        best = sc[j];
        best_j = j;
      }
    }
    std::uint32_t best_e = lane * 8 + static_cast<std::uint32_t>(best_j);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const float o_best = __shfl_xor(best, off);
      const std::uint32_t o_e = __shfl_xor(best_e, off);
      const bool take = o_best > best || (o_best == best && o_e < best_e);
      best = take ? o_best : best;
      best_e = take ? o_e : best_e;
    }
    const std::uint32_t win_lane = best_e >> 3;
    const float w_local = lane == win_lane ? pr[best_e & 7u] : 0.0F;
    const float w_win = __shfl_sync(~0ull, w_local, win_lane);
    if (lane == win_lane) {
      sc[best_e & 7u] = -INFINITY;
    }
    if (lane == 0) {
      ids[t * k + s] = static_cast<std::int32_t>(best_e);
      weights[t * k + s] = best > -INFINITY ? w_win : 0.0F;
      total += weights[t * k + s];
    }
  }
  if (lane == 0) {
    const float inv = total > 0.0F ? 1.0F / total : 0.0F;
    for (std::uint32_t s = 0; s < picks; ++s) {
      weights[t * k + s] *= inv;
    }
  }
}

__global__ void RouterSigmoidKernel(const float* logits, std::uint32_t stride,
                                    const float* bias, std::int32_t* ids,
                                    float* weights, std::uint32_t n_experts,
                                    std::uint32_t k) {
  // Bring-up-safe selection: thread 0 serializes the top-k over a shared
  // score array. 256 x 8 scalar iterations per token is trivially in-bounds
  // and deterministic; parallelize once the forward pass is proven.
  __shared__ float score[256];
  __shared__ float prob[256];
  __shared__ float picked[8];
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t e = threadIdx.x;
  const bool active = e < n_experts && n_experts <= 256;
  const float logit =
      active ? logits[static_cast<std::size_t>(t) * stride + e] : 0.0f;
  const float p = active ? 1.0f / (1.0f + __expf(-logit)) : 0.0f;
  score[e] = (active && bias != nullptr) ? p + bias[e] : p;
  prob[e] = p;
  __syncthreads();
  if (threadIdx.x != 0) {
    return;
  }
  const std::uint32_t picks = k < 8 ? k : 8;
  std::uint32_t chosen[8];
  float chosen_w[8];
  float total = 0.0f;
  for (std::uint32_t s = 0; s < picks; ++s) {
    std::uint32_t best = 0xFFFFFFFFu;
    float best_score = -INFINITY;
    for (std::uint32_t j = 0; j < n_experts && j < 256; ++j) {
      if (score[j] > best_score) {
        best_score = score[j];
        best = j;
      }
    }
    if (best == 0xFFFFFFFFu || best_score == -INFINITY) {
      chosen[s] = 0;
      chosen_w[s] = 0.0f;
      continue;
    }
    score[best] = -INFINITY;
    chosen[s] = best;
    chosen_w[s] = prob[best];
    total += prob[best];
  }
  const float inv = total > 0.0f ? 1.0f / total : 0.0f;
  for (std::uint32_t s = 0; s < picks; ++s) {
    ids[t * k + s] = static_cast<std::int32_t>(chosen[s]);
    weights[t * k + s] = chosen_w[s] * inv;
  }
  picked[0] = total;  // keep the shared array referenced for debug reads
}


void RouterTopKSigmoid(const float* logits, std::uint32_t stride,
                       const float* bias, std::int32_t* ids, float* weights,
                       std::uint32_t n_tokens, std::uint32_t n_experts,
                       std::uint32_t k, hipStream_t stream) {
  // The warp form covers eight experts per lane over one warp, so it needs
  // the production 256-expert shape; anything wider falls back to the serial
  // scan (bit-identical on ids and weights, checked by the router probe).
  static const bool router_legacy = getenv("GUFO_ROUTER_LEGACY") != nullptr;
  if (!router_legacy && n_experts <= 256) {
    hipLaunchKernelGGL(RouterSigmoidWarpKernel, dim3(n_tokens), dim3(32), 0,
                       stream, logits, stride, bias, ids, weights, n_experts,
                       k);
  } else {
    hipLaunchKernelGGL(RouterSigmoidKernel, dim3(n_tokens), dim3(kThreads), 0,
                       stream, logits, stride, bias, ids, weights, n_experts,
                       k);
  }
  SyncTrace("RouterTopKSigmoid", stream);
}

// ---------------------------------------------------------------------------
// MoE epilogue: weighted sum of the selected expert rows (no shared expert).
// ---------------------------------------------------------------------------

template <typename ExpertT>
__global__ void MimoMoeEpilogueKernel(const ExpertT* expert_out,
                                      const float* weights, float* out,
                                      std::uint32_t k, std::uint32_t dim) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t i = threadIdx.x;
  float acc = 0.0f;
  for (std::uint32_t s = 0; s < k; ++s) {
    if constexpr (std::is_same_v<ExpertT, float>) {
      acc += weights[t * k + s] *
             expert_out[(static_cast<std::size_t>(t) * k + s) * dim + i];
    } else {
      acc += weights[t * k + s] *
             __half2float(
                 expert_out[(static_cast<std::size_t>(t) * k + s) * dim + i]);
    }
  }
  out[static_cast<std::size_t>(t) * dim + i] = acc;
}

template <typename ExpertT>
__global__ void MimoMoeEpilogueVec4Kernel(const ExpertT* expert_out,
                                          const float* weights, float* out,
                                          std::uint32_t k, std::uint32_t dim) {
  const std::uint32_t t = blockIdx.x;
  const std::uint32_t i = threadIdx.x * 4;
  float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
  for (std::uint32_t s = 0; s < k; ++s) {
    const float w = weights[t * k + s];
    const ExpertT* row =
        expert_out + (static_cast<std::size_t>(t) * k + s) * dim + i;
    float4 v;
    if constexpr (std::is_same_v<ExpertT, float>) {
      v = *reinterpret_cast<const float4*>(row);
    } else {
      const uint2 packed = *reinterpret_cast<const uint2*>(row);
      const __half2* h2 = reinterpret_cast<const __half2*>(&packed);
      const float2 a = __half22float2(h2[0]);
      const float2 b = __half22float2(h2[1]);
      v = {a.x, a.y, b.x, b.y};
    }
    acc.x += w * v.x;
    acc.y += w * v.y;
    acc.z += w * v.z;
    acc.w += w * v.w;
  }
  *reinterpret_cast<float4*>(out + static_cast<std::size_t>(t) * dim + i) = acc;
}

void MoeEpilogue(const float* expert_out, const float* weights, float* out,
                 std::uint32_t n_tokens, std::uint32_t k, std::uint32_t dim,
                 hipStream_t stream) {
  hipLaunchKernelGGL(MimoMoeEpilogueKernel<float>, dim3(n_tokens),
                     dim3(dim), 0, stream, expert_out, weights, out, k, dim);
}

void MoeEpilogueVec4(const float* expert_out, const float* weights,
                     float* out, std::uint32_t n_tokens, std::uint32_t k,
                     std::uint32_t dim, hipStream_t stream) {
  if (dim % 4 != 0) {
    MoeEpilogue(expert_out, weights, out, n_tokens, k, dim, stream);
    return;
  }
  hipLaunchKernelGGL(MimoMoeEpilogueVec4Kernel<float>, dim3(n_tokens),
                     dim3(dim / 4), 0, stream, expert_out, weights, out, k,
                     dim);
}

__global__ void KvQuantizeHalfKernel(const __half* cache, std::int8_t* cache_q,
                                     __half* scales, std::uint32_t row_dim) {
  const std::uint32_t row = blockIdx.x;
  if (row_dim % kKvQuantBlock != 0) {
    return;
  }
  const std::uint32_t warp = threadIdx.x >> 5u;
  const std::uint32_t lane = threadIdx.x & 31u;
  const std::uint32_t warps = blockDim.x >> 5u;
  const std::uint32_t blocks = row_dim / kKvQuantBlock;
  for (std::uint32_t block = warp; block < blocks; block += warps) {
    const std::size_t base =
        static_cast<std::size_t>(row) * row_dim + block * kKvQuantBlock + lane;
    const float value = __half2float(cache[base]);
    float best = fabsf(value);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      best = fmaxf(best, __shfl_xor(best, offset));
    }
    const float scale = best / 127.0f;
    const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    cache_q[base] = static_cast<std::int8_t>(__float2int_rn(value * inv));
    if (lane == 0) {
      scales[static_cast<std::size_t>(row) * blocks + block] =
          __float2half(scale);
    }
  }
}

void KvQuantizeFromHalf(const __half* cache, std::int8_t* cache_q,
                        __half* scales, std::uint32_t rows,
                        std::uint32_t row_dim, hipStream_t stream) {
  if (cache == nullptr || cache_q == nullptr || scales == nullptr || rows == 0 ||
      row_dim == 0 || row_dim % kKvQuantBlock != 0) {
    return;
  }
  hipLaunchKernelGGL(KvQuantizeHalfKernel, dim3(rows), dim3(kThreads), 0,
                     stream, cache, cache_q, scales, row_dim);
}

__global__ void MxUpscaleF16Kernel(__half* x, std::size_t count) {
  const std::size_t i =
      blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
  if (i < count) {
    x[i] = __float2half(__half2float(x[i]) * 16.0F);
  }
}

void MxUpscaleF16(__half* x, std::size_t count, hipStream_t stream) {
  if (count != 0) {
    hipLaunchKernelGGL(MxUpscaleF16Kernel, dim3(Blocks(count)), dim3(kThreads),
                       0, stream, x, count);
  }
}

void MoeEpilogueVec4F16(const __half* expert_out, const float* weights,
                        float* out, std::uint32_t n_tokens, std::uint32_t k,
                        std::uint32_t dim, hipStream_t stream) {
  if (dim % 4 != 0) {
    // Scalar fallback reading the F16 rows element-wise.
    hipLaunchKernelGGL(MimoMoeEpilogueKernel<__half>, dim3(n_tokens),
                       dim3(dim), 0, stream, expert_out, weights, out, k, dim);
    return;
  }
  hipLaunchKernelGGL(MimoMoeEpilogueVec4Kernel<__half>, dim3(n_tokens),
                     dim3(dim / 4), 0, stream, expert_out, weights, out, k,
                     dim);
}

}  // namespace gufo::models::mimo2::rocm
