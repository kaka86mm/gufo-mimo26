// exl3_moe.hip.cpp -- EXL3 (trellis) routed experts for the MiMo-V2.6 engine.
//
// Port of the validated block in ~/exl3_moe_gemv.hip.cpp (window 37: bit-exact
// against the kyojin reference on the production shape) into the engine's
// shapes and dispatch. Per rank (TpPartition holds its share of every expert):
//
//   gate/up  trellis [experts, hidden/16, ff_share/16, bpb], suh [e, hidden],
//            svh [e, ff_share]
//   down     trellis [experts, ff_share/16, hidden/16, bpb], suh [e, ff_share],
//            svh [e, hidden]
//
//   K1   InputTransform  u[slot][side][k] = H128(x_token . suh_side_expert)
//   K2   GateUpSwiGLU    gate/up GEMV over the rank's kt tiles, per-slab
//                        output transform (H128 + svh), act = silu(gate)*up
//   K2b  DownInput       au[slot][i] = H128(act[slot] . suh_down_expert)
//   K3   DownScatter     down GEMV over the rank's kt tiles, per-slab output
//                        transform (H128 + svh), routing-weighted accumulate
//                        into the token row (the TP boundary all-reduce adds
//                        the peer's partial)
//
// Two bitrates appear in the sidecar (bpb 32 = K 2.0, bpb 40 = K 2.5); each
// gets its own ring table set, selected per launch from the tensor's bpb.
// The decode is the fast path from exl3_gemv_bench2 (word-table window
// extraction + the single SAD byte sum): numerically identical to the bit
// walk, 573 G weights/s on this die.
//
// Accumulation is float and sequential over k (the window-37 harness used
// fp64 for its own tolerance check); the Hadamard normalisation is 1/sqrt(128).

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>

#include "src/models/mimo2/kernels/rocm/exl3_moe.hpp"

namespace gufo::models::mimo2::rocm {
namespace {

/// Ring geometry for one bitrate: `start[p]` is the first bit of ring position
/// p's 16-bit window (frac.cu: S(p) = (p>>4)*bpb + sum_{j<=p&15} D(j) - 16).
/// `perm` maps a ring position to its row-major tile element and is shared by
/// both bitrates.
struct RingTables {
  int word_off[256];
  int word_next[256];
  int shift[256];
  int perm[256];
  int L;        ///< tile bit length (16 * bpb)
  int words32;  ///< 32-bit words per tile (L / 32)
};

__device__ __constant__ RingTables kRing[2];
__device__ __constant__ float kRingInv[2];
__device__ __constant__ float kRingBias[2];

__host__ __device__ __forceinline__ float BitsToFloat(unsigned short bits) {
  return __half2float(__ushort_as_half(bits));
}

RingTables BuildRing(int bpb) {
  RingTables t{};
  int bits[16];
  int total = 0;
  for (int j = 0; j < 16; ++j) {
    // Integer K: every ring position carries bpb/16 bits. Half-integer
    // K = KA + 0.5: odd ring positions carry KA + 1 (decode mask 0xAAAA).
    bits[j] = (bpb % 16 == 0) ? (bpb / 16) : (bpb / 16 + ((0xAAAA >> j) & 1));
    total += bits[j];
  }
  t.L = 16 * total;
  if (t.L != 16 * bpb) {
    fprintf(stderr, "EXL3 ring: bpb %d sums to %d bits, expected %d\n", bpb,
            total, bpb);
    std::abort();
  }
  t.words32 = t.L >> 5;
  for (int p = 0; p < 256; ++p) {
    int s = (p >> 4) * bpb;
    for (int j = 0; j <= (p & 15); ++j) {
      s += bits[j];
    }
    const int start = (s - 16 + t.L) % t.L;
    t.word_off[p] = start >> 5;
    t.word_next[p] = (t.word_off[p] + 1 >= t.words32) ? 0 : (t.word_off[p] + 1);
    t.shift[p] = start & 31;
  }
  int ring_to_rm[256];
  for (int q = 0; q < 32; ++q) {
    const int r0 = (q % 4) * 2, c0 = q / 4;
    ring_to_rm[q * 8 + 0] = r0 * 16 + c0;
    ring_to_rm[q * 8 + 1] = (r0 + 1) * 16 + c0;
    ring_to_rm[q * 8 + 2] = (r0 + 8) * 16 + c0;
    ring_to_rm[q * 8 + 3] = (r0 + 9) * 16 + c0;
    ring_to_rm[q * 8 + 4] = r0 * 16 + c0 + 8;
    ring_to_rm[q * 8 + 5] = (r0 + 1) * 16 + c0 + 8;
    ring_to_rm[q * 8 + 6] = (r0 + 8) * 16 + c0 + 8;
    ring_to_rm[q * 8 + 7] = (r0 + 9) * 16 + c0 + 8;
  }
  for (int i = 0; i < 256; ++i) {
    t.perm[ring_to_rm[i]] = i;
  }
  return t;
}

bool EnsureTables(std::string* error_msg) {
  static std::once_flag once;
  static bool ok = true;
  std::call_once(once, [&] {
    const int bpbs[2] = {32, 40};
    for (int i = 0; i < 2 && ok; ++i) {
      const RingTables t = BuildRing(bpbs[i]);
      const float inv = BitsToFloat(0x1EEE);
      const float bias = BitsToFloat(0xC931);
      if (hipMemcpyToSymbol(kRing, &t, sizeof(t), i * sizeof(RingTables),
                            hipMemcpyHostToDevice) != hipSuccess ||
          hipMemcpyToSymbol(kRingInv, &inv, sizeof(float), i * sizeof(float),
                            hipMemcpyHostToDevice) != hipSuccess ||
          hipMemcpyToSymbol(kRingBias, &bias, sizeof(float), i * sizeof(float),
                            hipMemcpyHostToDevice) != hipSuccess) {
        ok = false;
      }
    }
  });
  if (!ok && error_msg != nullptr) {
    *error_msg = "EXL3 ring table upload failed";
  }
  return ok;
}

__device__ __forceinline__ float DecodeWeight(const unsigned short* tile,
                                              int i, int ring) {
  const RingTables& rt = kRing[ring];
  const int p = rt.perm[i];
  const unsigned a = static_cast<unsigned>(tile[rt.word_off[p] * 2]) |
                     (static_cast<unsigned>(tile[rt.word_off[p] * 2 + 1]) << 16);
  const unsigned b = static_cast<unsigned>(tile[rt.word_next[p] * 2]) |
                     (static_cast<unsigned>(tile[rt.word_next[p] * 2 + 1]) << 16);
  const unsigned long long stream =
      (static_cast<unsigned long long>(a) << 32) | b;
  const unsigned window =
      static_cast<unsigned>((stream >> (48 - rt.shift[p])) & 0xFFFFULL);
  const unsigned x = window * 0x83DCD12DU;
  const unsigned s = __builtin_amdgcn_sad_u8(x, 0u, 0u) + 0x6400U;
  const float h = __half2float(__ushort_as_half(static_cast<unsigned short>(s)));
  return fmaf(h, kRingInv[ring], kRingBias[ring]);
}

/// In-place normalised Sylvester Hadamard of one 128-block in shared memory.
__device__ void Hadamard128(float* v, int lane) {
  for (int step = 1; step < 128; step <<= 1) {
    const float a = v[lane];
    const float b = v[lane ^ step];
    __syncthreads();
    v[lane] = ((lane & step) != 0) ? (b - a) : (a + b);
    __syncthreads();
  }
  v[lane] *= 0.08838834765F;
}

/// K1: u[slot][side][k] = H128(x_token . suh_side_expert) per 128-block.
/// grid = (hidden/128, slots*2), block = 128; side = blockIdx.y & 1.
__global__ void Exl3InputTransform(const float* __restrict__ x, int hidden,
                                   const __half* __restrict__ suh_gate,
                                   const __half* __restrict__ suh_up,
                                   const std::int32_t* __restrict__ ids,
                                   int used, __half* __restrict__ u) {
  __shared__ float v[128];
  const int lane = threadIdx.x;
  const int which = blockIdx.y;  // slot * 2 + side
  const int slot = which >> 1;
  const int token = slot / used;
  const int expert = ids[slot];
  const int base = blockIdx.x * 128;
  const __half* s =
      ((which & 1) != 0 ? suh_up : suh_gate) +
      static_cast<std::size_t>(expert) * hidden + base;
  const float* row = x + static_cast<std::size_t>(token) * hidden + base;
  v[lane] = row[lane] * __half2float(s[lane]);
  __syncthreads();
  Hadamard128(v, lane);
  u[static_cast<std::size_t>(which) * hidden + base + lane] =
      __float2half_rn(v[lane]);
}

/// K2: act[slot][j] = silu(gate) * up for one 128-wide slab of the rank's ff.
/// grid = (ff_share/128, slots), block = 128.
__global__ void Exl3GateUpSwiGLU(
    const __half* __restrict__ u, int hidden,
    const unsigned short* __restrict__ trellis_gate,
    const unsigned short* __restrict__ trellis_up, int nt, int bpb,
    const __half* __restrict__ svh_gate, const __half* __restrict__ svh_up,
    const std::int32_t* __restrict__ ids, int used, int ff_share, int ring,
    float* __restrict__ act) {
  __shared__ float vg[128];
  __shared__ float vu[128];
  const int lane = threadIdx.x;
  const int slot = blockIdx.y;
  const int expert = ids[slot];
  const int ktiles = hidden >> 4;
  const int j = blockIdx.x * 128 + lane;
  const int jt = j >> 4;
  const int jm = j & 15;
  // One tile per (kt, jt) cell; jt is fixed per lane, so the tile walks kt.
  const unsigned short* tg =
      trellis_gate + (static_cast<std::size_t>(expert) * ktiles * nt + jt) * bpb;
  const unsigned short* tu =
      trellis_up + (static_cast<std::size_t>(expert) * ktiles * nt + jt) * bpb;
  const __half* ug = u + static_cast<std::size_t>(slot * 2) * hidden;
  const __half* uu = ug + hidden;
  float accg = 0.0F;
  float accu = 0.0F;
  if (j < ff_share) {
    for (int kt = 0; kt < ktiles; ++kt) {
      const unsigned short* bg = tg + static_cast<std::size_t>(kt) * nt * bpb;
      const unsigned short* bu = tu + static_cast<std::size_t>(kt) * nt * bpb;
      const __half* xg = ug + (kt << 4);
      const __half* xu = uu + (kt << 4);
#pragma unroll 4
      for (int i = 0; i < 16; ++i) {
        const int idx = (i << 4) | jm;
        accg = fmaf(__half2float(xg[i]), DecodeWeight(bg, idx, ring), accg);
        accu = fmaf(__half2float(xu[i]), DecodeWeight(bu, idx, ring), accu);
      }
    }
  }
  vg[lane] = accg;
  vu[lane] = accu;
  __syncthreads();
  Hadamard128(vg, lane);
  Hadamard128(vu, lane);
  if (j < ff_share) {
    const __half* sg = svh_gate + static_cast<std::size_t>(expert) * ff_share;
    const __half* su = svh_up + static_cast<std::size_t>(expert) * ff_share;
    const float gate = vg[lane] * __half2float(sg[j]);
    const float up = vu[lane] * __half2float(su[j]);
    act[static_cast<std::size_t>(slot) * ff_share + j] =
        gate / (1.0F + expf(-gate)) * up;
  }
}

/// K2b: au[slot][i] = H128(act[slot] . suh_down_expert) per 128-block of ff.
/// grid = (ff_share/128, slots), block = 128.
__global__ void Exl3DownInput(const float* __restrict__ act, int ff_share,
                              const __half* __restrict__ suh,
                              const std::int32_t* __restrict__ ids, int used,
                              __half* __restrict__ au) {
  __shared__ float v[128];
  const int lane = threadIdx.x;
  const int slot = blockIdx.y;
  const int expert = ids[slot];
  const int base = blockIdx.x * 128;
  const __half* s = suh + static_cast<std::size_t>(expert) * ff_share + base;
  const float* a = act + static_cast<std::size_t>(slot) * ff_share + base;
  v[lane] = a[lane] * __half2float(s[lane]);
  __syncthreads();
  Hadamard128(v, lane);
  au[static_cast<std::size_t>(slot) * ff_share + base + lane] =
      __float2half_rn(v[lane]);
}

/// K3: out[token][j] = sum_slot weight_slot * svh_expert[j] . H128(au_slot.w)
/// grid = (hidden/128, n_tokens), block = 128.
__global__ void Exl3DownScatter(
    const __half* __restrict__ au, int ff_share,
    const unsigned short* __restrict__ trellis, int nt, int bpb,
    const __half* __restrict__ svh, const std::int32_t* __restrict__ ids,
    const float* __restrict__ weights, int used, int hidden, int ring,
    float* __restrict__ out) {
  __shared__ float v[128];
  const int lane = threadIdx.x;
  const int token = blockIdx.y;
  const int j = blockIdx.x * 128 + lane;
  const int ktiles = ff_share >> 4;
  const int jt = j >> 4;
  const int jm = j & 15;
  float acc = 0.0F;
  for (int s = 0; s < used; ++s) {
    const int slot = token * used + s;
    const int expert = ids[slot];
    const unsigned short* td =
        trellis + (static_cast<std::size_t>(expert) * ktiles * nt + jt) * bpb;
    const __half* a = au + static_cast<std::size_t>(slot) * ff_share;
    float local = 0.0F;
    for (int kt = 0; kt < ktiles; ++kt) {
      const unsigned short* b = td + static_cast<std::size_t>(kt) * nt * bpb;
      const __half* xa = a + (kt << 4);
#pragma unroll 4
      for (int i = 0; i < 16; ++i) {
        local = fmaf(__half2float(xa[i]), DecodeWeight(b, (i << 4) | jm, ring),
                     local);
      }
    }
    v[lane] = local;
    __syncthreads();
    Hadamard128(v, lane);
    acc += weights[slot] * v[lane] *
           __half2float(svh[static_cast<std::size_t>(expert) * hidden + j]);
    __syncthreads();
  }
  out[static_cast<std::size_t>(token) * hidden + j] = acc;
}

int RingForBpb(std::uint32_t bpb) {
  return bpb == 32 ? 0 : (bpb == 40 ? 1 : -1);
}

}  // namespace

bool LaunchExl3Moe(const Exl3MoeArgs& a, std::string* error_msg) {
  const auto fail = [&](const std::string& m) {
    if (error_msg != nullptr) {
      *error_msg = m;
    }
    return false;
  };
  if (!EnsureTables(error_msg)) {
    return false;
  }
  const int ring_g = RingForBpb(a.gate.bpb);
  const int ring_d = RingForBpb(a.down.bpb);
  if (ring_g < 0 || ring_d < 0 || a.gate.bpb != a.up.bpb) {
    return fail("EXL3: unexpected bitrate in the sidecar geometry");
  }
  if (a.hidden % 128 != 0 || a.ff_share % 128 != 0 || a.hidden % 16 != 0 ||
      a.ff_share % 16 != 0) {
    return fail("EXL3: hidden/ff must be 128-aligned");
  }
  const std::uint32_t slots = a.n_tokens * a.used;
  if (slots == 0) {
    return true;
  }
  // Grow-only scratch: u [slots*2, hidden] half, au [slots, ff] half,
  // act [slots, ff] float.
  const std::size_t want =
      (static_cast<std::size_t>(slots) * 2 * a.hidden +
       static_cast<std::size_t>(slots) * a.ff_share * 4) *
      sizeof(__half);
  static void* scratch = nullptr;
  static std::size_t have = 0;
  if (have < want) {
    if (scratch != nullptr) {
      (void)hipFree(scratch);
      scratch = nullptr;
    }
    if (hipMalloc(&scratch, want) != hipSuccess) {
      have = 0;
      return fail("EXL3 scratch allocation failed");
    }
    have = want;
  }
  auto* base = static_cast<unsigned char*>(scratch);
  auto* u = reinterpret_cast<__half*>(base);
  auto* au = u + static_cast<std::size_t>(slots) * 2 * a.hidden;
  auto* act = reinterpret_cast<float*>(au + static_cast<std::size_t>(slots) *
                                                a.ff_share);

  const auto check = [&](hipError_t e, const char* what) {
    return e == hipSuccess ? true : fail(std::string("EXL3 ") + what + ": " +
                                         hipGetErrorString(e));
  };
  const dim3 t128(128);
  // grid.y tops out at 65535: chunk the tokens so a prefill batch's slot count
  // stays inside it (a chunk never splits a token's `used` slots).
  const std::uint32_t per_chunk = 60000 / a.used;
  for (std::uint32_t t0 = 0; t0 < a.n_tokens; t0 += per_chunk) {
    const std::uint32_t tn = std::min(per_chunk, a.n_tokens - t0);
    const std::uint32_t slot0 = t0 * a.used;
    const std::uint32_t slotn = tn * a.used;
    const std::int32_t* ids = a.ids + slot0;
    const float* weights = a.weights + slot0;
    const float* x = a.x + static_cast<std::size_t>(t0) * a.hidden;
    float* out = a.out + static_cast<std::size_t>(t0) * a.hidden;
    auto* uc = u;
    auto* auc = au;
    auto* actc = act;
    if (!check(hipLaunchKernelGGL(Exl3InputTransform,
                                  dim3(a.hidden / 128, slotn * 2), t128, 0,
                                  a.stream, x, static_cast<int>(a.hidden),
                                  a.gate.suh, a.up.suh, ids,
                                  static_cast<int>(a.used), uc),
               "input transform") ||
        !check(hipLaunchKernelGGL(
                   Exl3GateUpSwiGLU, dim3(a.ff_share / 128, slotn), t128, 0,
                   a.stream, uc, static_cast<int>(a.hidden), a.gate.trellis,
                   a.up.trellis, static_cast<int>(a.gate.nt),
                   static_cast<int>(a.gate.bpb), a.gate.svh, a.up.svh, ids,
                   static_cast<int>(a.used), static_cast<int>(a.ff_share),
                   ring_g, actc),
               "gate/up") ||
        !check(hipLaunchKernelGGL(Exl3DownInput, dim3(a.ff_share / 128, slotn),
                                  t128, 0, a.stream, actc,
                                  static_cast<int>(a.ff_share), a.down.suh, ids,
                                  static_cast<int>(a.used), auc),
               "down input") ||
        !check(hipLaunchKernelGGL(
                   Exl3DownScatter, dim3(a.hidden / 128, tn), t128, 0, a.stream,
                   auc, static_cast<int>(a.ff_share), a.down.trellis,
                   static_cast<int>(a.down.nt), static_cast<int>(a.down.bpb),
                   a.down.svh, ids, weights, static_cast<int>(a.used),
                   static_cast<int>(a.hidden), ring_d, out),
               "down")) {
      return false;
    }
  }
  return true;
}

}  // namespace gufo::models::mimo2::rocm