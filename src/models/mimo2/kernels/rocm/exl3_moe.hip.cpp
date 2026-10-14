// exl3_moe.hip.cpp -- EXL3 (trellis) routed experts for the MiMo-V2.6 engine.
//
// Row-window decode (rewritten 2026-10-13). The previous version decoded one
// weight at a time through the engine's per-position tables (perm, word_off,
// word_next, shift): four divergent __constant__ lookups per weight, which is
// what capped decode at a fraction of the read wall. This version keeps the
// element <-> ring-position mapping bit-for-bit but hoists the extraction to
// the ring group:
//
//   - one ring group = 16 consecutive ring positions (p = 16g..16g+15); their
//     16-bit windows span <= 61 bits (bpb <= 48), so three consecutive u32
//     words (96 bits) cover them all.
//   - the group's words are assembled into a reversed-order 64-bit window V
//     aligned to the group base (two shifts + or); each of the 16 windows is
//     then one more constant shift + mask. Verified bit-exact against the
//     table decode on real sidecar data (exl3_rw_bench, 0/8192 diffs).
//   - the group's element structure is fixed: its 16 positions are 8 k-rows x
//     2 j-columns, the columns being {c0, c0+8} with c0 = g/2, and each row
//     appears twice per group (once per column). A lane therefore owns the
//     column pair {c0, c0+8} of one tile column jt and keeps only two
//     accumulators per matrix -- the previous shape needed a 16-wide
//     accumulator bank and spilled 544 bytes of scratch per thread.
//
// Measured on bpb48 real sidecar slices (8 experts, gate+up): 542 G weights/s
// at the L2 footprint and 542 G weights/s on an 805 MB DRAM footprint (~86%
// of a 237 GB/s plain-read wall); the table decode measured 147 / 182.
//
// Per rank (TpPartition holds its share of every expert):
//
//   gate/up  trellis [experts, hidden/16, ff_share/16, bpb], suh [e, hidden],
//            svh [e, ff_share]
//   down     trellis [experts, ff_share/16, hidden/16, bpb], suh [e, ff_share],
//            svh [e, hidden]
//
//   K1   InputTransform  u[slot][side][k] = H128(x_token . suh_side_expert)
//   K2   GateUpSwiGLU    gate/up GEMV (col-pair row-window) over the rank's
//                        kt tiles, per-slab output transform (H128 + svh),
//                        act = silu(gate)*up
//   K2b  DownInput       au[slot][i] = H128(act[slot] . suh_down_expert)
//   K3   DownScatter     down GEMV over the rank's kt tiles, per-slab output
//                        transform (H128 + svh), routing-weighted accumulate
//                        into the token row (the TP boundary all-reduce adds
//                        the peer's partial)
//
// Accumulation is float; the k sum is split into two halves (one per chunk
// lane) and re-added in chunk order, so results are deterministic and differ
// from the sequential order only by fp reassociation (~1e-6 on 4K terms).
// The Hadamard normalisation is 1/sqrt(128).

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <string>

#include "src/models/mimo2/kernels/rocm/exl3_moe.hpp"

namespace gufo::models::mimo2::rocm {
namespace {

constexpr int kChunk = 2;         ///< k halves per lane pair (ch = lane & 1)
constexpr int kRowLocal = 8;      ///< j-tile columns per block row (128 j / 16)
constexpr int kSlab = kRowLocal * 16;  ///< ff/hidden outputs per block

__device__ __constant__ float kInv = 0.0F;
__device__ __constant__ float kBias = 0.0F;

__host__ __device__ __forceinline__ float BitsToFloat(unsigned short bits) {
  return __half2float(__ushort_as_half(bits));
}

bool EnsureConstants(std::string* error_msg) {
  static std::once_flag once;
  static bool ok = true;
  std::call_once(once, [&] {
    const float inv = BitsToFloat(0x1EEE);
    const float bias = BitsToFloat(0xC931);
    if (hipMemcpyToSymbol(kInv, &inv, sizeof(float)) != hipSuccess ||
        hipMemcpyToSymbol(kBias, &bias, sizeof(float)) != hipSuccess) {
      ok = false;
    }
  });
  if (!ok && error_msg != nullptr) {
    *error_msg = "EXL3 decode constants upload failed";
  }
  return ok;
}

/// Ring geometry for one bitrate. Integer K carries bpb/16 bits per in-group
/// position; half-integer K = KA + 0.5 gives odd positions one extra bit
/// (0xAAAA mirrors the engine's previous BuildRing).
template <int BPB>
__host__ __device__ constexpr int BitsAt(int j) {
  return (BPB % 16 == 0) ? (BPB / 16) : (BPB / 16 + ((0xAAAA >> j) & 1));
}

/// Bit offset of in-group position c's 16-bit window, relative to the group's
/// first window (sum of the bits consumed by positions 1..c).
template <int BPB>
__host__ __device__ constexpr int OffAt(int c) {
  int s = 0;
  for (int j = 1; j <= c; ++j) {
    s += BitsAt<BPB>(j);
  }
  return s;
}

template <int BPB>
__host__ __device__ constexpr int RingL() {
  return 16 * BPB;
}

/// Source words of ring group g and the in-word alignment shift.
/// start(16g) = g*bpb + bits[0] - 16 (mod L).
template <int BPB>
__device__ __forceinline__ void GroupWords(int g, int* wo, int* wn, int* wn2,
                                           int* sh) {
  constexpr int kWords = RingL<BPB>() / 32;
  int rb = g * BPB + BPB / 16 - 16;
  if (rb < 0) {
    rb += RingL<BPB>();
  }
  const int w = rb >> 5;
  *sh = rb & 31;
  *wo = w;
  *wn = (w + 1 >= kWords) ? 0 : (w + 1);
  *wn2 = (w + 2 >= kWords) ? (w + 2 - kWords) : (w + 2);
}

/// Reversed-order 64-bit view of ring group g, aligned to the group's first
/// window: bit k of the result is stream bit (rowbase + 63 - k), so window c
/// is `(V >> (48 - OffAt<BPB>(c))) & 0xFFFF`.
template <int BPB>
__device__ __forceinline__ unsigned long long RowWindow(const unsigned* w,
                                                        int wo, int wn,
                                                        int wn2, int sh) {
  const unsigned long long lo =
      (static_cast<unsigned long long>(w[wo]) << 32) | w[wn];
  const unsigned long long hi =
      (static_cast<unsigned long long>(w[wn]) << 32) | w[wn2];
  return (lo << sh) | (hi >> (32 - sh));
}

template <int BPB>
__device__ __forceinline__ unsigned RowWin(unsigned long long v, int c) {
  return static_cast<unsigned>((v >> (48 - OffAt<BPB>(c))) & 0xFFFFULL);
}

__device__ __forceinline__ float DecodeWindow(unsigned window) {
  const unsigned x = window * 0x83DCD12DU;
  const unsigned s = __builtin_amdgcn_sad_u8(x, 0u, 0u) + 0x6400U;
  return fmaf(BitsToFloat(static_cast<unsigned short>(s)), kInv, kBias);
}

/// k-row (element row) of in-group position c, per group parity. The two
/// groups of a column pair cover all 16 rows (8 rows each, each row twice).
constexpr int kRowEven[16] = {0, 1, 8, 9, 0, 1, 8, 9,
                              2, 3, 10, 11, 2, 3, 10, 11};
constexpr int kRowOdd[16] = {4, 5, 12, 13, 4, 5, 12, 13,
                             6, 7, 14, 15, 6, 7, 14, 15};

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
/// Block = [jtl 8][c0 8][ch 2] (128 lanes) covers the 128 j of the slab;
/// grid = (ff_share/128, slots).
template <int BPB>
__global__ void Exl3GateUpSwiGLU(
    const __half* __restrict__ u, int hidden,
    const unsigned short* __restrict__ trellis_gate,
    const unsigned short* __restrict__ trellis_up, int nt,
    const __half* __restrict__ svh_gate, const __half* __restrict__ svh_up,
    const std::int32_t* __restrict__ ids, int ff_share,
    float* __restrict__ act) {
  __shared__ float vg[kSlab];
  __shared__ float vu[kSlab];
  __shared__ float sg[kSlab][kChunk];
  __shared__ float su[kSlab][kChunk];
  const int lane = threadIdx.x;
  const int ch = lane & (kChunk - 1);
  const int c0 = (lane >> 1) & 7;
  const int jtl = lane >> 4;
  const int slot = blockIdx.y;
  const int expert = ids[slot];
  const int ktiles = hidden >> 4;
  const int kt0 = ch * (ktiles >> 1);
  const int kt1 = kt0 + (ktiles >> 1);
  const int jt = blockIdx.x * kRowLocal + jtl;
  int woA;
  int wnA;
  int wn2A;
  int shA;
  int woB;
  int wnB;
  int wn2B;
  int shB;
  GroupWords<BPB>(2 * c0, &woA, &wnA, &wn2A, &shA);
  GroupWords<BPB>(2 * c0 + 1, &woB, &wnB, &wn2B, &shB);
  const int tw = nt * BPB / 2;  // u32 words per tile
  const unsigned* wg = reinterpret_cast<const unsigned*>(
      trellis_gate + (static_cast<std::size_t>(expert) * ktiles * nt + jt) * BPB);
  const unsigned* wu = reinterpret_cast<const unsigned*>(
      trellis_up + (static_cast<std::size_t>(expert) * ktiles * nt + jt) * BPB);
  const __half* ug = u + static_cast<std::size_t>(slot * 2) * hidden;
  const __half* uu = ug + hidden;
  float ga0 = 0.0F;
  float ga1 = 0.0F;
  float ua0 = 0.0F;
  float ua1 = 0.0F;
  for (int kt = kt0; kt < kt1; ++kt) {
    const unsigned* gw = wg + static_cast<std::size_t>(kt) * tw;
    const unsigned* uw = wu + static_cast<std::size_t>(kt) * tw;
    const __half* xg = ug + (kt << 4);
    const __half* xu = uu + (kt << 4);
    float xf[32];
#pragma unroll
    for (int r = 0; r < 16; ++r) {
      xf[r] = __half2float(xg[r]);
      xf[16 + r] = __half2float(xu[r]);
    }
    {
      const unsigned long long v = RowWindow<BPB>(gw, woA, wnA, wn2A, shA);
      const unsigned long long vu = RowWindow<BPB>(uw, woA, wnA, wn2A, shA);
#pragma unroll
      for (int cc = 0; cc < 4; ++cc) {
        ga0 = fmaf(xf[kRowEven[cc]], DecodeWindow(RowWin<BPB>(v, cc)), ga0);
        ua0 = fmaf(xf[16 + kRowEven[cc]], DecodeWindow(RowWin<BPB>(vu, cc)),
                   ua0);
      }
#pragma unroll
      for (int cc = 4; cc < 8; ++cc) {
        ga1 = fmaf(xf[kRowEven[cc]], DecodeWindow(RowWin<BPB>(v, cc)), ga1);
        ua1 = fmaf(xf[16 + kRowEven[cc]], DecodeWindow(RowWin<BPB>(vu, cc)),
                   ua1);
      }
#pragma unroll
      for (int cc = 8; cc < 12; ++cc) {
        ga0 = fmaf(xf[kRowEven[cc]], DecodeWindow(RowWin<BPB>(v, cc)), ga0);
        ua0 = fmaf(xf[16 + kRowEven[cc]], DecodeWindow(RowWin<BPB>(vu, cc)),
                   ua0);
      }
#pragma unroll
      for (int cc = 12; cc < 16; ++cc) {
        ga1 = fmaf(xf[kRowEven[cc]], DecodeWindow(RowWin<BPB>(v, cc)), ga1);
        ua1 = fmaf(xf[16 + kRowEven[cc]], DecodeWindow(RowWin<BPB>(vu, cc)),
                   ua1);
      }
    }
    {
      const unsigned long long v = RowWindow<BPB>(gw, woB, wnB, wn2B, shB);
      const unsigned long long vu = RowWindow<BPB>(uw, woB, wnB, wn2B, shB);
#pragma unroll
      for (int cc = 0; cc < 4; ++cc) {
        ga0 = fmaf(xf[kRowOdd[cc]], DecodeWindow(RowWin<BPB>(v, cc)), ga0);
        ua0 = fmaf(xf[16 + kRowOdd[cc]], DecodeWindow(RowWin<BPB>(vu, cc)),
                   ua0);
      }
#pragma unroll
      for (int cc = 4; cc < 8; ++cc) {
        ga1 = fmaf(xf[kRowOdd[cc]], DecodeWindow(RowWin<BPB>(v, cc)), ga1);
        ua1 = fmaf(xf[16 + kRowOdd[cc]], DecodeWindow(RowWin<BPB>(vu, cc)),
                   ua1);
      }
#pragma unroll
      for (int cc = 8; cc < 12; ++cc) {
        ga0 = fmaf(xf[kRowOdd[cc]], DecodeWindow(RowWin<BPB>(v, cc)), ga0);
        ua0 = fmaf(xf[16 + kRowOdd[cc]], DecodeWindow(RowWin<BPB>(vu, cc)),
                   ua0);
      }
#pragma unroll
      for (int cc = 12; cc < 16; ++cc) {
        ga1 = fmaf(xf[kRowOdd[cc]], DecodeWindow(RowWin<BPB>(v, cc)), ga1);
        ua1 = fmaf(xf[16 + kRowOdd[cc]], DecodeWindow(RowWin<BPB>(vu, cc)),
                   ua1);
      }
    }
  }
  // slab reduce over the two k chunks, then the output transform
  const int js = jtl * 16 + c0;
  sg[js][ch] = ga0;
  sg[js + 8][ch] = ga1;
  su[js][ch] = ua0;
  su[js + 8][ch] = ua1;
  __syncthreads();
  if (ch == 0) {
    vg[js] = sg[js][0] + sg[js][1];
    vg[js + 8] = sg[js + 8][0] + sg[js + 8][1];
    vu[js] = su[js][0] + su[js][1];
    vu[js + 8] = su[js + 8][0] + su[js + 8][1];
  }
  __syncthreads();
  Hadamard128(vg, lane);
  Hadamard128(vu, lane);
  const int j = blockIdx.x * kSlab + lane;
  if (j < ff_share) {
    const __half* sgt = svh_gate + static_cast<std::size_t>(expert) * ff_share;
    const __half* sut = svh_up + static_cast<std::size_t>(expert) * ff_share;
    const float gate = vg[lane] * __half2float(sgt[j]);
    const float up = vu[lane] * __half2float(sut[j]);
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
/// Block = [jtl 8][c0 8][ch 2] (128 lanes) covers 128 hidden outputs;
/// grid = (hidden/128, n_tokens).
template <int BPB>
__global__ void Exl3DownScatter(
    const __half* __restrict__ au, int ff_share,
    const unsigned short* __restrict__ trellis, int nt,
    const __half* __restrict__ svh, const std::int32_t* __restrict__ ids,
    const float* __restrict__ weights, int used, int hidden,
    float* __restrict__ out) {
  __shared__ float vd[kSlab];
  __shared__ float sd[kSlab][kChunk];
  const int lane = threadIdx.x;
  const int ch = lane & (kChunk - 1);
  const int c0 = (lane >> 1) & 7;
  const int jtl = lane >> 4;
  const int token = blockIdx.y;
  const int jt = blockIdx.x * kRowLocal + jtl;
  const int ktiles = ff_share >> 4;
  const int kt0 = ch * (ktiles >> 1);
  const int kt1 = kt0 + (ktiles >> 1);
  int woA;
  int wnA;
  int wn2A;
  int shA;
  int woB;
  int wnB;
  int wn2B;
  int shB;
  GroupWords<BPB>(2 * c0, &woA, &wnA, &wn2A, &shA);
  GroupWords<BPB>(2 * c0 + 1, &woB, &wnB, &wn2B, &shB);
  const int tw = nt * BPB / 2;
  float acc = 0.0F;
  for (int s = 0; s < used; ++s) {
    const int slot = token * used + s;
    const int expert = ids[slot];
    const unsigned* wd = reinterpret_cast<const unsigned*>(
        trellis + (static_cast<std::size_t>(expert) * ktiles * nt + jt) * BPB);
    const __half* a = au + static_cast<std::size_t>(slot) * ff_share;
    float l0 = 0.0F;
    float l1 = 0.0F;
    for (int kt = kt0; kt < kt1; ++kt) {
      const unsigned* dw = wd + static_cast<std::size_t>(kt) * tw;
      const __half* xa = a + (kt << 4);
      float xf[16];
#pragma unroll
      for (int r = 0; r < 16; ++r) {
        xf[r] = __half2float(xa[r]);
      }
      {
        const unsigned long long v = RowWindow<BPB>(dw, woA, wnA, wn2A, shA);
#pragma unroll
        for (int cc = 0; cc < 4; ++cc) {
          l0 = fmaf(xf[kRowEven[cc]], DecodeWindow(RowWin<BPB>(v, cc)), l0);
        }
#pragma unroll
        for (int cc = 4; cc < 8; ++cc) {
          l1 = fmaf(xf[kRowEven[cc]], DecodeWindow(RowWin<BPB>(v, cc)), l1);
        }
#pragma unroll
        for (int cc = 8; cc < 12; ++cc) {
          l0 = fmaf(xf[kRowEven[cc]], DecodeWindow(RowWin<BPB>(v, cc)), l0);
        }
#pragma unroll
        for (int cc = 12; cc < 16; ++cc) {
          l1 = fmaf(xf[kRowEven[cc]], DecodeWindow(RowWin<BPB>(v, cc)), l1);
        }
      }
      {
        const unsigned long long v = RowWindow<BPB>(dw, woB, wnB, wn2B, shB);
#pragma unroll
        for (int cc = 0; cc < 4; ++cc) {
          l0 = fmaf(xf[kRowOdd[cc]], DecodeWindow(RowWin<BPB>(v, cc)), l0);
        }
#pragma unroll
        for (int cc = 4; cc < 8; ++cc) {
          l1 = fmaf(xf[kRowOdd[cc]], DecodeWindow(RowWin<BPB>(v, cc)), l1);
        }
#pragma unroll
        for (int cc = 8; cc < 12; ++cc) {
          l0 = fmaf(xf[kRowOdd[cc]], DecodeWindow(RowWin<BPB>(v, cc)), l0);
        }
#pragma unroll
        for (int cc = 12; cc < 16; ++cc) {
          l1 = fmaf(xf[kRowOdd[cc]], DecodeWindow(RowWin<BPB>(v, cc)), l1);
        }
      }
    }
    const int js = jtl * 16 + c0;
    sd[js][ch] = l0;
    sd[js + 8][ch] = l1;
    __syncthreads();
    if (ch == 0) {
      vd[js] = sd[js][0] + sd[js][1];
      vd[js + 8] = sd[js + 8][0] + sd[js + 8][1];
    }
    __syncthreads();
    Hadamard128(vd, lane);
    const int j = blockIdx.x * kSlab + lane;
    if (j < hidden) {
      acc += weights[slot] * vd[lane] *
             __half2float(svh[static_cast<std::size_t>(expert) * hidden + j]);
    }
    __syncthreads();
  }
  const int j = blockIdx.x * kSlab + lane;
  if (j < hidden) {
    out[static_cast<std::size_t>(token) * hidden + j] = acc;
  }
}

int RingForBpb(std::uint32_t bpb) {
  return bpb == 32 ? 0 : (bpb == 40 ? 1 : (bpb == 48 ? 2 : -1));
}

}  // namespace

bool LaunchExl3Moe(const Exl3MoeArgs& a, std::string* error_msg) {
  const auto fail = [&](const std::string& m) {
    if (error_msg != nullptr) {
      *error_msg = m;
    }
    return false;
  };
  if (!EnsureConstants(error_msg)) {
    return false;
  }
  const int ring_g = RingForBpb(a.gate.bpb);
  const int ring_d = RingForBpb(a.down.bpb);
  if (ring_g < 0 || ring_d < 0 || a.gate.bpb != a.up.bpb) {
    return fail("EXL3: unexpected bitrate in the sidecar geometry");
  }
  if (a.hidden % 128 != 0 || a.ff_share % 128 != 0 || a.hidden % 32 != 0 ||
      a.ff_share % 32 != 0) {
    return fail("EXL3: hidden/ff must be 128-aligned and k-tile aligned");
  }
  if ((reinterpret_cast<std::uintptr_t>(a.gate.trellis) & 3) != 0 ||
      (reinterpret_cast<std::uintptr_t>(a.up.trellis) & 3) != 0 ||
      (reinterpret_cast<std::uintptr_t>(a.down.trellis) & 3) != 0) {
    return fail("EXL3: trellis base not 4-byte aligned");
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
    hipLaunchKernelGGL(Exl3InputTransform, dim3(a.hidden / 128, slotn * 2),
                       t128, 0, a.stream, x, static_cast<int>(a.hidden),
                       a.gate.suh, a.up.suh, ids, static_cast<int>(a.used), u);
    if (!check(hipGetLastError(), "input transform")) {
      return false;
    }
    bool ok = true;
    switch (static_cast<int>(a.gate.bpb)) {
      case 32:
        hipLaunchKernelGGL(Exl3GateUpSwiGLU<32>,
                           dim3(a.ff_share / 128, slotn), t128, 0, a.stream, u,
                           static_cast<int>(a.hidden), a.gate.trellis,
                           a.up.trellis, static_cast<int>(a.gate.nt),
                           a.gate.svh, a.up.svh, ids,
                           static_cast<int>(a.ff_share), act);
        ok = check(hipGetLastError(), "gate/up");
        break;
      case 40:
        hipLaunchKernelGGL(Exl3GateUpSwiGLU<40>,
                           dim3(a.ff_share / 128, slotn), t128, 0, a.stream, u,
                           static_cast<int>(a.hidden), a.gate.trellis,
                           a.up.trellis, static_cast<int>(a.gate.nt),
                           a.gate.svh, a.up.svh, ids,
                           static_cast<int>(a.ff_share), act);
        ok = check(hipGetLastError(), "gate/up");
        break;
      default:
        hipLaunchKernelGGL(Exl3GateUpSwiGLU<48>,
                           dim3(a.ff_share / 128, slotn), t128, 0, a.stream, u,
                           static_cast<int>(a.hidden), a.gate.trellis,
                           a.up.trellis, static_cast<int>(a.gate.nt),
                           a.gate.svh, a.up.svh, ids,
                           static_cast<int>(a.ff_share), act);
        ok = check(hipGetLastError(), "gate/up");
        break;
    }
    if (!ok) {
      return false;
    }
    hipLaunchKernelGGL(Exl3DownInput, dim3(a.ff_share / 128, slotn), t128, 0,
                       a.stream, act, static_cast<int>(a.ff_share), a.down.suh,
                       ids, static_cast<int>(a.used), au);
    if (!check(hipGetLastError(), "down input")) {
      return false;
    }
    switch (static_cast<int>(a.down.bpb)) {
      case 32:
        hipLaunchKernelGGL(Exl3DownScatter<32>, dim3(a.hidden / 128, tn), t128,
                           0, a.stream, au, static_cast<int>(a.ff_share),
                           a.down.trellis, static_cast<int>(a.down.nt),
                           a.down.svh, ids, weights, static_cast<int>(a.used),
                           static_cast<int>(a.hidden), out);
        ok = check(hipGetLastError(), "down");
        break;
      case 40:
        hipLaunchKernelGGL(Exl3DownScatter<40>, dim3(a.hidden / 128, tn), t128,
                           0, a.stream, au, static_cast<int>(a.ff_share),
                           a.down.trellis, static_cast<int>(a.down.nt),
                           a.down.svh, ids, weights, static_cast<int>(a.used),
                           static_cast<int>(a.hidden), out);
        ok = check(hipGetLastError(), "down");
        break;
      default:
        hipLaunchKernelGGL(Exl3DownScatter<48>, dim3(a.hidden / 128, tn), t128,
                           0, a.stream, au, static_cast<int>(a.ff_share),
                           a.down.trellis, static_cast<int>(a.down.nt),
                           a.down.svh, ids, weights, static_cast<int>(a.used),
                           static_cast<int>(a.hidden), out);
        ok = check(hipGetLastError(), "down");
        break;
    }
    if (!ok) {
      return false;
    }
  }
  return true;
}

}  // namespace gufo::models::mimo2::rocm