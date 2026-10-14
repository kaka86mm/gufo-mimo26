// Microbench: vendored mimo2_mmq_mxfp4_moe_vec at the MiMo decode shape,
// one rank's share (M=1024 gate/up over K=4096, down 4096x1024, 8 experts).
// Compare against GTT peak to see how far the mmvq path sits from
// bandwidth-bound. Random weights: timing only, no numerics.
//
//   hipcc -O2 -std=c++17 --offload-arch=gfx1151 \
//     mimo2_moe_vec_bench.hip -o /tmp/moebench \
//     <build>/src/models/mimo2/libgufo_mimo2.a -lamdhip64

#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" int mimo2_mmq_mxfp4_moe_vec(
    const void* W, const float* X, const int32_t* ids, float* out, int M,
    int K, int n_tokens, int n_experts, int n_expert_used,
    hipStream_t stream);

#define CK(x)                                                            \
  do {                                                                   \
    const hipError_t e_ = (x);                                           \
    if (e_ != hipSuccess) {                                              \
      fprintf(stderr, "HIP error %s at %d\n", hipGetErrorString(e_), __LINE__); \
      exit(1);                                                           \
    }                                                                    \
  } while (0)

struct Buf {
  void* p{nullptr};
  std::size_t bytes{0};
  void Alloc(std::size_t n) {
    bytes = n;
    CK(hipMalloc(&p, n));
    CK(hipMemset(p, 0, n));
  }
  template <typename T>
  T* as() {
    return static_cast<T*>(p);
  }
  ~Buf() {
    if (p) hipFree(p);
  }
};

static float TimeGemm(const char* what, const void* w, const float* x,
                      const int32_t* ids, float* out, int M, int K,
                      int n_tokens, int n_used, int n_experts,
                      hipStream_t s, int iters) {
  // Warmup (pool growth, first-touch).
  for (int i = 0; i < 5; ++i) {
    if (mimo2_mmq_mxfp4_moe_vec(w, x, ids, out, M, K, n_tokens, n_experts,
                                n_used, s) != 0) {
      fprintf(stderr, "%s failed\n", what);
      exit(1);
    }
  }
  CK(hipStreamSynchronize(s));
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    if (mimo2_mmq_mxfp4_moe_vec(w, x, ids, out, M, K, n_tokens, n_experts,
                                n_used, s) != 0) {
      fprintf(stderr, "%s failed\n", what);
      exit(1);
    }
  }
  CK(hipStreamSynchronize(s));
  const auto t1 = std::chrono::steady_clock::now();
  const double ms =
      std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
  // One MXFP4 block: 32 elements -> 16 B nibbles + 1 B e8m0.
  const double bytes = double(n_used) * M * (K / 32.0) * 17.0;
  printf("%-10s M=%4d K=%4d rows=%2d used=%d  %8.3f ms  %7.1f GB/s\n", what, M,
         K, n_tokens, n_used, ms, bytes / (ms * 1e6));
  return ms;
}

int main(int argc, char** argv) {
  const int iters = argc > 1 ? atoi(argv[1]) : 200;
  const int n_experts = 256;
  const int used = 8;
  hipStream_t s;
  CK(hipStreamCreate(&s));

  // gate/up: every expert row block lives in one tensor [n_experts, M, K/32]
  const std::size_t gu_blocks =
      std::size_t(n_experts) * 1024 * (4096 / 32);
  const std::size_t down_blocks = std::size_t(n_experts) * 4096 * (1024 / 32);
  Buf w_gu, w_down, x_in, x_mid, out;
  w_gu.Alloc(gu_blocks * 17);
  w_down.Alloc(down_blocks * 17);
  x_in.Alloc(4096 * 4);   // one token's hidden
  x_mid.Alloc(1024 * 4 * used);
  out.Alloc(4096 * 4 * 8);

  // Random-ish payload so nothing is a zero page.
  {
    std::vector<unsigned char> host(gu_blocks * 17);
    for (std::size_t i = 0; i < host.size(); ++i) host[i] = (unsigned char)(i * 131 + 17);
    CK(hipMemcpy(w_gu.p, host.data(), host.size(), hipMemcpyHostToDevice));
    host.resize(down_blocks * 17);
    for (std::size_t i = 0; i < host.size(); ++i) host[i] = (unsigned char)(i * 197 + 3);
    CK(hipMemcpy(w_down.p, host.data(), host.size(), hipMemcpyHostToDevice));
    std::vector<float> fx(4096);
    for (int i = 0; i < 4096; ++i) fx[i] = (i % 13) * 0.031f - 0.19f;
    CK(hipMemcpy(x_in.p, fx.data(), fx.size() * 4, hipMemcpyHostToDevice));
  }
  std::vector<int32_t> ids8(used);
  for (int s_i = 0; s_i < used; ++s_i) ids8[s_i] = (s_i * 31) % n_experts;
  Buf d_ids;
  d_ids.Alloc(used * 4);
  CK(hipMemcpy(d_ids.p, ids8.data(), used * 4, hipMemcpyHostToDevice));
  std::vector<int32_t> ids1(used, ids8[0]);
  Buf d_ids1;
  d_ids1.Alloc(used * 4);
  CK(hipMemcpy(d_ids1.p, ids1.data(), used * 4, hipMemcpyHostToDevice));

  printf("== decode shape (1 token), hot cache ==\n");
  const double g = TimeGemm("gate", w_gu.p, x_in.as<float>(), d_ids.as<int32_t>(),
                            out.as<float>(), 1024, 4096, 1, 8, n_experts, s, iters);
  const double u = TimeGemm("up", w_gu.p, x_in.as<float>(), d_ids.as<int32_t>(),
                            x_mid.as<float>(), 1024, 4096, 1, 8, n_experts, s, iters);
  const double d = TimeGemm("down", w_down.p, x_mid.as<float>(), d_ids1.as<int32_t>(),
                            out.as<float>(), 4096, 1024, 8, 1, n_experts, s, iters);

  // Cold-data pass: rotate through distinct weight copies so L2 cannot hold
  // the working set; approximates production where all 47 layers read
  // distinct experts every step.
  const int nlayers = 5;
  std::vector<Buf> gu_set(nlayers), down_set(nlayers);
  for (int i = 0; i < nlayers; ++i) {
    gu_set[i].Alloc(gu_blocks * 17);
    down_set[i].Alloc(down_blocks * 17);
    CK(hipMemcpy(gu_set[i].p, w_gu.p, gu_blocks * 17, hipMemcpyDeviceToDevice));
    CK(hipMemcpy(down_set[i].p, w_down.p, down_blocks * 17, hipMemcpyDeviceToDevice));
  }
  for (int i = 0; i < 5; ++i) {
    if (mimo2_mmq_mxfp4_moe_vec(gu_set[0].p, x_in.as<float>(), d_ids.as<int32_t>(),
                                out.as<float>(), 1024, 4096, 1, n_experts, 8, s) != 0)
      exit(1);
  }
  CK(hipStreamSynchronize(s));
  const auto c0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) {
    const int li = i % nlayers;
    if (mimo2_mmq_mxfp4_moe_vec(gu_set[li].p, x_in.as<float>(), d_ids.as<int32_t>(),
                                out.as<float>(), 1024, 4096, 1, n_experts, 8, s) != 0)
      exit(1);
    if (mimo2_mmq_mxfp4_moe_vec(gu_set[li].p, x_in.as<float>(), d_ids.as<int32_t>(),
                                x_mid.as<float>(), 1024, 4096, 1, n_experts, 8, s) != 0)
      exit(1);
    if (mimo2_mmq_mxfp4_moe_vec(down_set[li].p, x_mid.as<float>(), d_ids1.as<int32_t>(),
                                out.as<float>(), 4096, 1024, 8, n_experts, 1, s) != 0)
      exit(1);
  }
  CK(hipStreamSynchronize(s));
  const auto c1 = std::chrono::steady_clock::now();
  const double cold = std::chrono::duration<double, std::milli>(c1 - c0).count() / iters;
  const double used_bytes = 2.0 * 8 * 1024.0 * 128 * 17 + 8 * 4096.0 * 32 * 17;
  printf("COLD triple/layer %.3f ms  (%.1f MB live weights -> %.1f GB/s)\n",
         cold, used_bytes / 1e6, used_bytes / (cold * 1e6));

  printf("== verify shape (4 tokens) ==\n");
  Buf x4, mid4, out4;
  x4.Alloc(4096 * 4 * 4);
  mid4.Alloc(1024 * 4 * 4 * 2);
  out4.Alloc(4096 * 4 * 4);
  std::vector<int32_t> ids4(4 * used);
  for (int t = 0; t < 4; ++t)
    for (int s_i = 0; s_i < used; ++s_i) ids4[t * used + s_i] = ((t * 7 + s_i * 31) % n_experts);
  Buf d_ids4;
  d_ids4.Alloc(ids4.size() * 4);
  CK(hipMemcpy(d_ids4.p, ids4.data(), ids4.size() * 4, hipMemcpyHostToDevice));
  const double g4 = TimeGemm("gate4", w_gu.p, x4.as<float>(), d_ids4.as<int32_t>(),
                             out4.as<float>(), 1024, 4096, 4, 8, n_experts, s, iters);
  const double u4 = TimeGemm("up4", w_gu.p, x4.as<float>(), d_ids4.as<int32_t>(),
                             mid4.as<float>(), 1024, 4096, 4, 8, n_experts, s, iters);
  const double d4 = TimeGemm("down4", w_down.p, mid4.as<float>(), d_ids4.as<int32_t>(),
                             out4.as<float>(), 4096, 1024, 32, 1, n_experts, s, iters);
  printf("total/layer %.3f ms  (%.2f MB/token -> %.1f GB/s effective)\n", g4 + u4 + d4,
         (gu_blocks * 2 + down_blocks) * 17.0 / 4 / 1e6,
         (gu_blocks * 2 + down_blocks) * 17.0 / ((g4 + u4 + d4) * 1e6));
  return 0;
}
