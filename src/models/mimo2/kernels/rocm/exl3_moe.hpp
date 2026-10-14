// exl3_moe.hpp -- EXL3 routed-expert launch interface (see exl3_moe.hip.cpp).
#ifndef GUFO_MODELS_MIMO2_KERNELS_ROCM_EXL3_MOE_HPP_
#define GUFO_MODELS_MIMO2_KERNELS_ROCM_EXL3_MOE_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <string>

namespace gufo::models::mimo2::rocm {

/// One EXL3 expert tensor set as the graft left it on the device: the trellis
/// words (I16) plus the two Hadamard-domain scales, already sliced to this
/// rank's share of the intermediate dimension.
struct Exl3ExpsView {
  const unsigned short* trellis{nullptr};  ///< [experts, kt, nt, bpb]
  const __half* suh{nullptr};              ///< [experts, in]
  const __half* svh{nullptr};              ///< [experts, out]
  std::uint32_t kt{0};
  std::uint32_t nt{0};
  std::uint32_t bpb{0};
};

struct Exl3MoeArgs {
  /// gate/up: in = hidden, out = the rank's ff share; down: in = ff share,
  /// out = hidden.
  Exl3ExpsView gate;
  Exl3ExpsView up;
  Exl3ExpsView down;
  const float* x{nullptr};             ///< [n_tokens, hidden] f32 rows
  const std::int32_t* ids{nullptr};    ///< [n_tokens * used] expert per slot
  const float* weights{nullptr};       ///< [n_tokens * used] routing weights
  float* out{nullptr};                 ///< [n_tokens, hidden] partial, added
                                       ///< to the peer's by the boundary sum
  std::uint32_t n_tokens{1};
  std::uint32_t used{8};
  std::uint32_t hidden{4096};
  std::uint32_t ff_share{1024};
  hipStream_t stream{nullptr};
};

/// Runs the four EXL3 kernels. Returns false with a message on any failure.
[[nodiscard]] bool LaunchExl3Moe(const Exl3MoeArgs& args,
                                 std::string* error_msg);

}  // namespace gufo::models::mimo2::rocm

#endif  // GUFO_MODELS_MIMO2_KERNELS_ROCM_EXL3_MOE_HPP_