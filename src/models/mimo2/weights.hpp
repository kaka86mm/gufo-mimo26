#ifndef GUFO_MODELS_MIMO2_WEIGHTS_HPP_
#define GUFO_MODELS_MIMO2_WEIGHTS_HPP_

#include <optional>
#include <string_view>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/mimo2/config.hpp"

namespace gufo::models::mimo2 {

/// One GGUF tensor bound to its shard and geometry. `cols` is the input
/// extent (ne[0]); `rows` the output rows; `experts` the ne[2] stack count.
struct TensorRef {
  const void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint64_t cols{0};
  std::uint64_t rows{0};
  std::uint64_t experts{1};
  std::uint32_t shard{0};
  std::uint64_t file_offset{0};
  std::string_view name;

  [[nodiscard]] std::size_t RowBytes() const noexcept;
  [[nodiscard]] std::size_t SizeBytes() const noexcept {
    return RowBytes() * static_cast<std::size_t>(rows) *
           static_cast<std::size_t>(experts);
  }
  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
};

/// One block's weights. Layers 1..num_layers-1 are routed-MoE; layer 0 and
/// the NextN blocks carry a dense FFN instead. `is_nextn` layers hold the
/// eh_proj/enorm/hnorm tensors of the MTP draft.
struct LayerWeights {
  bool is_nextn{false};
  bool dense_ffn{false};

  // Attention. Either the split projections or the fused attn_qkv is bound
  // (the artifact carries one form; MTP blocks always use the fused form).
  TensorRef attn_qkv;
  TensorRef attn_q;
  TensorRef attn_k;
  TensorRef attn_v;
  TensorRef attn_out;
  TensorRef attn_norm;   ///< RMSNorm, f32.
  TensorRef attn_sinks;  ///< Optional per-head attention sinks, f32.

  TensorRef ffn_norm;    ///< RMSNorm, f32.

  // Dense FFN branch (layer 0 and NextN blocks).
  TensorRef ffn_gate;
  TensorRef ffn_up;
  TensorRef ffn_down;

  // Routed branch.
  TensorRef router;        ///< ffn_gate_inp, f32.
  TensorRef router_bias;   ///< Optional ffn_exp_probs_b, f32.
  TensorRef ffn_gate_exps;
  TensorRef ffn_up_exps;
  TensorRef ffn_down_exps;

  // NextN-only tensors. The head shares fall back to the trunk's
  /// output/output_norm when absent.
  TensorRef nextn_eh_proj;
  TensorRef nextn_enorm;
  TensorRef nextn_hnorm;
  TensorRef nextn_embed_tokens;
  TensorRef nextn_shared_head_head;
  TensorRef nextn_shared_head_norm;
  TensorRef layer_out_norm;

  [[nodiscard]] bool HasAttention() const noexcept {
    return !attn_qkv.empty() || !attn_q.empty();
  }
};

struct ModelWeights {
  Config config;
  TensorRef token_embd;
  TensorRef output;       ///< Empty falls back to token_embd (tied head).
  TensorRef output_norm;
  std::vector<LayerWeights> layers;

  [[nodiscard]] static std::optional<ModelWeights> Bind(
      const core::GgufReader& reader, std::string* error_msg);
};

/// The MTP sidecar: the trailing NextN blocks, possibly numbering their
/// layers from zero. Blocks are located by their nextn.eh_proj tensors.
struct MtpWeights {
  Config config;
  std::vector<LayerWeights> blocks;

  [[nodiscard]] static std::optional<MtpWeights> Bind(
      const core::GgufReader& reader, const Config& trunk,
      std::string* error_msg);
};

}  // namespace gufo::models::mimo2

#endif  // GUFO_MODELS_MIMO2_WEIGHTS_HPP_
