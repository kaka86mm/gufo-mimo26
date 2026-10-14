#ifndef GUFO_MODELS_MIMO2_CONFIG_HPP_
#define GUFO_MODELS_MIMO2_CONFIG_HPP_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::mimo2 {

/// Architecture parameters of a `mimo2` GGUF artifact (Xiaomi MiMo-V2.6).
/// Every value is read from the file; the fixed structure below only bounds
/// what this runtime was written for: a 48-layer trunk (1 dense-FFN layer
/// followed by 47 routed-MoE layers) whose attention alternates global GQA
/// with sliding-window GQA, plus up to three appended NextN (MTP) blocks.
///
/// The attention geometry is uniform across layers except for the KV head
/// count and the window type, which vary per layer: sliding layers carry more
/// KV heads and see only the trailing `sliding_window` positions. Query/key
/// heads are `head_dim_qk` wide while value heads are `head_dim_v` wide; the
/// two differ in this architecture.
struct Config {
  std::uint32_t num_layers{0};      ///< Trunk layers (MTP blocks excluded).
  std::uint32_t num_layers_all{0};  ///< block_count, includes the MTP blocks.
  std::uint32_t nextn_layers{0};    ///< Appended MTP blocks (0..3).

  std::uint32_t hidden_size{0};     ///< n_embd.
  std::uint32_t vocab_size{0};      ///< Filled from token_embd by the binder.
  std::uint32_t context_length{0};  ///< Trained context (1M on V2.6-Flash).
  float rms_eps{1e-6F};

  // Attention: shared geometry.
  std::uint32_t num_heads{0};       ///< Query heads (64).
  std::uint32_t head_dim_qk{0};     ///< Query/key head width (192).
  std::uint32_t head_dim_v{0};      ///< Value head width (128).
  std::uint32_t rotary_dim{0};      ///< rope.dimension_count.
  float rope_theta{10000.0F};       ///< Global-attention layers.
  float rope_theta_swa{0.0F};       ///< SWA layers; 0 falls back to rope_theta.
  float value_scale{1.0F};          ///< Post-attention multiply; 1 disables.
  std::uint32_t sliding_window{0};  ///< Trailing positions SWA layers see.

  // Attention: per-layer, `num_layers_all` entries each. One byte per layer
  // from attention.sliding_window.pattern (1 = sliding window).
  std::vector<std::uint8_t> is_swa;
  std::vector<std::uint32_t> kv_heads;

  // Feed-forward.
  std::uint32_t dense_ff{0};        ///< Dense layers (layer 0, MTP blocks).
  std::uint32_t num_experts{0};     ///< Routed experts (256).
  std::uint32_t num_experts_used{0};///< Experts per token (8).
  std::uint32_t expert_ff{0};       ///< Expert intermediate width (uniform).
  bool router_bias{false};          ///< ffn_exp_probs_b present (set at bind).

  /// An MTP sidecar must agree with the trunk on everything the draft shares.
  [[nodiscard]] bool MtpMatches(const Config& trunk) const noexcept;

  [[nodiscard]] std::uint32_t QDim() const noexcept {
    return num_heads * head_dim_qk;
  }
  /// K cache row width of a layer (its KV count times the key head width).
  [[nodiscard]] std::uint32_t KvDim(std::uint32_t layer) const noexcept {
    return kv_heads[layer] * head_dim_qk;
  }
  /// V cache row width of a layer (KV count times the value head width).
  [[nodiscard]] std::uint32_t VDim(std::uint32_t layer) const noexcept {
    return kv_heads[layer] * head_dim_v;
  }
  [[nodiscard]] bool LayerIsSwa(std::uint32_t layer) const noexcept {
    return layer < is_swa.size() && is_swa[layer] != 0;
  }
  [[nodiscard]] float RopeThetaFor(std::uint32_t layer) const noexcept {
    return LayerIsSwa(layer) && rope_theta_swa > 0.0F ? rope_theta_swa
                                                      : rope_theta;
  }
  /// First position of `layer` visible to a query at `position`; the
  /// exclusive end is always position + 1. SWA layers clamp to the window.
  [[nodiscard]] std::uint32_t VisibleBegin(std::uint32_t layer,
                                           std::uint32_t position) const noexcept {
    return LayerIsSwa(layer)
               ? (position + 1 > sliding_window ? position + 1 - sliding_window
                                                : 0)
               : 0;
  }

  /// Reads and cross-checks the hyperparameters. `require_trunk` only changes
  /// the error wording; the MTP sidecar repeats the trunk keys.
  [[nodiscard]] static std::optional<Config> FromGguf(
      const core::GgufReader& gguf, bool require_trunk,
      std::string* error_msg);
};

}  // namespace gufo::models::mimo2

#endif  // GUFO_MODELS_MIMO2_CONFIG_HPP_
