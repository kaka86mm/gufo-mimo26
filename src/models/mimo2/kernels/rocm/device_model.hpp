#ifndef GUFO_MODELS_MIMO2_KERNELS_ROCM_DEVICE_MODEL_HPP_
#define GUFO_MODELS_MIMO2_KERNELS_ROCM_DEVICE_MODEL_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/mimo2/distributed/tp_partition.hpp"
#include "src/models/mimo2/weights.hpp"

namespace gufo::models::mimo2::rocm {

/// One weight resident in device memory, still in its GGUF encoding.
struct DeviceTensor {
  void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint32_t cols{0};
  std::uint32_t rows{0};
  std::uint32_t experts{1};

  [[nodiscard]] bool empty() const noexcept { return data == nullptr; }
  [[nodiscard]] const float* f32() const noexcept {
    return static_cast<const float*>(data);
  }
};

/// One block on the device. Trunk layers are routed-MoE except layer 0; the
/// NextN blocks are dense. Attention geometry (`q_heads`, `kv_heads`) is
/// always this rank's share, so the session state and scratch of a layer
/// follow its DeviceLayer fields rather than a model-wide uniform value.
struct DeviceLayer {
  bool dense_ffn{false};  ///< Dense FFN (layer 0 and NextN blocks).
  bool swa{false};        ///< Trailing-window attention (Config::is_swa).
  float rope_theta{0.0F};

  // Attention: this rank's heads. Query and key heads are head_dim_qk wide,
  // value heads head_dim_v; the projections carry those widths and the
  // caches are laid out per Config::KvDim/VDim.
  std::uint32_t q_heads{0};
  std::uint32_t kv_heads{0};
  /// Under TP the heads are split, so the output projection yields this
  /// rank's partial (summed by the layer-boundary all-reduce). An odd KV
  /// count replicates the whole block instead; peers then contribute a zero
  /// partial. The draft blocks are never split: every rank keeps an
  /// identical whole copy of them.
  bool attn_split{false};

  /// The artifact fuses [Q | K | V] projection rows; the split into three
  /// DeviceTensors happens at upload so the executor always projects Q, K
  /// and V separately (PrepareQkv reads contiguous per-matrix rows).
  DeviceTensor attn_q, attn_k, attn_v;
  /// Input rows are the attention context: this rank's q_heads value heads
  /// (the value width, not the query width).
  DeviceTensor attn_out;
  DeviceTensor attn_norm;   ///< RMSNorm, f32.
  DeviceTensor attn_sinks;  ///< Optional per-head sinks, f32, this rank's
                            ///< heads (the data pointer is already offset);
                            ///< empty on layers without sinks.
  DeviceTensor ffn_norm;    ///< RMSNorm, f32.

  // Dense branch. Under TP the intermediate rows split like the routed
  /// experts (`ffn_split`); when a share would not fall on a quantization
  /// block boundary every rank holds the whole FFN and only rank zero's
  /// output survives the layer-boundary sum (peers zero theirs).
  DeviceTensor ffn_gate, ffn_up, ffn_down;
  bool ffn_split{false};

  // Routed branch.
  DeviceTensor router;       ///< ffn_gate_inp, f32 [hidden -> experts].
  DeviceTensor router_bias;  ///< Optional ffn_exp_probs_b, f32 [experts].
  DeviceTensor ffn_gate_exps, ffn_up_exps, ffn_down_exps;

  // NextN-only. The two head tensors fall back to the trunk's
  /// output/output_norm when empty.
  DeviceTensor nextn_eh_proj;  ///< [2*hidden -> hidden].
  DeviceTensor nextn_enorm, nextn_hnorm;
  DeviceTensor nextn_embed;     ///< Optional private embedding table.
  DeviceTensor nextn_head_head; ///< Optional private lm_head.
  DeviceTensor nextn_head_norm; ///< Optional private head norm.
  DeviceTensor layer_out_norm;  ///< Optional pre-head norm.
};

/// The trunk and the NextN draft blocks uploaded to the GPU. The draft
/// blocks come only from the MTP sidecar file: the trunk artifact declares
/// none (its block_count already excludes them).
class DeviceModel {
public:
  ~DeviceModel();
  DeviceModel(const DeviceModel&) = delete;
  DeviceModel& operator=(const DeviceModel&) = delete;

  /// Streams tensors from the same open files used to bind their metadata.
  /// `mtp`/`mtp_reader` carry the sidecar's own shard table. `dense_sidecar`
  /// (optional) replaces the attention projections with a requantized copy
  /// (tools/mimo2_requant_dense.py); see GraftDenseSidecar.
  [[nodiscard]] static std::unique_ptr<DeviceModel> Upload(
      const ModelWeights& weights, const core::GgufReader& reader,
      const MtpWeights* mtp, const core::GgufReader* mtp_reader,
      std::string* error_msg = nullptr,
      const distributed::TpPartition* partition = nullptr,
      const core::GgufReader* dense_sidecar = nullptr,
      const core::GgufReader* expert_sidecar = nullptr);

  const Config& config() const noexcept { return config_; }
  const DeviceTensor& token_embd() const noexcept { return token_embd_; }
  const DeviceTensor& output() const noexcept { return output_; }
  const DeviceTensor& output_norm() const noexcept { return output_norm_; }
  const std::vector<DeviceLayer>& layers() const noexcept { return layers_; }
  [[nodiscard]] bool has_mtp() const noexcept { return !nextn_.empty(); }
  const std::vector<DeviceLayer>& nextn_layers() const noexcept {
    return nextn_;
  }
  [[nodiscard]] std::size_t resident_bytes() const noexcept { return bytes_; }
  [[nodiscard]] std::uint32_t tp_rank() const noexcept { return tp_rank_; }
  [[nodiscard]] std::uint32_t tp_world_size() const noexcept {
    return tp_world_size_;
  }
  /// Widest K among the BF16/F16 matrices (activation staging for hipBLAS).
  [[nodiscard]] std::size_t max_half_cols() const noexcept {
    return max_half_cols_;
  }
  /// Widest K among the Q8_0 matrices (decode activation quantization).
  [[nodiscard]] std::size_t max_q8_cols() const noexcept {
    return max_q8_cols_;
  }

private:
  DeviceModel() = default;

  Config config_;
  DeviceTensor token_embd_;
  DeviceTensor output_;
  DeviceTensor output_norm_;
  std::vector<DeviceLayer> layers_;
  std::vector<DeviceLayer> nextn_;
  std::vector<void*> allocations_;
  std::size_t bytes_{0};
  std::uint32_t tp_rank_{0};
  std::uint32_t tp_world_size_{1};
  std::size_t max_half_cols_{1};
  std::size_t max_q8_cols_{32};
};

}  // namespace gufo::models::mimo2::rocm

#endif  // GUFO_MODELS_MIMO2_KERNELS_ROCM_DEVICE_MODEL_HPP_
