#ifndef GUFO_MODELS_MIMO2_DFLASH_V1_HPP_
#define GUFO_MODELS_MIMO2_DFLASH_V1_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/speculative/draft_backend.hpp"

/// DFlash v1 draft sidecar for MiMo-V2.6-Flash-RL: a five-block 1.5B model
/// that reads five trunk layer outputs (taps) as its context, projects them
/// through fc + enc.output_norm, keeps per-layer K/V in a ring, and drafts up
/// to block_size-1 tokens in one non-causal block forward pass. Unlike the
/// Qwen DFlash2 sidecar it has no dynamic convolutions and no candidate
/// selector: the block slots' logits are read directly with the trunk's
/// output head.
namespace gufo::models::mimo2 {

struct DFlashV1Config {
  std::uint32_t num_layers{0};
  std::uint32_t hidden_size{0};
  std::uint32_t intermediate_size{0};
  std::uint32_t num_heads{0};
  std::uint32_t num_kv_heads{0};
  std::uint32_t head_dim{0};
  std::uint32_t block_size{0};
  std::uint32_t sliding_window{0};
  std::uint32_t context_length{0};
  std::uint32_t mask_token_id{0};
  /// Head dims the rope rotates (`partial_rotary_factor` × head_dim); 0 means
  /// the full head. MiMo's drafter config sets partial_rotary_factor 0.5, so
  /// only the leading half of every head rotates.
  std::uint32_t rotary_dim{0};
  float rope_theta{10000.0F};
  float rms_eps{1e-6F};
  /// Post-attention multiply (`dflash_config.attention_value_scale`). The
  /// sidecar GGUF carries no key for it; the HF drafter config's 0.612 is the
  /// default and `GUFO_DFLASH_VALUE_SCALE` overrides it.
  float value_scale{0.612F};
  /// Block-row causality (`dflash.attention.causal`). MiMo's drafter is
  /// non-causal: every block slot reads the whole block. `GUFO_DFLASH_CAUSAL`
  /// overrides it for A/B.
  bool causal_block{false};
  /// Per-layer sliding-window flags (`dflash.attention.sliding_window_pattern`).
  /// A sliding layer masks its context to the trailing window.
  std::vector<bool> layer_swa;
  /// Zero-based trunk layer outputs whose residuals feed the encoder
  /// (GGUF `dflash.target_layers` minus one).
  std::vector<std::uint32_t> target_layer_ids;
};

/// One sidecar tensor. `data` is GPU-readable: the model registers a
/// host-side copy of the mapped GGUF and points here into that copy.
struct DFlashV1Tensor {
  const void* data{nullptr};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint32_t cols{0};  ///< contiguous (input) width
  std::uint32_t rows{0};  ///< output rows (1 for vectors)
};

struct DFlashV1LayerWeights {
  DFlashV1Tensor attn_norm;
  DFlashV1Tensor attn_q;
  DFlashV1Tensor attn_k;
  DFlashV1Tensor attn_v;
  DFlashV1Tensor attn_q_norm;
  DFlashV1Tensor attn_k_norm;
  DFlashV1Tensor attn_sinks;
  DFlashV1Tensor attn_output;
  DFlashV1Tensor ffn_norm;
  DFlashV1Tensor ffn_gate;
  DFlashV1Tensor ffn_up;
  DFlashV1Tensor ffn_down;
};

/// Tied matrices the drafter borrows from the trunk: the token embedding for
/// the block input and the output head for its logits.
struct DFlashV1TrunkTensors {
  const void* token_embedding{nullptr};
  core::GgmlType token_embedding_type{core::GgmlType::kQ8_0};
  const void* output{nullptr};
  core::GgmlType output_type{core::GgmlType::kQ8_0};
};

/// One sidecar GGUF mapped region copied into anonymous memory and
/// host-registered so the GPU reads the file-cheap copy in place.
struct DFlashV1MappedCopy {
  void* host{nullptr};
  const void* device{nullptr};
  const void* source{nullptr};
  std::size_t size{0};
};

/// The shared, immutable sidecar: parsed configuration and registered weight
/// copies. One instance serves every request's drafter (the Qwen DFlash model
/// plays the same role); it is expensive to build and cheap to share.
class DFlashV1Model {
public:
  ~DFlashV1Model();
  DFlashV1Model(const DFlashV1Model&) = delete;
  DFlashV1Model& operator=(const DFlashV1Model&) = delete;

  [[nodiscard]] static std::shared_ptr<const DFlashV1Model> Load(
      std::string_view sidecar_path, const DFlashV1TrunkTensors& trunk,
      std::uint32_t trunk_hidden_size, std::uint32_t trunk_vocab_size,
      float trunk_value_scale, std::string* error_msg);

  [[nodiscard]] const DFlashV1Config& config() const noexcept {
    return config_;
  }
  [[nodiscard]] std::span<const std::uint32_t> target_layer_ids()
      const noexcept {
    return config_.target_layer_ids;
  }
  /// Width of one target-feature row: taps * hidden.
  [[nodiscard]] std::size_t target_features_width() const noexcept {
    return config_.target_layer_ids.size() * config_.hidden_size;
  }
  [[nodiscard]] std::size_t weight_bytes() const noexcept {
    return weight_bytes_;
  }
  [[nodiscard]] const DFlashV1Tensor& token_embedding() const noexcept {
    return token_embedding_;
  }
  [[nodiscard]] const DFlashV1Tensor& output() const noexcept {
    return output_;
  }
  [[nodiscard]] const DFlashV1Tensor& fc() const noexcept { return fc_; }
  [[nodiscard]] const DFlashV1Tensor& enc_norm() const noexcept {
    return enc_norm_;
  }
  [[nodiscard]] const DFlashV1Tensor& output_norm() const noexcept {
    return output_norm_;
  }
  /// Trained mask-slot embedding (`dflash/mask_embedding.pt` in the HF repo).
  /// The trunk's embedding table holds zeros at the mask token, so the block's
  /// masked slots must be substituted with this vector. Empty when the file is
  /// absent.
  [[nodiscard]] std::span<const float> mask_embedding() const noexcept {
    return mask_embedding_;
  }
  [[nodiscard]] const std::vector<DFlashV1LayerWeights>& layers()
      const noexcept {
    return layers_;
  }

private:
  DFlashV1Model() = default;

  bool LoadWeights(core::GgufReader& reader, std::uint32_t trunk_hidden_size,
                   std::uint32_t trunk_vocab_size, std::string* error_msg);
  bool LoadMaskEmbedding(std::string_view sidecar_path, std::string* error_msg);

  std::unique_ptr<core::GgufReader> reader_;
  std::vector<DFlashV1MappedCopy> mapped_copies_;
  DFlashV1Config config_;
  DFlashV1Tensor token_embedding_;
  DFlashV1Tensor output_;
  DFlashV1Tensor fc_;
  DFlashV1Tensor enc_norm_;
  DFlashV1Tensor output_norm_;
  std::vector<float> mask_embedding_;
  std::vector<DFlashV1LayerWeights> layers_;
  std::size_t weight_bytes_{0};
};

/// One request's drafter state over a shared model: the ring K/V caches,
/// scratch and the forward passes. Every operation drains its stream before
/// returning, because callers consume results on the host.
class DFlashV1Drafter {
public:
  ~DFlashV1Drafter();
  DFlashV1Drafter(const DFlashV1Drafter&) = delete;
  DFlashV1Drafter& operator=(const DFlashV1Drafter&) = delete;

  [[nodiscard]] static std::unique_ptr<DFlashV1Drafter> Create(
      std::shared_ptr<const DFlashV1Model> model, std::uint32_t max_context,
      std::string* error_msg);

  [[nodiscard]] const DFlashV1Config& config() const noexcept {
    return model_->config();
  }
  [[nodiscard]] std::span<const std::uint32_t> target_layer_ids()
      const noexcept {
    return model_->target_layer_ids();
  }
  [[nodiscard]] std::size_t target_features_width() const noexcept {
    return model_->target_features_width();
  }
  /// Committed positions whose features have been injected.
  [[nodiscard]] std::uint32_t injected_length() const noexcept {
    return injected_length_;
  }
  [[nodiscard]] std::uint32_t history_capacity() const noexcept {
    return history_capacity_;
  }
  [[nodiscard]] std::size_t state_bytes() const noexcept { return state_bytes_; }

  /// Injects `num_tokens` rows of `features` (target_features_width floats
  /// each, tap-major) at `position`; position must equal injected_length().
  bool InjectTargetContext(std::span<const float> features,
                           std::uint32_t position, std::uint32_t num_tokens);

  /// Drafts up to `draft_count` tokens (block_size-1 max) after
  /// `anchor_token` at committed position `position` (== injected_length()).
  /// Greedy argmax over the block slots; empty on failure.
  [[nodiscard]] std::vector<std::int32_t> ForwardBlock(
      std::int32_t anchor_token, std::uint32_t position,
      std::uint32_t draft_count);

  /// Diagnostic snapshot of the last block forward for tests: the post-block
  /// hidden rows, the rows after the final output norm, the head logits and
  /// the drafted tokens. Offline tooling only.
  struct BlockDebug {
    std::uint32_t block_count{0};
    std::uint32_t draft_count{0};
    std::vector<float> hidden;  ///< [block_count][hidden]
    std::vector<float> normed;  ///< [block_count][hidden]
    std::vector<float> logits;  ///< [draft_count][vocab]
    std::vector<std::int32_t> drafts;
    /// Captured when SetDebug(true): the embedded rows right after the
    /// lookup, the first block's attention sub-stage tensors and the head of
    /// layer 0's injected K ring.
    std::vector<float> embedding{};
    std::vector<float> after_attention{};
    std::vector<float> query{};
    std::vector<float> raw_query{};
    std::vector<float> normed_query{};
    /// Row 0 of the block hidden at the end of every layer, in order.
    std::vector<float> layer_hidden{};
    std::vector<float> block_key{};
    std::vector<float> block_value{};
    std::vector<float> ring_key{};
  };
  [[nodiscard]] bool CopyBlockDebug(BlockDebug* out, std::string* error);
  /// Bring-up diagnostics: capture the embedding and first attention stage.
  void SetDebug(bool enabled) noexcept { debug_ = enabled; }
  /// Bring-up diagnostics: the last block's logits row `row`, copied to the
  /// host (row 0 predicts the position right after the block's anchor).
  [[nodiscard]] std::span<const float> CopyLogitsRow(std::size_t row);

  /// Drops the injected context (a new request starts at position zero).
  void Reset() noexcept { injected_length_ = 0; }

private:
  DFlashV1Drafter() = default;

  bool Allocate(std::string* error_msg);
  /// The block forward's device work, free of host synchronization so it can
  /// be captured once and replayed: embeddings, five layers, final norm,
  /// head GEMM and argmax.
  bool BlockBody(std::uint32_t block_count);
  bool InjectChunk(std::span<const float> features, std::uint32_t position,
                   std::uint32_t num_tokens);
  /// out[tokens][weight.rows] = weight @ x[tokens][weight.cols]; Q8_0 rides
  /// the small-projection kernel up to eight rows and the W8A8 tier above.
  void Gemm(const DFlashV1Tensor& weight, const float* x, float* out,
            std::uint32_t tokens);
  void Release() noexcept;

  std::shared_ptr<const DFlashV1Model> model_;
  hipStream_t stream_{nullptr};
  std::vector<void*> allocations_;
  std::size_t state_bytes_{0};
  std::uint32_t history_capacity_{0};
  std::uint32_t injection_capacity_{0};
  std::uint32_t injected_length_{0};

  /// Ring K/V per layer: [num_layers][history_capacity][kv_heads*head_dim].
  std::vector<float*> ring_k_;
  std::vector<float*> ring_v_;
  float* d_features_{nullptr};   ///< [injection_capacity][enc width]
  float* d_fused_{nullptr};      ///< [max(injection, block)][hidden]
  float* d_normed_{nullptr};     ///< [max(injection, block)][hidden]
  float* d_k_block_{nullptr};    ///< [max(injection, block)][kv width]
  float* d_v_block_{nullptr};    ///< [max(injection, block)][kv width]
  float* d_block_hidden_{nullptr};  ///< [block][hidden]
  float* d_q_{nullptr};             ///< [block][q width]
  float* d_attn_out_{nullptr};      ///< [block][q width]
  float* d_ffn_gate_{nullptr};      ///< [block][intermediate]
  float* d_ffn_up_{nullptr};        ///< [block][intermediate]
  float* d_ffn_down_{nullptr};      ///< [block][hidden]
  float* d_logits_{nullptr};        ///< [block][vocab]
  std::int32_t* d_block_tokens_{nullptr};  ///< [block]
  void* d_argmax_scratch_{nullptr};        ///< [block][kArgmaxParts]
  std::int32_t* d_drafts_{nullptr};        ///< [block]
  std::uint32_t* d_position_{nullptr};     ///< one device position
  float* d_mask_embedding_{nullptr};       ///< [hidden] or null when absent
  hipGraphExec_t block_graph_{nullptr};    ///< one replay per block forward
  std::vector<std::int32_t> host_drafts_;  ///< stable readback target
  std::uint32_t last_block_count_{0};
  std::uint32_t last_draft_count_{0};
  std::vector<std::int32_t> last_drafts_;
  bool debug_{false};
  std::vector<float> debug_embedding_;
  std::vector<float> debug_after_attention_;
  std::vector<float> debug_query_;
  std::vector<float> debug_raw_query_;
  std::vector<float> debug_normed_query_;
  std::vector<float> debug_layer_hidden_;
  std::vector<float> debug_block_key_;
  std::vector<float> debug_block_value_;
  std::vector<float> debug_ring_key_;
  std::vector<float> debug_logits_row_;
  void* d_w8a8_tiles_{nullptr};            ///< wide-activation Q8 tiling
  std::size_t w8a8_tiles_bytes_{0};
};

/// Draft width controller for the v1 sidecar. The cost model is empirical:
/// a step costs its draft block plus one verification row per drafted slot,
/// so the block is charged in row equivalents measured at runtime. That keeps
/// the width choice honest on both a compute-bound trunk (where rows are cheap
/// and wide drafting pays) and a memory-bound one (where a row costs about a
/// plain token and only high-acceptance slots are worth a row). Acceptance
/// keeps two EWMAs (depth one and the deeper conditional rate) so the
/// marginal slots' decay is modelled.
class DFlashV1LengthController {
public:
  static constexpr std::uint32_t kMaxDraftTokens = 7;

  explicit DFlashV1LengthController(std::uint32_t limit) noexcept
      : limit_(limit < 1 ? 1
                         : (limit > kMaxDraftTokens ? kMaxDraftTokens
                                                    : limit)) {}

  /// 0 means propose nothing (plain decode is the better expected value).
  [[nodiscard]] std::uint32_t Choose(std::uint32_t budget) const noexcept;
  void Observe(std::size_t accepted, std::size_t drafted) noexcept;
  /// Runtime cost observation: the draft block's wall time and the following
  /// verification's, each with the rows it covered.
  void ObserveCosts(double block_ms, double verify_ms, std::uint32_t rows) noexcept;
  /// Runtime observation of a plain (undrafted) step, the value a draft must
  /// beat.
  void ObservePlainStep(double ms) noexcept;
  void Reset() noexcept {
    p1_ = 0.55F;
    p2_ = 0.80F;
    explored_depth_ = 0;
    failed_depths_ = 0;
    probe_depth_ = 0;
    warmup_steps_ = 3;
    plain_steps_since_probe_ = 0;
    probe_interval_ = 24;
    observed_ = false;
  }
  [[nodiscard]] float DepthOneRate() const noexcept { return p1_; }
  [[nodiscard]] float DeeperRate() const noexcept { return p2_; }
  [[nodiscard]] std::uint32_t ProbeDepth() const noexcept {
    return probe_depth_;
  }
  /// Marks one step's outcome for the exploration counter: a proposal resets
  /// it, an undrafted step advances it.
  void NoteStep(std::uint32_t planned) noexcept {
    if (planned == 0) {
      if (plain_steps_since_probe_ < 1U << 20) {
        ++plain_steps_since_probe_;
      }
    } else {
      plain_steps_since_probe_ = 0;
    }
  }
  [[nodiscard]] std::uint32_t ExploredDepth() const noexcept {
    return explored_depth_;
  }
  [[nodiscard]] std::uint32_t FailedDepths() const noexcept {
    return failed_depths_;
  }
  [[nodiscard]] float BlockMs() const noexcept { return block_ms_; }
  [[nodiscard]] float RowMs() const noexcept { return row_ms_; }
  [[nodiscard]] float PlainMs() const noexcept { return plain_ms_; }

private:
  std::uint32_t limit_;
  float p1_{0.55F};  ///< EWMA depth-1 acceptance
  float p2_{0.80F};  ///< EWMA conditional acceptance of deeper slots
  /// EWMA of the draft block and of one verification row, in milliseconds.
  float block_ms_{0.0F};
  float row_ms_{0.0F};
  float plain_ms_{0.0F};  ///< one undrafted step, the per-token baseline
  /// Steps left that probe a small width regardless of the model, so the
  /// cost observations exist before the model is trusted.
  std::uint32_t warmup_steps_{3};
  /// Undrafted steps since the last proposal. An undrafted step yields no
  /// acceptance observation, so a controller that declined once would never
  /// learn that the content changed; every probe_interval_ plain steps it
  /// spends one small proposal to re-sample. The interval doubles each time a
  /// probe finds nothing (content that never accepts drafts pays for
  /// exploration and gets nothing back) and resets as soon as a draft lands.
  std::uint32_t plain_steps_since_probe_{0};
  std::uint32_t probe_interval_{24};
  std::uint32_t explored_depth_{0};
  std::uint32_t failed_depths_{0};  ///< bitmask of depths that rejected
  std::uint32_t probe_depth_{0};    ///< forced width after a full acceptance
  bool observed_{false};
};

/// IDraftBackend over one DFlashV1Drafter. Greedy-only: sampled requests fall
/// back to no speculation (SupportsSampledProposals false).
class DFlashV1DraftBackend final : public speculative::IDraftBackend {
public:
  ~DFlashV1DraftBackend() override = default;
  DFlashV1DraftBackend(const DFlashV1DraftBackend&) = delete;
  DFlashV1DraftBackend& operator=(const DFlashV1DraftBackend&) = delete;

  [[nodiscard]] static std::unique_ptr<DFlashV1DraftBackend> Create(
      std::shared_ptr<const DFlashV1Model> model, std::uint32_t max_context,
      std::uint32_t max_draft_tokens, std::string* error_msg);

  /// The drafter's per-request device state, mirroring DFlashV1Drafter::
  /// Allocate() (all buffers f32, so the context only enters through the
  /// history ring and the injection window). The serve layer's resource claim
  /// must cover this on top of the target session; without it a load-time
  /// reconcile rejects configurations that fit.
  [[nodiscard]] static std::size_t StateBytesFor(const DFlashV1Model& model,
                                                 std::uint32_t max_context) noexcept;

  [[nodiscard]] std::string_view Name() const noexcept override {
    return "Mimo2DFlashV1DraftBackend";
  }
  [[nodiscard]] bool RequiresTargetHiddenStates() const noexcept override {
    return true;
  }
  [[nodiscard]] std::span<const std::uint32_t> TargetHiddenLayerIds()
      const noexcept override {
    return drafter_->target_layer_ids();
  }
  [[nodiscard]] bool PrimeTargetContext(
      const speculative::DraftTargetContext& context) override;
  [[nodiscard]] bool AppendTargetContext(
      const speculative::DraftTargetContext& context,
      std::uint32_t position) override;
  void UpdateTargetHidden(std::span<const float> hidden) override;
  [[nodiscard]] speculative::DraftProposal Propose(
      std::span<const tokenization::TokenId> prompt_tokens,
      std::uint32_t current_pos, std::uint32_t max_tokens) override;
  void AcceptFeedback(std::span<const tokenization::TokenId> accepted,
                      tokenization::TokenId correction_token) override;
  void Reset() noexcept override;
  void BeginRequest() noexcept override { controller_.Reset(); }
  /// Cancellation rewind: clears features not yet injected (the injected
  /// frontier itself stays).
  void DiscardPendingTargetContext(std::uint32_t position);
  [[nodiscard]] DFlashV1Drafter& drafter() noexcept { return *drafter_; }
  /// Bring-up diagnostics: the last proposal's logits row (the prediction for
  /// the first drafted position), copied to the host.
  [[nodiscard]] std::span<const float> CopyProposalLogits(std::size_t row);
  /// Per-request device state the drafter holds (KV ring and scratch).
  [[nodiscard]] std::size_t DrafterStateBytes() const noexcept {
    return drafter_->state_bytes();
  }
  [[nodiscard]] const DFlashV1LengthController& controller() const noexcept {
    return controller_;
  }

private:
  DFlashV1DraftBackend(std::unique_ptr<DFlashV1Drafter> drafter,
                       std::uint32_t max_draft_tokens)
      : drafter_(std::move(drafter)),
        max_draft_tokens_(max_draft_tokens),
        controller_(max_draft_tokens) {}

  [[nodiscard]] std::uint32_t PendingFeatureCount(
      std::uint32_t current_pos) const;
  void InjectPendingFeatures(std::uint32_t current_pos);

  std::unique_ptr<DFlashV1Drafter> drafter_;
  std::uint32_t max_draft_tokens_{0};
  std::uint32_t max_context_{0};
  DFlashV1LengthController controller_;
  std::vector<float> pending_target_features_;
  std::vector<tokenization::TokenId> proposed_tokens_;
  /// Wall time of the last draft block and the moment it finished, so the
  /// feedback can charge the width controller the verification that followed.
  double last_block_ms_{0.0};
  double propose_end_ms_{0.0};
  double last_feedback_ms_{0.0};
  bool feedback_seen_{false};
  bool primed_{false};
  bool proposal_active_{false};
};

}  // namespace gufo::models::mimo2

#endif  // GUFO_MODELS_MIMO2_DFLASH_V1_HPP_