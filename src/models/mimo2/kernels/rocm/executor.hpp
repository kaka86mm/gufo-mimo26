#ifndef GUFO_MODELS_MIMO2_KERNELS_ROCM_EXECUTOR_HPP_
#define GUFO_MODELS_MIMO2_KERNELS_ROCM_EXECUTOR_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "src/core/session_mode.hpp"
#include "src/models/mimo2/kernels/rocm/blaslt.hpp"
#include "src/models/mimo2/kernels/rocm/communicator.hpp"
#include "src/models/mimo2/kernels/rocm/device_model.hpp"
#include "src/models/mimo2/kernels/rocm/kernels.hpp"
#include "src/models/mimo2/mtp_sampling.hpp"

namespace gufo::models::mimo2::rocm {

class Executor;
struct ArgmaxCandidate;
struct SnapshotHeader;

/// Per-sequence state on the device: every trunk layer's KV caches and the
/// NextN blocks' own caches plus their carried hidden rows. KV rows beyond
/// the position are stale until rewritten, so rewinding is only a position
/// update (no recurrent state exists to restore).
class Session {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  [[nodiscard]] bool mtp_enabled() const noexcept { return mtp_enabled_; }
  [[nodiscard]] std::uint32_t position() const noexcept { return position_; }
  [[nodiscard]] std::uint32_t max_context() const noexcept {
    return max_context_;
  }
  /// Drops every token; the next Forward starts at position 0.
  void Reset();
  void SetCancellationCheck(std::function<bool()> check);
  [[nodiscard]] bool CheckCancellation(std::string* error) const;
  [[nodiscard]] bool Cancelled() const noexcept { return cancelled_; }
  [[nodiscard]] std::uint64_t MutationEpoch() const noexcept {
    return mutation_epoch_;
  }
  [[nodiscard]] std::size_t AllocatedBytes() const noexcept;

private:
  friend class Executor;
  Session() = default;

  struct AttentionState {
    __half* k_cache{nullptr};  ///< [rows][KvDim(layer)] halves
    __half* v_cache{nullptr};  ///< [rows][VDim(layer)] halves
    /// int8 twin of the caches with one fp16 scale per 32 values (GUFO_MIMO_KV_INT8):
    /// the store writes both, and the decode attention reads the int8 pair, so
    /// a deep row's KV traffic halves. The fp16 rows stay authoritative for
    /// the prefill/wide kernels.
    std::int8_t* k_cache_q{nullptr};  ///< [rows][KvDim(layer)] bytes
    __half* k_scales{nullptr};        ///< [rows][KvDim(layer) / 32]
    std::int8_t* v_cache_q{nullptr};  ///< [rows][VDim(layer)] bytes
    __half* v_scales{nullptr};        ///< [rows][VDim(layer) / 32]
    /// 0 addresses the caches by absolute position (full-history layers);
    /// a nonzero power-of-two value is the sliding layer's ring size
    /// (pow2ceil(window + max_batch), capped at the context). Must exceed
    /// window + batch: a prefill chunk writes all its rows before
    /// attention reads them, and a tighter ring would clobber history its
    /// own earlier queries still need.
    std::uint32_t swa_rows{0};
  };
  /// Per-launch values the kernels read from device memory, so a captured
  /// graph replays at any position.
  struct Control {
    std::uint32_t position;      ///< first position of the trunk batch
    std::uint32_t mtp_position;  ///< first position of the draft batch
    std::int32_t hidden_row;     ///< kept trunk row the draft reads, or -1
  };
  /// One NextN block's private caches and carried hidden state.
  struct MtpState {
    __half* k_cache{nullptr};
    __half* v_cache{nullptr};
    float* h{nullptr};  ///< [hidden] wide residual handed to the next draft
    std::uint32_t position{0};
    std::uint32_t swa_rows{0};  ///< AttentionState::swa_rows semantics.
  };

  mutable bool cancelled_{false};
  std::uint64_t mutation_epoch_{0};
  const Executor* owner_{nullptr};
  std::uint32_t max_context_{0};
  bool mtp_enabled_{false};
  std::uint32_t position_{0};
  std::vector<AttentionState> attention_;
  /// One entry per bound NextN block; the draft chain runs the first block
  /// (the extra blocks stay dormant, see MtpForward).
  std::vector<MtpState> mtp_blocks_;
  float* target_hidden_{nullptr};  ///< last max_speculative trunk rows
  /// DFlash target taps: the trunk residual rows of the tapped layer outputs
  /// (zero-based), captured for the draft sidecar. Empty disarms capture.
  std::vector<std::uint32_t> tap_layers_;
  std::size_t tap_width_{0};  ///< tap_layers_.size() * hidden_size
  /// Device frontier row of the last forward: [tap_layers_.size()][hidden].
  float* target_taps_{nullptr};
  /// Device staging for a verification batch's tap rows
  /// ([tap_device_rows_][tap_width_]). A graph node writes here, so an armed
  /// verification still replays from a captured graph; the host copy happens
  /// once the launch finished.
  float* tap_device_{nullptr};
  std::size_t tap_device_rows_{0};
  /// Host staging, same row width as the taps: prompt rows accumulate across
  /// prefill chunks; verification rows are one batch; last_taps_ is the
  /// on-demand copy of the device frontier.
  std::vector<float> prompt_taps_;
  std::vector<float> verify_taps_;
  std::vector<float> last_taps_;
  /// Element offset of the running prefill chunk inside prompt_taps_.
  std::size_t tap_batch_offset_{0};
  /// What the running Forward captures (see Executor::CaptureTaps).
  enum class TapPhase { kNone, kPrompt, kLastRow, kVerify };
  TapPhase tap_phase_{TapPhase::kNone};
  std::uint32_t spec_base_{0};     ///< position before the speculative batch
  std::uint32_t spec_tokens_{0};   ///< tokens of the pending spec batch
  MtpState& mtp_() noexcept { return mtp_blocks_.front(); }
  const MtpState& mtp_() const noexcept { return mtp_blocks_.front(); }
  Control* control_{nullptr};
  /// Captured decode graphs by batch shape, and the shapes that ran eagerly
  /// once (the GEMM tier's arena must be grown before capture).
  std::unordered_map<std::uint64_t, hipGraphExec_t> graphs_;
  /// The communicator capture session each captured graph replays with.
  std::unordered_map<std::uint64_t, void*> graph_capture_sessions_;
  std::unordered_set<std::uint64_t> warmed_;
  std::vector<void*> allocations_;
  std::size_t allocated_bytes_{0};
  std::function<bool()> is_cancelled_;
};

/// Most target hidden-layer taps one draft sidecar may consume; the MiMo
/// DFlash sidecar reads five layer outputs.
inline constexpr std::uint32_t kMaxTargetLayerTaps = 5;

/// Runs the trunk graph on the GPU for one session at a time. Buffers are
/// sized once for `max_batch` tokens; longer prompts are fed in chunks.
class Executor {
public:
  /// A two-rank sum split into its start and its finish, so the GPU computes
  /// other rows while a partial is in flight. `start` queues the exchange of
  /// the partial at `data`, pending on the stream; `finish` waits for the
  /// oldest started exchange and adds the peer's partial into `data` on the
  /// stream. Sums finish in the order they started.
  struct SplitReduce {
    std::function<bool(const float*, std::size_t, hipStream_t, std::string*)>
        start;
    std::function<bool(float*, std::size_t, hipStream_t, std::string*)> finish;
  };

  struct Options {
    int device_index{0};
    /// Rows of one trunk batch; longer prompts are fed in chunks.
    std::uint32_t max_batch{1};
    /// Rows of logits (and hidden states) a Forward call may return.
    std::uint32_t max_logit_rows{1};
    /// Longest speculative batch; bounds the kept trunk rows.
    std::uint32_t max_speculative{1};
    /// The transport behind all_reduce, when there is one: a communicator
    /// that implements the capture protocol lets decode graphs capture the
    /// exchange host half inside the graph.
    std::shared_ptr<Communicator> communicator;
    /// GPU-visible sum reduction used by the layer boundaries. The callback
    /// is required only when DeviceModel carries world_size > 1.
    std::function<bool(float*, std::size_t, hipStream_t, std::string*)>
        all_reduce;
    /// The same sum, split. When set, a prefill chunk holds two trunk
    /// batches that run a layer apart, so each batch's exchange overlaps the
    /// other's computation (see PrefillChunk).
    SplitReduce split_reduce;
    /// Called once the stream has drained after sums were queued: false with
    /// the first failure among them. Required with all_reduce under TP.
    std::function<bool(std::string*)> reduce_status;
    /// Diagnostic observer called with every routed-FFN input and, under TP,
    /// every reduced FFN output, in forward order: a device buffer of rows x
    /// hidden floats pending on the given stream. Null in production.
    std::function<void(const float*, std::size_t, hipStream_t)> moe_observer;
  };

  /// The two-rank all-reduce over `communicator` for rows of `hidden` floats,
  /// queued on the stream without blocking the host: the GPU stages its
  /// partial, waits for the peer's and adds it. Failures surface through
  /// TwoRankReduceStatus once the stream has drained.
  [[nodiscard]] static std::function<bool(float*, std::size_t, hipStream_t,
                                          std::string*)>
  TwoRankAllReduce(std::shared_ptr<Communicator> communicator,
                   std::uint32_t hidden);
  [[nodiscard]] static std::function<bool(std::string*)> TwoRankReduceStatus(
      std::shared_ptr<Communicator> communicator);
  /// The overlapped form of TwoRankAllReduce.
  [[nodiscard]] static SplitReduce TwoRankSplitReduce(
      std::shared_ptr<Communicator> communicator, std::uint32_t hidden);

  ~Executor();
  [[nodiscard]] hipStream_t stream() const noexcept { return stream_; }
  Executor(const Executor&) = delete;
  Executor& operator=(const Executor&) = delete;

  [[nodiscard]] static std::unique_ptr<Executor> Create(
      const DeviceModel& model, Options options,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      core::SessionMode mode, std::uint32_t max_context,
      std::string* error_msg = nullptr) const;
  /// Kept for engine symmetry; recurrent state no longer needs scratch, so
  /// this only validates the depth against the speculative limit.
  [[nodiscard]] bool EnsureRollback(Session& session, std::uint32_t depth,
                                    std::string* error_msg) const;
  [[nodiscard]] std::size_t SessionBytes(
      core::SessionMode mode, std::uint32_t max_context,
      std::uint32_t rollback_depth) const noexcept;
  [[nodiscard]] std::size_t DeferredScratchBytes() const;

  enum class ForwardMode { kDecode, kVerify, kPrefill };

  /// Appends `tokens` (at most max_batch, or PrefillChunk for kPrefill) to
  /// the session and returns the logits of the last `n_logits` tokens in
  /// `logits` (n_logits * vocab floats, host memory). A null `logits` keeps
  /// the rows on the GPU for verification. The final residual of those
  /// tokens stays on the device for MtpForward. kVerify permits Rollback (at
  /// most max_speculative rows). kPrefill uses consistent prompt arithmetic
  /// at every chunk width. A prefill that PairLead splits runs as two trunk
  /// batches with the same results as two calls; the draft block then keeps
  /// the second batch's rows.
  [[nodiscard]] bool Forward(Session& session,
                             std::span<const std::int32_t> tokens,
                             std::uint32_t n_logits, float* logits,
                             ForwardMode mode, std::string* error_msg) const;

  struct BatchItem {
    Session* session;
    std::span<const std::int32_t> tokens;
    bool speculative;
  };
  /// Packs independent decode chains for shared projections. Every chain
  /// keeps its own KV state and the decode arithmetic (<=8 rows). Logit
  /// rows remain in batch order until SelectBatchLogits is called.
  [[nodiscard]] bool ForwardBatch(std::span<const BatchItem> items,
                                  std::string* error_msg) const;
  [[nodiscard]] bool SelectBatchLogits(std::uint32_t offset, std::uint32_t rows,
                                       float* logits,
                                       std::string* error_msg) const;
  /// Stage a contiguous run of the completed batch's logit rows in the pinned
  /// host buffer with one transfer and one sync. `SelectBatchLogits` serves
  /// rows inside the staging from there, so a cohort pays one readback instead
  /// of one blocking download per session (six of those were the per-cycle
  /// hole in the producer profile). The staging dies with the next head pass.
  [[nodiscard]] bool StageBatchLogits(std::uint32_t offset, std::uint32_t rows,
                                      std::string* error_msg) const;
  /// Key count from which a decode-width cohort splits each row's visible key
  /// range, and the number of blocks it splits into. Whether the range is
  /// split is the batch's only attention-shape variable, so the graph key
  /// carries that one bit; this is the threshold behind it.
  [[nodiscard]] static std::uint32_t AttentionSplitKeys();
  [[nodiscard]] static std::uint32_t AttentionSplitCount();

  /// Keeps the first `keep` (1..n) tokens of the last speculative batch and
  /// discards the rest: the position rewinds and the KV rows past it go
  /// stale until the next replay overwrites them. If `logits` is supplied,
  /// copies the kept frontier into it using the same synchronization.
  [[nodiscard]] bool Rollback(Session& session, std::uint32_t keep,
                              std::string* error_msg,
                              float* logits = nullptr) const;

  /// Rewinds the frontier to an earlier position of the pending or last
  /// batch: KV rows past it go stale until the next replay overwrites them.
  /// Unlike Rollback (which keeps a nonempty speculative prefix) this admits
  /// `position == spec_base_`, the generic verifier's restore-to-frontier.
  [[nodiscard]] bool Rewind(Session& session, std::uint32_t position,
                            std::string* error_msg = nullptr) const;

  /// Arms DFlash target-tap capture at the given zero-based trunk layer
  /// outputs (at most kMaxTargetLayerTaps, strictly increasing); an empty
  /// span disarms. The taps are the layer's post-FFN residual rows, tap-major
  /// within a row, fp32 — the layout the Qwen DFlash sidecar consumes. While
  /// armed, forward bodies run eagerly (a host readback cannot replay from a
  /// captured graph), a split-reduce prefill runs as one batch, and batched
  /// multi-session decode is refused: the draft path needs one sequence.
  [[nodiscard]] bool SetTargetLayerCapture(
      Session& session, std::span<const std::uint32_t> layers,
      std::string* error_msg = nullptr) const;
  /// Host tap rows the prefill chunks captured so far, `[rows][tap_width]`.
  [[nodiscard]] std::span<const float> GetPromptHiddenStates(
      const Session& session) const noexcept {
    return session.prompt_taps_;
  }
  /// Host tap rows of the last verification batch, `[rows][tap_width]`.
  [[nodiscard]] std::span<const float> GetVerificationHiddenStates(
      const Session& session) const noexcept {
    return session.verify_taps_;
  }
  /// Drains the stream and returns the last forward's frontier tap row
  /// (tap_width floats); empty when capture is disarmed or a copy failed.
  [[nodiscard]] std::span<const float> CopyLastHidden(Session& session) const;
  /// Forgets the accumulated prompt rows (a new request starts from zero).
  void ClearPromptHiddenStates(Session& session) noexcept {
    session.prompt_taps_.clear();
    session.tap_batch_offset_ = 0;
  }
  /// Whether the session captures target taps. Callers that reason about the
  /// pair-split prefill layout (see PairLead) must consult this: armed capture
  /// forces the single-batch path.
  [[nodiscard]] bool TargetCaptureArmed(
      const Session& session) const noexcept {
    return !session.tap_layers_.empty();
  }
  /// Width of one captured tap row (taps * hidden), zero when disarmed.
  [[nodiscard]] std::uint32_t TargetCaptureWidth(
      const Session& session) const noexcept {
    return static_cast<std::uint32_t>(session.tap_width_);
  }

  /// Runs the first NextN block over `tokens` (at most max_batch) at its own
  /// position. The hidden input of token i is the trunk residual of row
  /// `hidden_row + i` of the last Forward batch, or, with hidden_row < 0
  /// (single token), the block's own residual from the previous call. Only
  /// requested outputs are computed. Production requests a greedy token or
  /// compact candidate logits. Catch-up skips the output head when only the
  /// draft state is needed.
  ///
  /// The remaining NextN blocks of the sidecar stay dormant: their caches
  /// are allocated and snapshotted but the vertical chain that would feed
  /// them is not wired yet (each block ingests only the tokens the first
  /// block also sees, so running them needs the multi-depth draft loop).
  struct MtpOutput {
    std::int32_t* token{nullptr};
    MtpCandidateLogits* candidates{nullptr};
  };
  struct MtpHeadItem {
    Session* session;
    MtpOutput output;
  };
  struct MtpBatchItem {
    Session* session;
    std::span<const std::int32_t> tokens;
    std::int32_t hidden_row;
  };
  /// Runs independent short predictor chains with shared projections and
  /// private attention caches, positions and carried hidden states.
  [[nodiscard]] bool MtpForwardBatch(std::span<const MtpBatchItem> items,
                                     std::string* error_msg) const;
  /// Projects each session's carried draft hidden state with the shared
  /// head.
  [[nodiscard]] bool MtpHeads(std::span<const MtpHeadItem> items,
                              std::string* error_msg) const;
  [[nodiscard]] bool MtpForward(Session& session,
                                std::span<const std::int32_t> tokens,
                                std::int32_t hidden_row, MtpOutput output,
                                std::string* error_msg,
                                const float* hidden_source = nullptr) const;
  /// Leading kept trunk rows, for independent predictor qualification.
  [[nodiscard]] bool CopyTrunkHidden(const Session& session,
                                     std::span<float> hidden,
                                     std::string* error_msg) const;

  /// Plain greedy verification keeps full logit rows on the GPU.
  [[nodiscard]] bool GreedyMtpPredictions(
      std::span<ArgmaxCandidate> predictions, std::string* error_msg) const;

  /// GUFO_ROUTER_SELFCHECK: recompute the current layer's router logits with
  /// the plain tiled form and compare against `s_.router` (the split-K
  /// result) — rounding-only deltas, and the top-8 expert sets must match.
  void RouterSelfCheck(const __half* router_f16, std::uint32_t n_tokens,
                       std::uint32_t splits) const;

  /// A session's complete context as one host byte payload: every layer's KV
  /// caches up to the position, the draft block's caches plus carried
  /// residual, and the `hidden_rows` most recent kept trunk rows. The
  /// payload restores into any session of this executor whose context holds
  /// the position; the pending speculative batch must be empty.
  struct SnapshotInfo {
    std::uint32_t position{0};
    std::uint32_t hidden_rows{0};
  };
  [[nodiscard]] std::uint64_t SnapshotBytes(const Session& session,
                                            std::uint32_t hidden_rows) const;
  [[nodiscard]] bool SaveSnapshot(const Session& session,
                                  std::uint32_t hidden_rows,
                                  std::span<std::uint8_t> payload,
                                  std::string* error_msg) const;
  /// Restoration never grows session scratch; the draft rewinds with the
  /// position. `next_drafts` is accepted for call-shape symmetry and only
  /// bounds nothing (there is no rollback arena anymore).
  [[nodiscard]] bool RestoreSnapshot(Session& session,
                                     std::span<const std::uint8_t> payload,
                                     SnapshotInfo* info, std::string* error_msg,
                                     std::uint32_t next_drafts = 0) const;

  /// Rewinds the draft block's own context.
  void MtpRewind(Session& session, std::uint32_t position) const noexcept {
    auto& mtp = session.mtp_();
    mtp.position = std::min(mtp.position, position);
  }
  [[nodiscard]] std::uint32_t MtpPosition(
      const Session& session) const noexcept {
    return session.mtp_().position;
  }

  [[nodiscard]] const Config& config() const noexcept {
    return model_->config();
  }
  [[nodiscard]] std::uint32_t max_batch() const noexcept {
    return options_.max_batch;
  }
  /// Tokens one prefill Forward takes: two trunk batches with a split
  /// reduce, else one.
  [[nodiscard]] std::uint32_t PrefillChunk() const noexcept {
    return options_.split_reduce.start ? 2 * options_.max_batch
                                       : options_.max_batch;
  }
  /// Armed target capture shrinks a prefill chunk to one trunk batch: the
  /// pair path is unavailable, and a single batch of 2*max_batch rows
  /// overflows the all-reduce window (sized for max_batch rows).
  [[nodiscard]] std::uint32_t PrefillChunk(
      const Session& session) const noexcept {
    return TargetCaptureArmed(session) ? options_.max_batch : PrefillChunk();
  }
  /// With a split reduce, a prefill chunk of `n` tokens runs as two trunk
  /// batches of about half each, so each batch's sums cross the link while
  /// the other computes; returns the first batch's rows, or 0 when the chunk
  /// runs as one batch (without a split reduce, or too short to repay a
  /// second pass over each layer's weights).
  [[nodiscard]] std::uint32_t PairLead(std::uint32_t n) const noexcept {
    constexpr std::uint32_t kPairMinTokens = 384;
    return options_.split_reduce.start && n >= kPairMinTokens &&
                   n <= PrefillChunk()
               ? (n + 1) / 2
               : 0;
  }
  [[nodiscard]] std::uint32_t max_speculative() const noexcept {
    return options_.max_speculative;
  }
  [[nodiscard]] bool has_mtp() const noexcept { return model_->has_mtp(); }
  /// Slots in the batch decode tables (the batched bodies' session slots).
  static constexpr std::uint32_t kBatchSlotCount = 8;

private:
  Executor() = default;

  /// Visits every device region of a snapshot in payload order with
  /// (device pointer or null when sizing, payload offset, bytes, name).
  /// Returns the payload size, or 0 once a visit failed.
  template<typename Visit>
  std::uint64_t WalkSnapshot(const SnapshotHeader& h, const Session* session,
                             Visit&& visit) const;

  /// out = (up . x) * silu(gate . x) over the dense FFN; with `down` (the
  /// projection that consumes the result), a wide batch leaves only that
  /// projection's staged input form (F16 rows in s_.x_half or the tiled Q8
  /// layout in s_.x_q8t, registered in the input cache).
  bool GatedDense(const DeviceTensor& gate, const DeviceTensor& up,
                  const float* x, float* out, std::uint32_t n_tokens,
                  const DeviceTensor* down, std::string* error_msg) const;
  bool Dense(const DeviceTensor& w, const float* x, float* out,
             std::uint32_t n_tokens, std::string* error_msg) const;
  /// Reuse or populate the F16 activation staging buffer.
  void PrepareHalfInput(const float* x, std::uint32_t rows,
                        std::uint32_t cols) const;
  /// Whether a wide dense Q8_0 projection takes the F16 WMMA GEMM.
  bool DenseF16Route(const DeviceTensor& w, std::uint32_t n_tokens) const;
  /// Reads the routing of the current batch back and derives the tile hints
  /// for its expert GEMMs (tiled batches only).
  bool RouteHints(std::uint32_t n_tokens, std::string* error_msg) const;
  /// The routed experts after the router: either the RoutedF16Gemm family
  /// over the compacted assignments or, for MXFP4 weights, the vendored
  /// quantized GEMM tier.
  bool MoeExperts(const DeviceLayer& l, const float* x, float* out,
                  std::uint32_t n_tokens, std::string* error_msg) const;
  bool MoeExpertsWmma(const DeviceLayer& l, const float* x, float* out,
                      std::uint32_t n_tokens, std::string* error_msg) const;
  bool MoeExpertsMxfp4(const DeviceLayer& l, const float* x, float* out,
                       std::uint32_t n_tokens, std::string* error_msg) const;
  /// A block's feed-forward: the dense SwiGLU of layer 0 and the draft
  /// blocks, or the routed experts. Without `reduce`, a distributed rank's
  /// output stays its partial (an unsplit replicated block contributes a
  /// zeroed partial instead).
  bool FeedForward(const DeviceLayer& l, const float* x, float* out,
                   std::uint32_t n_tokens, std::string* error_msg,
                   bool reduce = true) const;
  /// Q/K/V projections (unless already staged), rope, cache store and the
  /// attention itself, then the output projection scaled by value_scale.
  /// With `slot_table` the caches and ring come from slot_table[slot] and
  /// `pos` points into the slot control table (graph-replay indirection);
  /// otherwise `s` carries them directly.
  bool AttentionPart(const DeviceLayer& l, Session::AttentionState& s,
                     const float* x, float* out, std::uint32_t n_tokens,
                     const std::uint32_t* pos, std::string* error_msg,
                     bool projections_ready = false,
                     bool project_output = true,
                     const SlotKv* slot_table = nullptr,
                     std::uint32_t slot = 0,
                     std::uint32_t visible_keys = 0,
                     std::uint32_t wide_splits = 1,
                     const std::uint32_t* row_slot = nullptr,
                     const std::uint32_t* row_pos = nullptr) const;
  /// Sums `rows` rows of a partial across ranks.
  bool AllReduce(float* data, std::size_t rows, std::string* error_msg) const;
  /// Selects the greedy token or compact candidates from the draft logits.
  bool MtpHead(const DeviceLayer& block, const float* res, std::uint32_t row,
               bool token, bool candidates, std::string* error_msg) const;
  /// Enqueues one trunk batch (control and token upload through logits).
  bool ForwardBody(Session& session, std::uint32_t n, std::uint32_t n_logits,
                   bool download_logits, bool speculative,
                   std::uint32_t start_pos, std::uint32_t first_layer,
                   std::uint32_t end_layer, std::string* error_msg) const;
  /// Copies the layer's residual rows into the session's tap staging when
  /// `layer` is a tapped output (see SetTargetLayerCapture).
  bool CaptureTaps(Session& session, std::uint32_t layer, std::uint32_t n,
                   std::string* error_msg) const;
  /// A prefill chunk that PairLead splits after `lead` rows (see Forward):
  /// runs the two trunk batches a layer apart, then the draft block over
  /// both.
  bool ForwardPair(Session& session, std::span<const std::int32_t> tokens,
                   std::uint32_t lead, std::uint32_t n_logits, float* logits,
                   std::string* error_msg) const;
  /// One trunk batch as the layers see it.
  struct TrunkBatch {
    std::uint32_t n;
    std::uint32_t start_pos;
    bool speculative;
    Session::Control* control;  ///< the positions its kernels read
  };
  /// A trunk layer's attention half: the norm, projections, attention and
  /// output projection, the result in s_.block_out (this rank's partial
  /// without `reduce`).
  bool AttentionHalf(Session& session, std::uint32_t il,
                     const TrunkBatch& batch, bool reduce,
                     std::string* error_msg) const;
  /// The layer's feed-forward half: the norm, the FFN and the residual
  /// update; s_.block_out is the FFN output (partial without `reduce`).
  bool FfnHalf(std::uint32_t il, std::uint32_t n_tokens, bool reduce,
               std::string* error_msg) const;
  /// res += block_out over `n_tokens` hidden-wide rows.
  void AddResidual(float* res, const float* block_out,
                   std::uint32_t n_tokens) const;
  bool MtpBody(Session& session, std::uint32_t n, std::uint32_t pos,
               bool token, bool candidates, std::string* error_msg,
               const float* hidden_source) const;
  /// Runs `body` eagerly, or as the session's captured graph for `key`
  /// when `graph` is set.
  bool Run(Session& session, std::uint64_t key, bool graph,
           const std::function<bool()>& body, std::string* error_msg,
           bool synchronize = true) const;
  /// Waits for the stream, then for the outcome of the sums queued on it.
  bool Drain(const char* what, std::string* error_msg) const;

  const DeviceModel* model_{nullptr};
  Options options_;
  hipStream_t stream_{nullptr};
  hipEvent_t counts_ready_{nullptr};
  hipblasHandle_t blas_{nullptr};
  std::unique_ptr<BlasLt> blaslt_;
  float value_scale_{1.0F};

  // Scratch, sized for max_batch tokens. Names follow reference.cpp.
  struct Scratch {
    std::int32_t* tokens;
    void* x_half;   ///< activations narrowed to the weight's 16-bit type
    void* x_q8t;    ///< tiled Q8 activations of a wide batch (W8A8 route)
    float* res;
    float* xn;
    float* block_out;  ///< the running block output (a partial under TP)
    // attention (the widest layer on this rank: split trunk or whole draft)
    float* q;   ///< [t][q_heads*head_dim_qk]
    float* k;   ///< [t][kv_heads*head_dim_qk]
    float* v;   ///< [t][kv_heads*head_dim_v]
    float* ctx;  ///< [t][q_heads*head_dim_v]
    float* attn_partials;  ///< split-key partials of a narrow batch
    // moe
    float* router;
    std::uint16_t* router_f16;  ///< every routed layer's router rows, F16
    std::int32_t* ids;
    std::uint32_t* expert_counts;
    // Routed WMMA route: 16-row padded bucket bounds, scatter cursors, the
    // compact row -> (token, slot) maps and the per-tile launch map.
    std::int32_t* routed_bounds;
    std::int32_t* routed_cursors;
    std::int32_t* rows_token;
    std::int32_t* rows_slot;
    std::int32_t* routed_tiles;
    float* weights;
    float* gate_e;  ///< [t*used][ff] routed gate rows (F32 or staged F16)
    float* up_e;    ///< [t*used][ff] routed up/SwiGLU rows (F16 on wmma)
    float* down_e;  ///< [t*used][hidden] routed down rows
    float* ff_act;   ///< dense FFN SwiGLU rows (the gate rows first)
    float* ff_up;    ///< dense FFN up rows while the gate fills `ff_act`
    __half* ff_half;  ///< dense SwiGLU rows staged for the F16 down route
    // head rows kept for the draft block
    float* logits;
    // mtp
    float* mtp_h;   ///< [t][hidden] draft hidden input
    float* mtp_eh;  ///< [t][2*hidden] concatenated [e_norm | h_norm]
    ArgmaxCandidate* mtp_argmax;
    std::int32_t* mtp_token;
    std::uint32_t* mtp_ids;
    std::uint32_t* mtp_scratch_ids;
    float* mtp_scores;
    // router
    float* router_partials;  ///< split-K partials [slices][t][experts]
    float* router_alt;       ///< self-check: the other form's logits
  };
  mutable Scratch s_{};
  /// Attention activation widths of one layer's rows inside the shared
  /// buffers (they differ between a split trunk layer and a whole draft
  /// block).
  struct AttnDims {
    std::uint32_t q;
    std::uint32_t k;
    std::uint32_t v;
    std::uint32_t ctx;
  };
  [[nodiscard]] AttnDims DimsOf(const DeviceLayer& l) const noexcept;
  [[nodiscard]] Scratch RowScratch(const Scratch& base,
                                   std::uint32_t offset) const;
  [[nodiscard]] Scratch RowScratch(const Scratch& base, std::uint32_t offset,
                                   const AttnDims& dims) const;
  void UseScratch(const Scratch& scratch) const;
  /// The second trunk batch of a prefill pair: its own residual and layer
  /// output rows, which stay live while the first batch runs; every other
  /// buffer is shared.
  [[nodiscard]] Scratch PairScratch(const Scratch& base) const;
  float* pair_res_{nullptr};
  float* pair_block_out_{nullptr};
  /// The second batch's positions; the first uses the session's.
  Session::Control* pair_control_{nullptr};
  bool DenseBatch(const DeviceTensor& w, const float* x, float* out,
                  std::uint32_t rows, std::string* error_msg) const;
  /// A block's feed-forward over a decode batch of independent rows. Without
  /// `reduce` the rows' partials stay local (including a non-owner rank's
  /// zeroed ones) for a caller that queues the sum itself — the batch form of
  /// the `reduce` flag FeedForward already carries for the pair pipeline.
  bool FeedForwardBatch(const DeviceLayer& l, const float* x, float* out,
                        std::uint32_t rows, std::string* error,
                        bool reduce = true) const;
  bool MoeBatch(const DeviceLayer& l, const float* x, float* out,
                std::uint32_t rows, std::string* error,
                bool reduce = true) const;
  bool AllocateBatch(std::string* error_msg) const;
  /// The slot indirection behind the executor-wide batch graphs: a pinned
  /// table (refreshed by the host before every launch) mirrored on the
  /// device by a copy node at the body's head, so one captured graph serves
  /// any session mix. Layout: ctrl[8] | ptrs[8] | kv[layers][8] | mtp_kv[8].
  struct SlotTables {
    void* host{nullptr};  ///< hipHostMalloc'd; layout below
    void* dev{nullptr};   ///< hipMalloc'd mirror
    std::uint32_t layers{0};
    std::size_t ctrl_off{0}, ptrs_off{0}, kv_off{0}, mtp_off{0}, bytes{0};
    /// Per-row attention tables: the session slot and absolute position of
    /// every row of a fused cohort step (up to eight rows per session). The
    /// host refreshes them with the rest of the table before each body.
    static constexpr std::uint32_t kRowRows = kBatchSlotCount * 8;
    std::uint32_t* row_slot_host{nullptr};
    std::uint32_t* row_pos_host{nullptr};
    std::uint32_t* row_slot_dev{nullptr};
    std::uint32_t* row_pos_dev{nullptr};

    SlotControl* HostCtrl() const {
      return reinterpret_cast<SlotControl*>(static_cast<char*>(host) +
                                            ctrl_off);
    }
    SlotPtrs* HostPtrs() const {
      return reinterpret_cast<SlotPtrs*>(static_cast<char*>(host) + ptrs_off);
    }
    SlotKv* HostKv(std::uint32_t layer) const {
      return reinterpret_cast<SlotKv*>(static_cast<char*>(host) + kv_off) +
             static_cast<std::size_t>(layer) * kBatchSlotCount;
    }
    SlotKv* HostMtpKv() const {
      return reinterpret_cast<SlotKv*>(static_cast<char*>(host) + mtp_off);
    }
    SlotControl* DevCtrl() const {
      return reinterpret_cast<SlotControl*>(static_cast<char*>(dev) +
                                            ctrl_off);
    }
    SlotPtrs* DevPtrs() const {
      return reinterpret_cast<SlotPtrs*>(static_cast<char*>(dev) + ptrs_off);
    }
    const SlotKv* DevKv(std::uint32_t layer) const {
      return reinterpret_cast<const SlotKv*>(static_cast<char*>(dev) +
                                             kv_off) +
             static_cast<std::size_t>(layer) * kBatchSlotCount;
    }
    const SlotKv* DevMtpKv() const {
      return reinterpret_cast<const SlotKv*>(static_cast<char*>(dev) +
                                             mtp_off);
    }
  };
  bool AllocateSlots(std::string* error_msg) const;
  /// Writes one batch row's destinations into the pinned slot table (the
  /// device mirror follows from the copy node at the body's head). A
  /// cancelled session gets null destinations: the kernels skip its private
  /// writes and clear its attention output, matching the eager skip.
  void FillSlotRow(std::uint32_t slot, const Session& session,
                   std::int32_t hidden_row) const;
  /// The two-lane decode batch: the cohort splits into a prefix and a suffix
  /// whose layer halves interleave, so each rank's exchange of one lane is in
  /// flight while the GPU computes the other — the batch form of the pipeline
  /// ForwardPair runs for prefill. Each lane's rows stay private to their own
  /// sessions, so unlike the pair form the lanes need no ordering between
  /// them. Requires an eager body: the overlapped exchange is host-driven and
  /// has no capture protocol.
  bool ForwardBatchLanes(std::span<const BatchItem> items,
                         std::span<const std::uint32_t> offsets,
                         std::uint32_t rows, const Scratch& base,
                         std::string* error) const;
  /// Whether batch bodies run under stream capture (the env gates RunCaptured
  /// applies). Both ranks read the same env, so a decision keyed on it — like
  /// taking the lane pipeline — stays in lockstep.
  [[nodiscard]] static bool GraphCaptureEnabled();
  /// Graph-captured replay of a decode batch body, keyed on the exact batch
  /// shape; cold shapes run eagerly (MTP chain lengths jitter, so only the
  /// hot shapes pay the capture). Shares the stream with the eager path.
  bool RunCaptured(std::uint64_t key, const std::function<bool()>& body,
                   std::string* error_msg) const;
  mutable SlotTables slots_;
  /// Allocated only when concurrent decoding is first requested.
  mutable float* batch_logits_{nullptr};
  mutable MtpCandidateLogits* batch_candidates_host_{nullptr};
  mutable std::uint32_t batch_rows_{0};
  /// Rows of the completed batch currently sitting in `logits_host_`
  /// (StageBatchLogits); zero whenever the head has not staged them.
  mutable std::uint32_t staged_logit_offset_{0};
  mutable std::uint32_t staged_logit_rows_{0};
  /// Visible keys of the widest row in the last batch: a cohort step reads
  /// each row's own cache, so the widest row decides whether the key range is
  /// split for the whole step (short rows only pay empty split blocks).
  mutable std::uint32_t batch_visible_keys_{0};
  mutable const float* selected_logits_{nullptr};
  [[nodiscard]] const float* VerificationLogits() const {
    return selected_logits_ != nullptr ? selected_logits_ : s_.logits;
  }
  // Pinned host staging the launched (or captured) work reads and writes.
  // control_host_[1] holds a prefill pair's second batch.
  Session::Control* control_host_{nullptr};
  std::int32_t* tokens_host_{nullptr};
  std::uint32_t* counts_host_{nullptr};
  std::int32_t* tiles_host_{nullptr};         ///< routed tile map staging
  mutable std::unordered_map<std::uint64_t, hipGraphExec_t> batch_graphs_;
  mutable std::unordered_map<std::uint64_t, void*> batch_graph_sessions_;
  mutable std::unordered_set<std::uint64_t> batch_warmed_;
  /// Captured batch graphs cost real host memory each (the exec plus the
  /// session's exchange slots), and the batch-MTP key space explodes with
  /// concurrent sessions — every session's chain length jitters per step, so
  /// cold tuples keep arriving. Past the cap a cold shape stays eager instead
  /// of capturing, keeping the hot ones graphed while the cache stays bounded.
  static constexpr std::size_t kMaxBatchGraphs = 32;
  mutable std::uint32_t routed_max_rows_{0};  ///< 0 = no readback yet
  /// Router weight F16 mirrors, filled once per process (pointer-keyed so
  /// the layer need not carry an index).
  mutable std::unordered_map<const void*, __half*> router_f16_map_;
  mutable std::once_flag router_f16_once_;
  mutable std::uint32_t router_checks_done_{0};  ///< GUFO_ROUTER_SELFCHECK
  /// Aligned Q8_0 artifacts for the projection weights the decode path
  /// streams every step (q/k/v/o): a one-time repack into the vendored
  /// dense_vec layout, keyed by the raw tensor pointer. Oversized tensors
  /// (the lm_head's vocab rows) are not repacked — they would add hundreds
  /// of MB for a projection that runs on selected rows only.
  mutable std::unordered_map<const void*, void*> aligned_q8_map_;
  mutable std::uint32_t routed_n_tiles_{0};   ///< down projection tiles
  mutable std::uint32_t routed_64_tiles_{0};  ///< appended 64-token tiles
  mutable std::uint32_t routed_pair_offset_{0};
  mutable std::uint32_t routed_pair_tiles_{0};
  mutable std::uint32_t routed_pair_rows_{64};
  mutable std::uint32_t routed_tile_rows_{48};  ///< token rows per tile
  mutable int routed_tile_cols_{0};
  float* logits_host_{nullptr};
  std::int32_t* mtp_token_host_{nullptr};
  MtpCandidateLogits* mtp_candidates_host_{nullptr};
  std::vector<void*> allocations_;
  // What s_.x_q8t / s_.x_half currently hold (input pointer, rows, cols,
  // and the half type), so a projection over the same rows skips its
  // activation pass. Cleared whenever the source buffer is rewritten.
  mutable const float* q8t_src_{nullptr};
  mutable std::uint32_t q8t_rows_{0};
  mutable std::size_t q8t_cols_{0};
  mutable const float* half_src_{nullptr};
  mutable std::uint32_t half_rows_{0};
  mutable std::size_t half_cols_{0};
  mutable bool half_bf16_{false};
  /// Set by GatedDense when s_.ff_half holds the SwiGLU rows its F16 down
  /// projection reads.
  mutable bool ff_half_ready_{false};
};

}  // namespace gufo::models::mimo2::rocm

#endif  // GUFO_MODELS_MIMO2_KERNELS_ROCM_EXECUTOR_HPP_
