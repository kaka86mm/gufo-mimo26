#ifndef GUFO_MODELS_MIMO2_ENGINE_HPP_
#define GUFO_MODELS_MIMO2_ENGINE_HPP_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/core/sampling.hpp"
#include "src/core/session_mode.hpp"
#include "src/models/mimo2/chat_template.hpp"
#include "src/models/mimo2/config.hpp"
#include "src/models/mimo2/mtp_policy.hpp"
#include "src/models/mimo2/tokenizer.hpp"

struct ihipStream_t;

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::mimo2 {

struct ModelWeights;
struct MtpWeights;
struct MtpCandidateLogits;
namespace rocm {
class Communicator;
class DeviceModel;
class Executor;
class Session;
}  // namespace rocm

struct ModelOptions {
  std::uint32_t max_context = 4096;
  /// Optional MTP draft sidecar (`mtp-...-*.gguf`). Empty leaves
  /// speculative decoding off.
  std::string mtp_model_path;
  /// Optional DFlash v1 draft sidecar (`dflash-...-*.gguf`). Mutually
  /// exclusive with the MTP sidecar.
  std::string dflash_model_path;
  /// Maximum DFlash proposals per cycle (the sidecar's block_size-1 caps it).
  std::uint32_t dflash_max_draft_tokens = 7;
  /// Maximum proposals per cycle; acceptance history selects the length.
  std::uint32_t max_draft_tokens = kMaxMtpDraftTokens;
  /// Fixed serving capacity used by the calibrated MTP cost model. Keeping
  /// it independent of scheduler timing preserves seeded request replay.
  std::uint32_t decode_concurrency = 1;
  /// Optional two-rank expert-parallel identity. World size one preserves
  /// the existing single-rank path and needs no communicator.
  std::uint32_t tp_rank = 0;
  std::uint32_t tp_world_size = 1;
  int hip_device = 0;
  std::shared_ptr<rocm::Communicator> communicator;
  /// Diagnostic observer for MoE inputs and outputs; see Executor::Options.
  /// The stream parameter is a `hipStream_t`, spelled without the HIP
  /// headers.
  std::function<void(const float*, std::size_t, ihipStream_t*)> moe_observer;
};

class Session;
class SessionSnapshot;

/// Gufo-owned API over the ROCm runtime: one resident model, any number of
/// sessions with shared projection batches and independent context state.
class Model final : public std::enable_shared_from_this<Model> {
public:
  ~Model();
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  [[nodiscard]] static std::shared_ptr<Model> Load(
      const std::string& model_path, const ModelOptions& options,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::unique_ptr<Session> CreateSession(
      core::SessionMode mode, std::uint32_t max_context,
      std::string* error_msg = nullptr);

  [[nodiscard]] std::vector<std::int32_t> Tokenize(std::string_view text) const;
  [[nodiscard]] std::string Decode(std::span<const std::int32_t> tokens) const;
  [[nodiscard]] std::string TokenText(std::int32_t token) const;
  [[nodiscard]] std::int32_t EosToken() const noexcept;
  [[nodiscard]] bool IsStopToken(std::int32_t token) const noexcept;
  [[nodiscard]] std::uint32_t VocabSize() const noexcept;
  /// Prompt tokens one prefill step feeds: under TP, up to two trunk
  /// batches run as an overlapped pair.
  [[nodiscard]] std::uint32_t PrefillCapacity() const noexcept;
  /// The first trunk batch's rows when a prefill step of `tokens` runs as a
  /// pair, else 0.
  [[nodiscard]] std::uint32_t PrefillPairLead(
      std::uint32_t tokens) const noexcept;
  [[nodiscard]] std::uint32_t MaxContext() const noexcept {
    return options_.max_context;
  }
  [[nodiscard]] bool HasMtp() const noexcept;
  /// DFlash v1 sidecar path (empty when unused).
  [[nodiscard]] const std::string& DFlashModelPath() const noexcept {
    return options_.dflash_model_path;
  }
  [[nodiscard]] std::uint32_t DFlashMaxDraftTokens() const noexcept {
    return options_.dflash_max_draft_tokens;
  }
  /// Tied matrices a DFlash drafter borrows: the token-embedding table and
  /// the output head, as device pointers plus their GGUF types.
  struct HeadTensorPointers {
    const void* token_embedding{nullptr};
    core::GgmlType token_embedding_type{core::GgmlType::kQ8_0};
    const void* output{nullptr};
    core::GgmlType output_type{core::GgmlType::kQ8_0};
  };
  [[nodiscard]] HeadTensorPointers HeadTensors() const;
  [[nodiscard]] std::uint32_t DecodeConcurrency() const noexcept {
    return options_.decode_concurrency;
  }
  [[nodiscard]] std::uint32_t TpRank() const noexcept {
    return options_.tp_rank;
  }
  [[nodiscard]] std::uint32_t TpWorldSize() const noexcept {
    return options_.tp_world_size;
  }
  [[nodiscard]] int HipDevice() const noexcept { return options_.hip_device; }
  [[nodiscard]] std::string ModelName() const;
  [[nodiscard]] const Config& config() const noexcept;
  [[nodiscard]] const MimoTokenizer& tokenizer() const noexcept {
    return *tokenizer_;
  }
  [[nodiscard]] std::size_t ResidentBytes() const noexcept;
  /// Worst-case private device state.
  [[nodiscard]] std::size_t SessionBytes(core::SessionMode mode,
                                         std::uint32_t context) const noexcept;
  [[nodiscard]] std::size_t DeferredScratchBytes() const;

private:
  Model() = default;

  ModelOptions options_;
  std::shared_ptr<core::GgufReader> reader_;
  std::shared_ptr<core::GgufReader> mtp_reader_;
  /// Optional requantized dense-projection sidecar (GUFO_MIMO_DENSE_SC).
  std::shared_ptr<core::GgufReader> dense_sc_reader_;
  /// Optional requantized expert sidecar (GUFO_EXPERT_SC): the routed expert
  /// tensors at Q3_K, grafted over the artifact's MXFP4 copies.
  std::shared_ptr<core::GgufReader> expert_sc_reader_;
  std::unique_ptr<ModelWeights> weights_;
  std::unique_ptr<MtpWeights> mtp_weights_;
  std::unique_ptr<MimoTokenizer> tokenizer_;
  std::unique_ptr<rocm::DeviceModel> device_;
  std::unique_ptr<rocm::Executor> executor_;
  MtpBatchController batch_policy_;

  friend class Session;
  friend class DFlashV1Target;
};

/// One conversation's context. Sync feeds a prompt (reusing whatever prefix
/// the session already holds), Evaluate appends one token, DecodeStep
/// runs a draft-verify cycle. The logits of the last token are kept.
class Session final {
public:
  ~Session();
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;

  /// Makes the session state equal to `prompt`: keeps the longest common
  /// prefix with the current tokens, reprocesses the rest.
  [[nodiscard]] bool Sync(std::span<const std::int32_t> prompt,
                          std::string* error_msg = nullptr);
  [[nodiscard]] bool Evaluate(std::int32_t token,
                              std::string* error_msg = nullptr);
  struct DecodeResult {
    std::vector<std::int32_t> tokens;
    bool stop{false};
  };
  /// Greedy decoding verifies deterministic drafts. Sampled decoding draws
  /// compact MTP proposals and uses target/draft rejection with a residual
  /// correction. History contains only committed tokens; stochastic RNG
  /// consumption includes proposal and verification draws. A one-token
  /// budget or a model without MTP uses ordinary decoding. Benchmarks may
  /// continue past EOS by setting stop_at_eos to false.
  [[nodiscard]] bool DecodeStep(std::size_t max_tokens,
                                sampling::SamplerState& sampler,
                                DecodeResult* result,
                                std::string* error_msg = nullptr,
                                bool stop_at_eos = true);
  struct BatchOutcome {
    bool completed{false};
    std::string error{};
  };
  struct DecodeRequest {
    Session* session;
    std::size_t max_tokens;
    sampling::SamplerState* sampler;
    DecodeResult* result;
    bool stop_at_eos{true};
    BatchOutcome* outcome{nullptr};
  };
  /// A false return may include completed peers. Inspect each outcome; only
  /// sessions whose persistent state was partly mutated become invalid.
  /// `plan`, when given, fixes the batch's draft count (none: each session
  /// chooses from its own acceptance history) instead of choosing it here.
  [[nodiscard]] static bool DecodeBatch(
      std::span<const DecodeRequest> requests, std::string* error_msg = nullptr,
      const std::optional<std::uint32_t>* plan = nullptr);
  /// The draft count DecodeBatch would choose for these requests: one for
  /// the whole batch when every request decodes greedily with MTP, from
  /// measured cycle costs, else none. A TP2 peer runs the batch with it.
  [[nodiscard]] static std::optional<std::uint32_t> PlanBatch(
      std::span<const DecodeRequest> requests);
  struct AdvanceRequest {
    Session* session;
    std::int32_t token;
    BatchOutcome* outcome{nullptr};
  };
  [[nodiscard]] static bool EvaluateBatch(
      std::span<const AdvanceRequest> requests,
      std::string* error_msg = nullptr);
  [[nodiscard]] std::span<const float> Logits() const noexcept {
    return valid_ ? std::span<const float>(logits_) : std::span<const float>{};
  }
  [[nodiscard]] std::uint32_t Position() const noexcept;
  [[nodiscard]] std::uint32_t ContextSize() const noexcept;
  [[nodiscard]] std::span<const std::int32_t> Tokens() const noexcept {
    return valid_ ? std::span<const std::int32_t>(tokens_)
                  : std::span<const std::int32_t>{};
  }
  void Reset();
  /// Cancellation is checked at model, layer and snapshot-copy boundaries.
  void SetCancellationCheck(std::function<bool()> check);
  [[nodiscard]] bool IsValid() const noexcept { return valid_; }
  [[nodiscard]] std::size_t AllocatedBytes() const noexcept;
  /// A new request reusing cached context starts its own acceptance history.
  void ResetDraftPolicy() noexcept { draft_length_.Reset(); }

  struct SpeculativeStats {
    std::uint64_t cycles{0};
    std::uint64_t drafted{0};
    std::uint64_t accepted{0};
  };
  [[nodiscard]] const SpeculativeStats& Statistics() const noexcept {
    return stats_;
  }

  /// Compatibility version; bump on payload or inference arithmetic changes.
  static constexpr std::uint32_t kSnapshotPayloadVersion = 1;
  /// Bytes a snapshot of the current context occupies.
  [[nodiscard]] std::uint64_t SnapshotBytes() const;
  /// Captures the whole context (tokens, device caches, draft-block state,
  /// last logits) into host memory. The speculative length controller is
  /// preserved for stochastic replay.
  [[nodiscard]] std::unique_ptr<SessionSnapshot> SaveSnapshot(
      std::string* error_msg = nullptr) const;
  /// Replaces this session's context with a snapshot of the same model.
  [[nodiscard]] bool RestoreSnapshot(const SessionSnapshot& snapshot,
                                     std::string* error_msg = nullptr);
  [[nodiscard]] bool RestoreSnapshot(std::span<const std::uint8_t> payload,
                                     std::string* error_msg = nullptr);

private:
  [[nodiscard]] bool RestoreSnapshotPayload(
      std::span<const std::uint8_t> payload, std::string* error_msg);
  friend class Model;
  Session(std::shared_ptr<Model> model, std::unique_ptr<rocm::Session> session);

  bool Feed(std::span<const std::int32_t> tokens, std::string* error_msg,
            bool prefill = false);
  /// Trunk rows the draft block may still read: [hidden_base_, size).
  [[nodiscard]] std::uint32_t KeptHiddenRows() const noexcept;
  bool DraftCatchUp(std::int32_t next_token, bool propose,
                    std::string* error_msg,
                    MtpCandidateLogits* candidates = nullptr);
  bool DraftReplay(std::int32_t next_token, std::vector<std::int32_t>* replay,
                   std::int32_t* hidden_row, std::string* error_msg) const;
  static bool DraftCatchUpBatch(std::span<const AdvanceRequest> requests,
                                std::string* error_msg);
  template<class Request>
  static bool RunIsolatedBatch(
      std::span<const Request> requests, std::string* error_msg,
      const std::optional<std::uint32_t>* plan = nullptr);
  static bool DecodeBatchImpl(std::span<const DecodeRequest> requests,
                              const std::optional<std::uint32_t>* plan,
                              std::string* error_msg);
  static bool EvaluateBatchImpl(std::span<const AdvanceRequest> requests,
                                std::string* error_msg);
  struct PendingDecode;
  bool PrepareDecode(const DecodeRequest& request, PendingDecode* pending,
                     std::string* error_msg, bool defer_head = false,
                     std::optional<std::uint32_t> batch_drafts = {});
  static void AppendDraft(PendingDecode& pending);
  bool FinishDecode(const DecodeRequest& request, const PendingDecode& pending,
                    std::string* error_msg);

  std::shared_ptr<Model> model_;
  std::unique_ptr<rocm::Session> session_;
  std::vector<std::int32_t> tokens_;
  std::vector<float> logits_;
  std::int32_t draft_token_{0};
  std::vector<float> verify_logits_;
  std::uint32_t hidden_base_{0};  ///< first position whose hidden row is kept
  MtpLengthController draft_length_;
  SpeculativeStats stats_;
  bool valid_{true};
  /// Set when the depth cost model latches this request to plain decoding:
  /// past the crossover the predictor's catch-up cannot be repaid, so the
  /// cycle stops paying it. Never cleared before Reset (the context only
  /// grows), which is what makes skipping the catch-up safe: the draft block
  /// is left behind deliberately and is never asked to replay again.
  bool mtp_depth_latched_{false};
  [[nodiscard]] bool MtpEnabled() const noexcept;
  /// Speculation this cycle may use. Differs from MtpEnabled() once the
  /// depth latch has fired.
  [[nodiscard]] bool MtpActive() const noexcept {
    return MtpEnabled() && !mtp_depth_latched_;
  }
};

/// Immutable host copy of a session context. The same bytes restore in
/// memory and persist to disk.
class SessionSnapshot final {
public:
  ~SessionSnapshot() = default;
  SessionSnapshot(const SessionSnapshot&) = delete;
  SessionSnapshot& operator=(const SessionSnapshot&) = delete;
  SessionSnapshot(SessionSnapshot&&) = delete;
  SessionSnapshot& operator=(SessionSnapshot&&) = delete;

  [[nodiscard]] std::uint64_t SizeBytes() const noexcept { return size_; }
  [[nodiscard]] std::span<const std::uint8_t> bytes() const noexcept {
    return {data_.get(), size_};
  }
  [[nodiscard]] bool CopyTo(std::span<std::uint8_t> destination) const noexcept;

private:
  explicit SessionSnapshot(std::uint64_t size);

  std::unique_ptr<std::uint8_t[]> data_;
  std::uint64_t size_{0};

  friend class Session;
};

}  // namespace gufo::models::mimo2

#endif  // GUFO_MODELS_MIMO2_ENGINE_HPP_
