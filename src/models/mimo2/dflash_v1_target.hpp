#ifndef GUFO_MODELS_MIMO2_DFLASH_V1_TARGET_HPP_
#define GUFO_MODELS_MIMO2_DFLASH_V1_TARGET_HPP_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/speculative/speculative_verifier.hpp"

#if defined(ENGINE_ENABLE_HIP)

namespace gufo::models::mimo2 {

class Model;
namespace rocm {
class Executor;
class Session;
}  // namespace rocm

/// ISpeculativeTargetExecutor over one mimo2 autoregressive session, for the
/// generic SpeculativeVerifier driving a DFlash draft backend: prompt prefill
/// with target-tap capture, single-token decode, one-batch verification with
/// per-row hidden/logits capture, and the save/rewind/commit transaction the
/// verifier's rollback path needs. Verification logits stay host-side, so the
/// default host rejection sampling applies to sampled requests.
class DFlashV1Target final : public speculative::ISpeculativeTargetExecutor {
public:
  ~DFlashV1Target() override;
  DFlashV1Target(const DFlashV1Target&) = delete;
  DFlashV1Target& operator=(const DFlashV1Target&) = delete;

  [[nodiscard]] static std::unique_ptr<DFlashV1Target> Create(
      std::shared_ptr<Model> model, std::uint32_t max_context,
      std::string* error_msg);

  // ── ISpeculativeTargetExecutor ───────────────────────────────────────────
  void Reset() noexcept override;
  [[nodiscard]] tokenization::TokenId ForwardPromptBatch(
      std::span<const tokenization::TokenId> prompt_tokens) override;
  [[nodiscard]] tokenization::TokenId ForwardPromptSuffix(
      std::span<const tokenization::TokenId> tokens, std::uint32_t position,
      bool compute_logits) override;
  [[nodiscard]] tokenization::TokenId ForwardToken(
      tokenization::TokenId token_id, std::uint32_t pos,
      bool compute_logits) override;
  void SaveState(std::uint32_t valid_context) override;
  void RestoreState() override;
  void SetPromptHiddenCapture(
      bool enabled,
      std::span<const std::uint32_t> target_layer_ids) override;
  [[nodiscard]] std::span<const float> GetPromptHiddenStates()
      const noexcept override;
  [[nodiscard]] std::span<const float> CopyLastHidden() override;
  [[nodiscard]] std::span<const float> CopyLastLogits() override;
  [[nodiscard]] std::span<const float> CopyVerificationLogits(
      std::size_t row) override;
  [[nodiscard]] speculative::VerificationChunkResult ForwardVerificationChunk(
      std::span<const tokenization::TokenId> candidate_tokens,
      std::uint32_t start_pos, bool capture_hidden,
      bool capture_logits) override;
  void CommitVerificationChunk(
      std::span<const tokenization::TokenId> committed_tokens,
      std::uint32_t start_pos) override;
  [[nodiscard]] tokenization::TokenId GetEosTokenId() const noexcept override;
  [[nodiscard]] bool IsStopToken(
      tokenization::TokenId token) const noexcept override;
  [[nodiscard]] std::size_t VocabularySize() const noexcept override;
  [[nodiscard]] std::string_view DecodeToken(
      tokenization::TokenId token_id) const noexcept override;

  // ── Runner-side state ────────────────────────────────────────────────────
  [[nodiscard]] const rocm::Executor& executor() const noexcept {
    return *executor_;
  }
  [[nodiscard]] rocm::Session& session() noexcept { return *session_; }
  [[nodiscard]] std::uint32_t Position() const noexcept;
  [[nodiscard]] std::size_t AllocatedBytes() const noexcept;
  /// Host logits of the last forward, `[rows][vocab]`.
  [[nodiscard]] std::span<const float> LastLogits() const noexcept {
    return logits_;
  }
  [[nodiscard]] std::size_t LastLogitRows() const noexcept {
    return last_logit_rows_;
  }
  /// Mirrors the engine's session feeding for runners that drive the target
  /// directly (a prompt exactly equal to the retained prefix does nothing).
  [[nodiscard]] bool Feed(std::span<const std::int32_t> tokens, bool prefill,
                          std::string* error_msg);
  /// Appends one token and keeps its logits (the plain AR step).
  [[nodiscard]] bool Evaluate(std::int32_t token, std::string* error_msg);
  /// One plain autoregressive row per target, merged into a single trunk pass:
  /// `tokens[i]` is fed to `targets[i]` and each target's own logits are
  /// refreshed for its row. No drafts and no tap capture, so a shared cycle
  /// pays the engine's own batch rate instead of one verification per session.
  [[nodiscard]] static bool ForwardRowsBatch(
      std::span<DFlashV1Target*> targets,
      std::span<const std::int32_t> tokens, std::string* error_msg);
  /// Drops the tap capture a speculation path armed: rows fed by the caller
  /// never reach the drafter, so their captures would only cost time.
  [[nodiscard]] bool DisarmCapture(std::string* error_msg);

private:
  DFlashV1Target(std::shared_ptr<Model> model,
                 std::unique_ptr<rocm::Session> session);

  [[nodiscard]] tokenization::TokenId GreedyFromRow(std::size_t row) const;
  /// Applies a deferred restore: the verifier restores the frontier and then
  /// commits the accepted prefix, and that commit is a rollback over the very
  /// batch a rewind would discard, so the rewind itself is deferred to the
  /// next forward call.
  void EnsureRestored();
  [[nodiscard]] tokenization::TokenId FeedPrompt(
      std::span<const tokenization::TokenId> tokens, std::uint32_t position);

  std::shared_ptr<Model> model_;
  std::unique_ptr<rocm::Session> session_;
  rocm::Executor* executor_{nullptr};
  std::vector<std::int32_t> token_scratch_;
  std::vector<float> logits_;  ///< [last_logit_rows_][vocab]
  std::size_t last_logit_rows_{0};
  std::vector<float> verification_logits_;  ///< last chunk, [rows][vocab]
  std::size_t verification_rows_{0};
  std::uint32_t saved_position_{0};
  bool restore_pending_{false};
  bool capture_armed_{false};
};

}  // namespace gufo::models::mimo2

#endif  // defined(ENGINE_ENABLE_HIP)

#endif  // GUFO_MODELS_MIMO2_DFLASH_V1_TARGET_HPP_