#if defined(ENGINE_ENABLE_HIP)
#include "src/models/mimo2/dflash_v1_target.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <utility>

#include "src/models/mimo2/engine.hpp"
#include "src/models/mimo2/kernels/rocm/executor.hpp"

namespace gufo::models::mimo2 {
namespace {

const bool kDfTrace = std::getenv("GUFO_DFLASH_TRACE") != nullptr;

void DfTrace(const char* what) {
  if (kDfTrace) {
    fprintf(stderr, "[DFTRACE] %s\n", what);
    fflush(stderr);
  }
}

[[noreturn]] void Throw(std::string_view what, const std::string& detail) {
  throw std::runtime_error(std::string(what) + ": " + detail);
}

std::int32_t ToEngineToken(tokenization::TokenId token) {
  return static_cast<std::int32_t>(token);
}

}  // namespace

DFlashV1Target::DFlashV1Target(std::shared_ptr<Model> model,
                               std::unique_ptr<rocm::Session> session)
    : model_(std::move(model)), session_(std::move(session)) {}

DFlashV1Target::~DFlashV1Target() = default;

std::unique_ptr<DFlashV1Target> DFlashV1Target::Create(
    std::shared_ptr<Model> model, std::uint32_t max_context,
    std::string* error_msg) {
  if (model == nullptr || max_context == 0) {
    if (error_msg != nullptr) {
      *error_msg = "DFlash v1 target needs a model and a context size";
    }
    return nullptr;
  }
  auto session = model->executor_->CreateSession(core::SessionMode::kAutoregressive,
                                                 max_context, error_msg);
  if (session == nullptr) {
    return nullptr;
  }
  std::unique_ptr<DFlashV1Target> target(
      new DFlashV1Target(std::move(model), std::move(session)));
  target->executor_ = target->model_->executor_.get();
  target->logits_.resize(target->model_->VocabSize());
  return target;
}

std::uint32_t DFlashV1Target::Position() const noexcept {
  return session_->position();
}

std::size_t DFlashV1Target::AllocatedBytes() const noexcept {
  return session_->AllocatedBytes();
}

void DFlashV1Target::Reset() noexcept {
  session_->Reset();
  last_logit_rows_ = 0;
  verification_rows_ = 0;
  verification_logits_.clear();
  saved_position_ = 0;
  if (capture_armed_) {
    executor_->ClearPromptHiddenStates(*session_);
  }
}

void DFlashV1Target::SetPromptHiddenCapture(
    bool enabled, std::span<const std::uint32_t> target_layer_ids) {
  DfTrace("arm capture enter");
  std::string error;
  if (enabled) {
    if (target_layer_ids.empty()) {
      Throw("DFlash v1 tap capture", "no target layer ids");
    }
    if (!executor_->SetTargetLayerCapture(*session_, target_layer_ids,
                                          &error)) {
      Throw("DFlash v1 tap capture", error);
    }
  } else {
    executor_->SetTargetLayerCapture(*session_, {}, &error);
  }
  capture_armed_ = enabled;
  if (enabled) {
    executor_->ClearPromptHiddenStates(*session_);
  }
  DfTrace("arm capture exit");
}

std::span<const float> DFlashV1Target::GetPromptHiddenStates() const noexcept {
  return executor_->GetPromptHiddenStates(*session_);
}

std::span<const float> DFlashV1Target::CopyLastHidden() {
  return executor_->CopyLastHidden(*session_);
}

std::span<const float> DFlashV1Target::CopyLastLogits() {
  if (last_logit_rows_ == 0) {
    return {};
  }
  return std::span<const float>(logits_).last(model_->VocabSize());
}

std::span<const float> DFlashV1Target::CopyVerificationLogits(
    std::size_t row) {
  const std::size_t vocab = model_->VocabSize();
  if (row >= verification_rows_ ||
      verification_logits_.size() != verification_rows_ * vocab) {
    return {};
  }
  return std::span<const float>(verification_logits_)
      .subspan(row * vocab, vocab);
}

tokenization::TokenId DFlashV1Target::GreedyFromRow(std::size_t row) const {
  const std::size_t vocab = model_->VocabSize();
  if ((row + 1) * vocab > logits_.size()) {
    throw std::logic_error("DFlash v1 target has no logits for the row");
  }
  const float* begin = logits_.data() + row * vocab;
  const float* best = std::max_element(begin, begin + vocab);
  return static_cast<tokenization::TokenId>(best - begin);
}

bool DFlashV1Target::Feed(std::span<const std::int32_t> tokens, bool prefill,
                          std::string* error_msg) {
  if (tokens.empty()) {
    return true;
  }
  if (kDfTrace) {
    fprintf(stderr, "[DFTRACE] feed prefill=%d n=%zu pos=%u\n",
            prefill ? 1 : 0, tokens.size(), session_->position());
    fflush(stderr);
  }
  const std::uint64_t end = static_cast<std::uint64_t>(session_->position()) +
                            tokens.size();
  if (end > session_->max_context()) {
    if (error_msg != nullptr) {
      *error_msg = "DFlash v1 target context is full";
    }
    return false;
  }
  const std::uint32_t chunk =
      prefill ? executor_->PrefillChunk(*session_) : executor_->max_batch();
  for (std::size_t offset = 0; offset < tokens.size(); offset += chunk) {
    const std::size_t rows = std::min<std::size_t>(chunk, tokens.size() - offset);
    last_logit_rows_ = 1;
    if (!executor_->Forward(
            *session_, tokens.subspan(offset, rows), 1, logits_.data(),
            prefill ? rocm::Executor::ForwardMode::kPrefill
                    : rocm::Executor::ForwardMode::kDecode,
            error_msg)) {
      last_logit_rows_ = 0;
      return false;
    }
  }
  return true;
}

bool DFlashV1Target::Evaluate(std::int32_t token, std::string* error_msg) {
  return Feed(std::span<const std::int32_t>(&token, 1), /*prefill=*/false,
              error_msg);
}

bool DFlashV1Target::DisarmCapture(std::string* error_msg) {
  if (!capture_armed_) {
    return true;
  }
  if (!executor_->SetTargetLayerCapture(*session_, {}, error_msg)) {
    return false;
  }
  executor_->ClearPromptHiddenStates(*session_);
  capture_armed_ = false;
  return true;
}

bool DFlashV1Target::ForwardRowsBatch(
    std::span<DFlashV1Target*> targets,
    std::span<const std::int32_t> tokens, std::string* error_msg) {
  if (targets.empty() || targets.size() != tokens.size()) {
    if (error_msg != nullptr) {
      *error_msg = "DFlash v1 row batch needs exactly one token per target";
    }
    return false;
  }
  rocm::Executor& exec = *targets.front()->executor_;
  for (const auto* target : targets) {
    if (target->executor_ != &exec) {
      if (error_msg != nullptr) {
        *error_msg = "DFlash v1 row batch spans two executors";
      }
      return false;
    }
  }
  std::vector<rocm::Executor::BatchItem> items;
  items.reserve(targets.size());
  for (std::size_t i = 0; i < targets.size(); ++i) {
    DFlashV1Target* target = targets[i];
    target->EnsureRestored();
    if (target->session_->position() >= target->session_->max_context()) {
      if (error_msg != nullptr) {
        *error_msg = "DFlash v1 row batch target context is full";
      }
      return false;
    }
    items.push_back({target->session_.get(), tokens.subspan(i, 1),
                     /*speculative=*/false});
  }
  std::string error;
  if (!exec.ForwardBatch(items, &error)) {
    if (error_msg != nullptr) {
      *error_msg = error;
    }
    return false;
  }
  for (std::size_t i = 0; i < targets.size(); ++i) {
    DFlashV1Target* target = targets[i];
    target->last_logit_rows_ = 1;
    if (!exec.SelectBatchLogits(static_cast<std::uint32_t>(i), 1,
                                target->logits_.data(), &error)) {
      target->last_logit_rows_ = 0;
      if (error_msg != nullptr) {
        *error_msg = error;
      }
      return false;
    }
  }
  return true;
}

tokenization::TokenId DFlashV1Target::FeedPrompt(
    std::span<const tokenization::TokenId> tokens, std::uint32_t position) {
  EnsureRestored();
  DfTrace("feed_prompt enter");
  if (position != session_->position()) {
    Throw("DFlash v1 prompt position", "does not match the session frontier");
  }
  token_scratch_.resize(tokens.size());
  std::transform(tokens.begin(), tokens.end(), token_scratch_.begin(),
                 ToEngineToken);
  std::string error;
  if (!Feed(token_scratch_, /*prefill=*/true, &error)) {
    Throw("DFlash v1 prompt prefill", error);
  }
  return GreedyFromRow(last_logit_rows_ - 1);
}

tokenization::TokenId DFlashV1Target::ForwardPromptBatch(
    std::span<const tokenization::TokenId> prompt_tokens) {
  return FeedPrompt(prompt_tokens, session_->position());
}

tokenization::TokenId DFlashV1Target::ForwardPromptSuffix(
    std::span<const tokenization::TokenId> tokens, std::uint32_t position,
    bool compute_logits) {
  (void)compute_logits;  // the head always downloads this session's logits
  // A suffix feeds only its own rows: the retained prefix keeps both its KV
  // and its captured tap rows.
  if (capture_armed_) {
    executor_->ClearPromptHiddenStates(*session_);
  }
  return FeedPrompt(tokens, position);
}

tokenization::TokenId DFlashV1Target::ForwardToken(
    tokenization::TokenId token_id, std::uint32_t pos, bool compute_logits) {
  EnsureRestored();
  if (pos != session_->position()) {
    Throw("DFlash v1 token position", "does not match the session frontier");
  }
  const std::int32_t token = ToEngineToken(token_id);
  std::string error;
  if (!Feed(std::span<const std::int32_t>(&token, 1), /*prefill=*/false,
            &error)) {
    Throw("DFlash v1 token forward", error);
  }
  if (!compute_logits) {
    last_logit_rows_ = 0;
    return 0;
  }
  return GreedyFromRow(last_logit_rows_ - 1);
}

void DFlashV1Target::SaveState(std::uint32_t valid_context) {
  if (valid_context != session_->position()) {
    Throw("DFlash v1 save state", "does not match the session frontier");
  }
  saved_position_ = valid_context;
}

void DFlashV1Target::RestoreState() { restore_pending_ = true; }

namespace {

double DfNowMs() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

}  // namespace

void DFlashV1Target::EnsureRestored() {
  if (!restore_pending_) {
    return;
  }
  restore_pending_ = false;
  std::string error;
  if (!executor_->Rewind(*session_, saved_position_, &error)) {
    Throw("DFlash v1 restore state", error);
  }
}

speculative::VerificationChunkResult DFlashV1Target::ForwardVerificationChunk(
    std::span<const tokenization::TokenId> candidate_tokens,
    std::uint32_t start_pos, bool capture_hidden, bool capture_logits) {
  EnsureRestored();
  if (kDfTrace) {
    fprintf(stderr, "[DFTRACE] verify chunk enter rows=%zu pos=%u hidden=%d\n",
            candidate_tokens.size(), start_pos, capture_hidden ? 1 : 0);
    fflush(stderr);
  }
  if (start_pos != session_->position()) {
    Throw("DFlash v1 verification position",
          "does not match the session frontier");
  }
  if (candidate_tokens.size() > executor_->max_speculative()) {
    Throw("DFlash v1 verification chunk", "exceeds the speculative batch");
  }
  token_scratch_.resize(candidate_tokens.size());
  std::transform(candidate_tokens.begin(), candidate_tokens.end(),
                 token_scratch_.begin(), ToEngineToken);
  const std::size_t rows = token_scratch_.size();
  verification_logits_.resize(rows * model_->VocabSize());
  std::string error;
  // One multi-row verification batch; every row's logits are downloaded so
  // predictions and the sampled-verification path share them.
  const double t0 = kDfTrace ? DfNowMs() : 0.0;
  if (!executor_->Forward(*session_, token_scratch_, static_cast<std::uint32_t>(rows),
                          verification_logits_.data(),
                          rocm::Executor::ForwardMode::kVerify, &error)) {
    Throw("DFlash v1 verification forward", error);
  }
  const double t1 = kDfTrace ? DfNowMs() : 0.0;
  verification_rows_ = rows;
  last_logit_rows_ = 0;

  speculative::VerificationChunkResult result;
  result.predictions.reserve(rows);
  const std::size_t vocab = model_->VocabSize();
  for (std::size_t row = 0; row < rows; ++row) {
    const float* begin = verification_logits_.data() + row * vocab;
    const float* best = std::max_element(begin, begin + vocab);
    result.predictions.push_back(
        static_cast<tokenization::TokenId>(best - begin));
  }
  if (const char* dir = std::getenv("GUFO_DFLASH_DUMP")) {
    static std::uint32_t dump_index = 0;
    char name[512];
    std::snprintf(name, sizeof(name), "%s/target_%u.f32", dir, dump_index);
    if (FILE* dump = std::fopen(name, "wb")) {
      std::fwrite(verification_logits_.data(), sizeof(float),
                  verification_rows_ * vocab, dump);
      std::fclose(dump);
    }
    std::snprintf(name, sizeof(name), "%s/target_%u.txt", dir, dump_index);
    if (FILE* dump = std::fopen(name, "w")) {
      for (const auto token : token_scratch_) {
        fprintf(dump, "token %d\n", token);
      }
      for (const auto token : result.predictions) {
        fprintf(dump, "predict %u\n", token);
      }
      std::fclose(dump);
    }
    fprintf(stderr, "[DFDUMP] target row -> %s/target_%u.f32\n", dir,
            dump_index++);
    fflush(stderr);
  }
  if (kDfTrace) {
    fprintf(stderr, "[DFTRACE] target predicts");
    for (const auto token : result.predictions) {
      fprintf(stderr, " %u", token);
    }
    fprintf(stderr, "\n");
    fflush(stderr);
  }
  const double t2 = kDfTrace ? DfNowMs() : 0.0;
  if (capture_hidden) {
    const auto hidden = executor_->GetVerificationHiddenStates(*session_);
    result.hidden_width = executor_->TargetCaptureWidth(*session_);
    if (result.hidden_width == 0 ||
        hidden.size() != rows * result.hidden_width) {
      Throw("DFlash v1 verification hidden states", "are incomplete");
    }
    result.hidden_states.assign(hidden.begin(), hidden.end());
  }
  if (capture_logits) {
    result.vocab_size = vocab;
    result.logits = verification_logits_;
  }
  if (kDfTrace) {
    const double t3 = DfNowMs();
    fprintf(stderr,
            "[DFTRACE] verify split fwd=%.1f post=%.1f hidden=%.1f rows=%zu\n",
            t1 - t0, t2 - t1, t3 - t2, rows);
    fflush(stderr);
  }
  return result;
}

void DFlashV1Target::CommitVerificationChunk(
    std::span<const tokenization::TokenId> committed_tokens,
    std::uint32_t start_pos) {
  (void)start_pos;
  // The verification batch already wrote a K/V row at every candidate's final
  // position, so the commit only drops the rejected tail: no second trunk
  // forward is needed, and the rows past the kept prefix go stale until the
  // next replay overwrites them.
  restore_pending_ = false;
  if (committed_tokens.empty()) {
    return;
  }
  std::string error;
  if (!executor_->Rollback(*session_,
                           static_cast<std::uint32_t>(committed_tokens.size()),
                           &error)) {
    Throw("DFlash v1 verification commit", error);
  }
}

tokenization::TokenId DFlashV1Target::GetEosTokenId() const noexcept {
  return model_->tokenizer().GetEosTokenId();
}

bool DFlashV1Target::IsStopToken(tokenization::TokenId token) const noexcept {
  return model_->tokenizer().IsStopToken(token);
}

std::size_t DFlashV1Target::VocabularySize() const noexcept {
  return model_->VocabSize();
}

std::string_view DFlashV1Target::DecodeToken(
    tokenization::TokenId token_id) const noexcept {
  return model_->tokenizer().DecodeToken(token_id);
}

}  // namespace gufo::models::mimo2
#endif  // defined(ENGINE_ENABLE_HIP)