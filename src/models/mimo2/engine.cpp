#include "src/models/mimo2/engine.hpp"

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <type_traits>

#include "src/core/gguf_reader.hpp"
#include "src/models/mimo2/distributed/tp_partition.hpp"
#include "src/models/mimo2/kernels/rocm/communicator.hpp"
#include "src/models/mimo2/kernels/rocm/device_model.hpp"
#include "src/models/mimo2/kernels/rocm/executor.hpp"
#include "src/models/mimo2/kernels/rocm/kernels.hpp"
#include "src/models/mimo2/mtp_sampling.hpp"
#include "src/models/mimo2/weights.hpp"

namespace gufo::models::mimo2 {
namespace {

// gfx1151 prefill sweep carried over from the Flash-Next tuning; larger
// chunks used more scratch without improving throughput.
constexpr std::uint32_t kPrefillChunkTokens = 2048;

/// DFlash bring-up audit (temporary scaffold; the draft backend will arm
/// capture itself later): ANY value arms the five MiMo target taps on every
/// fresh prefill, and "rows" additionally dumps a per-row hash of the
/// captured host rows so chunk offsets can be compared across runs.
const char* const kTapAudit = std::getenv("GUFO_TAP_AUDIT");
constexpr std::array<std::uint32_t, 5> kTapAuditLayers{0, 11, 23, 35, 47};
bool TapAuditRows() noexcept {
  return kTapAudit != nullptr && std::strcmp(kTapAudit, "rows") == 0;
}

/// Reports the rows a prefill captured: the summary keeps each tap's RMS of
/// the last row, "rows" hashes every row (FNV-1a over the raw floats).
void TapAuditReport(rocm::Executor& exec, rocm::Session& session,
                    std::size_t prompt_tokens) {
  const auto taps = exec.GetPromptHiddenStates(session);
  const std::size_t width = kTapAuditLayers.size() * exec.config().hidden_size;
  const std::size_t rows = width == 0 ? 0 : taps.size() / width;
  fprintf(stderr, "TAPAUDIT prompt=%zu rows=%zu width=%zu\n", prompt_tokens,
          rows, width);
  if (taps.empty() || width == 0) {
    return;
  }
  const auto stats = [&](const float* row, std::size_t begin,
                         std::size_t count, double* rms) {
    double sq = 0;
    for (std::size_t j = begin; j < begin + count; ++j) {
      sq += static_cast<double>(row[j]) * row[j];
    }
    *rms = std::sqrt(sq / static_cast<double>(count));
  };
  const float* last = taps.data() + (rows - 1) * width;
  fprintf(stderr, "TAPLAST");
  for (std::size_t t = 0; t < kTapAuditLayers.size(); ++t) {
    double rms = 0;
    stats(last, t * exec.config().hidden_size, exec.config().hidden_size, &rms);
    fprintf(stderr, " tap%u=%.6f", kTapAuditLayers[t], rms);
  }
  fprintf(stderr, "\n");
  if (!TapAuditRows()) {
    return;
  }
  for (std::size_t i = 0; i < rows; ++i) {
    const float* row = taps.data() + i * width;
    std::uint64_t hash = 1469598103934665603ULL;
    double sq = 0;
    float mn = row[0];
    float mx = row[0];
    bool finite = true;
    for (std::size_t j = 0; j < width; ++j) {
      const float v = row[j];
      std::uint32_t bits = 0;
      std::memcpy(&bits, &v, sizeof(bits));
      hash = (hash ^ bits) * 1099511628211ULL;
      finite = finite && std::isfinite(v);
      sq += static_cast<double>(v) * v;
      mn = std::min(mn, v);
      mx = std::max(mx, v);
    }
    fprintf(stderr, "TAPROW i=%zu hash=%016llx rms=%.6f min=%.4f max=%.4f%s\n",
            i, static_cast<unsigned long long>(hash),
            std::sqrt(sq / static_cast<double>(width)), mn, mx,
            finite ? "" : " NONFINITE");
  }
}

void AssignError(std::string* error_msg, std::string_view message) {
  if (error_msg != nullptr) {
    *error_msg = message;
  }
}

constexpr std::array<char, 8> kSnapshotMagic{'M', 'I', 'M', 'O', '2',
                                             'E', 'S', '1'};

/// Host-side session fields ahead of the executor payload: the token
/// history and the logits of the last token.
struct SessionSnapshotHeader {
  std::array<char, 8> magic;
  std::uint32_t version;
  std::uint32_t vocab_size;
  std::uint32_t token_count;
  std::uint32_t hidden_rows;
  std::uint64_t executor_bytes;
  MtpLengthState draft_policy;
  std::uint32_t policy_concurrency;
  /// Nonzero when the depth latch had fired: the restored session resumes
  /// with the draft block left behind, exactly as it was saved. This
  /// replaces the former reserved word, so the layout and payload version
  /// stay as they were and older snapshots read as "not latched".
  std::uint32_t mtp_depth_latched{0};
};
static_assert(std::is_trivially_copyable_v<SessionSnapshotHeader>);

std::uint64_t SessionSnapshotHostBytes(std::uint32_t token_count,
                                       std::uint32_t vocab_size) {
  return sizeof(SessionSnapshotHeader) +
         std::uint64_t{token_count} * sizeof(std::int32_t) +
         std::uint64_t{vocab_size} * sizeof(float);
}

}  // namespace

Model::~Model() = default;

std::shared_ptr<Model> Model::Load(const std::string& model_path,
                                   const ModelOptions& options,
                                   std::string* error_msg) {
  std::shared_ptr<Model> m(new Model());
  if (options.decode_concurrency == 0 || options.decode_concurrency > 8) {
    AssignError(error_msg, "decode concurrency must be between one and eight");
    return nullptr;
  }
  if (options.hip_device < 0) {
    AssignError(error_msg, "HIP device index must be nonnegative");
    return nullptr;
  }
  if (options.tp_world_size != 1 && options.tp_world_size != 2) {
    AssignError(error_msg,
                "MiMo-V2 currently supports TP world sizes one or two");
    return nullptr;
  }
  if (options.tp_world_size == 1 && options.tp_rank != 0) {
    AssignError(error_msg, "single-rank MiMo-V2 requires rank zero");
    return nullptr;
  }
  if (options.tp_world_size > 1 && !options.communicator) {
    AssignError(error_msg, "TP=2 MiMo-V2 requires a communicator");
    return nullptr;
  }
  if (options.tp_world_size > 1 &&
      (options.communicator->world_size() != options.tp_world_size ||
       options.communicator->rank() != options.tp_rank ||
       options.communicator->device_index() != options.hip_device)) {
    AssignError(error_msg,
                "TP communicator identity does not match model rank/device");
    return nullptr;
  }
  if (hipSetDevice(options.hip_device) != hipSuccess) {
    AssignError(error_msg, "HIP device selection failed");
    return nullptr;
  }
  if (options.max_draft_tokens == 0) {
    AssignError(error_msg, "draft token limit must be positive");
    return nullptr;
  }
  if (!options.mtp_model_path.empty() &&
      options.max_draft_tokens > kMaxMtpDraftTokens) {
    AssignError(error_msg, "MiMo-V2 MTP supports at most seven draft tokens");
    return nullptr;
  }
  if (!options.mtp_model_path.empty() && !options.dflash_model_path.empty()) {
    AssignError(error_msg,
                "MiMo-V2 loads either the MTP or the DFlash sidecar, not both");
    return nullptr;
  }
  if (options.dflash_model_path.empty() == false &&
      options.dflash_max_draft_tokens == 0) {
    AssignError(error_msg, "DFlash draft token limit must be positive");
    return nullptr;
  }
  m->options_ = options;
  m->reader_ = core::GgufReader::OpenFile(model_path, error_msg);
  if (!m->reader_) {
    return nullptr;
  }
  auto weights = ModelWeights::Bind(*m->reader_, error_msg);
  if (!weights) {
    return nullptr;
  }
  m->weights_ = std::make_unique<ModelWeights>(std::move(*weights));
  const Config& c = m->weights_->config;
  std::optional<distributed::TpPartition> partition;
  const distributed::TpPartition* partition_ptr = nullptr;
  if (options.tp_world_size > 1) {
    partition = distributed::TpPartition::Create(
        c.expert_ff, options.tp_rank, options.tp_world_size, error_msg);
    if (!partition) {
      return nullptr;
    }
    partition_ptr = &*partition;
  }
  if (options.max_context == 0 || options.max_context > c.context_length) {
    AssignError(error_msg, "context exceeds the model's " +
                               std::to_string(c.context_length) + " tokens");
    return nullptr;
  }
  m->tokenizer_ = MimoTokenizer::CreateFromGguf(*m->reader_, error_msg);
  if (!m->tokenizer_) {
    return nullptr;
  }
  if (!options.mtp_model_path.empty()) {
    m->mtp_reader_ =
        core::GgufReader::OpenFile(options.mtp_model_path, error_msg);
    if (!m->mtp_reader_) {
      return nullptr;
    }
    auto mtp = MtpWeights::Bind(*m->mtp_reader_, c, error_msg);
    if (!mtp) {
      return nullptr;
    }
    m->mtp_weights_ = std::make_unique<MtpWeights>(std::move(*mtp));
  }
  // Dense-projection requantization (tools/mimo2_requant_dense.py): the sidecar
  // carries the attention projections (and layer 0's dense FFN) at a 4-bit
  // type, roughly halving the per-token dense bytes.
  if (const char* dense_sc = std::getenv("GUFO_MIMO_DENSE_SC");
      dense_sc != nullptr && *dense_sc != '\0') {
    m->dense_sc_reader_ = core::GgufReader::OpenFile(dense_sc, error_msg);
    if (!m->dense_sc_reader_) {
      return nullptr;
    }
  }
  // Expert requantization (ops/mimo2_requant_experts.cpp): a sidecar carrying
  // the routed expert tensors at Q3_K. The experts are 96% of the artifact's
  // bytes, so this is the only tensor class whose bytes move the decode cycle.
  if (const char* expert_sc = std::getenv("GUFO_EXPERT_SC");
      expert_sc != nullptr && *expert_sc != '\0') {
    m->expert_sc_reader_ = core::GgufReader::OpenFile(expert_sc, error_msg);
    if (!m->expert_sc_reader_) {
      return nullptr;
    }
  }
  m->device_ = rocm::DeviceModel::Upload(
      *m->weights_, *m->reader_, m->mtp_weights_.get(), m->mtp_reader_.get(),
      error_msg, partition_ptr, m->dense_sc_reader_.get(),
      m->expert_sc_reader_.get());
  if (!m->device_) {
    return nullptr;
  }
  rocm::Executor::Options exec;
  exec.device_index = options.hip_device;
  // The verbs transport's receive window (32 MiB minus a 64-byte header; see
  // kWindowBytes/kDataOffset in kernels/rocm/verbs.cpp) caps any single
  // all-reduce or split exchange. The widest is a prefill lane carrying
  // max_batch rows of hidden floats (a PrefillChunk splits across two lanes),
  // and an over-wide exchange poisons the communicator, so the batch width
  // stays inside the window (2047 rows at hidden 4096 instead of 2048).
  constexpr std::uint64_t kAllReduceWindowBytes = (32U << 20) - 64;
  const std::uint64_t window_rows =
      kAllReduceWindowBytes /
      (std::uint64_t{c.hidden_size} * sizeof(float));
  exec.max_batch = static_cast<std::uint32_t>(std::min(
      {std::uint64_t{kPrefillChunkTokens},
       std::uint64_t{options.max_context},
       std::max<std::uint64_t>(window_rows, 1)}));
  const std::uint64_t draft_rows =
      m->mtp_weights_ ? options.max_draft_tokens
                      : (options.dflash_model_path.empty()
                             ? 0
                             : options.dflash_max_draft_tokens);
  exec.max_logit_rows =
      draft_rows == 0
          ? 1
          : static_cast<std::uint32_t>(
                std::min<std::uint64_t>(exec.max_batch, draft_rows + 1));
  exec.max_speculative = exec.max_logit_rows;
  if (options.communicator) {
    exec.communicator = options.communicator;
    exec.all_reduce =
        rocm::Executor::TwoRankAllReduce(options.communicator, c.hidden_size);
    exec.reduce_status =
        rocm::Executor::TwoRankReduceStatus(options.communicator);
    // Split-reduce lanes stay on: measured neck-and-neck with single-lane
    // prefill (the lanes' concurrent expert-tile kernels share L2, so the
    // second lane's weight traffic is mostly cached) while hiding the
    // boundary exchange.
    exec.split_reduce =
        rocm::Executor::TwoRankSplitReduce(options.communicator, c.hidden_size);
  }
  exec.moe_observer = options.moe_observer;
  m->executor_ = rocm::Executor::Create(*m->device_, exec, error_msg);
  if (!m->executor_) {
    return nullptr;
  }
  return m;
}

std::unique_ptr<Session> Model::CreateSession(core::SessionMode mode,
                                              std::uint32_t max_context,
                                              std::string* error_msg) {
  if (max_context == 0 || max_context > options_.max_context) {
    AssignError(error_msg, "session context is outside the model limits");
    return nullptr;
  }
  auto native = executor_->CreateSession(mode, max_context, error_msg);
  if (!native) {
    return nullptr;
  }
  return std::unique_ptr<Session>(
      new Session(shared_from_this(), std::move(native)));
}

std::vector<std::int32_t> Model::Tokenize(std::string_view text) const {
  std::vector<std::int32_t> out;
  for (auto id : tokenizer_->Encode(text)) {
    out.push_back(static_cast<std::int32_t>(id));
  }
  return out;
}

std::string Model::Decode(std::span<const std::int32_t> tokens) const {
  std::vector<tokenization::TokenId> ids(tokens.begin(), tokens.end());
  return tokenizer_->Decode(ids);
}

std::string Model::TokenText(std::int32_t token) const {
  return tokenizer_->DecodeTokenCopy(static_cast<tokenization::TokenId>(token));
}

std::int32_t Model::EosToken() const noexcept {
  return static_cast<std::int32_t>(tokenizer_->GetEosTokenId());
}

bool Model::IsStopToken(std::int32_t token) const noexcept {
  return tokenizer_->IsStopToken(static_cast<tokenization::TokenId>(token));
}

std::uint32_t Model::VocabSize() const noexcept {
  return weights_->config.vocab_size;
}

std::uint32_t Model::PrefillCapacity() const noexcept {
  return std::min(executor_->PrefillChunk(), options_.max_context);
}

std::uint32_t Model::PrefillPairLead(std::uint32_t tokens) const noexcept {
  return executor_->PairLead(tokens);
}

bool Model::HasMtp() const noexcept {
  return device_->has_mtp();
}

Model::HeadTensorPointers Model::HeadTensors() const {
  const auto& embedding = device_->token_embd();
  const auto& head = device_->output();
  return {.token_embedding = embedding.data,
          .token_embedding_type = embedding.type,
          .output = head.data,
          .output_type = head.type};
}

std::string Model::ModelName() const {
  return std::string(reader_->GetMetadataString("general.name")
                         .value_or("MiMo-V2.6-Flash"));
}

const Config& Model::config() const noexcept {
  return weights_->config;
}

std::size_t Model::ResidentBytes() const noexcept {
  return device_->resident_bytes();
}

std::size_t Model::SessionBytes(core::SessionMode mode,
                                std::uint32_t context) const noexcept {
  return executor_->SessionBytes(mode, context,
                                 mode == core::SessionMode::kSpeculative
                                     ? executor_->max_speculative() - 1
                                     : 0);
}

std::size_t Model::DeferredScratchBytes() const {
  return executor_->DeferredScratchBytes();
}

std::size_t Session::AllocatedBytes() const noexcept {
  return session_->AllocatedBytes();
}

bool Session::MtpEnabled() const noexcept {
  return session_->mtp_enabled();
}

/// Context from which a sustained "no drafts" verdict latches the request to
/// plain decoding. The verdict already means no width repays its cycle; the
/// latch then also drops the predictor catch-up those cycles pay, which the
/// policy alone cannot (it must stay in sync in case drafting resumes).
/// GUFO_MTP_LATCH_CTX overrides the default. There is no off switch: a small
/// value (0) makes the latch eligible from the first token, and disabling it
/// means a value no request can reach (2026-10-14 window: 0 measured as plain
/// decoding, not as drafting everywhere).
static std::uint32_t MtpLatchContext() {
  static const std::uint32_t context = [] {
    const char* env = getenv("GUFO_MTP_LATCH_CTX");
    const long v = env != nullptr ? std::strtol(env, nullptr, 10) : -1;
    return static_cast<std::uint32_t>(v >= 0 ? v : 32768);
  }();
  return context;
}

Session::Session(std::shared_ptr<Model> model,
                 std::unique_ptr<rocm::Session> session)
    : model_(std::move(model)),
      session_(std::move(session)),
      // This controller decides only for the lone-request path (the batch
      // path measures its own per-width costs), so it must not inherit the
      // serve session count: a six-session server still decodes width one
      // against the single-cohort cost curve.
      draft_length_(model_->options_.max_draft_tokens, 1) {
  logits_.resize(model_->VocabSize());
}

Session::~Session() = default;

std::uint32_t Session::Position() const noexcept {
  return session_->position();
}
std::uint32_t Session::ContextSize() const noexcept {
  return session_->max_context();
}

void Session::Reset() {
  valid_ = false;
  session_->Reset();
  tokens_.clear();
  hidden_base_ = 0;
  draft_length_.Reset();
  mtp_depth_latched_ = false;
  model_->executor_->MtpRewind(*session_, 0);
  valid_ = true;
}

void Session::SetCancellationCheck(std::function<bool()> check) {
  session_->SetCancellationCheck(std::move(check));
}

std::uint32_t Session::KeptHiddenRows() const noexcept {
  return MtpEnabled()
             ? static_cast<std::uint32_t>(tokens_.size() - hidden_base_)
             : 0;
}

std::uint64_t Session::SnapshotBytes() const {
  if (!valid_)
    return 0;
  return SessionSnapshotHostBytes(static_cast<std::uint32_t>(tokens_.size()),
                                  model_->VocabSize()) +
         model_->executor_->SnapshotBytes(*session_, KeptHiddenRows());
}

std::size_t Session::SnapshotHeaderBytes() noexcept {
  return sizeof(SessionSnapshotHeader);
}

std::uint32_t Session::MtpPosition() const noexcept {
  return MtpEnabled() ? session_->mtp_position() : 0;
}

std::unique_ptr<SessionSnapshot> Session::SaveSnapshot(
    std::string* error_msg) const {
  if (!valid_ || tokens_.empty() || tokens_.size() != session_->position() ||
      tokens_.size() > std::numeric_limits<std::uint32_t>::max()) {
    AssignError(error_msg, "snapshot needs a synced, non-empty context");
    return nullptr;
  }
  const auto token_count = static_cast<std::uint32_t>(tokens_.size());
  const std::uint32_t hidden_rows = KeptHiddenRows();
  const std::uint64_t executor_bytes =
      model_->executor_->SnapshotBytes(*session_, hidden_rows);
  const std::uint64_t host_bytes =
      SessionSnapshotHostBytes(token_count, model_->VocabSize());
  std::unique_ptr<SessionSnapshot> snapshot(
      new SessionSnapshot(host_bytes + executor_bytes));
  std::uint8_t* out = snapshot->data_.get();
  const SessionSnapshotHeader header{
      .magic = kSnapshotMagic,
      .version = kSnapshotPayloadVersion,
      .vocab_size = model_->VocabSize(),
      .token_count = token_count,
      .hidden_rows = hidden_rows,
      .executor_bytes = executor_bytes,
      .draft_policy = draft_length_.State(),
      .policy_concurrency = model_->DecodeConcurrency(),
      .mtp_depth_latched = mtp_depth_latched_ ? 1U : 0U,
  };
  std::memcpy(out, &header, sizeof(header));
  out += sizeof(header);
  std::memcpy(out, tokens_.data(), tokens_.size() * sizeof(std::int32_t));
  out += tokens_.size() * sizeof(std::int32_t);
  std::memcpy(out, logits_.data(), logits_.size() * sizeof(float));
  out += logits_.size() * sizeof(float);
  if (!model_->executor_->SaveSnapshot(
          *session_, hidden_rows,
          std::span<std::uint8_t>(out,
                                  static_cast<std::size_t>(executor_bytes)),
          error_msg)) {
    return nullptr;
  }
  return snapshot;
}

bool Session::RestoreSnapshot(const SessionSnapshot& snapshot,
                              std::string* error_msg) {
  return RestoreSnapshotPayload(snapshot.bytes(), error_msg);
}

bool Session::RestoreSnapshot(std::span<const std::uint8_t> payload,
                              std::string* error_msg) {
  return RestoreSnapshotPayload(payload, error_msg);
}

bool Session::RestoreSnapshotPayload(std::span<const std::uint8_t> payload,
                                     std::string* error_msg) {
  SessionSnapshotHeader header{};
  if (payload.size() < sizeof(header)) {
    AssignError(error_msg, "session snapshot is truncated");
    return false;
  }
  std::memcpy(&header, payload.data(), sizeof(header));
  if (header.magic != kSnapshotMagic ||
      header.version != kSnapshotPayloadVersion) {
    AssignError(error_msg, "session snapshot format is not supported");
    return false;
  }
  if (header.vocab_size != model_->VocabSize() || header.token_count == 0 ||
      header.policy_concurrency != model_->DecodeConcurrency() ||
      header.token_count > ContextSize() ||
      payload.size() != SessionSnapshotHostBytes(header.token_count,
                                                 header.vocab_size) +
                            header.executor_bytes) {
    AssignError(error_msg, "session snapshot does not fit this session");
    return false;
  }
  auto restored_policy = draft_length_;
  if (!restored_policy.Restore(header.draft_policy)) {
    AssignError(error_msg, "session snapshot draft policy is invalid");
    return false;
  }
  const std::uint8_t* in = payload.data() + sizeof(header);
  std::vector<std::int32_t> tokens(header.token_count);
  std::memcpy(tokens.data(), in, tokens.size() * sizeof(std::int32_t));
  in += tokens.size() * sizeof(std::int32_t);
  std::vector<float> logits(header.vocab_size);
  std::memcpy(logits.data(), in, logits.size() * sizeof(float));
  in += logits.size() * sizeof(float);

  valid_ = false;
  mtp_depth_latched_ = header.mtp_depth_latched != 0;
  rocm::Executor::SnapshotInfo info;
  const auto remaining = ContextSize() - header.token_count;
  const auto next_drafts =
      MtpActive() ? restored_policy.Choose(remaining ? remaining - 1 : 0,
                                           header.token_count)
                  : 0;
  if (!model_->executor_->RestoreSnapshot(
          *session_,
          std::span<const std::uint8_t>(
              in, static_cast<std::size_t>(header.executor_bytes)),
          &info, error_msg, next_drafts)) {
    Reset();
    return false;
  }
  if (info.position != header.token_count ||
      info.hidden_rows != header.hidden_rows) {
    Reset();
    AssignError(error_msg, "session snapshot positions are inconsistent");
    return false;
  }
  tokens_ = std::move(tokens);
  logits_ = std::move(logits);
  hidden_base_ = info.position - info.hidden_rows;
  draft_token_ = 0;
  draft_length_ = restored_policy;
  stats_ = {};
  valid_ = true;
  return true;
}

SessionSnapshot::SessionSnapshot(std::uint64_t size)
    : data_(new std::uint8_t[size]), size_(size) {
  // Snapshot copies first-touch hundreds of MiB. Let Linux back the interior
  // with transparent huge pages instead of faulting one 4 KiB page at a time.
  // Advise only complete pages belonging to this allocation; this is optional
  // and does not pin memory or change the serialized payload.
  const long page = sysconf(_SC_PAGESIZE);
  if (page > 0) {
    const auto address = reinterpret_cast<std::uintptr_t>(data_.get());
    const auto skip = (page - address % page) % page;
    if (size > skip) {
      const auto length = (size - skip) / page * page;
      if (length != 0)
        (void)madvise(data_.get() + skip, length, MADV_HUGEPAGE);
    }
  }
}

bool SessionSnapshot::CopyTo(
    std::span<std::uint8_t> destination) const noexcept {
  if (destination.size() != size_) {
    return false;
  }
  std::memcpy(destination.data(), data_.get(), size_);
  return true;
}

bool Session::DraftReplay(std::int32_t next_token,
                          std::vector<std::int32_t>* replay,
                          std::int32_t* hidden_row,
                          std::string* error_msg) const {
  // The draft block trails the trunk: MTP position i consumes token i+1 and
  // the trunk's hidden of position i, so positions up to the current one are
  // replayed once their successor token is known. The session keeps hidden
  // rows of positions [hidden_base_, tokens_.size()).
  rocm::Executor& exec = *model_->executor_;
  const auto size = static_cast<std::uint32_t>(tokens_.size());
  const std::uint32_t mp = exec.MtpPosition(*session_);
  if (mp >= size) {
    return true;
  }
  if (mp < hidden_base_) {
    AssignError(error_msg, "draft block fell behind the kept hidden rows");
    return false;
  }
  replay->assign(tokens_.begin() + mp + 1, tokens_.end());
  replay->push_back(next_token);
  *hidden_row = static_cast<std::int32_t>(mp - hidden_base_);
  return true;
}

bool Session::DraftCatchUp(std::int32_t next_token, bool propose,
                           std::string* error_msg,
                           MtpCandidateLogits* candidates) {
  std::vector<std::int32_t> replay;
  std::int32_t hidden_row = 0;
  if (!DraftReplay(next_token, &replay, &hidden_row, error_msg))
    return false;
  if (replay.empty())
    return true;
  auto& exec = *model_->executor_;
  if (!exec.MtpForward(
          *session_, replay, hidden_row,
          {.token = propose && candidates == nullptr ? &draft_token_ : nullptr,
           .candidates = candidates},
          error_msg)) {
    return false;
  }
  return true;
}

bool Session::DraftCatchUpBatch(std::span<const AdvanceRequest> requests,
                                std::string* error_msg) {
  if (requests.empty())
    return true;
  auto& exec = *requests.front().session->model_->executor_;
  std::vector<std::vector<std::int32_t>> replays(requests.size());
  std::vector<rocm::Executor::MtpBatchItem> items;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    auto& session = *requests[i].session;
    // A latched session left its draft block behind on purpose; replaying it
    // now would read hidden rows it no longer keeps.
    if (!session.MtpActive())
      continue;
    std::int32_t hidden_row = 0;
    if (!session.DraftReplay(requests[i].token, &replays[i], &hidden_row,
                             error_msg))
      return false;
    if (replays[i].empty())
      continue;
    items.push_back({session.session_.get(), replays[i], hidden_row});
  }
  return items.empty() || exec.MtpForwardBatch(items, error_msg);
}

bool Session::Feed(std::span<const std::int32_t> tokens, std::string* error_msg,
                   bool prefill) {
  rocm::Executor& exec = *model_->executor_;
  const std::size_t step =
      prefill ? exec.PrefillChunk(*session_) : exec.max_batch();
  for (std::size_t off = 0; off < tokens.size(); off += step) {
    const std::size_t n = std::min<std::size_t>(step, tokens.size() - off);
    const auto chunk = tokens.subspan(off, n);
    if (MtpActive() && !tokens_.empty() &&
        !DraftCatchUp(chunk[0], false, error_msg)) {
      return false;
    }
    const auto mode = prefill ? rocm::Executor::ForwardMode::kPrefill
                              : rocm::Executor::ForwardMode::kDecode;
    if (!exec.Forward(*session_, chunk, 1, logits_.data(), mode, error_msg)) {
      return false;
    }
    // A chunk of two trunk batches keeps the second one's rows; armed tap
    // capture forces the single-batch prefill path (see PairLead).
    const std::uint32_t lead =
        prefill && !exec.TargetCaptureArmed(*session_)
            ? exec.PairLead(static_cast<std::uint32_t>(n))
            : 0;
    const std::size_t last = n - lead;
    hidden_base_ = static_cast<std::uint32_t>(
        tokens_.size() + n -
        std::min<std::size_t>(last, exec.max_speculative()));
    tokens_.insert(tokens_.end(), chunk.begin(), chunk.end());
  }
  return true;
}

bool Session::Sync(std::span<const std::int32_t> prompt,
                   std::string* error_msg) {
  if (prompt.empty()) {
    AssignError(error_msg, "prompt is empty");
    return false;
  }
  if (prompt.size() > ContextSize()) {
    AssignError(error_msg, "prompt exceeds the session context");
    return false;
  }
  if (!valid_)
    Reset();
  // KV rows cannot be rewound to a divergent prefix, so any divergence
  // restarts the session; an extension only feeds the new tail.
  std::size_t common = 0;
  while (common < tokens_.size() && common < prompt.size() &&
         tokens_[common] == prompt[common]) {
    ++common;
  }
  if (common == prompt.size() && common == tokens_.size()) {
    return true;
  }
  if (common != tokens_.size()) {
    Reset();
    common = 0;
  }
  if (kTapAudit != nullptr && common == 0) {
    std::string tap_error;
    if (!model_->executor_->SetTargetLayerCapture(*session_, kTapAuditLayers,
                                                  &tap_error)) {
      AssignError(error_msg, "tap audit arming failed: " + tap_error);
      return false;
    }
  }
  valid_ = false;
  const bool ok = Feed(prompt.subspan(common), error_msg, true);
  valid_ = ok;
  if (ok && kTapAudit != nullptr) {
    TapAuditReport(*model_->executor_, *session_, prompt.size());
  }
  return ok;
}

bool Session::Evaluate(std::int32_t token, std::string* error_msg) {
  if (!valid_) {
    AssignError(error_msg,
                "session needs a successful Sync after a failed operation");
    return false;
  }
  if (tokens_.size() >= ContextSize()) {
    AssignError(error_msg, "session context is full");
    return false;
  }
  valid_ = false;
  const bool ok = Feed(std::span<const std::int32_t>(&token, 1), error_msg);
  valid_ = ok;
  return ok;
}

struct Session::PendingDecode {
  std::vector<std::int32_t> chain;
  std::vector<MtpProposal> proposals;
  std::uint32_t base{0};
  bool speculative{false};
  bool sampled{false};
  bool gpu_greedy{false};
  bool gpu_verification{false};
  std::size_t width{0};
  std::optional<sampling::SamplerState> draft_sampler;
  std::uint64_t draft_rng{0};
  MtpCandidateLogits candidates;
  std::int32_t draft{0};
};

void Session::AppendDraft(PendingDecode& pending) {
  if (pending.sampled) {
    pending.proposals.push_back(SampleMtpProposal(
        pending.candidates, *pending.draft_sampler, &pending.draft_rng));
    pending.draft = static_cast<std::int32_t>(pending.proposals.back().token);
    pending.draft_sampler->Accept(pending.proposals.back().token);
  }
  pending.chain.push_back(pending.draft);
}

bool Session::PrepareDecode(const DecodeRequest& request,
                            PendingDecode* pending, std::string* error_msg,
                            bool defer_head,
                            std::optional<std::uint32_t> batch_drafts) {
  const auto max_tokens = request.max_tokens;
  auto& sampler = *request.sampler;
  auto* result = request.result;
  const bool stop_at_eos = request.stop_at_eos;
  if (result == nullptr || max_tokens == 0 || tokens_.empty()) {
    AssignError(
        error_msg,
        "decode needs an output, a positive budget and a synced prompt");
    return false;
  }
  *result = {};
  const auto is_stop = [&](std::int32_t token) {
    return stop_at_eos && model_->IsStopToken(token);
  };
  rocm::Executor& exec = *model_->executor_;
  const std::size_t room = ContextSize() - tokens_.size();
  const std::size_t cap =
      std::min<std::size_t>({max_tokens, room, exec.max_speculative()});
  std::size_t width = 0;
  if (MtpActive() && cap > 1) {
    const auto drafts =
        batch_drafts
            ? std::min<std::uint32_t>(*batch_drafts, cap - 1)
            : draft_length_.Choose(static_cast<std::uint32_t>(cap - 1),
                                   static_cast<std::uint32_t>(tokens_.size()));
    // Past the crossover no width repays the predictor catch-up, and the
    // context only grows, so one sustained verdict ends speculation for this
    // request. Latching is what lets the AR cycles below stop paying it.
    if (!batch_drafts && tokens_.size() >= MtpLatchContext() &&
        draft_length_.LatchAr(static_cast<std::uint32_t>(cap - 1),
                              static_cast<std::uint32_t>(tokens_.size()))) {
      mtp_depth_latched_ = true;
    }
    width = MtpActive() ? 1 + drafts : cap;
  } else {
    width = cap;
  }
  // GUFO_MTP_FORCE_WIDTH pins the cycle width for the lone-request path so
  // per-width cycle costs and acceptance can be measured without the policy
  // in the way. A comma-separated list rotates one width per cycle, which
  // samples several widths at matched context in a single run.
  if (!defer_head && MtpActive() && cap > 1) {
    static const std::vector<std::uint32_t> forced = [] {
      std::vector<std::uint32_t> list;
      const char* env = getenv("GUFO_MTP_FORCE_WIDTH");
      for (const char* p = env; p != nullptr && *p != '\0';) {
        char* end = nullptr;
        const long value = std::strtol(p, &end, 10);
        if (end == p) {
          ++p;
          continue;
        }
        if (value > 0) {
          list.push_back(static_cast<std::uint32_t>(value));
        }
        p = end;
      }
      return list;
    }();
    if (!forced.empty()) {
      static std::size_t forced_cycle = 0;
      width =
          std::min<std::size_t>(cap, forced[forced_cycle++ % forced.size()]);
    }
  }
  if (width == 0) {
    result->stop = true;
    return true;
  }
  const auto anchor = static_cast<std::int32_t>(sampler.Sample(logits_));
  if (is_stop(anchor)) {
    result->stop = true;
    return true;
  }
  if (!MtpActive() || width < 2) {
    if (MtpActive() && !defer_head &&
        !DraftCatchUp(anchor, false, error_msg)) {
      return false;
    }
    pending->chain = {anchor};
    pending->base = static_cast<std::uint32_t>(tokens_.size());
    return true;
  }

  const std::uint32_t base = static_cast<std::uint32_t>(tokens_.size());
  const bool sampled = sampler.config().uses_random_sampling();
  const bool gpu_greedy = sampler.config().can_use_unmodified_argmax();
  const bool gpu_verification = gpu_greedy;
  if (!defer_head && !DraftCatchUp(anchor, true, error_msg,
                                   sampled ? &pending->candidates : nullptr)) {
    return false;
  }
  // A cycle-local proposal stream needs no pending RNG state in snapshots.
  // Target verification keeps its own draws after this independent seed.
  pending->draft_rng =
      sampled ? sampling::NextRandom(sampler.mutable_rng_state()) : 0;
  pending->draft_sampler = sampler.WithoutConstraint();
  pending->draft_sampler->Accept(static_cast<sampling::TokenId>(anchor));
  pending->chain = {anchor};
  pending->draft = draft_token_;
  pending->width = width;
  pending->base = base;
  pending->speculative = true;
  pending->sampled = sampled;
  pending->gpu_greedy = gpu_greedy;
  pending->gpu_verification = gpu_verification;
  if (!gpu_verification && verify_logits_.empty()) {
    verify_logits_.resize(exec.max_speculative() * model_->VocabSize());
  }
  if (!defer_head) {
    while (pending->chain.size() < width) {
      AppendDraft(*pending);
      if (pending->chain.size() < width &&
          !exec.MtpForward(
              *session_, std::span<const std::int32_t>(&pending->draft, 1), -1,
              {.token = sampled ? nullptr : &pending->draft,
               .candidates = sampled ? &pending->candidates : nullptr},
              error_msg)) {
        return false;
      }
    }
  }
  return true;
}

bool Session::FinishDecode(const DecodeRequest& request,
                           const PendingDecode& pending,
                           std::string* error_msg) {
  auto& sampler = *request.sampler;
  auto* result = request.result;
  auto& exec = *model_->executor_;
  const auto& chain = pending.chain;
  const auto& proposals = pending.proposals;
  const auto base = pending.base;
  const bool sampled = pending.sampled;
  const bool gpu_greedy = pending.gpu_greedy;
  const bool gpu_verification = pending.gpu_verification;
  const auto anchor = chain.front();
  const auto k = static_cast<std::uint32_t>(chain.size());
  const auto vocab = model_->VocabSize();
  const auto is_stop = [&](std::int32_t token) {
    return request.stop_at_eos && model_->IsStopToken(token);
  };
  if (!pending.speculative) {
    draft_length_.ObserveArToken();
    hidden_base_ = base;
    tokens_.push_back(anchor);
    sampler.Accept(static_cast<sampling::TokenId>(anchor));
    result->tokens.push_back(anchor);
    return true;
  }
  sampler.Accept(static_cast<sampling::TokenId>(anchor));
  std::array<rocm::ArgmaxCandidate, kMaxMtpDraftTokens> greedy{};
  if (gpu_greedy &&
      !exec.GreedyMtpPredictions(std::span(greedy).first(k - 1), error_msg)) {
    return false;
  }
  // GUFO_MTP_DEBUG: side-by-side first draft cycles — the draft chain vs
  // the verify argmax, to tell a broken drafter from a mis-aligned
  // comparison at a glance.
  static const bool mtp_debug = getenv("GUFO_MTP_DEBUG") != nullptr;
  static std::uint32_t mtp_debug_n = 0;
  std::uint32_t keep = 1;
  std::optional<std::int32_t> correction;
  while (keep < k) {
    if (gpu_greedy) {
      const auto& prediction = greedy[keep - 1];
      if (!std::isfinite(prediction.value)) {
        AssignError(error_msg, "logit distribution contains no finite values");
        return false;
      }
      if (is_stop(prediction.index)) {
        result->stop = true;
        break;
      }
      if (prediction.index != chain[keep]) {
        break;
      }
      sampler.Accept(static_cast<sampling::TokenId>(prediction.index));
      ++keep;
      continue;
    }
    if (sampled) {
      const auto verified =
          VerifyMtpProposal(std::span<const float>(verify_logits_)
                                .subspan((keep - 1) * vocab, vocab),
                            proposals[keep - 1], sampler);
      const auto token = static_cast<std::int32_t>(verified.token);
      const bool accepted = verified.accepted;
      if (is_stop(token)) {
        result->stop = true;
        break;
      }
      if (!accepted) {
        correction = token;
        break;
      }
      sampler.Accept(static_cast<sampling::TokenId>(token));
      ++keep;
      continue;
    }
    const auto decision = VerifyDraft(std::span<const float>(verify_logits_)
                                          .subspan((keep - 1) * vocab, vocab),
                                      chain[keep], sampler, is_stop);
    if (decision != DraftDecision::kAccept) {
      result->stop = decision == DraftDecision::kStop;
      break;
    }
    ++keep;
  }
  if (!exec.Rollback(*session_, keep, error_msg,
                     gpu_verification ? logits_.data() : nullptr)) {
    return false;
  }
  if (!gpu_verification) {
    std::copy_n(verify_logits_.data() + (keep - 1) * vocab, vocab,
                logits_.begin());
  }
  if (mtp_debug && gpu_greedy && ++mtp_debug_n <= 64) {
    fprintf(stderr, "MTPDBG k=%u keep=%u base=%u chain=[", k, keep, base);
    for (const auto t : chain) fprintf(stderr, " %d", t);
    fprintf(stderr, " ] greedy=[");
    for (std::uint32_t i = 0; i + 1 < k; ++i)
      fprintf(stderr, " %d", greedy[i].index);
    fprintf(stderr, " ] hist=[");
    const std::size_t hn = tokens_.size();
    for (std::size_t i = hn > 10 ? hn - 10 : 0; i < hn; ++i)
      fprintf(stderr, " %d", tokens_[i]);
    fprintf(stderr, " ] mp=%u hb=%u n=%zu%s\n",
            model_->executor_->MtpPosition(*session_) & 0x7fffffff,
            hidden_base_ & 0x7fffffff, hn,
            correction ? " (correction)" : "");
    fflush(stderr);
  }
  hidden_base_ = base;
  tokens_.insert(tokens_.end(), chain.begin(), chain.begin() + keep);
  result->tokens.assign(chain.begin(), chain.begin() + keep);
  stats_.cycles += 1;
  stats_.drafted += k - 1;
  stats_.accepted += keep - 1;
  // A target stop ends the request; it does not classify the remaining
  // proposals as failed predictions.
  draft_length_.Observe(keep - 1, result->stop ? keep - 1 : k - 1, base);

  // The next call knows the next sampled anchor. Defer draft catch-up until
  // then, retaining this session's target hidden rows across interleaving.
  exec.MtpRewind(*session_, base);
  if (correction) {
    // Evaluate the residual as the next cycle's anchor, avoiding a separate
    // target pass. Preserve the actual draw: resampling p would be biased.
    sampler.DeferSample(static_cast<sampling::TokenId>(*correction));
  }
  return true;
}

bool Session::DecodeStep(std::size_t max_tokens,
                         sampling::SamplerState& sampler, DecodeResult* result,
                         std::string* error_msg, bool stop_at_eos) {
  if (!valid_) {
    AssignError(error_msg,
                "session needs a successful Sync after a failed operation");
    return false;
  }
  valid_ = false;
  const DecodeRequest request{this, max_tokens, &sampler, result, stop_at_eos};
  PendingDecode pending;
  // GUFO_MTP_COST: per-cycle wall time by width, for the cost table. The
  // cycle ends after FinishDecode's verification sync, so `tot` is a true
  // wall time even in the non-blocking eager configuration.
  static const bool mtp_cost = getenv("GUFO_MTP_COST") != nullptr;
  const auto t0 = std::chrono::steady_clock::now();
  if (!PrepareDecode(request, &pending, error_msg)) {
    return false;
  }
  const auto t1 = std::chrono::steady_clock::now();
  if (pending.chain.empty()) {
    valid_ = true;
    return true;
  }
  float* logits = !pending.speculative       ? logits_.data()
                  : pending.gpu_verification ? nullptr
                                             : verify_logits_.data();
  if (!model_->executor_->Forward(
          *session_, pending.chain, pending.chain.size(), logits,
          pending.speculative ? rocm::Executor::ForwardMode::kVerify
                              : rocm::Executor::ForwardMode::kDecode,
          error_msg)) {
    return false;
  }
  const auto t2 = std::chrono::steady_clock::now();
  const bool ok = FinishDecode(request, pending, error_msg);
  if (mtp_cost) {
    const auto t3 = std::chrono::steady_clock::now();
    const auto ms = [](auto a, auto b) {
      return std::chrono::duration<float, std::milli>(b - a).count();
    };
    const auto state = draft_length_.State();
    const float p1 =
        state.successes[0] / (state.successes[0] + state.failures[0]);
    fprintf(stderr,
            "MTPCOST w=%zu keep=%zu base=%u p1=%.3f retry=%u prep=%.2f "
            "fwd=%.2f fin=%.2f tot=%.2f\n",
            pending.chain.size(), result->tokens.size(), pending.base, p1,
            state.retry_tokens, ms(t0, t1), ms(t1, t2), ms(t2, t3),
            ms(t0, t3));
    fflush(stderr);
  }
  valid_ = ok;
  return ok;
}

template<class Request>
bool Session::RunIsolatedBatch(std::span<const Request> requests,
                               std::string* error_msg,
                               const std::optional<std::uint32_t>* plan) {
  if (requests.empty() || requests.size() > 8) {
    AssignError(error_msg, "batch must contain 1..8 sessions");
    return false;
  }
  std::array<BatchOutcome, 8> outcomes{};
  std::array<std::uint64_t, 8> epochs{};
  std::array<sampling::SamplerState::DrawState, 8> draws{};
  std::vector<Request> active;
  active.reserve(requests.size());
  Model* model = nullptr;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    auto& outcome = outcomes[i];
    if (r.session)
      epochs[i] = r.session->session_->MutationEpoch();
    bool valid = r.session && r.session->valid_;
    if constexpr (std::is_same_v<Request, DecodeRequest>) {
      valid = valid && r.sampler && r.result && r.max_tokens > 0 &&
              !r.session->tokens_.empty();
    } else {
      valid = valid && r.token >= 0 &&
              static_cast<std::uint32_t>(r.token) <
                  r.session->model_->VocabSize() &&
              r.session->Position() < r.session->ContextSize();
    }
    for (std::size_t j = 0; j < requests.size(); ++j) {
      if (i == j)
        continue;
      valid = valid && r.session != requests[j].session;
      if constexpr (std::is_same_v<Request, DecodeRequest>)
        valid = valid && r.sampler != requests[j].sampler &&
                r.result != requests[j].result;
    }
    if (valid && model && r.session->model_.get() != model)
      valid = false;
    if (!valid) {
      outcome.error = "invalid or non-independent batch request";
      continue;
    }
    if (!r.session->session_->CheckCancellation(&outcome.error))
      continue;
    model = r.session->model_.get();
    if constexpr (std::is_same_v<Request, DecodeRequest>) {
      draws[i] = r.sampler->SaveDrawState();
    }
    auto copy = r;
    copy.outcome = &outcome;
    active.push_back(copy);
  }
  std::string shared_error;
  try {
    if (!active.empty()) {
      if constexpr (std::is_same_v<Request, DecodeRequest>)
        (void)DecodeBatchImpl(active, plan, &shared_error);
      else
        (void)EvaluateBatchImpl(active, &shared_error);
    }
  } catch (const std::exception& exception) {
    shared_error = exception.what();
  }
  bool success = true;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    auto& outcome = outcomes[i];
    if (!outcome.completed && outcome.error.empty()) {
      auto& session = *r.session;
      if (session.session_->Cancelled()) {
        outcome.error = "generation cancelled";
      } else if (session.session_->MutationEpoch() == epochs[i]) {
        // A preparation/batch-allocation failure did not touch this peer.
        // Retry it independently, without changing its cached frontier.
        session.valid_ = true;
        if constexpr (std::is_same_v<Request, DecodeRequest>) {
          r.sampler->RestoreDrawState(draws[i]);
        }
        try {
          if constexpr (std::is_same_v<Request, DecodeRequest>)
            outcome.completed =
                session.DecodeStep(r.max_tokens, *r.sampler, r.result,
                                   &outcome.error, r.stop_at_eos);
          else
            outcome.completed = session.Evaluate(r.token, &outcome.error);
        } catch (const std::exception& exception) {
          outcome.error = exception.what();
        }
      } else {
        outcome.error =
            shared_error.empty() ? "batch execution failed" : shared_error;
      }
    }
    if (!outcome.completed) {
      if (outcome.error.empty())
        outcome.error = "batch request failed";
      if (r.session && r.session->session_->MutationEpoch() != epochs[i])
        r.session->valid_ = false;
      if (success)
        AssignError(error_msg, outcome.error);
      success = false;
    } else {
      r.session->valid_ = true;
    }
    if (r.outcome)
      *r.outcome = std::move(outcome);
  }
  return success;
}

bool Session::DecodeBatch(std::span<const DecodeRequest> requests,
                          std::string* error_msg,
                          const std::optional<std::uint32_t>* plan) {
  return RunIsolatedBatch(requests, error_msg, plan);
}

std::optional<std::uint32_t> Session::PlanBatch(
    std::span<const DecodeRequest> requests) {
  if (requests.size() > 8)
    return std::nullopt;
  // Only the requests the batch would run: a TP2 peer plans before the batch
  // sets aside invalid ones.
  std::array<MtpBatchController::Row, 8> rows{};
  std::size_t count = 0;
  std::uint32_t batch_context = 0;
  const Session* first = nullptr;
  for (const auto& r : requests) {
    if (r.session == nullptr || r.sampler == nullptr || !r.session->valid_ ||
        r.session->tokens_.empty() || r.max_tokens == 0)
      continue;
    if (!r.session->MtpEnabled() || r.sampler->config().uses_random_sampling())
      return std::nullopt;
    const auto& exec = *r.session->model_->executor_;
    const auto cap = std::min<std::size_t>(
        r.max_tokens, r.session->ContextSize() - r.session->Position());
    // A latched session contributes no draft budget, so the cohort-wide
    // width falls to plain decoding: one row's depth verdict applies to the
    // shared step, and its draft block is not asked to replay.
    const std::uint32_t budget =
        !r.session->MtpActive()
            ? 0
            : static_cast<std::uint32_t>(
                  std::min<std::size_t>(cap, exec.max_speculative())) -
                  (cap != 0);
    rows[count++] = {&r.session->draft_length_, budget};
    batch_context = std::max(batch_context, r.session->Position());
    if (first == nullptr)
      first = r.session;
  }
  if (count < 2)
    return std::nullopt;
  auto& policy = first->model_->batch_policy_;
  return policy.Choose(std::span(rows).first(count), batch_context);
}

bool Session::EvaluateBatch(std::span<const AdvanceRequest> requests,
                            std::string* error_msg) {
  return RunIsolatedBatch(requests, error_msg);
}

bool Session::DecodeBatchImpl(std::span<const DecodeRequest> requests,
                              const std::optional<std::uint32_t>* plan,
                              std::string* error_msg) {
  if (requests.size() == 1) {
    const auto& r = requests.front();
    r.outcome->completed = r.session->DecodeStep(
        r.max_tokens, *r.sampler, r.result, &r.outcome->error, r.stop_at_eos);
    return r.outcome->completed;
  }
  auto& exec = *requests.front().session->model_->executor_;
  std::optional<std::uint32_t> batch_drafts;
  std::uint32_t batch_context = 0;
  auto& policy = requests.front().session->model_->batch_policy_;
  // A given plan (a TP2 peer's) replaces the timing-based choice.
  batch_drafts = plan != nullptr ? *plan : PlanBatch(requests);
  if (batch_drafts) {
    for (const auto& r : requests) {
      batch_context = std::max(batch_context, r.session->Position());
    }
  }
  const auto cycle_start = std::chrono::steady_clock::now();
  static const bool cycle_trace = getenv("GUFO_BATCH_CYCLE") != nullptr;
  std::array<std::size_t, 8> cycle_before{};
  if (cycle_trace) {
    for (std::size_t i = 0; i < requests.size() && i < cycle_before.size();
         ++i) {
      const auto* s = requests[i].session;
      cycle_before[i] = s != nullptr ? s->tokens_.size() : 0;
    }
  }
  std::vector<PendingDecode> pending(requests.size());
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    try {
      if (!r.session->PrepareDecode(r, &pending[i], &r.outcome->error, true,
                                    batch_drafts))
        pending[i] = {};
    } catch (const std::exception& exception) {
      r.outcome->error = exception.what();
      pending[i] = {};
    }
  }
  std::vector<AdvanceRequest> catchup;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    if (!pending[i].chain.empty())
      catchup.push_back({requests[i].session, pending[i].chain.front()});
  }
  if (!DraftCatchUpBatch(catchup, error_msg))
    return false;
  // Each round shares predictor projections across ready sessions.
  // Attention state, proposal distributions and RNG streams stay private.
  for (;;) {
    std::vector<rocm::Executor::MtpHeadItem> heads;
    std::vector<rocm::Executor::MtpBatchItem> bodies;
    for (std::size_t i = 0; i < requests.size(); ++i) {
      auto& p = pending[i];
      if (!requests[i].session->session_->Cancelled() && p.speculative &&
          p.chain.size() < p.width) {
        heads.push_back({requests[i].session->session_.get(),
                         {.token = p.sampled ? nullptr : &p.draft,
                          .candidates = p.sampled ? &p.candidates : nullptr}});
      }
    }
    if (heads.empty())
      break;
    if (!exec.MtpHeads(heads, error_msg))
      return false;
    for (std::size_t i = 0; i < requests.size(); ++i) {
      auto& p = pending[i];
      if (requests[i].session->session_->Cancelled() || !p.speculative ||
          p.chain.size() >= p.width)
        continue;
      AppendDraft(p);
      if (p.chain.size() < p.width)
        bodies.push_back({requests[i].session->session_.get(),
                          std::span<const std::int32_t>(&p.draft, 1), -1});
    }
    if (!bodies.empty() && !exec.MtpForwardBatch(bodies, error_msg))
      return false;
  }
  std::vector<rocm::Executor::BatchItem> items;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& r = requests[i];
    if (!r.session->session_->Cancelled() && !pending[i].chain.empty()) {
      items.push_back({r.session->session_.get(), pending[i].chain,
                       pending[i].speculative});
    }
  }
  if (!items.empty() && !exec.ForwardBatch(items, error_msg))
    return false;
  // One readback for the whole cohort: the per-session selects each paid a
  // blocking download plus a stream sync, which is what the producer profile
  // saw as six memcpy+sync pairs inside the inter-cycle hole. Rows outside
  // the staging (or a failed transfer) fall back to the per-session path.
  {
    std::uint32_t total_rows = 0;
    for (const auto& item : items)
      total_rows += static_cast<std::uint32_t>(item.tokens.size());
    if (total_rows > 0) {
      // Rows beyond the pinned staging (or a failed transfer) leave the
      // staging empty, and the per-session selects fall back to the device.
      std::string staging_error;
      (void)exec.StageBatchLogits(0, total_rows, &staging_error);
    }
  }
  std::uint32_t offset = 0;
  for (std::size_t i = 0; i < requests.size(); ++i) {
    const auto& p = pending[i];
    const auto& r = requests[i];
    auto& session = *r.session;
    const bool included =
        std::any_of(items.begin(), items.end(), [&](const auto& item) {
          return item.session == session.session_.get();
        });
    const auto row_offset = offset;
    if (included)
      offset += p.chain.size();
    if (!r.outcome->error.empty())
      continue;
    if (session.session_->Cancelled()) {
      r.outcome->error = "generation cancelled";
      continue;
    }
    if (p.chain.empty()) {
      r.outcome->completed = true;
      continue;
    }
    float* logits = !p.speculative       ? session.logits_.data()
                    : p.gpu_verification ? nullptr
                                         : session.verify_logits_.data();
    try {
      r.outcome->completed =
          exec.SelectBatchLogits(row_offset, p.chain.size(), logits,
                                 &r.outcome->error) &&
          session.FinishDecode(r, p, &r.outcome->error);
    } catch (const std::exception& exception) {
      r.outcome->error = exception.what();
    }
  }
  if (batch_drafts && items.size() == requests.size() &&
      std::ranges::all_of(requests,
                          [](const auto& r) { return r.outcome->completed; })) {
    const auto ms = std::chrono::duration<float, std::milli>(
                        std::chrono::steady_clock::now() - cycle_start)
                        .count();
    policy.Observe(requests.size(), batch_context, *batch_drafts, ms);
    if (cycle_trace) {
      std::size_t emitted = 0;
      for (std::size_t i = 0; i < requests.size() && i < cycle_before.size();
           ++i) {
        const auto* s = requests[i].session;
        if (s != nullptr && s->tokens_.size() >= cycle_before[i]) {
          emitted += s->tokens_.size() - cycle_before[i];
        }
      }
      fprintf(stderr, "BATCHCYCLE width=%u sessions=%zu ms=%.2f emitted=%zu\n",
              *batch_drafts, requests.size(), ms, emitted);
      fflush(stderr);
    }
  }
  return true;
}

bool Session::EvaluateBatchImpl(std::span<const AdvanceRequest> requests,
                                std::string* error_msg) {
  if (requests.size() == 1) {
    const auto& r = requests.front();
    r.outcome->completed = r.session->Evaluate(r.token, &r.outcome->error);
    return r.outcome->completed;
  }
  auto& exec = *requests.front().session->model_->executor_;
  if (!DraftCatchUpBatch(requests, error_msg))
    return false;
  std::vector<rocm::Executor::BatchItem> items;
  for (const auto& r : requests) {
    items.push_back({r.session->session_.get(), {&r.token, 1}, false});
  }
  if (!exec.ForwardBatch(items, error_msg)) {
    return false;
  }
  for (std::size_t i = 0; i < requests.size(); ++i) {
    auto& session = *requests[i].session;
    auto& outcome = *requests[i].outcome;
    if (session.session_->Cancelled()) {
      outcome.error = "generation cancelled";
      continue;
    }
    try {
      if (!exec.SelectBatchLogits(i, 1, session.logits_.data(), &outcome.error))
        continue;
      session.hidden_base_ = static_cast<std::uint32_t>(session.tokens_.size());
      session.tokens_.push_back(requests[i].token);
      outcome.completed = true;
    } catch (const std::exception& exception) {
      outcome.error = exception.what();
    }
  }
  return true;
}

}  // namespace gufo::models::mimo2
