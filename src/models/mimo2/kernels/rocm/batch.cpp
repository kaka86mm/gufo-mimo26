#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

#include "src/models/mimo2/kernels/rocm/executor.hpp"
#include "src/models/mimo2/kernels/rocm/kernels.hpp"
#include "src/models/mimo2/mtp_sampling.hpp"

namespace gufo::models::mimo2::rocm {
static const bool kEntryTrace = [] {
  return getenv("GUFO_SYNC_TRACE") != nullptr;
}();
namespace {

/// Per-step phase accounting for the fused multi-session batch. Host timers
/// are accurate when the caller synchronizes per kernel (HIP_LAUNCH_BLOCKING)
/// or per step; with plain eager launches they measure submission order. One
/// aggregate line per 25 steps, so a six-session step's budget is visible
/// without a profiler (rocprof v1 hangs on gfx1151).
struct BatchPhases {
  const bool on = getenv("GUFO_BATCH_TRACE") != nullptr;
  double proj = 0.0, attn = 0.0, oproj = 0.0, ar = 0.0, ffn = 0.0, head = 0.0;
  double total = 0.0;
  std::uint32_t steps = 0;
  std::uint32_t rows_max = 0;

  [[nodiscard]] static double Now() {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }
  void Tick() {
    if (!on) {
      return;
    }
    if (++steps < 25) {
      return;
    }
    const auto d = static_cast<double>(steps);
    fprintf(stderr,
            "BATCHTRACE steps=%u rows=%u proj=%.2f attn=%.2f oproj=%.2f "
            "ar=%.2f ffn=%.2f head=%.2f total=%.2f\n",
            steps, rows_max, proj / d, attn / d, oproj / d, ar / d, ffn / d,
            head / d, total / d);
    fflush(stderr);
    proj = attn = oproj = ar = ffn = head = total = 0.0;
    steps = 0;
    rows_max = 0;
  }
};

constexpr std::uint32_t kDecodeRows = 8;
constexpr std::uint32_t kBatchSessions = Executor::kBatchSlotCount;
static_assert(kBatchSessions >= kDecodeRows);

bool Fail(std::string* error, const std::string& message) {
  if (error != nullptr) {
    *error = message;
  }
  return false;
}

bool Check(hipError_t status, std::string* error) {
  return status == hipSuccess || Fail(error, std::string("batched forward: ") +
                                                 hipGetErrorString(status));
}

// A cancelled row stops touching its private state. Shared projections retain
// their row layout so cancellation cannot change a peer's arithmetic/order.
template<class Item>
bool AnyActive(std::span<const Item> items) {
  bool active = false;
  for (const auto& item : items)
    active |= item.session->CheckCancellation(nullptr);
  return active;
}

WeightType WeightTypeOf(core::GgmlType type) {
  switch (type) {
    case core::GgmlType::kBF16:
      return WeightType::kBF16;
    case core::GgmlType::kF16:
      return WeightType::kF16;
    case core::GgmlType::kQ8_0:
      return WeightType::kQ8_0;
    case core::GgmlType::kF32:
      return WeightType::kF32;
    default:
      throw std::logic_error("unsupported MiMo-V2 embedding format");
  }
}

}  // namespace

Executor::AttnDims Executor::DimsOf(const DeviceLayer& l) const noexcept {
  const Config& c = config();
  return {l.q_heads * c.head_dim_qk, l.kv_heads * c.head_dim_qk,
          l.kv_heads * c.head_dim_v, l.q_heads * c.head_dim_v};
}

Executor::Scratch Executor::RowScratch(const Scratch& b,
                                       std::uint32_t offset) const {
  const Config& c = config();
  const std::size_t r = offset;
  auto s = b;
  s.tokens += r;
  s.res += r * c.hidden_size;
  s.xn += r * c.hidden_size;
  s.block_out += r * c.hidden_size;
  s.router += r * c.num_experts;
  s.ids += r * c.num_experts_used;
  s.weights += r * c.num_experts_used;
  const std::uint32_t expert_ff =
      model_->layers().empty() ? c.expert_ff
                               : model_->layers().front().ffn_gate_exps.rows;
  s.gate_e += r * c.num_experts_used * expert_ff;
  s.up_e += r * c.num_experts_used * expert_ff;
  s.down_e += r * c.num_experts_used * c.hidden_size;
  std::uint32_t dense_ff = c.dense_ff;
  for (const auto& layer : model_->layers()) {
    dense_ff = std::max(dense_ff, layer.ffn_gate.rows);
  }
  for (const auto& block : model_->nextn_layers()) {
    dense_ff = std::max(dense_ff, block.ffn_gate.rows);
  }
  s.ff_act += r * dense_ff;
  s.ff_up += r * dense_ff;
  if (has_mtp()) {
    s.mtp_h += r * c.hidden_size;
    s.mtp_eh += r * 2 * c.hidden_size;
  }
  return s;
}

Executor::Scratch Executor::RowScratch(const Scratch& b, std::uint32_t offset,
                                       const AttnDims& dims) const {
  auto s = RowScratch(b, offset);
  const std::size_t r = offset;
  s.q += r * dims.q;
  s.k += r * dims.k;
  s.v += r * dims.v;
  s.ctx += r * dims.ctx;
  return s;
}

void Executor::UseScratch(const Scratch& scratch) const {
    if (kEntryTrace) { fprintf(stderr, "ENTRY batch:Executor::UseScratch\n"); fflush(stderr); }
  s_ = scratch;
  q8t_src_ = nullptr;
  half_src_ = nullptr;
  ff_half_ready_ = false;
}

bool Executor::AllocateBatch(std::string* error) const {
    if (kEntryTrace) { fprintf(stderr, "ENTRY batch:Executor::AllocateBatch\n"); fflush(stderr); }
  return batch_logits_ != nullptr ||
         Check(hipMalloc(&batch_logits_,
                         static_cast<std::size_t>(kBatchSessions) *
                             std::min(kDecodeRows, options_.max_logit_rows) *
                             config().vocab_size * sizeof(float)),
               error);
}

/// Rows per batched dense/FFN pass. The trunk scratch covers max_batch rows
/// and the fused top-8 MoE tile takes up to GUFO_MOE_FUSED_ROWS, so a cohort
/// step can stream each weight tensor once instead of once per eight-row
/// chunk: MoE expert reads are the union over the whole pass, which measured
/// ~20-30%% cheaper at a six-session step than three separate eight-row
/// passes. The knob stays for A/B and for artifacts with narrower scratch.
static std::uint32_t BatchPassRows() {
  static const std::uint32_t rows = [] {
    const char* env = getenv("GUFO_BATCH_PASS_ROWS");
    const long v = env != nullptr ? std::strtol(env, nullptr, 10) : -1;
    return static_cast<std::uint32_t>(v > 0 ? v : 24);
  }();
  return rows;
}

bool Executor::DenseBatch(const DeviceTensor& w, const float* x, float* out,
                          std::uint32_t rows, std::string* error) const {
    if (kEntryTrace) { fprintf(stderr, "ENTRY batch:Executor::DenseBatch\n"); fflush(stderr); }
  // Every request contains at most eight decode rows, so the shared
  // projection runs as small GEMMs over the concatenated batch in chunks
  // that keep the single-request arithmetic.
  const std::uint32_t pass_rows = BatchPassRows();
  for (std::uint32_t r = 0; r < rows; r += pass_rows) {
    if (kEntryTrace) { fprintf(stderr, "ENTRY batch:Executor::MoeBatch\n"); fflush(stderr); }
    const auto n = std::min(pass_rows, rows - r);
    if (!Dense(w, x + std::size_t{r} * w.cols,
               out + std::size_t{r} * w.rows, n, error)) {
      return false;
    }
  }
  return true;
}

bool Executor::MoeBatch(const DeviceLayer& l, const float* x, float* out,
                        std::uint32_t rows, std::string* error,
                        bool reduce) const {
  const std::uint32_t pass_rows = BatchPassRows();
  if (rows <= pass_rows)
    return FeedForward(l, x, out, rows, error, reduce);
  const auto& c = config();
  if (options_.moe_observer && !l.dense_ffn) {
    options_.moe_observer(
        x, static_cast<std::size_t>(rows) * c.hidden_size * sizeof(float),
        stream_);
  }
  const Scratch base = s_;
  for (std::uint32_t r = 0; r < rows; r += pass_rows) {
    UseScratch(RowScratch(base, r));
    const bool ok = FeedForward(
        l, x + std::size_t{r} * c.hidden_size,
        out + std::size_t{r} * c.hidden_size,
        std::min(pass_rows, rows - r), error, /*reduce=*/false);
    UseScratch(base);
    if (!ok)
      return false;
  }
  if (options_.moe_observer && model_->tp_world_size() > 1) {
    options_.moe_observer(
        out, static_cast<std::size_t>(rows) * c.hidden_size * sizeof(float),
        stream_);
  }
  return !reduce || AllReduce(out, rows, error);
}

bool Executor::FeedForwardBatch(const DeviceLayer& l, const float* x,
                                float* out, std::uint32_t rows,
                                std::string* error, bool reduce) const {
  const auto& c = config();
  if (!l.dense_ffn) {
    return MoeBatch(l, x, out, rows, error, reduce);
  }
  const Scratch base = s_;
  for (std::uint32_t r = 0; r < rows; r += kDecodeRows) {
    const auto n = std::min(kDecodeRows, rows - r);
    UseScratch(RowScratch(base, r));
    const bool ok =
        GatedDense(l.ffn_gate, l.ffn_up, x + std::size_t{r} * c.hidden_size,
                   s_.ff_act, n, &l.ffn_down, error) &&
        (ff_half_ready_
             ? (ff_half_ready_ = false,
                DenseF16Gemm(l.ffn_down.data, s_.ff_half,
                             out + std::size_t{r} * c.hidden_size, n,
                             l.ffn_down.rows, l.ffn_down.cols, stream_))
             : Dense(l.ffn_down, s_.ff_act,
                     out + std::size_t{r} * c.hidden_size, n, error));
    UseScratch(base);
    if (!ok)
      return false;
  }
  // An unsplit replicated FFN belongs to rank zero; peers ran the chunks
  // (keeping their staging state in step) and contribute zero partials.
  if (model_->tp_world_size() > 1 && !l.ffn_split &&
      model_->tp_rank() != 0 &&
      !Check(hipMemsetAsync(
                 out, 0,
                 static_cast<std::size_t>(rows) * c.hidden_size * sizeof(float),
                 stream_),
             error)) {
    if (kEntryTrace) { fprintf(stderr, "ENTRY batch:Executor::MtpForwardBatch\n"); fflush(stderr); }
    return false;
  }
  return !reduce || model_->tp_world_size() == 1 ||
         AllReduce(out, rows, error);
}

bool Executor::MtpForwardBatch(std::span<const MtpBatchItem> items,
                               std::string* error) const {
  if (getenv("GUFO_GRAPH_DEBUG") != nullptr) {
    fprintf(stderr, "GRAPHDBG MtpForwardBatch items=%zu\n", items.size());
    fflush(stderr);
  }
  const Config& c = config();
  if (!has_mtp() || items.empty() || items.size() > kBatchSessions) {
    return Fail(error, "invalid MTP body batch");
  }
  std::array<std::uint32_t, kBatchSessions> offsets{};
  std::array<std::uint32_t, kBatchSessions> final_rows{};
  std::uint32_t rows = 0;
  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto& item = items[i];
    const auto n = item.tokens.size();
    if (item.session == nullptr || item.session->owner_ != this ||
        !item.session->mtp_enabled_ || n == 0 || n > kDecodeRows ||
        (item.hidden_row < 0 && n != 1) ||
        (item.hidden_row >= 0 &&
         static_cast<std::uint32_t>(item.hidden_row) + n >
             options_.max_speculative) ||
        item.session->mtp_().position + n > item.session->max_context_) {
      return Fail(error, "invalid MTP body request");
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (items[j].session == item.session)
        return Fail(error, "MTP body requests must be independent");
    }
    for (const auto token : item.tokens) {
      if (token < 0 || static_cast<std::uint32_t>(token) >= c.vocab_size)
        return Fail(error, "MTP token out of range");
    }
    offsets[i] = rows;
    rows += n;
    final_rows[i] = rows - 1;
  }
  if (rows > options_.max_batch)
    return Fail(error, "MTP body batch exceeds executor capacity");
  if (!AnyActive(items))
    return true;
  if (items.size() == 1) {
    const auto& item = items.front();
    return MtpForward(*item.session, item.tokens, item.hidden_row, {}, error);
  }
  if (!AllocateBatch(error) || !AllocateSlots(error)) {
    return false;
  }
  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto& item = items[i];
    auto& session = *item.session;
    ++session.mutation_epoch_;
    FillSlotRow(static_cast<std::uint32_t>(i), session, item.hidden_row);
    std::copy(item.tokens.begin(), item.tokens.end(),
              tokens_host_ + offsets[i]);
  }
  // The body replays as the executor-wide graph for this exact shape: every
  // per-session destination is read from the slot tables at run time (the
  // pinned mirror is refreshed above and copied in at the body's head), so
  // the captured kernels stay correct across session mixes.
  std::uint64_t shape_key = rows;
  for (std::size_t i = 0; i < items.size(); ++i) {
    shape_key = shape_key * 33u + items[i].tokens.size();
  }
  shape_key |= std::uint64_t{1} << 62;  // batch-MTP key space
  const Scratch base = s_;
  const DeviceLayer& block = model_->nextn_layers().front();
  const AttnDims dims = DimsOf(block);
  const DeviceTensor& table = !block.nextn_embed.empty() ? block.nextn_embed
                                                         : model_->token_embd();
  const auto body = [&]() -> bool {
    if (!Check(hipMemcpyAsync(slots_.dev, slots_.host, slots_.bytes,
                              hipMemcpyHostToDevice, stream_),
               error) ||
        !Check(hipMemcpyAsync(base.tokens, tokens_host_,
                              rows * sizeof(std::int32_t),
                              hipMemcpyHostToDevice, stream_),
               error)) {
      return false;
    }
    // e side: the private embedding table when the sidecar carries one.
    EmbedTokens(table.data, WeightTypeOf(table.type), base.tokens, base.xn,
                rows, c.hidden_size, 1, stream_);
    RmsNormRows(base.xn, block.nextn_enorm.f32(), base.xn, rows, c.hidden_size,
                1, c.rms_eps, stream_);
    for (std::size_t i = 0; i < items.size(); ++i) {
      const auto n = static_cast<std::uint32_t>(items[i].tokens.size());
      UseScratch(RowScratch(base, offsets[i], dims));
      // A cancelled slot's null table pointers write nothing here and clear
      // its attention output below, matching the eager skip.
      MtpHiddenSlot(slots_.DevPtrs(), slots_.DevCtrl(),
                    static_cast<std::uint32_t>(i), s_.mtp_h, n, c.hidden_size,
                    stream_);
    }
    RmsNormRows(base.mtp_h, block.nextn_hnorm.f32(), base.mtp_h, rows,
                c.hidden_size, 1, c.rms_eps, stream_);
    // Concatenate [e_norm | h_norm] once over the whole batch (strided
    // device-to-device copies), then share the eh projection and the block.
    if (!Check(hipMemcpy2DAsync(
                   base.mtp_eh, 2 * c.hidden_size * sizeof(float), base.xn,
                   c.hidden_size * sizeof(float),
                   c.hidden_size * sizeof(float), rows,
                   hipMemcpyDeviceToDevice, stream_),
               error) ||
        !Check(hipMemcpy2DAsync(
                   base.mtp_eh + c.hidden_size,
                   2 * c.hidden_size * sizeof(float), base.mtp_h,
                   c.hidden_size * sizeof(float),
                   c.hidden_size * sizeof(float), rows,
                   hipMemcpyDeviceToDevice, stream_),
               error) ||
        !DenseBatch(block.nextn_eh_proj, base.mtp_eh, base.res, rows, error)) {
      return false;
    }
    RmsNormRows(base.res, block.attn_norm.f32(), base.xn, rows, c.hidden_size,
                1, c.rms_eps, stream_);
    UseScratch(base);
    if (!DenseBatch(block.attn_q, base.xn, base.q, rows, error) ||
        !DenseBatch(block.attn_k, base.xn, base.k, rows, error) ||
        !DenseBatch(block.attn_v, base.xn, base.v, rows, error)) {
      return false;
    }
    {
      // One fused call for the cohort's draft rows: the row tables carry each
      // row's session slot and MTP position, so the per-session launch triples
      // collapse exactly as in the trunk.
      Session::AttentionState attn;  // unused values under the slot table
      std::uint32_t row = 0;
      for (std::size_t i = 0; i < items.size(); ++i) {
        const auto n = static_cast<std::uint32_t>(items[i].tokens.size());
        const auto start =
            static_cast<std::uint32_t>(slots_.HostCtrl()[i].mtp_position);
        for (std::uint32_t j = 0; j < n; ++j) {
          slots_.row_slot_host[row] = static_cast<std::uint32_t>(i);
          slots_.row_pos_host[row] = start + j;
          ++row;
        }
      }
      if (!Check(hipMemcpyAsync(slots_.row_slot_dev, slots_.row_slot_host,
                                row * sizeof(std::uint32_t),
                                hipMemcpyHostToDevice, stream_),
                 error) ||
          !Check(hipMemcpyAsync(slots_.row_pos_dev, slots_.row_pos_host,
                                row * sizeof(std::uint32_t),
                                hipMemcpyHostToDevice, stream_),
                 error)) {
        return false;
      }
      UseScratch(base);
      if (!AttentionPart(block, attn, s_.xn, s_.ctx, row, nullptr, error,
                         /*projections_ready=*/true,
                         /*project_output=*/false, slots_.DevMtpKv(), 0u,
                         /*visible_keys=*/0, /*wide_splits=*/1,
                         slots_.row_slot_dev, slots_.row_pos_dev)) {
        return false;
      }
    }
    // Attention retains every catch-up KV row. Only each final residual and
    // context row contributes to the next proposal, so compact those rows
    // into idle trunk scratch before the output projection and FFN: a
    // one-row tail in the original rows-wide projection has different
    // scalar rounding. The projection buffers are dead by then.
    const bool compact_tail = rows > kDecodeRows && rows % kDecodeRows != 1;
    Scratch tail = base;
    if (compact_tail) {
      tail.res = base.mtp_eh;
      tail.ctx = base.q;
      for (std::size_t i = 0; i < items.size(); ++i) {
        const auto copy = [&](float* dst, const float* src,
                              std::size_t width) {
          return Check(hipMemcpyAsync(dst + i * width,
                                      src + std::size_t{final_rows[i]} * width,
                                      width * sizeof(float),
                                      hipMemcpyDeviceToDevice, stream_),
                       error);
        };
        if (!copy(tail.res, base.res, c.hidden_size) ||
            !copy(tail.ctx, base.ctx, dims.ctx)) {
          return false;
        }
      }
    }
    const auto tail_rows =
        compact_tail ? static_cast<std::uint32_t>(items.size()) : rows;
    UseScratch(tail);
    if (!DenseBatch(block.attn_out, tail.ctx, tail.block_out, tail_rows,
                    error)) {
      return false;
    }
    if (value_scale_ != 1.0F) {
      ScaleInPlace(tail.block_out, value_scale_,
                   static_cast<std::size_t>(tail_rows) * c.hidden_size,
                   stream_);
    }
    AddResidual(tail.res, tail.block_out, tail_rows);
    RmsNormRows(tail.res, block.ffn_norm.f32(), tail.xn, tail_rows,
                c.hidden_size, 1, c.rms_eps, stream_);
    if (!FeedForwardBatch(block, tail.xn, tail.block_out, tail_rows, error))
      return false;
    AddResidual(tail.res, tail.block_out, tail_rows);
    for (std::size_t i = 0; i < items.size(); ++i) {
      const auto row = compact_tail ? i : final_rows[i];
      CopyRowsSlot(tail.res + std::size_t{row} * c.hidden_size,
                   &slots_.DevPtrs()[i].mtp_h, 1, c.hidden_size, stream_);
    }
    return true;
  };
  bool ok = false;
  try {
    ok = RunCaptured(shape_key, body, error);
  } catch (...) {
    (void)hipStreamSynchronize(stream_);
    UseScratch(base);
    throw;
  }
  const auto status = hipStreamSynchronize(stream_);
  UseScratch(base);
  if (!ok || !Check(status, error) ||
      (options_.reduce_status && !options_.reduce_status(error)))
    return false;
  for (const auto& item : items) {
    if (item.session->Cancelled())
      continue;
    item.session->mtp_().position += item.tokens.size();
  }
  return true;
}

bool Executor::MtpHeads(std::span<const MtpHeadItem> items,
                        std::string* error) const {
  selected_logits_ = nullptr;
  if (!has_mtp() || items.empty() || items.size() > kBatchSessions ||
      items.size() > options_.max_batch) {
    return Fail(error, "invalid MTP head batch");
  }
  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto& item = items[i];
    if (item.session == nullptr || item.session->owner_ != this ||
        !item.session->mtp_enabled_ || item.session->mtp_().position == 0 ||
        (item.output.token == nullptr && item.output.candidates == nullptr)) {
      return Fail(error, "invalid MTP head request");
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (items[j].session == item.session) {
        return Fail(error, "MTP head requests must be independent");
      }
    }
  }
  const DeviceLayer& block = model_->nextn_layers().front();
  const DeviceTensor& norm = !block.nextn_head_norm.empty()
                                 ? block.nextn_head_norm
                             : !block.layer_out_norm.empty()
                                 ? block.layer_out_norm
                                 : model_->output_norm();
  const DeviceTensor& head = !block.nextn_head_head.empty()
                                 ? block.nextn_head_head
                                 : model_->output();
  if (!AnyActive(items))
    return true;
  if (items.size() == 1) {
    for (const auto& item : items) {
      if (!item.session->CheckCancellation(nullptr))
        continue;
      if (!MtpHead(block, item.session->mtp_().h, 0,
                   item.output.token != nullptr,
                   item.output.candidates != nullptr, error) ||
          !Check(hipStreamSynchronize(stream_), error)) {
        return false;
      }
      if (item.output.token != nullptr)
        *item.output.token = *mtp_token_host_;
      if (item.output.candidates != nullptr)
        *item.output.candidates = *mtp_candidates_host_;
    }
    return true;
  }
  if (!AllocateBatch(error)) {
    return false;
  }
  if (batch_candidates_host_ == nullptr) {
    if (!Check(hipHostMalloc(&batch_candidates_host_,
                             kBatchSessions * sizeof(MtpCandidateLogits)),
               error))
      return false;
    for (std::uint32_t i = 0; i < kBatchSessions; ++i)
      std::construct_at(batch_candidates_host_ + i);
  }
  const auto n = static_cast<std::uint32_t>(items.size());
  const auto count = std::min<std::size_t>(head.rows, kMtpCandidates);
  const Scratch base = s_;
  const auto body = [&]() {
    if (!AnyActive(items))
      return true;
    for (std::uint32_t i = 0; i < n; ++i) {
      RmsNormRows(items[i].session->mtp_().h, norm.f32(),
                  base.xn + std::size_t{i} * config().hidden_size, 1,
                  config().hidden_size, 1, config().rms_eps, stream_);
    }
    UseScratch(base);
    if (!DenseBatch(head, base.xn, batch_logits_, n, error))
      return false;
    staged_logit_rows_ = 0;
    if (!AnyActive(items))
      return true;
    for (std::uint32_t i = 0; i < n; ++i) {
      if (!items[i].session->CheckCancellation(nullptr))
        continue;
      if (items[i].output.candidates == nullptr) {
        Argmax(batch_logits_ + std::size_t(i) * head.rows, s_.mtp_argmax,
               s_.mtp_token, 1, head.rows, stream_);
        if (!Check(hipMemcpyAsync(batch_candidates_host_[i].ids.data(),
                                  s_.mtp_token, sizeof(std::int32_t),
                                  hipMemcpyDeviceToHost, stream_),
                   error))
          return false;
        continue;
      }
      MtpTopCandidates(batch_logits_ + std::size_t(i) * head.rows, s_.mtp_ids,
                       s_.mtp_scratch_ids, s_.mtp_scores, head.rows, stream_);
      if (!Check(hipMemcpyAsync(batch_candidates_host_ + i, s_.mtp_ids,
                                offsetof(MtpCandidateLogits, logits) +
                                    count * sizeof(float),
                                hipMemcpyDeviceToHost, stream_),
                 error)) {
        return false;
      }
      batch_candidates_host_[i].size = count;
    }
    return true;
  };
  bool ok = false;
  try {
    if (kEntryTrace) { fprintf(stderr, "ENTRY batch:Executor::ForwardBatch\n"); fflush(stderr); }
    ok = body();
  } catch (...) {
    (void)hipStreamSynchronize(stream_);
    UseScratch(base);
    throw;
  }
  const auto status = hipStreamSynchronize(stream_);
  UseScratch(base);
  if (!ok || !Check(status, error) ||
      (options_.reduce_status && !options_.reduce_status(error)))
    return false;
  for (std::size_t i = 0; i < items.size(); ++i) {
    if (items[i].session->Cancelled())
      continue;
    if (items[i].output.token != nullptr)
      *items[i].output.token = batch_candidates_host_[i].ids[0];
    if (items[i].output.candidates != nullptr)
      *items[i].output.candidates = batch_candidates_host_[i];
  }
  return true;
}

bool Executor::ForwardBatch(std::span<const BatchItem> items,
                            std::string* error) const {
  if (getenv("GUFO_GRAPH_DEBUG") != nullptr) {
    fprintf(stderr, "GRAPHDBG ForwardBatch items=%zu\n", items.size());
    fflush(stderr);
  }
  selected_logits_ = nullptr;
  const Config& c = config();
  // Staging caches are keyed on (pointer, shape) and every one of these
  // buffers is rewritten in place between layers/chunks/tokens: invalidate at
  // the boundary so a re-staged buffer cannot false-hit. Within a layer the
  // legal reuse (gate/up sharing one input) survives.
  q8t_src_ = nullptr;
  q8t_rows_ = 0;
  q8t_cols_ = 0;
  half_src_ = nullptr;
  half_rows_ = 0;
  half_cols_ = 0;
  half_bf16_ = false;
  q8t_src_ = nullptr;
  q8t_rows_ = 0;
  q8t_cols_ = 0;
  if (items.empty() || items.size() > kBatchSessions) {
    return Fail(error, "decode batch must contain 1..8 sessions");
  }
  std::array<std::uint32_t, kBatchSessions> offsets{};
  std::uint32_t rows = 0;
  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto& item = items[i];
    if (item.session == nullptr || item.session->owner_ != this ||
        item.tokens.empty() || item.tokens.size() > kDecodeRows ||
        item.tokens.size() > options_.max_logit_rows ||
        (item.speculative && item.tokens.size() > options_.max_speculative) ||
        item.session->position_ + item.tokens.size() >
            item.session->max_context_) {
      return Fail(error, "invalid session or chain in decode batch");
    }
    for (std::size_t j = 0; j < i; ++j) {
      if (items[j].session == item.session) {
        return Fail(error, "decode batch contains a duplicate session");
      }
    }
    if (!item.session->tap_layers_.empty()) {
      return Fail(error, "decode batch does not support target tap capture");
    }
    for (const auto token : item.tokens) {
      if (token < 0 || static_cast<std::uint32_t>(token) >= c.vocab_size) {
        return Fail(error, "token out of range");
      }
    }
    offsets[i] = rows;
    rows += static_cast<std::uint32_t>(item.tokens.size());
  }
  if (rows > options_.max_batch) {
    return Fail(error, "decode batch exceeds executor capacity");
  }
  if (!AnyActive(items))
    return true;
  for (const auto& item : items) {
    if (item.speculative &&
        !EnsureRollback(*item.session, item.tokens.size() - 1, error))
      return false;
  }
  if (!AllocateBatch(error)) {
    return false;
  }
  if (!AllocateSlots(error)) {
    return false;
  }
  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto& item = items[i];
    auto& session = *item.session;
    ++session.mutation_epoch_;
    session.spec_base_ = session.position_;
    session.spec_tokens_ = item.speculative ? item.tokens.size() : 0;
    FillSlotRow(static_cast<std::uint32_t>(i), session, -1);
    std::copy(item.tokens.begin(), item.tokens.end(),
              tokens_host_ + offsets[i]);
  }
  // The cohort attention reads each row's own cache, so the widest row decides
  // whether the step splits its rows' key ranges (deep rows would otherwise
  // walk thousands of key tiles inside one block).
  batch_visible_keys_ = 0;
  for (const auto& item : items) {
    const auto visible = item.session->position_ + item.tokens.size();
    if (visible > batch_visible_keys_) {
      batch_visible_keys_ = visible;
    }
  }

  const Scratch base = s_;
  // The two-lane pipeline interleaves the cohort's layer halves so one lane's
  // exchange is in flight while the GPU computes the other. Measured slower
  // than the single sweep on both shapes (2026-10-07, six-way counting: AR
  // 62.8 vs 82.8 t/s, production MTP 59.9 vs 73.8), because this transport's
  // overlapped exchange is host-synchronous: `finish` blocks the enqueueing
  // thread (BATCHTRACE showed 74 ms/cycle inside it) and a stream-ordered
  // wait cannot be covered by work queued behind it, so the split only adds
  // the second chain — an extra exchange handshake, a second set of small
  // GEMMs, and for overlapping expert sets a second read of the same
  // experts. It stays behind GUFO_BATCH_LANES=1 for follow-up on a
  // two-stream executor, which is what covering the wait would actually need.
  static const bool lanes_enabled = [] {
    const char* env = getenv("GUFO_BATCH_LANES");
    return env != nullptr && env[0] == '1' && env[1] == '\0';
  }();
  const bool lanes =
      lanes_enabled && options_.split_reduce.start != nullptr &&
      model_->tp_world_size() > 1 && items.size() >= 2 && rows >= 4 &&
      !(GraphCaptureEnabled() &&
        (options_.communicator && options_.communicator->CaptureSafe()));
  // The body replays as the executor-wide graph for this exact shape: every
  // per-session destination (KV caches, control values, hidden rows) is read
  // from the slot tables at run time — the pinned mirror was refreshed above
  // and the copy node at the body's head re-reads it on every replay. The
  // attention split is the batch's only data-dependent launch shape, so its
  // threshold crossing belongs in the key.
  std::uint64_t shape_key = rows;
  for (std::size_t i = 0; i < items.size(); ++i) {
    shape_key = shape_key * 33u +
                (items[i].tokens.size() * 2u + (items[i].speculative ? 1u : 0u));
  }
  const std::uint32_t split_keys = Executor::AttentionSplitKeys();
  shape_key = shape_key * 2u +
              (split_keys != 0 && batch_visible_keys_ >= split_keys ? 1u : 0u);
  // Every exit restores the ordinary single-session scratch view. Enqueued
  // work is drained before a caller can reuse its pinned staging buffers.
  static BatchPhases phases;
  const double phase_t0 = phases.on ? BatchPhases::Now() : 0.0;
  const auto body = [&]() -> bool {
    if (!Check(hipMemcpyAsync(slots_.dev, slots_.host, slots_.bytes,
                              hipMemcpyHostToDevice, stream_),
               error)) {
      return false;
    }
    if (!Check(hipMemcpyAsync(base.tokens, tokens_host_,
                              rows * sizeof(std::int32_t),
                              hipMemcpyHostToDevice, stream_),
               error)) {
      return false;
    }
    EmbedTokens(model_->token_embd().data,
                WeightTypeOf(model_->token_embd().type), base.tokens, base.res,
                rows, c.hidden_size, 1, stream_);
    const auto& layers = model_->layers();
    for (std::uint32_t il = 0; il < c.num_layers; ++il) {
      // Per layer: the norm/dense buffers are rewritten at the same addresses
      // every layer (see the staging-cache note in Executor::Dense).
      q8t_src_ = nullptr;
      q8t_rows_ = 0;
      q8t_cols_ = 0;
      half_src_ = nullptr;
      half_rows_ = 0;
      half_cols_ = 0;
      half_bf16_ = false;
      const auto& l = layers[il];
      const AttnDims dims = DimsOf(l);
      const double t_proj = phases.on ? BatchPhases::Now() : 0.0;
      RmsNormRows(base.res, l.attn_norm.f32(), base.xn, rows, c.hidden_size, 1,
                  c.rms_eps, stream_);
      UseScratch(base);
      if (!DenseBatch(l.attn_q, base.xn, base.q, rows, error) ||
          !DenseBatch(l.attn_k, base.xn, base.k, rows, error) ||
          !DenseBatch(l.attn_v, base.xn, base.v, rows, error)) {
        return false;
      }
      const double t_attn = phases.on ? BatchPhases::Now() : 0.0;
      if (phases.on) {
        phases.proj += t_attn - t_proj;
      }
      {
        // Unused values under the slot table; a cancelled slot's null table
        // pointers store no KV and clear its context rows (the eager skip
        // plus memset), so the kernel sequence never depends on
        // cancellation state.
        //
        // One fused call covers the whole cohort: the row tables carry each
        // row's session slot and absolute position, so the six per-session
        // rope/store/attention launch triples collapse into one. The rows sit
        // consecutively in the shared scratch, which is what lets a single
        // call read them with per-row destinations.
        Session::AttentionState attn;
        std::uint32_t row = 0;
        for (std::size_t i = 0; i < items.size(); ++i) {
          const auto n = static_cast<std::uint32_t>(items[i].tokens.size());
          const auto start = static_cast<std::uint32_t>(slots_.HostCtrl()[i].position);
          for (std::uint32_t j = 0; j < n; ++j) {
            slots_.row_slot_host[row] = static_cast<std::uint32_t>(i);
            slots_.row_pos_host[row] = start + j;
            ++row;
          }
        }
        if (!Check(hipMemcpyAsync(slots_.row_slot_dev, slots_.row_slot_host,
                                  row * sizeof(std::uint32_t),
                                  hipMemcpyHostToDevice, stream_),
                   error) ||
            !Check(hipMemcpyAsync(slots_.row_pos_dev, slots_.row_pos_host,
                                  row * sizeof(std::uint32_t),
                                  hipMemcpyHostToDevice, stream_),
                   error)) {
          return false;
        }
        UseScratch(base);
        if (!AttentionPart(l, attn, s_.xn, s_.ctx, row, nullptr, error,
                           /*projections_ready=*/true,
                           /*project_output=*/false, slots_.DevKv(il), 0u,
                           /*visible_keys=*/0, /*wide_splits=*/1,
                           slots_.row_slot_dev, slots_.row_pos_dev)) {
          return false;
        }
      }
      const double t_oproj = phases.on ? BatchPhases::Now() : 0.0;
      if (phases.on) {
        phases.attn += t_oproj - t_attn;
      }
      // With split heads each rank's projection is a partial the layer's
      // exchange completes; a replicated block contributes a zeroed peer.
      if (!DenseBatch(l.attn_out, base.ctx, base.block_out, rows, error)) {
        return false;
      }
      const double t_ar = phases.on ? BatchPhases::Now() : 0.0;
      if (phases.on) {
        phases.oproj += t_ar - t_oproj;
      }
      if (getenv("GUFO_SYNC_TRACE") != nullptr) {
        hipStreamSynchronize(stream_);
        const hipError_t e = hipGetLastError();
        if (e != hipSuccess) {
          return Fail(error, std::string("TRACE after attn_out DenseBatch: ") +
                                 hipGetErrorString(e));
        }
      }
      if (value_scale_ != 1.0F) {
        // Same rocBLAS avoidance as the single-request path: the scale is a
        // plain elementwise multiply.
        ScaleInPlace(base.block_out, value_scale_,
                     static_cast<std::size_t>(rows) * c.hidden_size, stream_);
      }
      if (model_->tp_world_size() > 1) {
        if (!l.attn_split && model_->tp_rank() != 0 &&
            !Check(hipMemsetAsync(
                       base.block_out, 0,
                       static_cast<std::size_t>(rows) * c.hidden_size *
                           sizeof(float),
                       stream_),
                   error)) {
          return false;
        }
        if (getenv("GUFO_SYNC_TRACE") != nullptr) {
          hipStreamSynchronize(stream_);
          const hipError_t e0 = hipGetLastError();
          if (e0 != hipSuccess) {
            return Fail(error, std::string("TRACE before AllReduce: ") +
                                   hipGetErrorString(e0));
          }
        }
        if (!AllReduce(base.block_out, rows, error)) {
          return false;
        }
        if (phases.on) {
          phases.ar += BatchPhases::Now() - t_ar;
        }
        if (getenv("GUFO_SYNC_TRACE") != nullptr) {
          hipStreamSynchronize(stream_);
          const hipError_t e1 = hipGetLastError();
          if (e1 != hipSuccess) {
            return Fail(error, std::string("TRACE after AllReduce: ") +
                                   hipGetErrorString(e1));
          }
        }
      }
      AddResidual(base.res, base.block_out, rows);
      RmsNormRows(base.res, l.ffn_norm.f32(), base.xn, rows, c.hidden_size, 1,
                  c.rms_eps, stream_);
      UseScratch(base);
      if (getenv("GUFO_DUMP_DIR") != nullptr && il == 1) {
        static bool dumped_in = false;
        if (!dumped_in) {
          dumped_in = true;
          std::vector<float> host(4096);
          hipStreamSynchronize(stream_);
          hipMemcpy(host.data(), base.res, 16384, hipMemcpyDeviceToHost);
          FILE* g = fopen((std::string(getenv("GUFO_DUMP_DIR")) + "/ffn-res-in.txt").c_str(), "w");
          for (float v : host) fprintf(g, "%.6f\n", v);
          fclose(g);
          hipMemcpy(host.data(), base.xn, 16384, hipMemcpyDeviceToHost);
          g = fopen((std::string(getenv("GUFO_DUMP_DIR")) + "/ffn-xn.txt").c_str(), "w");
          for (float v : host) fprintf(g, "%.6f\n", v);
          fclose(g);
        }
      }
      const double t_ffn = phases.on ? BatchPhases::Now() : 0.0;
      if (!FeedForwardBatch(l, base.xn, base.block_out, rows, error)) {
        return false;
      }
      if (phases.on) {
        phases.ffn += BatchPhases::Now() - t_ffn;
      }
      if (getenv("GUFO_DUMP_DIR") != nullptr && il == 1) {
        std::vector<float> host(4096);
        hipStreamSynchronize(stream_);
        hipMemcpy(host.data(), base.block_out, 16384, hipMemcpyDeviceToHost);
        FILE* f = fopen((std::string(getenv("GUFO_DUMP_DIR")) + "/ffn-out.txt").c_str(), "w");
        for (float v : host) fprintf(f, "%.6f\n", v);
        fclose(f);
      }
      AddResidual(base.res, base.block_out, rows);
    }
    for (std::size_t i = 0; i < items.size(); ++i) {
      const auto n = static_cast<std::uint32_t>(items[i].tokens.size());
      UseScratch(RowScratch(base, offsets[i]));
      // The slot table carries each session's destination (a memcpy node
      // cannot read it); a cancelled slot's null pointer writes nothing.
      CopyRowsSlot(s_.res, &slots_.DevPtrs()[i].target_hidden, n,
                   c.hidden_size, stream_);
    }
    UseScratch(base);
    const double t_head = phases.on ? BatchPhases::Now() : 0.0;
    RmsNormRows(base.res, model_->output_norm().f32(), base.xn, rows,
                c.hidden_size, 1, c.rms_eps, stream_);
    const bool ok = DenseBatch(model_->output(), base.xn, batch_logits_, rows,
                               error);
    staged_logit_rows_ = 0;
    if (phases.on) {
      phases.head += BatchPhases::Now() - t_head;
    }
    return ok;
  };
  bool ok = false;
  try {
    ok = lanes ? ForwardBatchLanes(items, offsets, rows, base, error)
               : RunCaptured(shape_key, body, error);
  } catch (...) {
    (void)hipStreamSynchronize(stream_);
    UseScratch(base);
    throw;
  }
  const auto status = hipStreamSynchronize(stream_);
  UseScratch(base);
  if (phases.on) {
    phases.total += BatchPhases::Now() - phase_t0;
    phases.rows_max = std::max(phases.rows_max, rows);
    phases.Tick();
  }
  if (!ok || !Check(status, error) ||
      (options_.reduce_status && !options_.reduce_status(error))) {
    return false;
  }
  for (std::size_t i = 0; i < items.size(); ++i) {
    if (items[i].session->Cancelled())
      continue;
    items[i].session->position_ += items[i].tokens.size();
  }
  batch_rows_ = rows;
  return true;
}

bool Executor::ForwardBatchLanes(std::span<const BatchItem> items,
                                 std::span<const std::uint32_t> offsets,
                                 std::uint32_t rows, const Scratch& base,
                                 std::string* error) const {
  const Config& c = config();
  const auto& exchange = options_.split_reduce;
  const auto& layers = model_->layers();
  // Prefix split with the closest row counts; the rule reads only the row
  // layout, so both ranks cut the cohort in the same place.
  std::size_t lead = 1;
  {
    std::uint32_t acc = 0;
    for (std::size_t i = 0; i + 1 < items.size(); ++i) {
      acc += static_cast<std::uint32_t>(items[i].tokens.size());
      lead = i + 1;
      if (acc * 2 >= rows) {
        break;
      }
    }
  }
  // A lane's scratch is the batch scratch with its own residual and layer
  // output rows (the pair buffers the prefill pipeline also uses). Halves run
  // one at a time and each re-derives `xn` from its own residual, so the
  // shared projections, activations and attention buffers need no second
  // copy.
  struct Lane {
    Scratch scratch;
    std::uint32_t row_off;  ///< first row of this lane in the batch tables
    std::uint32_t n;        ///< its rows
    std::uint32_t visible;  ///< its widest row's visible key count
    std::uint32_t part;     ///< the next half to run (two per layer)
    bool pending;           ///< its last half's exchange is in flight
  };
  std::array<Lane, 2> lanes{
      Lane{.scratch = base,
           .row_off = 0,
           .n = offsets[lead],
           .visible = 0,
           .part = 0,
           .pending = false},
      Lane{.scratch = PairScratch(base),
           .row_off = offsets[lead],
           .n = rows - offsets[lead],
           .visible = 0,
           .part = 0,
           .pending = false}};
  for (std::size_t i = 0; i < items.size(); ++i) {
    Lane& lane = lanes[i < lead ? 0 : 1];
    const auto visible = static_cast<std::uint32_t>(
        items[i].session->position_ + items[i].tokens.size());
    lane.visible = std::max(lane.visible, visible);
  }
  static std::uint32_t lanes_logged = 0;
  if (getenv("GUFO_GRAPH_DEBUG") != nullptr && lanes_logged < 4) {
    ++lanes_logged;
    fprintf(stderr, "BATCHLANES rows=%u lead=%zu rows_a=%u rows_b=%u\n", rows,
            lead, lanes[0].n, lanes[1].n);
    fflush(stderr);
  }
  static BatchPhases phases;
  const double phase_t0 = phases.on ? BatchPhases::Now() : 0.0;
  std::array<const Lane*, 2> in_flight{};
  std::size_t flying = 0;
  // Sums finish in the order they started; a failed forward still waits for
  // the exchanges it started, so both ranks' sequences stay aligned.
  struct Settle {
    const Executor* executor;
    const Scratch& base;
    std::array<const Lane*, 2>& in_flight;
    std::size_t& flying;
    ~Settle() {
      for (std::size_t i = 0; i < flying; ++i) {
        const Lane& lane = *in_flight[i];
        std::string ignored;
        (void)executor->options_.split_reduce.finish(
            lane.scratch.block_out,
            static_cast<std::size_t>(lane.n) *
                executor->config().hidden_size * sizeof(float),
            executor->stream_, &ignored);
      }
      executor->UseScratch(base);
    }
  } settle{this, base, in_flight, flying};
  const auto bytes = [&](const Lane& lane) {
    return static_cast<std::size_t>(lane.n) * c.hidden_size * sizeof(float);
  };
  const auto enter = [&](const Lane& lane) { UseScratch(lane.scratch); };

  // The layer's attention half: the norm, the fused cohort attention over
  // this lane's row tables and the output projection, leaving the rank's
  // partial of the layer output in the lane's block_out.
  const auto attn_half = [&](const Lane& lane, std::uint32_t il) -> bool {
    const DeviceLayer& l = layers[il];
    enter(lane);
    const double t_proj = phases.on ? BatchPhases::Now() : 0.0;
    RmsNormRows(lane.scratch.res, l.attn_norm.f32(), lane.scratch.xn, lane.n,
                c.hidden_size, 1, c.rms_eps, stream_);
    if (!DenseBatch(l.attn_q, lane.scratch.xn, lane.scratch.q, lane.n, error) ||
        !DenseBatch(l.attn_k, lane.scratch.xn, lane.scratch.k, lane.n, error) ||
        !DenseBatch(l.attn_v, lane.scratch.xn, lane.scratch.v, lane.n, error)) {
      return false;
    }
    const double t_attn = phases.on ? BatchPhases::Now() : 0.0;
    if (phases.on) {
      phases.proj += t_attn - t_proj;
    }
    // The row tables carry each row's own session slot and absolute
    // position, so a lane's rows read their own sessions' caches and the
    // lanes stay independent. AttentionPart takes the key-range split
    // threshold from the batch member, which this lane's widest row sets.
    batch_visible_keys_ = lane.visible;
    Session::AttentionState attn;
    if (!AttentionPart(l, attn, lane.scratch.xn, lane.scratch.ctx, lane.n,
                       nullptr, error, /*projections_ready=*/true,
                       /*project_output=*/false, slots_.DevKv(il), 0u,
                       /*visible_keys=*/0, /*wide_splits=*/1,
                       slots_.row_slot_dev + lane.row_off,
                       slots_.row_pos_dev + lane.row_off)) {
      return false;
    }
    const double t_oproj = phases.on ? BatchPhases::Now() : 0.0;
    if (phases.on) {
      phases.attn += t_oproj - t_attn;
    }
    // With split heads each rank's projection is a partial the layer's
    // exchange completes; a replicated block contributes a zeroed peer.
    if (!DenseBatch(l.attn_out, lane.scratch.ctx, lane.scratch.block_out,
                    lane.n, error)) {
      return false;
    }
    if (phases.on) {
      phases.oproj += BatchPhases::Now() - t_oproj;
    }
    if (value_scale_ != 1.0F) {
      ScaleInPlace(lane.scratch.block_out, value_scale_,
                   static_cast<std::size_t>(lane.n) * c.hidden_size, stream_);
    }
    if (model_->tp_world_size() > 1 && !l.attn_split &&
        model_->tp_rank() != 0 &&
        !Check(hipMemsetAsync(lane.scratch.block_out, 0,
                              static_cast<std::size_t>(lane.n) *
                                  c.hidden_size * sizeof(float),
                              stream_),
               error)) {
      return false;
    }
    return true;
  };

  // The layer's feed-forward half: the norm and the FFN, its partial left in
  // the lane's block_out.
  const auto ffn_half = [&](const Lane& lane, std::uint32_t il) -> bool {
    const DeviceLayer& l = layers[il];
    enter(lane);
    RmsNormRows(lane.scratch.res, l.ffn_norm.f32(), lane.scratch.xn, lane.n,
                c.hidden_size, 1, c.rms_eps, stream_);
    const double t_ffn = phases.on ? BatchPhases::Now() : 0.0;
    const bool half =
        FeedForwardBatch(l, lane.scratch.xn, lane.scratch.block_out, lane.n,
                         error, /*reduce=*/false);
    if (phases.on) {
      phases.ffn += BatchPhases::Now() - t_ffn;
    }
    return half;
  };

  // Runs the lane's halves until one ends in a partial, whose exchange it
  // starts, then yields to the other lane.
  const std::uint32_t parts = 2 * c.num_layers;
  const auto advance = [&](Lane& lane) -> bool {
    while (lane.part < parts) {
      const std::uint32_t part = lane.part++;
      const std::uint32_t il = part / 2;
      if (part % 2 == 0 ? !attn_half(lane, il) : !ffn_half(lane, il)) {
        return false;
      }
      if (!exchange.start(lane.scratch.block_out, bytes(lane), stream_,
                          error)) {
        return false;
      }
      in_flight[flying++] = &lane;
      lane.pending = true;
      return true;
    }
    return true;
  };
  // Adds the peer's partial to the lane's and folds the sum into its
  // residual. The attention half's sum must land before the FFN half's norm
  // reads it, exactly as the single-sweep path adds both.
  const auto finish = [&](Lane& lane) -> bool {
    if (flying == 0 || in_flight[0] != &lane) {
      Fail(error, "decode lane sums finished out of order");
      return false;
    }
    in_flight[0] = in_flight[1];
    --flying;
    lane.pending = false;
    const double t_ar = phases.on ? BatchPhases::Now() : 0.0;
    if (!exchange.finish(lane.scratch.block_out, bytes(lane), stream_,
                         error)) {
      return false;
    }
    if (phases.on) {
      phases.ar += BatchPhases::Now() - t_ar;
    }
    enter(lane);
    AddResidual(lane.scratch.res, lane.scratch.block_out, lane.n);
    return true;
  };

  if (!Check(hipMemcpyAsync(slots_.dev, slots_.host, slots_.bytes,
                            hipMemcpyHostToDevice, stream_),
             error) ||
      !Check(hipMemcpyAsync(base.tokens, tokens_host_,
                            rows * sizeof(std::int32_t),
                            hipMemcpyHostToDevice, stream_),
             error)) {
    return false;
  }
  // The fused cohort's per-row tables, rebuilt as the single sweep rebuilds
  // them inside its body. A lane reads the slice its own rows occupy, and
  // those rows sit at row zero of its scratch.
  std::uint32_t row = 0;
  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto n = static_cast<std::uint32_t>(items[i].tokens.size());
    const auto start =
        static_cast<std::uint32_t>(slots_.HostCtrl()[i].position);
    for (std::uint32_t j = 0; j < n; ++j) {
      slots_.row_slot_host[row] = static_cast<std::uint32_t>(i);
      slots_.row_pos_host[row] = start + j;
      ++row;
    }
  }
  if (!Check(hipMemcpyAsync(slots_.row_slot_dev, slots_.row_slot_host,
                            row * sizeof(std::uint32_t),
                            hipMemcpyHostToDevice, stream_),
             error) ||
      !Check(hipMemcpyAsync(slots_.row_pos_dev, slots_.row_pos_host,
                            row * sizeof(std::uint32_t),
                            hipMemcpyHostToDevice, stream_),
             error)) {
    return false;
  }
  for (Lane& lane : lanes) {
    enter(lane);
    EmbedTokens(model_->token_embd().data,
                WeightTypeOf(model_->token_embd().type),
                base.tokens + lane.row_off, lane.scratch.res, lane.n,
                c.hidden_size, 1, stream_);
    if (!advance(lane)) {
      return false;
    }
  }
  while (lanes[0].pending || lanes[1].pending) {
    for (Lane& lane : lanes) {
      if (!lane.pending) {
        continue;
      }
      enter(lane);
      if (!finish(lane) || !advance(lane)) {
        return false;
      }
    }
  }
  for (std::size_t i = 0; i < items.size(); ++i) {
    const auto n = static_cast<std::uint32_t>(items[i].tokens.size());
    const Lane& lane = lanes[i < lead ? 0 : 1];
    enter(lane);
    // The slot table carries each session's destination (a memcpy node
    // cannot read it); a cancelled slot's null pointer writes nothing.
    CopyRowsSlot(lane.scratch.res +
                     static_cast<std::size_t>(offsets[i] - lane.row_off) *
                         c.hidden_size,
                 &slots_.DevPtrs()[i].target_hidden, n, c.hidden_size,
                 stream_);
  }
  for (const Lane& lane : lanes) {
    enter(lane);
    const double t_head = phases.on ? BatchPhases::Now() : 0.0;
    RmsNormRows(lane.scratch.res, model_->output_norm().f32(), lane.scratch.xn,
                lane.n, c.hidden_size, 1, c.rms_eps, stream_);
    if (!DenseBatch(model_->output(), lane.scratch.xn,
                    batch_logits_ + static_cast<std::size_t>(lane.row_off) *
                                        c.vocab_size,
                    lane.n, error)) {
      return false;
    }
    if (phases.on) {
      phases.head += BatchPhases::Now() - t_head;
    }
  }
  UseScratch(base);
  if (phases.on) {
    phases.total += BatchPhases::Now() - phase_t0;
    phases.rows_max = std::max(phases.rows_max, rows);
    phases.Tick();
  }
  return true;
}

bool Executor::SelectBatchLogits(std::uint32_t offset, std::uint32_t rows,
                                 float* logits, std::string* error) const {
  if (rows == 0 || rows > options_.max_logit_rows || offset > batch_rows_ ||
      rows > batch_rows_ - offset) {
    return Fail(error, "logit rows outside the completed decode batch");
  }
  const std::size_t count =
      static_cast<std::size_t>(rows) * config().vocab_size;
  selected_logits_ =
      batch_logits_ + static_cast<std::size_t>(offset) * config().vocab_size;
  if (getenv("GUFO_GRAPH_DEBUG") != nullptr) {
    std::vector<float> head(count);
    hipMemcpyAsync(head.data(), selected_logits_, count * sizeof(float),
                   hipMemcpyDeviceToHost, stream_);
    hipStreamSynchronize(stream_);
    double sum = 0;
    for (float v : head) sum += v;
    fprintf(stderr, "LOGITS off=%u rows=%u first=%.6f sum=%.3f\n", offset,
            rows, head.empty() ? 0.0f : head[0], sum);
    fflush(stderr);
  }
  if (logits != nullptr) {
    const std::uint32_t base = offset - staged_logit_offset_;
    if (staged_logit_rows_ != 0 && offset >= staged_logit_offset_ &&
        base <= staged_logit_rows_ &&
        rows <= staged_logit_rows_ - base) {
      std::copy_n(logits_host_ + static_cast<std::size_t>(base) *
                                     config().vocab_size,
                  count, logits);
      return true;
    }
    if (!Check(hipMemcpyAsync(logits_host_, selected_logits_,
                              count * sizeof(float), hipMemcpyDeviceToHost,
                              stream_),
               error) ||
        !Check(hipStreamSynchronize(stream_), error)) {
      return false;
    }
    std::copy_n(logits_host_, count, logits);
  }
  return true;
}

bool Executor::StageBatchLogits(std::uint32_t offset, std::uint32_t rows,
                                std::string* error) const {
  if (rows == 0 || rows > options_.max_logit_rows || offset > batch_rows_ ||
      rows > batch_rows_ - offset) {
    return Fail(error, "logit rows outside the completed decode batch");
  }
  const std::size_t count =
      static_cast<std::size_t>(rows) * config().vocab_size;
  if (!Check(hipMemcpyAsync(logits_host_,
                            batch_logits_ + static_cast<std::size_t>(offset) *
                                                config().vocab_size,
                            count * sizeof(float), hipMemcpyDeviceToHost,
                            stream_),
             error) ||
      !Check(hipStreamSynchronize(stream_), error)) {
    return false;
  }
  staged_logit_offset_ = offset;
  staged_logit_rows_ = rows;
  return true;
}

}  // namespace gufo::models::mimo2::rocm
