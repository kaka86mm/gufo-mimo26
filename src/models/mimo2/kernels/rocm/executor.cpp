#include "src/models/mimo2/kernels/rocm/executor.hpp"
#include <chrono>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <vector>

#include "mimo2_mmq.h"
#include "src/core/hip/snapshot_transfer.hpp"

namespace gufo::models::mimo2::rocm {
static const bool kEntryTrace = [] {
  return getenv("GUFO_SYNC_TRACE") != nullptr;
}();

// Tap-capture bring-up tracing (host-side steps of an armed forward).
static const bool kTapTrace = [] {
  return getenv("GUFO_TAP_TRACE") != nullptr;
}();
#define TAPTRACE(...)                         \
  do {                                        \
    if (kTapTrace) {                          \
      fprintf(stderr, "[TAPTRACE] " __VA_ARGS__); \
      fputc('\n', stderr);                   \
      fflush(stderr);                         \
    }                                         \
  } while (0)

/// One-shot layer-0 dump for the numpy oracle comparison.
static bool kDumpOnce = [] { return getenv("GUFO_DUMP_DIR") != nullptr; }();
static void DumpVec(const char* name, const void* dev, std::size_t floats,
                    hipStream_t stream, bool half = false) {
  if (!kDumpOnce) {
    return;
  }
  hipStreamSynchronize(stream);
  const std::size_t n = half ? floats / 2 + floats % 2 : floats;
  std::vector<float> host(n);
  std::vector<__half> host_h(n);
  void* dst = half ? static_cast<void*>(host_h.data())
                   : static_cast<void*>(host.data());
  hipMemcpy(dst, dev, n * (half ? 2 : 4), hipMemcpyDeviceToHost);
  std::string path = std::string(getenv("GUFO_DUMP_DIR")) + "/" + name + ".txt";
  FILE* f = fopen(path.c_str(), "w");
  for (std::size_t i = 0; i < n; ++i) {
    fprintf(f, "%.6f\n", half ? __half2float(host_h[i]) : host[i]);
  }
  fclose(f);
}


// Coarse phase timing for perf triage; printed once at process exit.
struct PhaseTimers {
  double attn_ms{0}, ffn_ms{0}, reduce_ms{0}, mtp_ms{0};
  std::uint64_t attn_n{0}, ffn_n{0}, reduce_n{0}, mtp_n{0};
  ~PhaseTimers() {
    if (attn_n == 0 && ffn_n == 0) {
      return;
    }
    fprintf(stderr,
            "PHASES attn=%.1fms/%lu ffn=%.1fms/%lu reduce=%.1fms/%lu "
            "mtp=%.1fms/%lu\n",
            attn_ms, (unsigned long)attn_n, ffn_ms, (unsigned long)ffn_n,
            reduce_ms, (unsigned long)reduce_n, mtp_ms, (unsigned long)mtp_n);
    fflush(stderr);
  }
};
static PhaseTimers kPhases;
static const bool kPhaseTiming = [] {
  return getenv("GUFO_PHASE_TIMING") != nullptr;
}();
struct PhaseScope {
  double* bucket;
  std::uint64_t* count;
  std::chrono::steady_clock::time_point t0;
  PhaseScope(double* b, std::uint64_t* c)
      : bucket(b), count(c), t0(std::chrono::steady_clock::now()) {}
  ~PhaseScope() {
    if (kPhaseTiming) {
      *bucket += std::chrono::duration<double, std::milli>(
                     std::chrono::steady_clock::now() - t0)
                     .count();
      ++*count;
    }
  }
};

static const bool kLayerTrace = [] {
  return getenv("GUFO_LAYER_TRACE") != nullptr;
}();
/// Prints row-0 stats of a device float range (norm, first values).
static void DumpRow(const char* what, const float* p, std::uint32_t n,
                    hipStream_t stream) {
  if (!kLayerTrace) {
    return;
  }
  hipStreamSynchronize(stream);
  std::vector<float> host(std::min<std::uint32_t>(n, 4096));
  hipMemcpy(host.data(), p, host.size() * sizeof(float),
            hipMemcpyDeviceToHost);
  double sq = 0;
  for (float v : host) {
    sq += static_cast<double>(v) * v;
  }
  fprintf(stderr, "LAYERTRACE %s norm2=%.2f v0..3=%.3f %.3f %.3f %.3f\n",
          what, sq, host[0], host[1], host[2], host[3]);
  fflush(stderr);
}
namespace {

using core::GgmlType;

void AssignError(std::string* error_msg, const std::string& message) {
  if (error_msg != nullptr) {
    *error_msg = message;
  }
}

bool Check(hipError_t err, const char* what, std::string* error_msg) {
  if (err == hipSuccess) {
    return true;
  }
  AssignError(error_msg, std::string(what) + ": " + hipGetErrorString(err));
  return false;
}

/// Allocates `count` elements of T and records the allocation.
template<typename T>
T* Alloc(std::vector<void*>& allocations, std::size_t count,
         std::string* error_msg, std::size_t* allocated_bytes = nullptr) {
  void* p = nullptr;
  const std::size_t bytes = std::max<std::size_t>(1, count) * sizeof(T);
  if (hipMalloc(&p, bytes) != hipSuccess) {
    AssignError(error_msg,
                "hipMalloc of " + std::to_string(bytes) + " bytes failed");
    allocations.push_back(nullptr);
    return nullptr;
  }
  (void)hipMemset(p, 0, bytes);
  allocations.push_back(p);
  if (allocated_bytes)
    *allocated_bytes += bytes;
  return static_cast<T*>(p);
}

WeightType SmallType(GgmlType type) {
  switch (type) {
    case GgmlType::kBF16:
      return WeightType::kBF16;
    case GgmlType::kF16:
      return WeightType::kF16;
    case GgmlType::kQ8_0:
      return WeightType::kQ8_0;
    case GgmlType::kF32:
      return WeightType::kF32;
    default:
      throw std::logic_error("unsupported MiMo-V2 matrix format");
  }
}

WeightType ExpertType(GgmlType type) {
  switch (type) {
    case GgmlType::kQ2_K:
      return WeightType::kQ2_K;
    case GgmlType::kQ4_K:
      return WeightType::kQ4_K;
    case GgmlType::kQ5_K:
      return WeightType::kQ5_K;
    case GgmlType::kQ6_K:
      return WeightType::kQ6_K;
    case GgmlType::kQ5_1:
      return WeightType::kQ5_1;
    case GgmlType::kQ8_0:
      return WeightType::kQ8_0;
    case GgmlType::kMXFP4:
      return WeightType::kMXFP4;
    default:
      throw std::logic_error("unsupported MiMo-V2 expert format");
  }
}

// The tier's tiled kernels compute whole column tiles; below this width the
// matrix-vector kernels read each weight once per row and win outright.
constexpr std::uint32_t kVecBatch = 8;
/// Rows a single attention call can carry into the merge-partial buffer: the
/// wide split path now covers whole 16-row matrix tiles.
constexpr std::uint32_t kAttnRowsMax = 16;
// Keep prompt arithmetic independent of chunk width. A scoped, per-thread
// policy also lets the output head retain its existing logits-row arithmetic.
thread_local bool prefill_phase = false;
struct PrefillPhase {
  bool previous;
  explicit PrefillPhase(bool enabled) : previous(prefill_phase) {
    prefill_phase = enabled;
  }
  ~PrefillPhase() { prefill_phase = previous; }
};
// Mirrors kernels.hip.cpp's env-gated stream tracing; the executor's own
// unconditional syncs must not run on the capture path or the hot loop.
static const bool kExecSyncTrace = getenv("GUFO_SYNC_TRACE") != nullptr;
bool MatrixRows(std::uint32_t rows) {
  return prefill_phase || rows > kVecBatch;
}
bool ExpertMatrixRows(std::uint32_t rows) {
  return prefill_phase || rows > 4 * kVecBatch;
}

/// GUFO_MX_WMMA_ROWS moves decode batches with at least that many rows onto
/// the WMMA route so the row-count crossover can be measured (0 keeps decode
/// on the vendored mmq tier, as shipped). Read once: the route decision and
/// the expert-counts gate must agree.
std::uint32_t MxWmmaRows() {
  static const std::uint32_t rows = [] {
    const char* env = getenv("GUFO_MX_WMMA_ROWS");
    return env != nullptr
               ? static_cast<std::uint32_t>(std::strtoul(env, nullptr, 10))
               : 0U;
  }();
  return rows;
}

/// Whether this layer and width take the RoutedCompact (WMMA) expert tier —
/// the decode path's only reader of the per-expert row counts. The vendored
/// MXFP4 tier (MoeExpertsMxfp4) never reads them.
bool UsesRoutedCompact(const DeviceLayer& l, std::uint32_t n_tokens) {
  if (l.ffn_gate_exps.type == GgmlType::kQ3_K) {
    // The vendored tier carries Q3_K expert GEMMs (gate/up/down, fused and
    // tiled); the RoutedF16 WMMA kernel has no Q3_K branch, so requantized
    // experts stay on the mmq route at every width, prefill included.
    return false;
  }
  if (l.ffn_gate_exps.type != GgmlType::kMXFP4) {
    return true;  // K-quants always take WMMA.
  }
  const bool wmma_route =
      prefill_phase || (MxWmmaRows() != 0 && n_tokens >= MxWmmaRows());
  return wmma_route && getenv("GUFO_MX_OFF") == nullptr;
}

/// Key-tile splits per row of a narrow attention batch (decode at depth).
constexpr std::uint32_t kAttnSplits = 8;
/// Upper bound on the split count; sizes the per-row partial buffer.
/// Raised from 32 for the long-context experiment: at 100K+ the shared form
/// wants one split per eight keys, and the old cap left each block walking
/// thousands of keys serially (150K measured ~4.4ms per layer against
/// 0.83ms at 65K). The scratch this sizes is 16 rows x q-heads x cap x
/// (d_v + 2) floats: 34 MiB at 64.
constexpr std::uint32_t kAttnSplitsMax = 64;

/// Whether the decode attention reads the int8 K/V twin. The fp16 rows stay
/// authoritative (the prefill tiles read them), so this only changes the
/// decode side's traffic — which is what a deep context pays for (the block
/// size is kKvQuantBlock from kernels.hpp).
bool KvInt8Enabled() {
  static const bool enabled = getenv("GUFO_MIMO_KV_INT8") != nullptr;
  return enabled;
}
/// Visible-key count from which a narrow global-attention row splits its key
/// range over kAttnSplits blocks. One split walks the range serially in a
/// single block: at a 145K-token context that is ~2300 key tiles and the
/// measured decode step ran far off the KV read bound. Below the threshold
/// (and on sliding layers, whose window is 128 keys) the merge costs more
/// than the parallelism returns.
constexpr std::uint32_t kAttnSplitKeys = 2048;
/// Wide dense Q8_0 projections take the F16 WMMA GEMM (F16 activation
/// rows, weights dequantized as they are staged) when its 256-row tiles
/// fill the device and the K sweep is short enough that the F16 rows stay
/// cache-resident; thresholds carried over from the Flash-Next tuning.
constexpr std::uint32_t kDenseF16MinRows = 2048;
constexpr std::uint32_t kDenseF16MaxCols = 2560;

/// Token rows per routed F16 expert GEMM tile: the narrow tile for batches
/// whose buckets pad to one or two 16-row tiles, the wide one when the
/// mean bucket fills most of it (the weight dequantization is per tile).
constexpr std::uint32_t kRoutedTileRowsNarrow = 16;
constexpr std::uint32_t kRoutedTileRowsWide = 48;

std::uint32_t RoutedTileRows(std::size_t slots, std::uint32_t n_experts) {
  return slots >= static_cast<std::size_t>(16) * n_experts
             ? kRoutedTileRowsWide
             : kRoutedTileRowsNarrow;
}

/// Upper bound on launched routed tiles: every 16-padded bucket contributes
/// at most one partial tile beyond its rows.
std::size_t RoutedTileCapacity(std::size_t slots, std::uint32_t n_experts) {
  return (slots + static_cast<std::size_t>(n_experts) * 15) /
             kRoutedTileRowsNarrow +
         n_experts + 1;
}

/// Column-tile width for the vendored MXFP4 expert GEMMs: the tile at or
/// above twice the mean bucket, so most experts fit one tile.
int RoutedTileCols(std::uint32_t n_tokens, std::uint32_t n_used,
                   std::uint32_t n_experts) {
  const std::uint32_t mean =
      std::max<std::uint32_t>(1, n_tokens * n_used / std::max(n_experts, 1u));
  for (int cols = 16; cols < 80; cols += 16) {
    if (static_cast<std::uint32_t>(cols) >= 2 * mean) {
      return cols;
    }
  }
  return 80;
}

/// Repacks a Q8_0 projection tensor into the vendored dense_vec artifact
/// (per-row block scales front, 16B-aligned codes behind) so decode rows can
/// stream it with the dp4a GEMV instead of the tiled W8A8 GEMM. One device
/// copy in, one out, once per weight: the decode loop repeats these tensors
/// every step, so the repack amortizes immediately. Returns nullptr for
/// shapes the artifact cannot serve (or too large to be worth carrying).
void* AlignedQ8Artifact(std::unordered_map<const void*, void*>& cache,
                        const void* data, std::uint32_t rows,
                        std::uint32_t cols) {
  if (const auto it = cache.find(data); it != cache.end()) {
    return it->second;
  }
  if (data == nullptr || cols % 1024 != 0 || rows == 0 || rows > 16384) {
    return nullptr;
  }
  const std::uint64_t bytes =
      mimo2_mmq_q8_0_aligned_bytes(static_cast<int>(rows),
                                   static_cast<int>(cols));
  if (bytes == 0) {
    return nullptr;
  }
  void* device = nullptr;
  if (hipMalloc(&device, bytes) != hipSuccess) {
    return nullptr;
  }
  // block_q8_0: a 2-byte scale, then the 32 codes.
  const std::size_t blocks =
      static_cast<std::size_t>(rows) * (cols / 32);
  const std::size_t raw_bytes = blocks * 34;
  std::vector<std::uint8_t> host(raw_bytes);
  std::vector<std::uint8_t> artifact(bytes, 0);
  if (hipMemcpy(host.data(), data, raw_bytes, hipMemcpyDeviceToHost) !=
      hipSuccess) {
    (void)hipFree(device);
    return nullptr;
  }
  const std::uint64_t dq_bytes = (blocks * 2u + 63u) & ~std::uint64_t{63};
  std::uint8_t* dq = artifact.data();
  std::uint8_t* qs = artifact.data() + dq_bytes;
  for (std::size_t b = 0; b < blocks; ++b) {
    std::memcpy(dq + b * 2, host.data() + b * 34, 2);
    std::memcpy(qs + b * 32, host.data() + b * 34 + 2, 32);
  }
  if (hipMemcpy(device, artifact.data(), bytes, hipMemcpyHostToDevice) !=
      hipSuccess) {
    (void)hipFree(device);
    return nullptr;
  }
  // GUFO_ALIGNED_LOG: one line per artifact while the host copy is live. A
  // correct Q8_0 repack shows plausible per-block scales (finite, small);
  // a tensor whose uploaded layout is not plain row-major blocks shows
  // NaN/garbage scales or absurd magnitudes — the 10-03 poisoning hunt.
  if (getenv("GUFO_ALIGNED_LOG") != nullptr) {
    const __half* dq = reinterpret_cast<const __half*>(artifact.data());
    double dmin = 1e30, dmax = 0, dsum = 0;
    std::size_t nan_count = 0, zero_codes = 0;
    for (std::size_t b = 0; b < blocks; ++b) {
      const float d = __half2float(dq[b]);
      if (!std::isfinite(d)) {
        ++nan_count;
      } else {
        dmin = std::min(dmin, static_cast<double>(std::fabs(d)));
        dmax = std::max(dmax, static_cast<double>(std::fabs(d)));
        dsum += std::fabs(d);
      }
      for (int j = 0; j < 32; ++j) {
        zero_codes += qs[b * 32 + j] == 0 ? 1 : 0;
      }
    }
    fprintf(stderr,
            "ALIGNEDLOG data=%p rows=%u cols=%u blocks=%zu dq_abs[min=%.3e "
            "max=%.3e mean=%.3e] nonfinite=%zu zero_codes=%zu/%zu\n",
            data, rows, cols, static_cast<std::size_t>(blocks), dmin, dmax,
            dsum / static_cast<double>(blocks), nan_count, zero_codes,
            blocks * 32);
    fflush(stderr);
  }
  cache.emplace(data, device);
  return device;
}

}  // namespace

Session::~Session() {
  for (auto& [key, exec] : graphs_) {
    (void)hipGraphExecDestroy(exec);
  }
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

void Session::SetCancellationCheck(std::function<bool()> check) {
  cancelled_ = false;
  is_cancelled_ = std::move(check);
}

bool Session::CheckCancellation(std::string* error) const {
  try {
    cancelled_ = cancelled_ || (is_cancelled_ && is_cancelled_());
  } catch (...) {
    cancelled_ = true;
  }
  if (cancelled_) {
    AssignError(error, "generation cancelled");
    return false;
  }
  return true;
}

std::size_t Session::AllocatedBytes() const noexcept { return allocated_bytes_; }

void Session::Reset() {
  for (auto& [key, graph] : graphs_) {
    (void)hipGraphExecDestroy(graph);
  }
  graphs_.clear();
  warmed_.clear();
  position_ = 0;
  spec_tokens_ = 0;
  prompt_taps_.clear();
  verify_taps_.clear();
  tap_batch_offset_ = 0;
  for (auto& block : mtp_blocks_) {
    block.position = 0;
  }
}

bool Executor::SetTargetLayerCapture(Session& session,
                                     std::span<const std::uint32_t> layers,
                                     std::string* error_msg) const {
  const Config& c = config();
  if (layers.size() > kMaxTargetLayerTaps) {
    AssignError(error_msg, "too many target hidden-layer taps");
    return false;
  }
  for (std::size_t i = 0; i < layers.size(); ++i) {
    if (layers[i] >= c.num_layers) {
      AssignError(error_msg, "target hidden-layer tap is out of range");
      return false;
    }
    if (i > 0 && layers[i] <= layers[i - 1]) {
      AssignError(error_msg, "target hidden-layer taps must be increasing");
      return false;
    }
  }
  if (!layers.empty() && session.target_taps_ == nullptr) {
    session.target_taps_ =
        Alloc<float>(session.allocations_,
                     static_cast<std::size_t>(kMaxTargetLayerTaps) *
                         c.hidden_size,
                     error_msg, &session.allocated_bytes_);
    if (session.target_taps_ == nullptr ||
        !Check(hipDeviceSynchronize(), "tap allocation", error_msg)) {
      return false;
    }
  }
  session.tap_layers_.assign(layers.begin(), layers.end());
  session.tap_width_ = layers.size() * c.hidden_size;
  if (!layers.empty() && session.tap_device_ == nullptr) {
    // One verification batch at a time; slack for the largest spec window.
    session.tap_device_rows_ = std::max<std::size_t>(32, kVecBatch + 2);
    session.tap_device_ =
        Alloc<float>(session.allocations_,
                     session.tap_device_rows_ * session.tap_width_, error_msg,
                     &session.allocated_bytes_);
    if (session.tap_device_ == nullptr ||
        !Check(hipDeviceSynchronize(), "tap staging allocation", error_msg)) {
      return false;
    }
  }
  session.prompt_taps_.clear();
  session.verify_taps_.clear();
  session.tap_batch_offset_ = 0;
  session.tap_phase_ = Session::TapPhase::kNone;
  return true;
}

bool Executor::CaptureTaps(Session& session, std::uint32_t layer,
                           std::uint32_t n, std::string* error_msg) const {
  const auto it = std::lower_bound(session.tap_layers_.begin(),
                                   session.tap_layers_.end(), layer);
  if (it == session.tap_layers_.end() || *it != layer) {
    return true;
  }
  static const bool reversed_taps =
      getenv("GUFO_DFLASH_TAPS_REVERSED") != nullptr;
  const std::size_t slot =
      reversed_taps ? session.tap_layers_.size() - 1 -
                          static_cast<std::size_t>(it - session.tap_layers_.begin())
                    : static_cast<std::size_t>(it - session.tap_layers_.begin());
  const std::size_t tap = slot;
  const std::size_t hidden = config().hidden_size;
  const std::size_t width = session.tap_width_;
  const float* const rows = s_.res;
  const float* const frontier =
      rows + static_cast<std::size_t>(n - 1) * hidden;
  switch (session.tap_phase_) {
    case Session::TapPhase::kPrompt:
      return Check(hipMemcpy2DAsync(session.prompt_taps_.data() +
                                        session.tap_batch_offset_ +
                                        tap * hidden,
                                    width * sizeof(float), rows,
                                    hidden * sizeof(float), hidden * sizeof(float),
                                    n, hipMemcpyDeviceToHost, stream_),
                   "prompt tap readback", error_msg);
    case Session::TapPhase::kVerify:
      return Check(hipMemcpy2DAsync(session.tap_device_ + tap * hidden,
                                    width * sizeof(float), rows,
                                    hidden * sizeof(float), hidden * sizeof(float),
                                    n, hipMemcpyDeviceToDevice, stream_),
                   "verification tap staging", error_msg) &&
             Check(hipMemcpyAsync(session.target_taps_ + tap * hidden, frontier,
                                  hidden * sizeof(float),
                                  hipMemcpyDeviceToDevice, stream_),
                   "tap frontier copy", error_msg);
    case Session::TapPhase::kLastRow:
      return Check(hipMemcpyAsync(session.target_taps_ + tap * hidden, frontier,
                                  hidden * sizeof(float),
                                  hipMemcpyDeviceToDevice, stream_),
                   "tap frontier copy", error_msg);
    case Session::TapPhase::kNone:
      break;
  }
  return true;
}

std::span<const float> Executor::CopyLastHidden(Session& session) const {
  const std::size_t width = session.tap_width_;
  if (width == 0 || session.target_taps_ == nullptr) {
    return {};
  }
  session.last_taps_.resize(width);
  if (hipMemcpyAsync(session.last_taps_.data(), session.target_taps_,
                     width * sizeof(float), hipMemcpyDeviceToHost,
                     stream_) != hipSuccess ||
      hipStreamSynchronize(stream_) != hipSuccess) {
    fprintf(stderr, "tap frontier readback failed: %s\n",
            hipGetErrorString(hipGetLastError()));
    session.last_taps_.clear();
    return {};
  }
  return session.last_taps_;
}

Executor::~Executor() {
  (void)hipFree(batch_logits_);
  (void)hipHostFree(batch_candidates_host_);
  for (auto& [key, exec] : batch_graphs_) {
    (void)hipGraphExecDestroy(exec);
  }
  if (slots_.dev != nullptr) {
    (void)hipFree(slots_.dev);
  }
  if (slots_.row_slot_host != nullptr) {
    (void)hipHostFree(slots_.row_slot_host);
    slots_.row_slot_host = nullptr;
  }
  if (slots_.row_pos_host != nullptr) {
    (void)hipHostFree(slots_.row_pos_host);
    slots_.row_pos_host = nullptr;
  }
  if (slots_.row_slot_dev != nullptr) {
    (void)hipFree(slots_.row_slot_dev);
    slots_.row_slot_dev = nullptr;
  }
  if (slots_.row_pos_dev != nullptr) {
    (void)hipFree(slots_.row_pos_dev);
    slots_.row_pos_dev = nullptr;
  }
  if (slots_.host != nullptr) {
    (void)hipHostFree(slots_.host);
  }
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
  for (void* p :
       {static_cast<void*>(control_host_), static_cast<void*>(tokens_host_),
        static_cast<void*>(logits_host_), static_cast<void*>(mtp_token_host_),
        static_cast<void*>(counts_host_),
        static_cast<void*>(mtp_candidates_host_),
        static_cast<void*>(tiles_host_)}) {
    if (p != nullptr) {
      (void)hipHostFree(p);
    }
  }
  if (blas_ != nullptr) {
    (void)hipblasDestroy(blas_);
  }
  if (counts_ready_ != nullptr) {
    (void)hipEventDestroy(counts_ready_);
  }
  if (stream_ != nullptr) {
    (void)hipStreamDestroy(stream_);
  }
}

std::unique_ptr<Executor> Executor::Create(const DeviceModel& model,
                                           Options options,
                                           std::string* error_msg) {
  std::unique_ptr<Executor> e(new Executor());
  e->model_ = &model;
  e->options_ = options;
  if (model.tp_world_size() > 1 &&
      (!options.all_reduce || !options.reduce_status)) {
    AssignError(error_msg,
                "distributed MiMo-V2 execution requires an all-reduce "
                "communicator");
    return nullptr;
  }
  if (static_cast<bool>(options.split_reduce.start) !=
          static_cast<bool>(options.split_reduce.finish) ||
      (options.split_reduce.start && model.tp_world_size() == 1)) {
    AssignError(error_msg,
                "a split reduce needs both halves and distributed execution");
    return nullptr;
  }
  e->options_.max_batch = std::max<std::uint32_t>(1, options.max_batch);
  e->options_.max_logit_rows = std::clamp<std::uint32_t>(
      options.max_logit_rows, 1, e->options_.max_batch);
  e->options_.max_speculative = std::clamp<std::uint32_t>(
      options.max_speculative, 1, e->options_.max_logit_rows);
  if (hipSetDevice(e->options_.device_index) != hipSuccess) {
    AssignError(error_msg, "HIP device selection failed");
    return nullptr;
  }
  if (mimo2_mmq_init(e->options_.device_index) != 0) {
    AssignError(error_msg, "quantized GEMM tier initialization failed");
    return nullptr;
  }
  if (!Check(hipStreamCreate(&e->stream_), "hipStreamCreate", error_msg)) {
    return nullptr;
  }
  // The vendored tier's internal pool must allocate on this executor's
  // stream so graph capture sees stream-ordered allocations.
  mimo2_pool_set_stream(e->stream_);
  if (!Check(hipEventCreateWithFlags(&e->counts_ready_, hipEventDisableTiming),
             "expert counts event", error_msg)) {
    return nullptr;
  }
  if (hipblasCreate(&e->blas_) != HIPBLAS_STATUS_SUCCESS ||
      hipblasSetStream(e->blas_, e->stream_) != HIPBLAS_STATUS_SUCCESS) {
    AssignError(error_msg, "hipblasCreate failed");
    return nullptr;
  }
  const Config& c = model.config();
  const std::size_t T = e->options_.max_batch;
  const std::size_t hidden = c.hidden_size;
  const std::size_t slots = T * c.num_experts_used;
  const std::size_t chunk = e->PrefillChunk();
  e->value_scale_ = c.value_scale;
  e->blaslt_ = BlasLt::Create(e->stream_, error_msg);
  if (e->blaslt_ == nullptr) {
    return nullptr;
  }
  auto& a = e->allocations_;
  Scratch& s = e->s_;
  auto f32 = [&](std::size_t n) { return Alloc<float>(a, n, error_msg); };
  s.tokens = Alloc<std::int32_t>(a, chunk, error_msg);
  // The half staging buffer serves both the F16 matrix route and the
  // Wmma MoE's activation narrowing (hidden-wide rows). max_half_cols only
  // accumulates F16/BF16 tensor widths — zero on a fully quantized artifact
  // like MiMo — so the hidden width must be included explicitly or
  // NarrowActivations writes past the buffer (intermittent illegal access
  // that the K-quant models never saw, their attention weights being F16).
  s.x_half = Alloc<std::uint16_t>(
      a, T * std::max<std::size_t>(model.max_half_cols(), hidden), error_msg);
  s.x_q8t =
      Alloc<std::uint8_t>(a, Q8TiledBytes(T, model.max_q8_cols()), error_msg);
  s.res = f32(T * hidden);
  s.xn = f32(T * hidden);
  s.block_out = f32(T * hidden);
  // The split trunk layers hold their share of the heads and the draft
  // blocks all of them; size for the widest.
  const DeviceLayer* widest = nullptr;
  for (const auto& l : model.layers()) {
    if (widest == nullptr || l.q_heads > widest->q_heads) {
      widest = &l;
    }
  }
  std::uint32_t max_q_heads = widest != nullptr ? widest->q_heads : 0;
  for (const auto& block : model.nextn_layers()) {
    max_q_heads = std::max(max_q_heads, block.q_heads);
  }
  const std::size_t q_dim = std::size_t{max_q_heads} * c.head_dim_qk;
  const std::size_t v_dim = std::size_t{max_q_heads} * c.head_dim_v;
  std::size_t k_dim = 0;
  for (const auto& l : model.layers()) {
    k_dim = std::max<std::size_t>(k_dim, l.kv_heads);
  }
  for (const auto& block : model.nextn_layers()) {
    k_dim = std::max<std::size_t>(k_dim, block.kv_heads);
  }
  k_dim *= c.head_dim_qk;
  s.q = f32(T * q_dim);
  s.k = f32(T * k_dim);
  s.v = f32(T * (k_dim / c.head_dim_qk * c.head_dim_v));
  s.ctx = f32(T * v_dim);
  s.attn_partials = f32(static_cast<std::size_t>(kAttnRowsMax) * max_q_heads *
                        kAttnSplitsMax * (c.head_dim_v + 2));
  s.router = f32(T * c.num_experts);
  s.router_partials =
      f32(static_cast<std::size_t>(kRouterSplitMaxSlices) *
          kRouterSplitMaxTokens * c.num_experts);
  s.router_alt =
      f32(static_cast<std::size_t>(kRouterSplitMaxTokens) * c.num_experts);
  {
    std::size_t routed_layers = 0;
    for (const auto& layer : model.layers()) {
      routed_layers += !layer.dense_ffn && !layer.router.empty();
    }
    s.router_f16 =
        Alloc<std::uint16_t>(a, routed_layers * std::size_t{c.num_experts} *
                                    hidden, error_msg);
  }
  s.ids = Alloc<std::int32_t>(a, slots, error_msg);
  s.expert_counts = Alloc<std::uint32_t>(a, c.num_experts, error_msg);
  {
    const std::size_t compact = RoutedCompactRows(slots, c.num_experts);
    s.routed_bounds = Alloc<std::int32_t>(a, c.num_experts + 1, error_msg);
    s.routed_cursors = Alloc<std::int32_t>(a, c.num_experts, error_msg);
    s.rows_token = Alloc<std::int32_t>(a, compact, error_msg);
    s.rows_slot = Alloc<std::int32_t>(a, compact, error_msg);
    s.routed_tiles = Alloc<std::int32_t>(
        a, 3 * RoutedTileCapacity(slots, c.num_experts), error_msg);
  }
  s.weights = f32(slots);
  std::size_t expert_ff = c.expert_ff;
  for (const auto& layer : model.layers()) {
    if (!layer.dense_ffn && !layer.ffn_gate_exps.empty()) {
      // A TP rank holds its share of every expert's intermediate rows.
      expert_ff = layer.ffn_gate_exps.rows;
      break;
    }
  }
  s.gate_e = f32(slots * expert_ff);
  s.up_e = f32(slots * expert_ff);
  s.down_e = f32(slots * hidden);
  std::size_t dense_ff = c.dense_ff;
  for (const auto& block : model.nextn_layers()) {
    dense_ff = std::max<std::size_t>(dense_ff, block.ffn_gate.rows);
  }
  if (!model.layers().empty()) {
    dense_ff = std::max<std::size_t>(dense_ff,
                                     model.layers().front().ffn_gate.rows);
  }
  s.ff_act = f32(T * dense_ff);
  s.ff_up = f32(T * dense_ff);
  s.ff_half = Alloc<__half>(a, T * dense_ff, error_msg);
  s.logits =
      f32(static_cast<std::size_t>(e->options_.max_logit_rows) * c.vocab_size);
  if (chunk > T) {
    e->pair_res_ = f32(T * hidden);
    e->pair_block_out_ = f32(T * hidden);
    e->pair_control_ = Alloc<Session::Control>(a, 1, error_msg);
  }
  if (model.has_mtp()) {
    s.mtp_h = f32(T * hidden);
    s.mtp_eh = f32(T * 2 * hidden);
    s.mtp_argmax = Alloc<ArgmaxCandidate>(a, kArgmaxParts, error_msg);
    s.mtp_token = Alloc<std::int32_t>(a, 1, error_msg);
    const auto candidate_ids = MtpCandidateWorkspaceSize(c.vocab_size);
    s.mtp_ids = Alloc<std::uint32_t>(a, candidate_ids, error_msg);
    s.mtp_scratch_ids = Alloc<std::uint32_t>(a, candidate_ids, error_msg);
    // Final selection consumes intermediate IDs before writing its scores.
    // Pack those scores behind the 64 returned IDs for one host transfer.
    s.mtp_scores = reinterpret_cast<float*>(s.mtp_ids + kMtpCandidates);
    void* draft_token = nullptr;
    void* draft_candidates = nullptr;
    if (!Check(hipHostMalloc(&draft_token, sizeof(std::int32_t)),
               "pinned draft token", error_msg) ||
        !Check(hipHostMalloc(&draft_candidates, sizeof(MtpCandidateLogits)),
               "pinned draft candidates", error_msg)) {
      return nullptr;
    }
    e->mtp_token_host_ = static_cast<std::int32_t*>(draft_token);
    e->mtp_candidates_host_ =
        static_cast<MtpCandidateLogits*>(draft_candidates);
    std::construct_at(e->mtp_candidates_host_);
    e->mtp_candidates_host_->size =
        std::min<std::size_t>(c.vocab_size, kMtpCandidates);
  }
  {
    void* control = nullptr;
    void* tokens = nullptr;
    void* logits = nullptr;
    if (!Check(hipHostMalloc(&control, 2 * sizeof(Session::Control)),
               "pinned control buffer", error_msg) ||
        !Check(hipHostMalloc(&tokens, chunk * sizeof(std::int32_t)),
               "pinned token buffer", error_msg) ||
        !Check(hipHostMalloc(&logits, static_cast<std::size_t>(
                                          e->options_.max_logit_rows) *
                                          c.vocab_size * sizeof(float)),
               "pinned logits buffer", error_msg)) {
      return nullptr;
    }
    void* counts = nullptr;
    if (!Check(hipHostMalloc(&counts, c.num_experts * sizeof(std::uint32_t)),
               "pinned expert counts", error_msg)) {
      return nullptr;
    }
    e->counts_host_ = static_cast<std::uint32_t*>(counts);
    void* tiles = nullptr;
    if (!Check(
            hipHostMalloc(&tiles, 3 * RoutedTileCapacity(slots, c.num_experts) *
                                      sizeof(std::int32_t)),
            "pinned routed tile map", error_msg)) {
      return nullptr;
    }
    e->tiles_host_ = static_cast<std::int32_t*>(tiles);
    e->control_host_ = static_cast<Session::Control*>(control);
    e->tokens_host_ = static_cast<std::int32_t*>(tokens);
    e->logits_host_ = static_cast<float*>(logits);
    // The static narrow-batch tile map: every expert contributes at most one
    // 16-row tile (each token's top-8 experts are distinct, so a bucket
    // holds at most kVecBatch rows). Graph replays reuse these bytes.
    for (std::uint32_t expert = 0; expert < c.num_experts; ++expert) {
      e->tiles_host_[expert] = static_cast<std::int32_t>(expert);
    }
    if (!Check(hipMemcpyAsync(s.routed_tiles, e->tiles_host_,
                              c.num_experts * sizeof(std::int32_t),
                              hipMemcpyHostToDevice, e->stream_),
               "routed tile map upload", error_msg)) {
      return nullptr;
    }
  }
  for (void* p : a) {
    if (p == nullptr) {
      return nullptr;
    }
  }
  return e;
}

std::unique_ptr<Session> Executor::CreateSession(core::SessionMode mode,
                                                 std::uint32_t max_context,
                                                 std::string* error_msg) const {
  std::unique_ptr<Session> s(new Session());
  s->owner_ = this;
  s->mtp_enabled_ = mode == core::SessionMode::kSpeculative;
  if (s->mtp_enabled_ && !has_mtp()) {
    AssignError(error_msg, "MTP session requires a loaded predictor");
    return nullptr;
  }
  const Config& c = config();
  if (max_context == 0 || max_context > c.context_length) {
    AssignError(error_msg, "session context exceeds the model context");
    return nullptr;
  }
  s->max_context_ = max_context;
  s->attention_.resize(c.num_layers);
  auto& a = s->allocations_;
  // Cache rows per layer: full-history layers need one row per position;
  // sliding layers get a power-of-two ring the kernels address as
  // position & (rows - 1). The ring must exceed window + max batch: a
  // prefill batch writes all its rows before attention reads them, so a
  // ring of window rows would let later tokens clobber history that
  // earlier queries in the same batch still attend to.
  const auto layer_rows = [&](const DeviceLayer& l) -> std::uint32_t {
    if (!l.swa) {
      return max_context;
    }
    std::uint32_t rows = std::min<std::uint64_t>(
        static_cast<std::uint64_t>(max_context),
        static_cast<std::uint64_t>(c.sliding_window) +
            options_.max_batch);
    if (rows < 2) {
      AssignError(error_msg, "sliding-window ring degenerated");
      return 0;
    }
    rows -= 1;
    rows |= rows >> 1;
    rows |= rows >> 2;
    rows |= rows >> 4;
    rows |= rows >> 8;
    rows |= rows >> 16;
    return rows + 1;
  };
  // Every layer's caches hold this rank's KV heads (the draft blocks', all
  // of them) at the layer's own key and value widths.
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    const DeviceLayer& l = model_->layers()[il];
    const std::uint32_t rows = layer_rows(l);
    if (rows == 0) {
      return nullptr;
    }
    const std::size_t k_row =
        static_cast<std::size_t>(l.kv_heads) * c.head_dim_qk;
    const std::size_t v_row =
        static_cast<std::size_t>(l.kv_heads) * c.head_dim_v;
    s->attention_[il].k_cache =
        Alloc<__half>(a, static_cast<std::size_t>(rows) * k_row,
                      error_msg, &s->allocated_bytes_);
    s->attention_[il].v_cache =
        Alloc<__half>(a, static_cast<std::size_t>(rows) * v_row,
                      error_msg, &s->allocated_bytes_);
    if (KvInt8Enabled()) {
      // Decode reads the int8 twin; the fp16 rows stay for the prefill tiles.
      s->attention_[il].k_cache_q = Alloc<std::int8_t>(
          a, static_cast<std::size_t>(rows) * k_row, error_msg,
          &s->allocated_bytes_);
      s->attention_[il].k_scales = Alloc<__half>(
          a, static_cast<std::size_t>(rows) * (k_row / kKvQuantBlock),
          error_msg, &s->allocated_bytes_);
      s->attention_[il].v_cache_q = Alloc<std::int8_t>(
          a, static_cast<std::size_t>(rows) * v_row, error_msg,
          &s->allocated_bytes_);
      s->attention_[il].v_scales = Alloc<__half>(
          a, static_cast<std::size_t>(rows) * (v_row / kKvQuantBlock),
          error_msg, &s->allocated_bytes_);
    }
    s->attention_[il].swa_rows = l.swa ? rows : 0;
  }
  s->control_ = Alloc<Session::Control>(a, 1, error_msg, &s->allocated_bytes_);
  // Autoregressive sessions keep one dormant block: every forward stages
  // `mtp_().position` into the control block, so an empty vector would fault
  // on a plain session (the DFlash target runs one).
  if (s->mtp_blocks_.empty()) {
    s->mtp_blocks_.resize(1);
  }
  if (s->mtp_enabled_) {
    s->mtp_blocks_.resize(model_->nextn_layers().size());
    for (std::size_t j = 0; j < s->mtp_blocks_.size(); ++j) {
      const DeviceLayer& l = model_->nextn_layers()[j];
      const std::uint32_t rows = layer_rows(l);
      if (rows == 0) {
        return nullptr;
      }
      const std::size_t k_row =
          static_cast<std::size_t>(l.kv_heads) * c.head_dim_qk;
      const std::size_t v_row =
          static_cast<std::size_t>(l.kv_heads) * c.head_dim_v;
      auto& state = s->mtp_blocks_[j];
      state.k_cache =
          Alloc<__half>(a, static_cast<std::size_t>(rows) * k_row,
                        error_msg, &s->allocated_bytes_);
      state.v_cache =
          Alloc<__half>(a, static_cast<std::size_t>(rows) * v_row,
                        error_msg, &s->allocated_bytes_);
      state.swa_rows = l.swa ? rows : 0;
      state.h = Alloc<float>(a, c.hidden_size, error_msg, &s->allocated_bytes_);
    }
    s->target_hidden_ = Alloc<float>(
        a, static_cast<std::size_t>(options_.max_speculative) * c.hidden_size,
        error_msg, &s->allocated_bytes_);
  }
  for (void* p : a) {
    if (p == nullptr) {
      return nullptr;
    }
  }
  // The zero fills above run on the null stream; nothing may read the new
  // buffers until they have landed.
  if (!Check(hipDeviceSynchronize(), "session init", error_msg)) {
    return nullptr;
  }
  return s;
}

bool Executor::EnsureRollback(Session& session, std::uint32_t depth,
                              std::string* error_msg) const {
  // KV rows past a rewound position go stale until replayed over, so no
  // rollback scratch exists; only validate the request.
  if (depth >= options_.max_speculative) {
    AssignError(error_msg, "invalid rollback depth");
    return false;
  }
  return true;
}

std::size_t Executor::SessionBytes(
    core::SessionMode mode, std::uint32_t max_context,
    std::uint32_t rollback_depth) const noexcept {
  (void)rollback_depth;  // KV rewind needs no arena.
  const auto& c = config();
  const bool kv_int8 = KvInt8Enabled();
  // Must mirror the allocator (CreateSession): sliding layers hold a ring
  // of pow2ceil(window + max_batch) rows capped at the context.
  const auto layer_rows = [&](const DeviceLayer& l) -> std::size_t {
    if (!l.swa) {
      return max_context;
    }
    std::size_t rows = std::min<std::uint64_t>(
        max_context,
        static_cast<std::uint64_t>(c.sliding_window) + options_.max_batch);
    rows -= 1;
    rows |= rows >> 1;
    rows |= rows >> 2;
    rows |= rows >> 4;
    rows |= rows >> 8;
    rows |= rows >> 16;
    return rows + 1;
  };
  std::size_t bytes = sizeof(Session::Control);
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    const DeviceLayer& l = model_->layers()[il];
    // K and V together are kv_heads * (qk + v) halfs per row — the two caches
    // are allocated at exactly those widths in CreateSession. An earlier
    // version doubled this, which made the serve layer's capacity claim
    // reject configurations that fit (for example three 500K sessions whose
    // real KV is 37 GB against a ~50 GB budget).
    const std::size_t halves =
        static_cast<std::size_t>(layer_rows(l)) * l.kv_heads *
        (c.head_dim_qk + c.head_dim_v);
    bytes += halves * sizeof(__half);
    if (kv_int8) {
      // The twin costs one byte per value plus a scale every kKvQuantBlock.
      bytes += halves * sizeof(std::int8_t) +
               (halves / kKvQuantBlock) * sizeof(__half);
    }
  }
  if (mode == core::SessionMode::kSpeculative) {
    for (const auto& l : model_->nextn_layers()) {
      bytes += static_cast<std::size_t>(layer_rows(l)) * l.kv_heads *
               (c.head_dim_qk + c.head_dim_v) * sizeof(__half);
      bytes += c.hidden_size * sizeof(float);
    }
    bytes += std::size_t{options_.max_speculative} * c.hidden_size *
             sizeof(float);
  }
  return bytes;
}

std::size_t Executor::DeferredScratchBytes() const {
  const auto& c = config();
  std::size_t bytes = batch_logits_ == nullptr
                          ? std::size_t{8} *
                                std::min(8U, options_.max_logit_rows) *
                                c.vocab_size * sizeof(float)
                          : 0;
  return bytes;
}

void Executor::PrepareHalfInput(const float* x, std::uint32_t rows,
                                std::uint32_t cols) const {
  if (half_src_ == x && half_rows_ == rows && half_cols_ == cols &&
      !half_bf16_) {
    return;
  }
  NarrowActivations(x, s_.x_half, false, static_cast<std::size_t>(rows) * cols,
                    stream_);
  half_src_ = x;
  half_rows_ = rows;
  half_cols_ = cols;
  half_bf16_ = false;
}

bool Executor::DenseF16Route(const DeviceTensor& w,
                             std::uint32_t n_tokens) const {
  return w.type == GgmlType::kQ8_0 && MatrixRows(n_tokens) &&
         w.rows >= kDenseF16MinRows && w.cols <= kDenseF16MaxCols &&
         w.cols <= model_->max_half_cols();
}

bool Executor::Dense(const DeviceTensor& w, const float* x, float* out,
                     std::uint32_t n_tokens, std::string* error_msg) const {
  if (!MatrixRows(n_tokens)) {
    // Requantized attn projections (Q4_K/Q4_0/Q5_1): the vendored vec
    // entries are type-templated, so the Q8_0 narrow route below carries over
    // and the tokens read about half the weight bytes.
    if (n_tokens <= 8 && w.cols % 256 == 0) {
      int vec_rc = -1;
      switch (w.type) {
        case GgmlType::kQ4_K:
          vec_rc = mimo2_mmq_q4_K_dense_vec(w.data, x, out,
                                            static_cast<int>(w.rows),
                                            static_cast<int>(n_tokens),
                                            static_cast<int>(w.cols), stream_);
          break;
        case GgmlType::kQ4_0:
          vec_rc = mimo2_mmq_q4_0_dense_vec(w.data, x, out,
                                            static_cast<int>(w.rows),
                                            static_cast<int>(n_tokens),
                                            static_cast<int>(w.cols), stream_);
          break;
        case GgmlType::kQ5_1:
          vec_rc = mimo2_mmq_q5_1_dense_vec(w.data, x, out,
                                            static_cast<int>(w.rows),
                                            static_cast<int>(n_tokens),
                                            static_cast<int>(w.cols), stream_);
          break;
        default:
          break;
      }
      if (vec_rc == 0) {
        return true;
      }
    }
    // Decode-sized batches: the vendored vector kernel quantizes the
    // activation itself (it needs a 256-multiple K); everything else takes
    // the grouped small GEMM.
    // The W8A8 route is clean on narrow batches now: a nine-shape harness
    // (batch 1-96, m 358-2048) ran 200 non-blocking quantize+GEMM rounds
    // per shape with canary guards — no out-of-bounds writes, no run-to-run
    // nondeterminism, and outputs matching a quantization-semantics
    // reference — and the full cluster passed serial and concurrent decode
    // with MTP chains live. The corruption that forced the opt-in here was
    // the zero-sized staging allocation since fixed; GUFO_NO_W8A8 remains
    // as the escape hatch.
    static const bool use_w8a8 = getenv("GUFO_NO_W8A8") == nullptr;
    if (w.type == GgmlType::kQ8_0 && w.cols % 32 == 0 && use_w8a8) {
      // Decode rows CAN try the aligned dense_vec GEMV (activation quantized
      // internally, weight streamed once, warp per output row), but it is
      // opt-in: a heavy A/B on 10-03 (identical 6K-token load, long-form
      // probe) showed the process degenerating into token loops after the
      // verify widths exercised the NC kernels, while the same binary with
      // GUFO_NO_ALIGNED_VEC=1 stayed clean. The +4% single-stream is not
      // worth an undiagnosed corruption; root-cause it before enabling.
      static const bool use_aligned_vec =
          getenv("GUFO_USE_ALIGNED_VEC") != nullptr;
      if (use_aligned_vec && n_tokens <= 8) {
        if (void* aligned = AlignedQ8Artifact(aligned_q8_map_, w.data, w.rows,
                                              w.cols);
            aligned != nullptr &&
            mimo2_mmq_q8_0_aligned_dense_vec(aligned, x, out,
                                             static_cast<int>(w.rows),
                                             static_cast<int>(n_tokens),
                                             static_cast<int>(w.cols),
                                             stream_) == 0) {
          return true;
        }
      }
      // The vec entry is the vendor's narrow-batch path by contract
      // ("routed through mmvq for batch counts that favour the vec path,
      // n_tokens <= 8"): dp4a GEMVs, activation quantized internally (the
      // separate Q8 tiling pass disappears). Measured standalone on the
      // production shapes it beats both the generic mmq entry (which the
      // first wiring used by mistake: v 2.7x, o 1.65x, q 1.12x) and W8A8.
      // GUFO_NO_DENSE_VEC restores the W8A8 route.
      static const bool use_dense_vec = getenv("GUFO_NO_DENSE_VEC") == nullptr;
      if (use_dense_vec && n_tokens <= 8 && w.cols % 256 == 0) {
        if (mimo2_mmq_q8_0_dense_vec(w.data, x, out,
                                     static_cast<int>(w.rows),
                                     static_cast<int>(n_tokens),
                                     static_cast<int>(w.cols),
                                     stream_) == 0) {
          return true;
        }
      }
      // The template's own W8A8 route: int8-tiled activations against the
      // Q8_0 weights (the Flash-Next production path), far faster than the
      // scalar grouped GEMM for these narrow batches.
      const bool cached = q8t_src_ == x && q8t_rows_ == n_tokens &&
                          q8t_cols_ == w.cols;
      if (!cached) {
        QuantizeQ8Tiled(x, s_.x_q8t, n_tokens, w.cols, stream_);
        q8t_src_ = x;
        q8t_rows_ = n_tokens;
        q8t_cols_ = w.cols;
      }
      if (W8A8Gemm(w.data, s_.x_q8t, out, n_tokens, w.rows, w.cols,
                   stream_)) {
        return true;
      }
      AssignError(error_msg, "W8A8 GEMM failed (narrow)");
      return false;
    }
    SmallGemm(w.data, SmallType(w.type), x, out, n_tokens, w.rows, w.cols,
              stream_);
    return true;
  }
  if (w.type == GgmlType::kQ8_0) {
    // Wide batches: the F16 route for the shapes it wins (F16 rows in
    // s_.x_half, often left there by a producer), else activations
    // quantized per 32-wide block into the tiled layout and the int8 WMMA
    // GEMM. Both staging buffers hold max_batch rows; a wider call runs in
    // pieces.
    const bool f16 = DenseF16Route(w, n_tokens);
    const std::uint32_t piece = static_cast<std::uint32_t>(options_.max_batch);
    for (std::uint32_t r0 = 0; r0 < n_tokens; r0 += piece) {
      const std::uint32_t rows = std::min(piece, n_tokens - r0);
      const float* src = x + static_cast<std::size_t>(r0) * w.cols;
      if (f16) {
        PrepareHalfInput(src, rows, w.cols);
        if (!DenseF16Gemm(w.data, static_cast<const __half*>(s_.x_half),
                          out + static_cast<std::size_t>(r0) * w.rows, rows,
                          w.rows, w.cols, stream_)) {
          AssignError(error_msg, "dense F16 GEMM failed");
          return false;
        }
        continue;
      }
      if (!(q8t_src_ == src && q8t_rows_ == rows && q8t_cols_ == w.cols)) {
        QuantizeQ8Tiled(src, s_.x_q8t, rows, w.cols, stream_);
        q8t_src_ = src;
        q8t_rows_ = rows;
        q8t_cols_ = w.cols;
      }
      if (!W8A8Gemm(w.data, s_.x_q8t,
                    out + static_cast<std::size_t>(r0) * w.rows, rows, w.rows,
                    w.cols, stream_)) {
        AssignError(error_msg, "W8A8 GEMM failed");
        return false;
      }
    }
    return true;
  }
  if (w.type == GgmlType::kQ4_K || w.type == GgmlType::kQ4_0 ||
      w.type == GgmlType::kQ5_1) {
    // Prefill batches of a requantized dense projection. The mmq entry takes
    // F32 activations (it quantizes them itself) and writes the same
    // row-major [token][row] layout this path expects.
    int dense_rc = -1;
    switch (w.type) {
      case GgmlType::kQ4_K:
        dense_rc = mimo2_mmq_q4_K_dense(w.data, x, out,
                                        static_cast<int>(w.rows),
                                        static_cast<int>(n_tokens),
                                        static_cast<int>(w.cols), stream_);
        break;
      case GgmlType::kQ4_0:
        dense_rc = mimo2_mmq_q4_0_dense(w.data, x, out,
                                        static_cast<int>(w.rows),
                                        static_cast<int>(n_tokens),
                                        static_cast<int>(w.cols), stream_);
        break;
      case GgmlType::kQ5_1:
        dense_rc = mimo2_mmq_q5_1_dense(w.data, x, out,
                                        static_cast<int>(w.rows),
                                        static_cast<int>(n_tokens),
                                        static_cast<int>(w.cols), stream_);
        break;
      default:
        break;
    }
    if (dense_rc != 0) {
      AssignError(error_msg, "dense quantized mmq projection failed");
      return false;
    }
    return true;
  }
  // Wide batches of the unquantized projections (the router): F32 weights
  // go straight to hipBLAS; 16-bit weights meet narrowed activations.
  const int m = static_cast<int>(w.rows);
  const int k = static_cast<int>(w.cols);
  const int n = static_cast<int>(n_tokens);
  if (w.type == GgmlType::kF32) {
    // Same rocBLAS avoidance: the own SmallGemm covers every batch width.
    SmallGemm(w.data, WeightType::kF32, x, out, n_tokens,
              static_cast<std::uint32_t>(m), static_cast<std::uint32_t>(k),
              stream_);
    return true;
  }
  const bool bf16 = w.type == GgmlType::kBF16;
  const hipDataType type = bf16 ? HIP_R_16BF : HIP_R_16F;
  if (!(half_src_ == x && half_rows_ == n_tokens && half_cols_ == w.cols &&
        half_bf16_ == bf16)) {
    NarrowActivations(x, s_.x_half, bf16, static_cast<std::size_t>(n) * k,
                      stream_);
    half_src_ = x;
    half_rows_ = n_tokens;
    half_cols_ = w.cols;
    half_bf16_ = bf16;
  }
  return blaslt_->Gemm(w.data, s_.x_half, out, type, m, n, k, error_msg);
}

bool Executor::GatedDense(const DeviceTensor& gate, const DeviceTensor& up,
                          const float* x, float* out, std::uint32_t n_tokens,
                          const DeviceTensor* down,
                          std::string* error_msg) const {
  // Decode and verification share the same projection arithmetic. The gate
  // rows land in `out` and the up rows in s_.ff_up; SwiGLU is in place over
  // its first operand.
  if (!Dense(gate, x, out, n_tokens, error_msg) ||
      !Dense(up, x, s_.ff_up, n_tokens, error_msg)) {
    return false;
  }
  // A wide batch writes the down projection's staged input directly (the
  // F32 rows are read by nothing else): F16 rows for the F16 route, else
  // the tiled Q8 layout; the cache lets Dense skip its activation pass.
  ff_half_ready_ = false;
  if (down != nullptr && down->cols == up.rows && MatrixRows(n_tokens) &&
      n_tokens <= options_.max_batch) {
    if (DenseF16Route(*down, n_tokens)) {
      SwigluHalf(out, s_.ff_up, s_.ff_half,
                 static_cast<std::size_t>(n_tokens) * up.rows, stream_);
      ff_half_ready_ = true;
      return true;
    }
    if (down->type == GgmlType::kQ8_0 &&
        SwigluQ8Tiled(out, s_.ff_up, s_.x_q8t, n_tokens, up.rows, stream_)) {
      q8t_src_ = out;
      q8t_rows_ = n_tokens;
      q8t_cols_ = up.rows;
      return true;
    }
  }
  Swiglu(out, s_.ff_up, static_cast<std::size_t>(n_tokens) * up.rows, stream_);
  return true;
}

bool Executor::RouteHints(std::uint32_t n_tokens,
                          std::string* error_msg) const {
  // Every column tile past an expert's bucket still costs a dispatch and a
  // full shared-memory reservation, so the MXFP4 grid is cut to the real
  // largest bucket and the widest useful tile. Counts are downloaded before
  // the routed GEMMs; wait only for that download.
  const Config& c = config();
  routed_max_rows_ = 0;
  routed_64_tiles_ = 0;
  routed_pair_tiles_ = 0;
  routed_pair_offset_ = 0;
  routed_pair_rows_ = 64;
  routed_n_tiles_ = 0;
  if (!ExpertMatrixRows(n_tokens)) {
    return true;
  }
  if (!Check(hipEventSynchronize(counts_ready_), "expert counts", error_msg)) {
    return false;
  }
  // The F16 expert GEMM launches one block per (expert, row tile of its
  // 16-padded bucket): the map is built here and uploaded ahead of the
  // launches on the same stream.
  std::uint32_t max_rows = 0;
  std::uint32_t n_tiles = 0;
  routed_tile_rows_ = RoutedTileRows(
      static_cast<std::size_t>(n_tokens) * c.num_experts_used, c.num_experts);
  for (std::uint32_t e = 0; e < c.num_experts; ++e) {
    const std::uint32_t padded = (counts_host_[e] + 15u) / 16u * 16u;
    max_rows = std::max(max_rows, counts_host_[e]);
    for (std::uint32_t j = 0;
         j < (padded + routed_tile_rows_ - 1) / routed_tile_rows_; ++j) {
      tiles_host_[n_tiles++] = static_cast<std::int32_t>(e | (j << 16));
    }
  }
  routed_max_rows_ = std::max<std::uint32_t>(1, max_rows);
  routed_n_tiles_ = n_tiles;
  // Append the 64-token map for gate/up and eligible down projections.
  // Wide expert buckets also get a 128-token gate/up map: it amortizes
  // weight decoding, while short buckets retain the cheaper 64-token tile.
  if (n_tokens >= 1024 && routed_tile_rows_ == kRoutedTileRowsWide) {
    for (std::uint32_t e = 0; e < c.num_experts; ++e) {
      const std::uint32_t padded = (counts_host_[e] + 15u) / 16u * 16u;
      for (std::uint32_t j = 0; j < (padded + 63u) / 64u; ++j)
        tiles_host_[n_tiles++] = static_cast<std::int32_t>(e | (j << 16));
    }
    routed_64_tiles_ = n_tiles - routed_n_tiles_;
    routed_pair_offset_ = routed_n_tiles_;
    routed_pair_tiles_ = routed_64_tiles_;
    std::uint32_t tiles_128 = 0;
    for (std::uint32_t e = 0; e < c.num_experts; ++e) {
      tiles_128 += (counts_host_[e] + 127u) / 128u;
    }
    if (tiles_128 * 4 <= routed_64_tiles_ * 3) {
      routed_pair_rows_ = 128;
      routed_pair_offset_ = n_tiles;
      routed_pair_tiles_ = tiles_128;
      for (std::uint32_t e = 0; e < c.num_experts; ++e) {
        for (std::uint32_t j = 0; j < (counts_host_[e] + 127u) / 128u; ++j) {
          tiles_host_[n_tiles++] = static_cast<std::int32_t>(e | (j << 16));
        }
      }
    }
  }
  // The measured launch-geometry globals corrupt narrow buckets (1-3
  // routed rows: short prompts and draft batches go out of bounds under
  // the tuned tile geometry). Wide prefill chunks average >=16 rows per
  // expert and are the only shape the tuning ever exercised, so the tuned
  // values are gated on this pass's row count: RouteHints itself only
  // runs for matrix-shaped batches, and n_tokens is the whole chunk here.
  if (n_tokens >= 512) {
    mimo2_mmq_set_routed_max_expert_rows(static_cast<int>(routed_max_rows_));
    routed_tile_cols_ = mimo2_mmq_routed_tile_cols_for_counts(
        counts_host_, static_cast<int>(c.num_experts));
    mimo2_mmq_set_routed_tile_cols(routed_tile_cols_);
  } else {
    mimo2_mmq_set_routed_max_expert_rows(0);
    routed_tile_cols_ = 0;
    mimo2_mmq_set_routed_tile_cols(0);
  }
  return n_tiles == 0 || Check(hipMemcpyAsync(s_.routed_tiles, tiles_host_,
                                              n_tiles * sizeof(std::int32_t),
                                              hipMemcpyHostToDevice, stream_),
                               "routed tile map upload", error_msg);
}

bool Executor::MoeExpertsWmma(const DeviceLayer& l, const float* x, float* out,
                              std::uint32_t n_tokens,
                              std::string* error_msg) const {
  const Config& c = config();
  const std::uint32_t used = c.num_experts_used;
  // One host holds each expert's full intermediate width; a TP2 rank holds
  // its share of every expert.
  const std::uint32_t expert_ff = l.ffn_gate_exps.rows;
  RoutedCompact(s_.ids, s_.expert_counts, s_.routed_bounds, s_.routed_cursors,
                s_.rows_token, s_.rows_slot, n_tokens, used, c.num_experts,
                stream_);
  if (kExecSyncTrace) {
    hipStreamSynchronize(stream_);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) {
      AssignError(error_msg, std::string("TRACE: after RoutedCompact: ") +
                                 hipGetErrorString(e));
      return false;
    }
  }
  // The GEMMs read F16 token rows. The half_src_ pointer/shape cache is NOT
  // consulted here: the dense FFN's F16 route stages the same s_.xn pointer
  // with the same shape on the previous layer, and this layer's norm output
  // lives at that identical address — a "hit" would stage last layer's
  // activations. Re-stage unconditionally; the MoE runs once per layer.
  // The router stage (or a previous projection) may already hold exactly
  // these rows: the pointer/shape key matches only when the same layer
  // staged the same norm output, which is the one safe hit.
  if (half_src_ != x || half_rows_ != n_tokens ||
      half_cols_ != c.hidden_size || half_bf16_) {
    NarrowActivations(x, s_.x_half, false,
                      static_cast<std::size_t>(n_tokens) * c.hidden_size,
                      stream_);
    half_src_ = x;
    half_rows_ = n_tokens;
    half_cols_ = c.hidden_size;
    half_bf16_ = false;
  }
  // Invalidate the staging cache afterwards: Dense() keys it on the
  // (pointer, shape), and the FFN norm buffer is rewritten at that same
  // address every layer. Leaving a valid entry lets the next layer's
  // projections false-hit on these F16 rows and multiply the previous
  // layer's activations.
  half_src_ = nullptr;
  half_rows_ = 0;
  half_cols_ = 0;
  half_bf16_ = false;
  q8t_src_ = nullptr;
  q8t_rows_ = 0;
  q8t_cols_ = 0;
  const auto* x_half = static_cast<const __half*>(s_.x_half);
  if (kExecSyncTrace) {
    hipStreamSynchronize(stream_);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) {
      AssignError(error_msg, std::string("TRACE: after Narrow: ") +
                                 hipGetErrorString(e));
      return false;
    }
  }
  // Narrow decode batches keep the static one-tile-per-expert map; wide
  // batches use the measured map (RouteHints).
  const std::int32_t* tiles = s_.routed_tiles;
  std::uint32_t n_tiles = c.num_experts;
  std::uint32_t tile_rows = kRoutedTileRowsNarrow;
  if (ExpertMatrixRows(n_tokens)) {
    n_tiles = routed_n_tiles_;
    tile_rows = routed_tile_rows_;
  }
  auto* up_half = reinterpret_cast<__half*>(s_.up_e);
  const WeightType gate_type = ExpertType(l.ffn_gate_exps.type);
  const auto sync_trace = [&](const char* what) {
    if (getenv("GUFO_SYNC_TRACE") == nullptr) {
      return true;
    }
    hipStreamSynchronize(stream_);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) {
      AssignError(error_msg,
                  std::string("TRACE: after ") + what + ": " +
                      hipGetErrorString(e));
      return false;
    }
    if (getenv("GUFO_MX_DUMP") != nullptr &&
        std::string(what) == "gate GEMM") {
      const std::size_t n = static_cast<std::size_t>(n_tokens) * used *
                            expert_ff;
      std::vector<float> host(n);
      if (hipMemcpy(host.data(), s_.gate_e, n * 4, hipMemcpyDeviceToHost) ==
          hipSuccess) {
        std::size_t nans = 0;
        for (float v : host) if (!std::isfinite(v)) ++nans;
        fprintf(stderr, "MXNAN right-after-gate: %zu/%zu nonfinite\n", nans,
                n);
        fflush(stderr);
      }
    }
    return true;
  };
  if (getenv("GUFO_MX_DUMP") != nullptr) {
    static bool dumped = false;
    if (!dumped) {
      dumped = true;
      hipStreamSynchronize(stream_);
      system("mkdir -p /tmp/mxdump");
      // Bisection: a zeroed destination rules out stale-buffer reads.
      hipMemsetAsync(s_.gate_e, 0,
                     static_cast<std::size_t>(n_tokens) * used * expert_ff * 4,
                     stream_);
      hipMemsetAsync(s_.up_e, 0,
                     static_cast<std::size_t>(n_tokens) * used * expert_ff * 4,
                     stream_);
      auto save = [&](const void* p, std::size_t bytes, const char* name) {
        std::vector<char> host(bytes);
        if (hipMemcpy(host.data(), p, bytes, hipMemcpyDeviceToHost) ==
            hipSuccess) {
          std::string path = std::string("/tmp/mxdump/") + name;
          FILE* f = fopen(path.c_str(), "wb");
          if (f) {
            fwrite(host.data(), 1, bytes, f);
            fclose(f);
          }
          fprintf(stderr, "MXDUMP %s %zu bytes\n", path.c_str(), bytes);
        }
      };
      const std::size_t row_bytes =
          static_cast<std::size_t>(expert_ff) * (c.hidden_size / 32) * 17;
      save(l.ffn_gate_exps.data,
           row_bytes * static_cast<std::size_t>(c.num_experts), "w_all.bin");
      save(tiles, n_tiles * 4, "tiles.bin");
      save(x_half, static_cast<std::size_t>(n_tokens) * c.hidden_size * 2,
           "x_half.bin");
      save(s_.routed_bounds, (c.num_experts + 1) * 4, "pad_bounds.bin");
      const std::size_t compact =
          RoutedCompactRows(static_cast<std::size_t>(n_tokens) * used,
                            c.num_experts);
      save(s_.rows_token, compact * 4, "rows_token.bin");
      save(s_.rows_slot, compact * 4, "rows_slot.bin");
      save(s_.ids, static_cast<std::size_t>(n_tokens) * used * 4, "ids.bin");
      fprintf(stderr,
              "MXDUMP meta n_tokens=%u n_tiles=%u tile_rows=%u m=%u k=%u "
              "experts=%u used=%u compact=%zu\n",
              n_tokens, n_tiles, tile_rows, expert_ff, c.hidden_size,
              c.num_experts, used, compact);
      fflush(stderr);
    }
  }
  static const bool moe_trace = getenv("GUFO_FFN_TRACE") != nullptr;
  static std::uint32_t moe_trace_n = 0;
  const bool moe_do_trace =
      moe_trace && (++moe_trace_n <= 2 || moe_trace_n % 47 == 0);
  const auto mt0 = std::chrono::steady_clock::now();
  // Wide batches also run gate/up on the 64-token map: fewer tiles trim the
  // 16-row padding tail (measured 1.12x per GEMM at the 5-blocks-per-SM
  // residency the fragment barrier enabled, bit-identical outputs). When
  // the 128-token pair map qualifies, the fused gate/up form takes over:
  // one kernel reads the staged activations once, pairs the gate and up
  // partials through shared memory, and never writes the gate's F32 rows
  // (1.11x over the separate pair on the same shape).
  const bool wide_gate_up =
      routed_64_tiles_ != 0 && routed_tile_rows_ == kRoutedTileRowsWide;
  const bool pair_gate_up =
      routed_pair_tiles_ != 0 && gate_type == WeightType::kMXFP4;
  if (pair_gate_up) {
    if (!RoutedGatedF16Gemm(l.ffn_gate_exps.data, l.ffn_up_exps.data,
                            gate_type, x_half,
                            s_.routed_tiles + routed_pair_offset_,
                            routed_pair_tiles_, routed_pair_rows_,
                            s_.routed_bounds, s_.rows_token, s_.rows_slot,
                            up_half, expert_ff, c.hidden_size, stream_) ||
        !sync_trace("up GEMM")) {
      if (error_msg == nullptr || error_msg->empty()) {
        AssignError(error_msg, "routed pair F16 gate/up GEMM failed");
      } else {
        fprintf(stderr, "WMMA fail: %s\n", error_msg->c_str());
        fflush(stderr);
      }
      return false;
    }
  } else if (!RoutedF16Gemm(
                 l.ffn_gate_exps.data, gate_type, x_half,
                 wide_gate_up ? s_.routed_tiles + routed_n_tiles_ : tiles,
                 wide_gate_up ? routed_64_tiles_ : n_tiles,
                 wide_gate_up ? 64 : tile_rows, s_.routed_bounds,
                 s_.rows_token, s_.rows_slot, nullptr, s_.gate_e, nullptr,
                 expert_ff, c.hidden_size, stream_) ||
             !sync_trace("gate GEMM") ||
             !RoutedF16Gemm(
                 l.ffn_up_exps.data, gate_type, x_half,
                 wide_gate_up ? s_.routed_tiles + routed_n_tiles_ : tiles,
                 wide_gate_up ? routed_64_tiles_ : n_tiles,
                 wide_gate_up ? 64 : tile_rows, s_.routed_bounds,
                 s_.rows_token, s_.rows_slot, s_.gate_e, nullptr, up_half,
                 expert_ff, c.hidden_size, stream_) ||
             !sync_trace("up GEMM")) {
    if (error_msg == nullptr || error_msg->empty()) {
      AssignError(error_msg, "routed F16 gate/up GEMM failed");
    } else {
      fprintf(stderr, "WMMA fail: %s\n", error_msg->c_str());
      fflush(stderr);
    }
    return false;
  }
  const auto mt1 = std::chrono::steady_clock::now();
  const WeightType down_type = ExpertType(l.ffn_down_exps.type);
  // Rows of down_e outside the tile map hold stale staging values; the
  // descale pass would blow them past F16 and the weighted sum would turn
  // them into NaN (0 * inf). Zero the buffer for the same reason the mmq
  // path memsets its outputs.
  (void)hipMemsetAsync(
      s_.down_e, 0,
      static_cast<std::size_t>(n_tokens) * used * c.hidden_size * 2, stream_);
  // Larger down tiles amortize weight decoding. Reuse the 64-token map
  // when it has no more padded rows than the 48-token map.
  const bool wide_down =
      routed_64_tiles_ != 0 && routed_tile_rows_ == kRoutedTileRowsWide &&
      routed_64_tiles_ * 4 <= routed_n_tiles_ * 3;
  if (!RoutedF16Gemm(l.ffn_down_exps.data, down_type, up_half,
                     s_.routed_tiles + (wide_down ? routed_n_tiles_ : 0),
                     wide_down ? routed_64_tiles_ : n_tiles,
                     wide_down ? 64 : tile_rows, s_.routed_bounds,
                     s_.rows_slot, s_.rows_slot, nullptr, nullptr,
                     reinterpret_cast<__half*>(s_.down_e), c.hidden_size,
                     expert_ff, stream_) ||
      !sync_trace("down GEMM")) {
    AssignError(error_msg, "routed F16 down GEMM failed");
    return false;
  }
  // Undo the up pass's F16 overflow guard before the weighted sum: one 2^-4
  // park (up pass only) meets one 2^4 here and the chain restores exactly.
  MxUpscaleF16(reinterpret_cast<__half*>(s_.down_e),
               static_cast<std::size_t>(n_tokens) * used * c.hidden_size,
               stream_);
  MoeEpilogueVec4F16(reinterpret_cast<const __half*>(s_.down_e), s_.weights,
                     out, n_tokens, used, c.hidden_size, stream_);
  if (!sync_trace("epilogue")) {
    return false;
  }
  if (moe_do_trace) {
    const auto mt2 = std::chrono::steady_clock::now();
    using ms_t = std::chrono::duration<double, std::milli>;
    fprintf(stderr, "MOETRACE L%u gateup_enq=%.2f down_enq=%.2f\n",
            moe_trace_n, ms_t(mt1 - mt0).count(), ms_t(mt2 - mt1).count());
    fflush(stderr);
  }
  if (getenv("GUFO_MX_DUMP") != nullptr) {
    // One-shot chain dump after the epilogue (the proven-safe spot): the
    // real up/down weights, router weights, and the restored down rows, so
    // the offline replay can rebuild and diff every remaining stage.
    static bool dumped_chain = false;
    if (!dumped_chain) {
      dumped_chain = true;
      hipStreamSynchronize(stream_);
      auto save_chain = [&](const void* p, std::size_t bytes,
                            const char* name) {
        std::vector<char> host(bytes);
        if (hipMemcpy(host.data(), p, bytes, hipMemcpyDeviceToHost) !=
            hipSuccess) {
          fprintf(stderr, "MXDUMP %s copy FAILED\n", name);
          return;
        }
        std::string path = std::string("/tmp/mxdump/") + name;
        FILE* f = fopen(path.c_str(), "wb");
        if (f) {
          fwrite(host.data(), 1, bytes, f);
          fclose(f);
          fprintf(stderr, "MXDUMP %s %zu bytes\n", path.c_str(), bytes);
        }
      };
      const std::size_t gate_row_bytes =
          static_cast<std::size_t>(expert_ff) * (c.hidden_size / 32) * 17;
      save_chain(l.ffn_up_exps.data, gate_row_bytes * c.num_experts,
                 "w_up.bin");
      const std::size_t down_row_bytes =
          static_cast<std::size_t>(c.hidden_size) * (expert_ff / 32) * 17;
      save_chain(l.ffn_down_exps.data, down_row_bytes * c.num_experts,
                 "w_down.bin");
      save_chain(s_.weights, static_cast<std::size_t>(n_tokens) * used * 4,
                 "weights.bin");
      save_chain(s_.down_e,
                 static_cast<std::size_t>(n_tokens) * used * c.hidden_size * 2,
                 "down_e.bin");
      fflush(stderr);
    }
  }
  if (getenv("GUFO_MX_DUMP") != nullptr) {
    hipStreamSynchronize(stream_);
    auto nan_scan = [&](const char* what, const void* p_f32,
                        std::size_t floats) {
      std::vector<float> host(floats);
      if (hipMemcpy(host.data(), p_f32, floats * 4, hipMemcpyDeviceToHost) !=
          hipSuccess) {
        fprintf(stderr, "MXNAN %s: copy failed\n", what);
        return;
      }
      std::size_t nans = 0, first = SIZE_MAX;
      for (std::size_t i = 0; i < floats; ++i) {
        if (!std::isfinite(host[i])) {
          ++nans;
          if (first == SIZE_MAX) first = i;
        }
      }
      fprintf(stderr, "MXNAN %s: %zu/%zu nonfinite, first@%zu=%f\n", what,
              nans, floats, first,
              first == SIZE_MAX ? 0.0f : host[first]);
    };
    const std::size_t slots_now = static_cast<std::size_t>(n_tokens) * used;
    nan_scan("x_f32_in", x, static_cast<std::size_t>(n_tokens) * c.hidden_size);
    nan_scan("gate_e", s_.gate_e, slots_now * expert_ff);
    nan_scan("down_out(out)", out,
             static_cast<std::size_t>(n_tokens) * c.hidden_size);
    // CPU reference over the dumped gate/up/down for slot 0 of token 0:
    // dump the F16 up buffer and down buffer heads for offline analysis.
    {
      std::vector<__half> up_h(std::min<std::size_t>(slots_now * expert_ff, 8192));
      hipMemcpy(up_h.data(), s_.up_e, up_h.size() * 2, hipMemcpyDeviceToHost);
      std::vector<__half> down_h(std::min<std::size_t>(slots_now * c.hidden_size, 4096));
      hipMemcpy(down_h.data(), s_.down_e, down_h.size() * 2, hipMemcpyDeviceToHost);
      std::vector<float> w_h(std::min<std::size_t>(slots_now, 8));
      hipMemcpy(w_h.data(), s_.weights, w_h.size() * 4, hipMemcpyDeviceToHost);
      fprintf(stderr, "MXVAL up0[:6]=");
      for (int i = 0; i < 6; ++i) fprintf(stderr, "%.4f ", __half2float(up_h[i]));
      fprintf(stderr, " down0[:6]=");
      for (int i = 0; i < 6; ++i) fprintf(stderr, "%.4f ", __half2float(down_h[i]));
      fprintf(stderr, " w[:8]=");
      for (float v : w_h) fprintf(stderr, "%.4f ", v);
      fprintf(stderr, "\n");
      fflush(stderr);
      {
        static bool out_dumped = false;
        if (!out_dumped) {
          out_dumped = true;
          FILE* f = fopen("/tmp/mxdump/moe_out.bin", "wb");
          if (f) {
            std::vector<float> o(static_cast<std::size_t>(n_tokens) *
                                c.hidden_size);
            if (hipMemcpy(o.data(), out, o.size() * 4,
                          hipMemcpyDeviceToHost) == hipSuccess) {
              fwrite(o.data(), 4, o.size(), f);
            }
            fclose(f);
          }
          f = fopen("/tmp/mxdump/x_half_full.bin", "wb");
          if (f) {
            fwrite(up_h.data(), 2, 0, f);  // placeholder no-op
            fclose(f);
          }
        }
      }
    }
    fflush(stderr);
  }
  return true;
}

namespace {

/// The five vendored MoE entry points a quantized expert tier needs: the
/// fused decode pair (gate+up+SwiGLU and down+slot-sum), the single-token
/// vector form, the paired gate/up form for wider batches, and the tiled
/// form. MXFP4 and Q3_K expose the same signatures over the same impls.
struct MmqMoeEntries {
  int (*gate_up_mid8)(const void*, const void*, const float*, const int32_t*,
                      const float*, float*, int, int, int, int, int, float,
                      hipStream_t);
  int (*down_sum8)(const void*, const float*, const int32_t*, float*, int, int,
                   int, int, int, hipStream_t);
  int (*vec)(const void*, const float*, const int32_t*, float*, int, int, int,
             int, int, hipStream_t);
  int (*pair)(const void*, const void*, const float*, const int32_t*, float*,
              float*, int, int, int, int, int, hipStream_t);
  int (*tiled)(const void*, const float*, const int32_t*, float*, int, int, int,
               int, int, hipStream_t);
};

constexpr MmqMoeEntries kMxfp4MoeEntries{
    mimo2_mmq_mxfp4_moe_gate_up_mid8_vec, mimo2_mmq_mxfp4_moe_down_sum8_vec,
    mimo2_mmq_mxfp4_moe_vec, mimo2_mmq_mxfp4_moe_pair, mimo2_mmq_mxfp4_moe};
constexpr MmqMoeEntries kQ3KMoeEntries{
    mimo2_mmq_q3_K_moe_gate_up_mid8_vec, mimo2_mmq_q3_K_moe_down_sum8_vec,
    mimo2_mmq_q3_K_moe_vec, mimo2_mmq_q3_K_moe_pair, mimo2_mmq_q3_K_moe};

}  // namespace

bool Executor::MoeExpertsMxfp4(const DeviceLayer& l, const float* x, float* out,
                               std::uint32_t n_tokens,
                               std::string* error_msg) const {
  // The vendored tier's MoE entries compute out[(t*used + s)*M + row] from
  // F32 token rows and an id map — the same layout MoeEpilogue consumes.
  // Its kernels sweep K in whole 256-element iterations.
  const Config& c = config();
  const std::uint32_t used = c.num_experts_used;
  const std::uint32_t expert_ff = l.ffn_gate_exps.rows;
  const auto fail = [&](const char* what) {
    AssignError(error_msg, std::string("MXFP4 expert ") + what + " failed");
    return false;
  };
  // Requantized experts (Q3_K sidecar) ride the same tier through the same
  // impls; the type is the only difference.
  const MmqMoeEntries& entries = l.ffn_gate_exps.type == GgmlType::kQ3_K
                                     ? kQ3KMoeEntries
                                     : kMxfp4MoeEntries;
  if (c.hidden_size % 256 != 0 || expert_ff % 256 != 0) {
    AssignError(error_msg,
                "MXFP4 experts need K a multiple of 256 (hidden and the "
                "rank's expert-ff share)");
    return false;
  }
  const std::uint32_t slots = n_tokens * used;
  // GUFO_MOE_UNFUSED: take the three-pass vec tier for the top-8 decode
  // shape too, to A/B the fused tier's launch/bandwidth split.
  static const bool moe_unfused = getenv("GUFO_MOE_UNFUSED") != nullptr;
  // Row bound for the fused top-8 tier (decode widths). The default covers a
  // six-session cohort with up to three rows each; measured at six rows the
  // fused tier's expert reads run ~150-170 GB/s against the pair form's 109,
  // which lifted the six-stream aggregate from 24.3 to 35.5 t/s
  // (2026-10-04). Prefill shapes never come here: prefill_phase routes to the
  // WMMA kernels first. GUFO_MOE_FUSED_ROWS sweeps the bound without a
  // rebuild.
  static const std::uint32_t fused_rows = [] {
    const char* env = getenv("GUFO_MOE_FUSED_ROWS");
    return env != nullptr ? static_cast<std::uint32_t>(std::atoi(env)) : 24U;
  }();
  if (n_tokens <= fused_rows && used == 8 && !moe_unfused) {
    // Decode batches through the top-8 fused tier: one gate+up+SwiGLU
    // launch with the router weight folded into the mid rows (per-32-block
    // absmax quantization is scale-free, so pre-weighting costs nothing),
    // then one down+slot-sum launch straight into the block output. Three
    // device operations replace the eleven of the three-pass form (two
    // duplicate activation quantizations, three output memsets, a separate
    // SwiGLU and a slot-sum epilogue). The kernels index (token, slot)
    // pairs, and the folding is per slot, so every decode width up to the
    // speculative batch takes this tier; at six rows the expert reads then
    // run ~170 GB/s against the pair form's ~109 (2026-10-04 measurement).
    static const bool vec_trace = getenv("GUFO_FFN_TRACE") != nullptr;
    static std::uint32_t vec_trace_n = 0;
    const bool vec_do_trace =
        vec_trace && (++vec_trace_n <= 2 || vec_trace_n % 47 == 0);
    const auto mv0 = std::chrono::steady_clock::now();
    if (entries.gate_up_mid8(
            l.ffn_gate_exps.data, l.ffn_up_exps.data, x, s_.ids, s_.weights,
            s_.gate_e, static_cast<int>(expert_ff),
            static_cast<int>(c.hidden_size), static_cast<int>(n_tokens),
            static_cast<int>(c.num_experts),
            static_cast<int>(used), 0.0F, stream_) != 0) {
      return fail("fused gate/up vector GEMM");
    }
    const auto mv1 = std::chrono::steady_clock::now();
    if (entries.down_sum8(
            l.ffn_down_exps.data, s_.gate_e, s_.ids, out,
            static_cast<int>(c.hidden_size), static_cast<int>(expert_ff),
            static_cast<int>(n_tokens),
            static_cast<int>(c.num_experts), static_cast<int>(used),
            stream_) != 0) {
      return fail("fused down vector GEMM");
    }
    if (vec_do_trace) {
      const auto mv2 = std::chrono::steady_clock::now();
      using ms_t = std::chrono::duration<double, std::milli>;
      fprintf(stderr, "MOEVEC n=%u rows=%u gateup=%.3f down=%.3f\n",
              vec_trace_n, n_tokens, ms_t(mv1 - mv0).count(),
              ms_t(mv2 - mv1).count());
      fflush(stderr);
    }
    return true;
  }
  if (n_tokens == 1) {
    // Fallback single-token decode (non top-8 routing). The fused pair_vec
    // launch (silu(gate)*up in one go) leaves a ~25% nondeterministic
    // corruption even with the W8A8 out-of-bounds route disabled; the
    // three-pass form is the stable one.
    if (entries.vec(l.ffn_gate_exps.data, x, s_.ids, s_.gate_e,
                                static_cast<int>(expert_ff),
                                static_cast<int>(c.hidden_size), 1,
                                static_cast<int>(c.num_experts),
                                static_cast<int>(used), stream_) != 0 ||
        entries.vec(l.ffn_up_exps.data, x, s_.ids, s_.up_e,
                                static_cast<int>(expert_ff),
                                static_cast<int>(c.hidden_size), 1,
                                static_cast<int>(c.num_experts),
                                static_cast<int>(used), stream_) != 0) {
      return fail("gate/up vector GEMM");
    }
    Swiglu(s_.gate_e, s_.up_e, static_cast<std::size_t>(slots) * expert_ff,
           stream_);
    if (entries.vec(
            l.ffn_down_exps.data, s_.gate_e, s_.ids, s_.down_e,
            static_cast<int>(c.hidden_size), static_cast<int>(expert_ff),
            static_cast<int>(slots), static_cast<int>(c.num_experts), 1,
            stream_) != 0) {
      return fail("down vector GEMM");
    }
    MoeEpilogueVec4(s_.down_e, s_.weights, out, n_tokens, used, c.hidden_size,
                    stream_);
    return true;
  }
  if (entries.pair(
          l.ffn_gate_exps.data, l.ffn_up_exps.data, x, s_.ids, s_.gate_e,
          s_.up_e, static_cast<int>(expert_ff), static_cast<int>(c.hidden_size),
          static_cast<int>(n_tokens), static_cast<int>(c.num_experts),
          static_cast<int>(used), stream_) != 0) {
    return fail("gate/up GEMM");
  }
  Swiglu(s_.gate_e, s_.up_e, static_cast<std::size_t>(slots) * expert_ff,
         stream_);
  if (entries.tiled(
          l.ffn_down_exps.data, s_.gate_e, s_.ids, s_.down_e,
          static_cast<int>(c.hidden_size), static_cast<int>(expert_ff),
          static_cast<int>(slots), static_cast<int>(c.num_experts), 1,
          stream_) != 0) {
    return fail("down GEMM");
  }
  MoeEpilogueVec4(s_.down_e, s_.weights, out, n_tokens, used, c.hidden_size,
                  stream_);
  return true;
}

bool Executor::MoeExperts(const DeviceLayer& l, const float* x, float* out,
                          std::uint32_t n_tokens,
                          std::string* error_msg) const {
  // MXFP4 expert weights: single-token decode and verification stay on the
  // vendored mmvq tier (DP4A vec-dots, near peak bandwidth at M=1); true
  // prefill batches take the RoutedF16Gemm WMMA route, whose MXFP4 branch
  // now matches the vendored decode bit-for-bit in semantics (Q4_0 nibble
  // order — elements 0-15 are the low nibbles of the 16 code bytes — the
  // E2M1 table through a half-sized LUT, and the 2^-4 overflow park on the
  // up pass only) and measured ~490 t/s prefill against the mmq tier's
  // ~333 on the same cluster. GUFO_MX_OFF forces the mmq tier back on.
  // K-quants always take WMMA.
  // Decode batches keep the mmq tier: on the one-to-three-row buckets of a
  // decode step the WMMA tile's 128-row blocks run at ~1% row occupancy and
  // measure 7x slower (1.9 vs 13 t/s single-stream; numerics stayed green).
  //
  // GUFO_MX_WMMA_ROWS moves decode batches with at least that many rows onto
  // the WMMA route, so the row-count crossover can be measured instead of
  // assumed. Off by default: 0 leaves decode on the mmq tier as shipped.
  // Route decision shared with UsesRoutedCompact, which the expert-counts
  // gate consults: the two must never disagree.
  if (!UsesRoutedCompact(l, n_tokens)) {
    return MoeExpertsMxfp4(l, x, out, n_tokens, error_msg);
  }
  return MoeExpertsWmma(l, x, out, n_tokens, error_msg);
}

bool Executor::FeedForward(const DeviceLayer& l, const float* x, float* out,
                           std::uint32_t n_tokens, std::string* error_msg,
                           bool reduce) const {
  const PhaseScope _ps(&kPhases.ffn_ms, &kPhases.ffn_n);
  const Config& c = config();
  if (l.dense_ffn) {
    if (!GatedDense(l.ffn_gate, l.ffn_up, x, s_.ff_act, n_tokens, &l.ffn_down,
                    error_msg)) {
      return false;
    }
    if (ff_half_ready_) {
      ff_half_ready_ = false;
      if (!DenseF16Gemm(l.ffn_down.data, s_.ff_half, out, n_tokens,
                        l.ffn_down.rows, l.ffn_down.cols, stream_)) {
        AssignError(error_msg, "dense FFN F16 GEMM failed");
        return false;
      }
    } else if (!Dense(l.ffn_down, s_.ff_act, out, n_tokens, error_msg)) {
      return false;
    }
    // An unsplit replicated FFN belongs to rank zero; peers still ran the
    // projections (their staging side effects keep the ranks in step) and
    // contribute a zero partial to the boundary sum.
    if (model_->tp_world_size() > 1 && !l.ffn_split &&
        model_->tp_rank() != 0 &&
        !Check(hipMemsetAsync(
                   out, 0,
                   static_cast<std::size_t>(n_tokens) * c.hidden_size *
                       sizeof(float),
                   stream_),
               "non-owner dense FFN clear", error_msg)) {
      return false;
    }
    return !reduce || AllReduce(out, n_tokens, error_msg);
  }
  if (options_.moe_observer) {
    options_.moe_observer(
        x, static_cast<std::size_t>(n_tokens) * c.hidden_size * sizeof(float),
        stream_);
  }
  static const bool ffn_trace = getenv("GUFO_FFN_TRACE") != nullptr;
  static std::uint32_t trace_layer = 0;
  const auto tt0 = std::chrono::steady_clock::now();
  const __half* router_f16 = nullptr;
  std::call_once(router_f16_once_, [&]() {
    const auto& layers = model_->layers();
    std::size_t off_elems = 0;
    for (const auto& layer : layers) {
      if (layer.dense_ffn || layer.router.empty()) {
        continue;
      }
      __half* dst = reinterpret_cast<__half*>(s_.router_f16) + off_elems;
      NarrowActivations(layer.router.f32(), dst, false,
                        static_cast<std::size_t>(c.num_experts) *
                            c.hidden_size,
                        stream_);
      router_f16_map_[layer.router.data] = dst;
      off_elems += static_cast<std::size_t>(c.num_experts) * c.hidden_size;
    }
  });
  const auto rf_it = router_f16_map_.find(l.router.data);
  router_f16 = rf_it != router_f16_map_.end() ? rf_it->second : nullptr;
  if (router_f16 != nullptr) {
    // The staged rows are exactly what the expert GEMMs consume below;
    // keying the half cache here lets MoeExpertsWmma skip its own
    // narrowing pass this layer.
    NarrowActivations(x, s_.x_half, false,
                      static_cast<std::size_t>(n_tokens) * c.hidden_size,
                      stream_);
    half_src_ = x;
    half_rows_ = n_tokens;
    half_cols_ = c.hidden_size;
    half_bf16_ = false;
    // Decode rows (1-2) run the warp-per-expert GEMV: one warp per expert row,
    // the row read once for both token rows, whole-row walk, fp32
    // accumulation — 14us vs 28us for the tiled kernel at n=1 and ~100x
    // closer to the fp32 reference. Wider batches stay on the WMMA tiles,
    // which win from n=3 up (the narrow form pays one x conversion per token
    // per step; the router probe has the crossover). GUFO_ROUTER_WMMA forces
    // the tiled kernel (A/B escape hatch).
    static const bool router_wmma = getenv("GUFO_ROUTER_WMMA") != nullptr;
    // Decode batches go through the split-K tiled form: sixteen k-slices take
    // the router GEMM from 74.7us to 22.2us a layer (measured on the six-row
    // counting cohort; the tiled form's two blocks cannot cover the 2 MB
    // sweep) with identical expert selection (self-check: top-8 flips 0/6,
    // max|delta| 2.6e-4 against a F16-input selection gap of ~2e-2).
    // GUFO_ROUTER_SPLITK=0 keeps the plain tiled kernel; GUFO_ROUTER_WMMA
    // forces it globally.
    static const std::uint32_t router_splitk = [] {
      const char* env = getenv("GUFO_ROUTER_SPLITK");
      if (env == nullptr) {
        return kRouterSplitMaxSlices;
      }
      const unsigned long value = std::strtoul(env, nullptr, 10);
      return static_cast<std::uint32_t>(value < kRouterSplitMaxSlices
                                            ? value
                                            : kRouterSplitMaxSlices);
    }();
    static const std::uint32_t router_check = [] {
      const char* env = getenv("GUFO_ROUTER_SELFCHECK");
      return env != nullptr
                 ? static_cast<std::uint32_t>(std::strtoul(env, nullptr, 10))
                 : 0U;
    }();
    const bool split_ok =
        router_splitk > 1 && s_.router_partials != nullptr &&
        s_.router_alt != nullptr && n_tokens <= kRouterSplitMaxTokens &&
        c.hidden_size % (16U * router_splitk) == 0;
    if (!router_wmma && n_tokens <= 2 && c.hidden_size % 4 == 0) {
      RouterGemmNarrow(router_f16, static_cast<const __half*>(s_.x_half),
                       s_.router, n_tokens, c.num_experts, c.hidden_size,
                       stream_);
    } else if (!router_wmma && split_ok) {
      RouterGemmWmmaSplit(router_f16, static_cast<const __half*>(s_.x_half),
                          s_.router, s_.router_partials, n_tokens,
                          c.num_experts, c.hidden_size, router_splitk,
                          stream_);
      if (router_check > 0 && router_checks_done_ < router_check) {
        ++router_checks_done_;
        RouterSelfCheck(router_f16, n_tokens, router_splitk);
      }
    } else {
      RouterGemmWmma(router_f16, static_cast<const __half*>(s_.x_half),
                     s_.router, n_tokens, c.num_experts, c.hidden_size,
                     stream_);
    }
  } else if (!Dense(l.router, x, s_.router, n_tokens, error_msg)) {
    return false;
  }
  if (getenv("GUFO_RT_DUMP") != nullptr) {
    static bool dumped = false;
    if (!dumped) {
      dumped = true;
      hipStreamSynchronize(stream_);
      system("mkdir -p /tmp/rtdump");
      auto save = [&](const void* p, std::size_t bytes, const char* name) {
        std::vector<char> host(bytes);
        if (hipMemcpy(host.data(), p, bytes, hipMemcpyDeviceToHost) ==
            hipSuccess) {
          FILE* f = fopen((std::string("/tmp/rtdump/") + name).c_str(), "wb");
          if (f) {
            fwrite(host.data(), 1, bytes, f);
            fclose(f);
          }
          fprintf(stderr, "RTDUMP %s %zu bytes\n", name, bytes);
        }
      };
      save(l.router.data,
           static_cast<std::size_t>(c.num_experts) * c.hidden_size * 4,
           "router_w.bin");
      if (!l.router_bias.empty()) {
        save(l.router_bias.data, c.num_experts * 4, "router_bias.bin");
      }
      save(x, static_cast<std::size_t>(n_tokens) * c.hidden_size * 4,
           "rt_x.bin");
      save(s_.router,
           static_cast<std::size_t>(n_tokens) * c.num_experts * 4,
           "router_out.bin");
      fprintf(stderr,
              "RTDUMP meta n_tokens=%u m=%u k=%u bias=%d\n", n_tokens,
              c.num_experts, c.hidden_size, (int)!l.router_bias.empty());
      fflush(stderr);
    }
  }
  RouterTopKSigmoid(s_.router, c.num_experts,
                    l.router_bias.empty() ? nullptr : l.router_bias.f32(),
                    s_.ids, s_.weights, n_tokens, c.num_experts,
                    c.num_experts_used, stream_);
  // RoutedCompact reads the per-expert row counts; the vendored MXFP4 tier
  // never does, and wide batches read them back to bound their tile grids.
  // At decode widths on the mmq tier the kernel and the memset inside it are
  // two dead device operations per MoE layer (94 per decode step).
  if (UsesRoutedCompact(l, n_tokens) || ExpertMatrixRows(n_tokens) ||
      getenv("GUFO_MOE_COUNTS") != nullptr) {
    ExpertCounts(s_.ids, s_.expert_counts, n_tokens, c.num_experts,
                 c.num_experts_used, stream_);
  }
  const auto tt1 = std::chrono::steady_clock::now();
  if (ExpertMatrixRows(n_tokens) &&
      (!Check(hipMemcpyAsync(counts_host_, s_.expert_counts,
                             c.num_experts * sizeof(std::uint32_t),
                             hipMemcpyDeviceToHost, stream_),
              "expert counts download", error_msg) ||
       !Check(hipEventRecord(counts_ready_, stream_), "expert counts event",
              error_msg) ||
       !RouteHints(n_tokens, error_msg))) {
    return false;
  }
  const auto tt2 = std::chrono::steady_clock::now();
  if (!MoeExperts(l, x, out, n_tokens, error_msg)) {
    return false;
  }
  // GUFO_MOE_COUNTS: how many distinct experts a batch's rows route to. A
  // verify batch that repeats experts can amortize their weight reads; rows
  // on disjoint expert sets cannot, so this bounds the batch's upside.
  if (n_tokens > 1 && getenv("GUFO_MOE_COUNTS") != nullptr) {
    static std::uint32_t count_dumps = 0;
    if (count_dumps++ < 600) {
      std::vector<std::uint32_t> expert_counts(c.num_experts);
      if (hipMemcpy(expert_counts.data(), s_.expert_counts,
                    expert_counts.size() * sizeof(std::uint32_t),
                    hipMemcpyDeviceToHost) == hipSuccess) {
        std::uint32_t distinct = 0;
        std::uint64_t slots = 0;
        for (const auto count : expert_counts) {
          distinct += count != 0 ? 1 : 0;
          slots += count;
        }
        const double per_expert =
            distinct != 0
                ? static_cast<double>(slots) / static_cast<double>(distinct)
                : 0.0;
        fprintf(stderr,
                "MOECOUNTS n=%u slots=%llu distinct=%u rows_per=%0.3f\n",
                n_tokens, static_cast<unsigned long long>(slots), distinct,
                per_expert);
        fflush(stderr);
      }
    }
  }
  const auto tt3 = std::chrono::steady_clock::now();
  if (ffn_trace && (++trace_layer <= 2 || trace_layer % 47 == 0)) {
    hipStreamSynchronize(stream_);
    const auto tt4 = std::chrono::steady_clock::now();
    using ms_t = std::chrono::duration<double, std::milli>;
    fprintf(stderr,
            "FFNTRACE L%u n=%u router=%.2f hints=%.2f moe_enq=%.2f "
            "moe_run=%.2f\n",
            trace_layer, n_tokens, ms_t(tt1 - tt0).count(),
            ms_t(tt2 - tt1).count(), ms_t(tt3 - tt2).count(),
            ms_t(tt4 - tt3).count());
    fflush(stderr);
  }
  if (getenv("GUFO_MOE_OUT") != nullptr) {
    static int dump_n = 0;
    // Capture the first layers' MoE input/output for path comparison.
    if (dump_n < 6) {
      hipStreamSynchronize(stream_);
      {
        std::vector<float> xi(static_cast<std::size_t>(n_tokens) *
                              c.hidden_size);
        if (hipMemcpy(xi.data(), x, xi.size() * 4, hipMemcpyDeviceToHost) ==
            hipSuccess) {
          char xpath[128];
          snprintf(xpath, sizeof(xpath), "/tmp/mxdump/x_L%d.bin", dump_n);
          FILE* xf = fopen(xpath, "wb");
          if (xf) {
            fwrite(xi.data(), 4, xi.size(), xf);
            fclose(xf);
          }
        }
      }
      std::vector<float> o(static_cast<std::size_t>(n_tokens) *
                           c.hidden_size);
      if (hipMemcpy(o.data(), out, o.size() * 4, hipMemcpyDeviceToHost) ==
          hipSuccess) {
        char path[128];
        snprintf(path, sizeof(path), "/tmp/mxdump/gate_e_L%d.bin", dump_n);
        {
          std::vector<float> g(static_cast<std::size_t>(n_tokens) *
                              c.num_experts_used * l.ffn_gate_exps.rows);
          if (hipMemcpy(g.data(), s_.gate_e, g.size() * 4,
                        hipMemcpyDeviceToHost) == hipSuccess) {
            FILE* gf = fopen(path, "wb");
            if (gf) {
              fwrite(g.data(), 4, g.size(), gf);
              fclose(gf);
            }
          }
        }
        snprintf(path, sizeof(path), "/tmp/mxdump/up_e_L%d.bin", dump_n);
        {
          std::vector<__half> g(static_cast<std::size_t>(n_tokens) *
                                c.num_experts_used *
                                l.ffn_gate_exps.rows);
          if (hipMemcpy(g.data(), s_.up_e, g.size() * 2,
                        hipMemcpyDeviceToHost) == hipSuccess) {
            FILE* gf = fopen(path, "wb");
            if (gf) {
              fwrite(g.data(), 2, g.size(), gf);
              fclose(gf);
            }
          }
        }
        snprintf(path, sizeof(path), "/tmp/mxdump/moe_out_L%d.bin", dump_n);
        FILE* f = fopen(path, "wb");
        if (f) {
          fwrite(o.data(), 4, o.size(), f);
          fclose(f);
        }
        double sum = 0;
        for (float v : o) sum += std::abs(v);
        // Direct gate comparison anchor: slot 0's first six gate values.
        std::vector<float> g6(6);
        bool g6ok = hipMemcpy(g6.data(), s_.gate_e, 24,
                              hipMemcpyDeviceToHost) == hipSuccess;
        fprintf(stderr,
                "MOEOUT L%d n=%u meanabs=%.5f gate0[:6]=%.4f %.4f %.4f %.4f "
                "%.4f %.4f (%d)\n",
                dump_n, n_tokens, sum / o.size(), g6[0], g6[1], g6[2], g6[3],
                g6[4], g6[5], (int)g6ok);
        fflush(stderr);
      }
      ++dump_n;
    }
  }
  // A pair lane keeps its FFN partial local (the boundary sum combines it);
  // the dense branch above has the same shape as `return !reduce || ...`.
  if (reduce && !AllReduce(out, n_tokens, error_msg)) {
    return false;
  }
  if (options_.moe_observer && model_->tp_world_size() > 1) {
    options_.moe_observer(
        out, static_cast<std::size_t>(n_tokens) * c.hidden_size * sizeof(float),
        stream_);
  }
  return true;
}

bool Executor::AttentionPart(const DeviceLayer& l, Session::AttentionState& s,
                             const float* x, float* out,
                             std::uint32_t n_tokens, const std::uint32_t* pos,
                             std::string* error_msg, bool projections_ready,
                             bool project_output, const SlotKv* slot_table,
                             std::uint32_t slot,
                             std::uint32_t visible_keys,
                             std::uint32_t wide_splits,
                             const std::uint32_t* row_slot,
                             const std::uint32_t* row_pos) const {
  const PhaseScope _ps(&kPhases.attn_ms, &kPhases.attn_n);
    if (kEntryTrace) { fprintf(stderr, "ENTRY exec:AttentionPart\n"); fflush(stderr); }
  const Config& c = config();
  static const bool attn_trace = getenv("GUFO_FFN_TRACE") != nullptr;
  static std::uint32_t attn_trace_n = 0;
  // Sampling keeps the steady-state trace to ~one call per 48-layer sweep;
  // GUFO_ATTN_TRACE_ALL prints every call (cycle-structure work).
  static const bool attn_trace_all = getenv("GUFO_ATTN_TRACE_ALL") != nullptr;
  if (attn_trace) {
    ++attn_trace_n;
  }
  const bool attn_do_trace =
      attn_trace &&
      (attn_trace_all || attn_trace_n <= 2 || attn_trace_n % 47 == 0);
  const auto at0 = std::chrono::steady_clock::now();
  if (!projections_ready &&
      (!Dense(l.attn_q, x, s_.q, n_tokens, error_msg) ||
       !Dense(l.attn_k, x, s_.k, n_tokens, error_msg) ||
       !Dense(l.attn_v, x, s_.v, n_tokens, error_msg))) {
    if (kEntryTrace) { fprintf(stderr, "ENTRY AttentionPart\n"); fflush(stderr); }
    return false;
  }
  const auto at1 = std::chrono::steady_clock::now();
  // Ropes the query in place and rounds keys and values into the layer's
  // caches at *pos..*pos + n_tokens; the graph replays at any position.
  // Sliding layers write through the window ring (s.swa_rows rows). With a
  // slot table the caches live in the table row instead of `s`.
  if (slot_table != nullptr) {
    PrepareQkvSlot(s_.q, s_.k, s_.v, slot_table, slot, n_tokens, l.q_heads,
                   l.kv_heads, c.head_dim_qk, c.head_dim_v, c.rotary_dim, pos,
                   l.rope_theta, stream_, row_slot, row_pos);
  } else {
    PrepareQkv(s_.q, s_.k, s_.v, s_.q, s.k_cache, s.v_cache, n_tokens,
               l.q_heads, l.kv_heads, c.head_dim_qk, c.head_dim_v,
               c.rotary_dim, s.swa_rows, pos, l.rope_theta, stream_,
               s.k_cache_q, s.v_cache_q, s.k_scales, s.v_scales);
  }
  if (getenv("GUFO_SYNC_TRACE") != nullptr) {
    hipStreamSynchronize(stream_);
    const hipError_t e1 = hipGetLastError();
    if (e1 != hipSuccess) {
      AssignError(error_msg, std::string("TRACE: after PrepareQkv: ") +
                                 hipGetErrorString(e1));
      return false;
    }
  }
  // Narrow batches split each row's key tiles over kAttnSplits blocks so a
  // decode step at depth fills the device. Sliding layers pass their
  // window; global layers attend everything below.
  if (kLayerTrace) {
    std::uint32_t pos0 = 0xFFFFFFFF;
    hipStreamSynchronize(stream_);
    hipMemcpy(&pos0, pos, 4, hipMemcpyDeviceToHost);
    fprintf(stderr, "LAYERTRACE pos=%u n_tokens=%u heads=%u kv=%u dqk=%u dv=%u window=%u sinks=%p\n",
            pos0, n_tokens, l.q_heads, l.kv_heads, c.head_dim_qk, c.head_dim_v,
            s.swa_rows,
            static_cast<const void*>(l.attn_sinks.empty()
                                         ? nullptr
                                         : l.attn_sinks.f32()));
    fflush(stderr);
    DumpRow("att-q", s_.q, 8, stream_);
    if (kLayerTrace) {
      __half hk[4], hv[4];
      hipStreamSynchronize(stream_);
      hipMemcpy(hk, s.k_cache, sizeof(hk), hipMemcpyDeviceToHost);
      hipMemcpy(hv, s.v_cache, sizeof(hv), hipMemcpyDeviceToHost);
      fprintf(stderr, "LAYERTRACE k0 half=%f %f v0 half=%f %f\n",
              __half2float(hk[0]), __half2float(hk[1]),
              __half2float(hv[0]), __half2float(hv[1]));
      fflush(stderr);
    }
  }
  const auto at2 = std::chrono::steady_clock::now();
  // Split-K stays off: it measured no win on this workload, and the merge
  // path is one more place for narrow-batch divergence to hide.
  // Decode-width global attention splits its key range once the context is
  // long enough: the single-split form walks the range serially in one
  // block, which at long contexts is thousands of key tiles per row. The
  // prefill form (n_tokens >= 16) keeps its matrix kernel.
  const std::uint32_t split_keys = AttentionSplitKeys();
  // The wide (16-row tile) path pays a fixed per-block cost, so it splits
  // only once a single block would walk an unwieldy key range: 17.5K keys
  // measured faster unsplit (n=2 1.51ms vs 2.67ms split).
  static const std::uint32_t wide_split_keys = [] {
    const char* env = getenv("GUFO_ATTN_WSM_KEYS");
    const long v = env != nullptr ? std::strtol(env, nullptr, 10) : -1;
    if (v < 0) {
      return 32768U;
    }
    return static_cast<std::uint32_t>(v);  // 0 = always split
  }();
  const std::uint32_t split_count = AttentionSplitCount();
  // A cohort step reads each row's own cache at its own position, so the
  // widest row sets whether the range is split for the whole step.
  const std::uint32_t cohort_keys =
      (row_slot != nullptr || row_pos != nullptr) ? batch_visible_keys_
                                                  : visible_keys;
  const std::uint32_t attn_splits =
      (!l.swa && n_tokens < 16 && split_keys != 0 &&
       cohort_keys >= split_keys)
          ? split_count
          : 1;
  const std::uint32_t wide = (attn_splits > 1 &&
                              (wide_split_keys == 0 ||
                               visible_keys >= wide_split_keys))
                                 ? attn_splits
                                 : 1;
  if (slot_table != nullptr) {
    if (row_slot != nullptr || row_pos != nullptr) {
      // Fused cohort step: one launch covers every session's rows, each row
      // walking its own visible key range in `attn_splits` blocks.
      AttentionRows(s_.q, slot_table, row_slot, row_pos,
                    l.attn_sinks.empty() ? nullptr : l.attn_sinks.f32(),
                    s_.ctx, attn_splits > 1 ? s_.attn_partials : nullptr,
                    attn_splits, n_tokens, l.q_heads, l.kv_heads,
                    c.head_dim_qk, c.head_dim_v,
                    l.swa ? c.sliding_window : 0, cohort_keys, stream_);
    } else {
      // One row, one session: the split count is the only parallelism the
      // shared-KV form has at depth, so the single-token path needs it too.
      AttentionSlot(s_.q, slot_table, slot,
                    l.attn_sinks.empty() ? nullptr : l.attn_sinks.f32(), s_.ctx,
                    attn_splits > 1 ? s_.attn_partials : nullptr, attn_splits,
                    n_tokens, pos, l.q_heads, l.kv_heads, c.head_dim_qk,
                    c.head_dim_v, l.swa ? c.sliding_window : 0, cohort_keys,
                    stream_);
    }
  } else {
    Attention(s_.q, s.k_cache, s.v_cache,
              l.attn_sinks.empty() ? nullptr : l.attn_sinks.f32(), s_.ctx,
              attn_splits > 1 ? s_.attn_partials : nullptr, attn_splits,
              wide, n_tokens, pos,
              l.q_heads, l.kv_heads, c.head_dim_qk, c.head_dim_v,
              l.swa ? c.sliding_window : 0, s.swa_rows, cohort_keys, stream_,
              s.k_cache_q, s.v_cache_q, s.k_scales, s.v_scales);
  }
  const auto at3 = std::chrono::steady_clock::now();
  if (getenv("GUFO_SYNC_TRACE") != nullptr) {
    hipStreamSynchronize(stream_);
    const hipError_t e2 = hipGetLastError();
    if (e2 != hipSuccess) {
      AssignError(error_msg, std::string("TRACE: after Attention: ") +
                                 hipGetErrorString(e2));
      return false;
    }
  }
  if (kLayerTrace) {
    DumpRow("att-ctx", s_.ctx, 8, stream_);
  }
  if (kDumpOnce && l.kv_heads == 2 && !l.swa) {
    static bool done = false;
    if (!done) {
      done = true;
      DumpVec("q", s_.q, 6144, stream_);
      DumpVec("xn", x, 4096, stream_);
      DumpVec("kcache", s.k_cache, 384 * 16, stream_, /*half=*/true);
      DumpVec("vcache", s.v_cache, 256 * 16, stream_, /*half=*/true);
      DumpVec("ctx", s_.ctx, 4096, stream_);
      fprintf(stderr, "DUMPED layer0 attention buffers\n");
      fflush(stderr);
    }
  }
  if (!project_output) {
    return true;
  }
  if (!Dense(l.attn_out, s_.ctx, out, n_tokens, error_msg)) {
    return false;
  }
  if (attn_do_trace) {
    const auto at4 = std::chrono::steady_clock::now();
    using ms_t = std::chrono::duration<double, std::milli>;
    fprintf(stderr,
            "ATTNTRACE L%u n=%u qkv=%.2f prep=%.2f fa=%.2f o=%.2f swa=%u "
            "addr=%llu\n",
            attn_trace_n, n_tokens, ms_t(at1 - at0).count(),
            ms_t(at2 - at1).count(), ms_t(at3 - at2).count(),
            ms_t(at4 - at3).count(), s.swa_rows,
            static_cast<unsigned long long>(
                reinterpret_cast<std::uintptr_t>(&l)));
    fflush(stderr);
  }
  // The attention output scales by value_scale after the projection (the
  // linear projection commutes, but this is the reference order).
  if (value_scale_ != 1.0F) {
    // rocBLAS is unusable in this environment (Tensile faults on first
    // call); the scale is a plain elementwise multiply anyway.
    ScaleInPlace(out, value_scale_,
                 static_cast<std::size_t>(n_tokens) * c.hidden_size, stream_);
  }
  return true;
}

namespace {
/// Stream host function a captured graph replays: the exchange's host half
/// (post once staged, wait for the peer, release the stream's wait). The
/// context storage outlives every graph: it sits in a process-lifetime pool.
struct CapturedExchangeCtx {
  Communicator* communicator;
  Communicator::CapturedPartial partial;
};
void CapturedExchangeThunk(void* user_data) {
  auto* ctx = static_cast<CapturedExchangeCtx*>(user_data);
  ctx->communicator->RunCapturedExchange(ctx->partial);
}
std::vector<std::unique_ptr<CapturedExchangeCtx>>& CapturedCtxPool() {
  static std::vector<std::unique_ptr<CapturedExchangeCtx>> pool;
  return pool;
}
}  // namespace

std::function<bool(float*, std::size_t, hipStream_t, std::string*)>
Executor::TwoRankAllReduce(std::shared_ptr<Communicator> communicator,
                           std::uint32_t hidden) {
  return [communicator = std::move(communicator), hidden](
             float* data, std::size_t bytes, hipStream_t stream,
             std::string* error) {
    // Under stream capture the background-thread exchange cannot be queued
    // (it would never replay); the capture protocol stages the same kernels
    // with slotted targets and takes the host half from a stream host
    // function the graph replays.
    hipStreamCaptureStatus capture_status = hipStreamCaptureStatusNone;
    if (hipStreamIsCapturing(stream, &capture_status) != hipSuccess ||
        capture_status != hipStreamCaptureStatusNone) {
      Communicator::CapturedPartial captured;
      if (!communicator->CaptureSafe() ||
          !communicator->CapturePartial(bytes, &captured, error)) {
        return false;
      }
      auto& ctx_pool = CapturedCtxPool();
      ctx_pool.push_back(std::make_unique<CapturedExchangeCtx>(
          CapturedExchangeCtx{communicator.get(), captured}));
      StagePartialSlotted(
          data, captured.queued.send, bytes / sizeof(float),
          captured.queued.arrivals, captured.queued.ready,
          captured.value_slot, stream);
      if (hipLaunchHostFunc(stream, CapturedExchangeThunk,
                            ctx_pool.back().get()) != hipSuccess) {
        AssignError(error, "captured exchange host function failed");
        return false;
      }
      WaitValueSlotted(captured.queued.arrived, captured.value_slot,
                       captured.queued.wait_ticks, captured.queued.timed_out,
                       stream);
      const std::size_t rows = bytes / (sizeof(float) * hidden);
      if (rows != 0) {
        AddRowsBroadcastSlotted(captured.queued.peer, data,
                                static_cast<std::uint32_t>(rows), hidden, 1,
                                captured.value_slot, stream);
      }
      return true;
    }
    // GUFO_AR_F16 halves the exchange payload (both ranks must agree: the
    // start script sets it on each). The local partial and the received sum
    // stay f32; only the wire format narrows.
    static const bool kHalfPayload = getenv("GUFO_AR_F16") != nullptr;
    Communicator::QueuedPartial queued;
    if (!communicator->QueuePartial(kHalfPayload ? bytes / 2 : bytes, &queued,
                                   error)) {
      return false;
    }
    if (kEntryTrace) {
      fprintf(stderr,
              "EXCHANGE bytes=%zu data=%p send=%p peer=%p arrivals=%p "
              "ready=%p arrived=%p value=%llu ticks=%llu\n",
              bytes, static_cast<void*>(data),
              static_cast<const void*>(queued.send),
              static_cast<const void*>(queued.peer),
              static_cast<const void*>(queued.arrivals),
              static_cast<const void*>(queued.ready),
              static_cast<const void*>(queued.arrived),
              static_cast<unsigned long long>(queued.value),
              static_cast<unsigned long long>(queued.wait_ticks));
      fflush(stderr);
    }
    if (kHalfPayload) {
      StagePartialHalf(data, reinterpret_cast<__half*>(queued.send),
                       bytes / sizeof(float), queued.arrivals, queued.ready,
                       queued.value, stream);
    } else {
      StagePartial(data, queued.send, bytes / sizeof(float), queued.arrivals,
                   queued.ready, queued.value, stream);
    }
    WaitValue(queued.arrived, queued.value, queued.wait_ticks, queued.timed_out,
              stream);
    const std::size_t rows = bytes / (sizeof(float) * hidden);
    if (rows != 0) {
      if (kHalfPayload) {
        AddRowsBroadcastHalf(reinterpret_cast<const __half*>(queued.peer), data,
                             static_cast<std::uint32_t>(rows), hidden, 1,
                             stream);
      } else {
        AddRowsBroadcast(queued.peer, data, static_cast<std::uint32_t>(rows),
                         hidden, 1, stream);
      }
    }
    return true;
  };
}

std::function<bool(std::string*)> Executor::TwoRankReduceStatus(
    std::shared_ptr<Communicator> communicator) {
  return [communicator = std::move(communicator)](std::string* error) {
    if (kEntryTrace) { fprintf(stderr, "ENTRY exec:Drain\n"); fflush(stderr); }
    return communicator->CheckQueued(error);
  };
}

bool Executor::Drain(const char* what, std::string* error_msg) const {
  return Check(hipStreamSynchronize(stream_), what, error_msg) &&
         (!options_.reduce_status || options_.reduce_status(error_msg));
}

Executor::SplitReduce Executor::TwoRankSplitReduce(
    std::shared_ptr<Communicator> communicator, std::uint32_t hidden) {
  SplitReduce reduce;
  reduce.start = [communicator](const float* data, std::size_t bytes,
                                hipStream_t stream, std::string* error) {
    return communicator->StartPartial(data, bytes, stream, error);
  };
  reduce.finish = [communicator = std::move(communicator), hidden](
                      float* data, std::size_t bytes, hipStream_t stream,
                      std::string* error) {
    const float* peer = communicator->FinishPartial(bytes, error);
    if (peer == nullptr) {
      return false;
    }
    const std::size_t rows = bytes / (sizeof(float) * hidden);
    if (rows != 0) {
    if (kEntryTrace) { fprintf(stderr, "ENTRY exec:AllReduce\n"); fflush(stderr); }
      AddRowsBroadcast(peer, data, static_cast<std::uint32_t>(rows), hidden, 1,
                       stream);
    }
    return true;
  };
  return reduce;
}

bool Executor::AllReduce(float* data, std::size_t rows,
                         std::string* error_msg) const {
  const PhaseScope _ps(&kPhases.reduce_ms, &kPhases.reduce_n);
  if (model_->tp_world_size() == 1) {
    return true;
  }
  const std::size_t hidden = config().hidden_size;
  if (rows > std::numeric_limits<std::size_t>::max() / hidden ||
      rows * hidden > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    if (error_msg != nullptr) {
      *error_msg = "MiMo-V2 all-reduce byte count overflows";
    }
    return false;
  }
  const std::size_t bytes = rows * hidden * sizeof(float);
  if (!options_.all_reduce ||
      !options_.all_reduce(data, bytes, stream_, error_msg)) {
    if (kEntryTrace) { fprintf(stderr, "ENTRY exec:MtpHead\n"); fflush(stderr); }
    if (error_msg != nullptr && error_msg->empty()) {
      *error_msg = "MiMo-V2 all-reduce failed";
    }
    return false;
  }
  return true;
}

bool Executor::MtpHead(const DeviceLayer& block, const float* res,
                       std::uint32_t row, bool token, bool candidates,
                       std::string* error_msg) const {
  const Config& c = config();
  const DeviceTensor& norm = !block.nextn_head_norm.empty()
                                 ? block.nextn_head_norm
                             : !block.layer_out_norm.empty()
                                 ? block.layer_out_norm
                                 : model_->output_norm();
  const DeviceTensor& head = !block.nextn_head_head.empty()
                                 ? block.nextn_head_head
                                 : model_->output();
  RmsNormRows(res + static_cast<std::size_t>(row) * c.hidden_size, norm.f32(),
              s_.xn, 1, c.hidden_size, 1, c.rms_eps, stream_);
  if (!Dense(head, s_.xn, s_.logits, 1, error_msg)) {
    return false;
  }
  if (candidates)
    MtpTopCandidates(s_.logits, s_.mtp_ids, s_.mtp_scratch_ids, s_.mtp_scores,
                     head.rows, stream_);
  if (token) {
    Argmax(s_.logits, s_.mtp_argmax, s_.mtp_token, 1, head.rows, stream_);
    if (!Check(
            hipMemcpyAsync(mtp_token_host_, s_.mtp_token, sizeof(std::int32_t),
                           hipMemcpyDeviceToHost, stream_),
            "draft token download", error_msg)) {
      return false;
    }
  }
  if (candidates) {
    const auto count = std::min<std::size_t>(head.rows, kMtpCandidates);
    static_assert(offsetof(MtpCandidateLogits, logits) ==
                  kMtpCandidates * sizeof(std::uint32_t));
    const auto bytes =
        offsetof(MtpCandidateLogits, logits) + count * sizeof(float);
    if (!Check(hipMemcpyAsync(mtp_candidates_host_, s_.mtp_ids, bytes,
                              hipMemcpyDeviceToHost, stream_),
               "draft candidates download", error_msg)) {
      return false;
    }
  }
  return true;
}

bool Executor::AllocateSlots(std::string* error_msg) const {
  if (slots_.host != nullptr) {
    return true;
  }
  slots_.layers = config().num_layers;
  slots_.ctrl_off = 0;
  slots_.ptrs_off = slots_.ctrl_off + kBatchSlotCount * sizeof(SlotControl);
  slots_.kv_off = slots_.ptrs_off + kBatchSlotCount * sizeof(SlotPtrs);
  slots_.mtp_off =
      slots_.kv_off + static_cast<std::size_t>(slots_.layers) *
                          kBatchSlotCount * sizeof(SlotKv);
  slots_.bytes = slots_.mtp_off + kBatchSlotCount * sizeof(SlotKv);
  void* host = nullptr;
  void* dev = nullptr;
  if (!Check(hipHostMalloc(&host, slots_.bytes), "slot table host",
             error_msg) ||
      !Check(hipMalloc(&dev, slots_.bytes), "slot table device", error_msg)) {
    (void)hipHostFree(host);
    (void)hipFree(dev);
    return false;
  }
  std::memset(host, 0, slots_.bytes);
  slots_.host = host;
  slots_.dev = dev;
  if (!Check(hipHostMalloc(reinterpret_cast<void**>(&slots_.row_slot_host),
                           SlotTables::kRowRows * sizeof(std::uint32_t)),
             "row slot table host", error_msg) ||
      !Check(hipHostMalloc(reinterpret_cast<void**>(&slots_.row_pos_host),
                           SlotTables::kRowRows * sizeof(std::uint32_t)),
             "row position table host", error_msg) ||
      !Check(hipMalloc(reinterpret_cast<void**>(&slots_.row_slot_dev),
                       SlotTables::kRowRows * sizeof(std::uint32_t)),
             "row slot table device", error_msg) ||
      !Check(hipMalloc(reinterpret_cast<void**>(&slots_.row_pos_dev),
                       SlotTables::kRowRows * sizeof(std::uint32_t)),
             "row position table device", error_msg)) {
    return false;
  }
  std::memset(slots_.row_slot_host, 0,
              SlotTables::kRowRows * sizeof(std::uint32_t));
  std::memset(slots_.row_pos_host, 0,
              SlotTables::kRowRows * sizeof(std::uint32_t));
  return true;
}

std::uint32_t Executor::AttentionSplitKeys() {
  static const std::uint32_t value = [] {
    const char* env = getenv("GUFO_ATTN_SPLIT_KEYS");
    const long v = env != nullptr ? std::strtol(env, nullptr, 10) : -1;
    if (v < 0) {
      return kAttnSplitKeys;
    }
    return static_cast<std::uint32_t>(v);
  }();
  return value;
}

std::uint32_t Executor::AttentionSplitCount() {
  static const std::uint32_t value = [] {
    const char* env = getenv("GUFO_ATTN_SPLITS");
    const long v = env != nullptr ? std::strtol(env, nullptr, 10) : -1;
    if (v < 0) {
      return kAttnSplits;
    }
    return static_cast<std::uint32_t>(
        v < 1 ? 1
              : (v > static_cast<long>(kAttnSplitsMax) ? kAttnSplitsMax : v));
  }();
  return value;
}

void Executor::FillSlotRow(std::uint32_t slot, const Session& session,
                           std::int32_t hidden_row) const {
  // A non-MTP session carries no draft block; its table row nulls the draft
  // destinations (mtp_blocks_ is empty, so mtp_() would be invalid).
  const auto* mtp = session.mtp_enabled_ ? &session.mtp_() : nullptr;
  slots_.HostCtrl()[slot] = {session.position_,
                             mtp != nullptr ? mtp->position : 0, hidden_row, 0};
  const bool active = !session.Cancelled();
  auto& ptrs = slots_.HostPtrs()[slot];
  // target_hidden is only the MTP sessions' kept trunk rows; a non-MTP
  // session's null leaves its copy out of the replayed graph.
  ptrs.target_hidden =
      active && session.mtp_enabled_ ? session.target_hidden_ : nullptr;
  ptrs.mtp_h = active && mtp != nullptr ? mtp->h : nullptr;
  for (std::uint32_t il = 0; il < slots_.layers; ++il) {
    const auto& a = session.attention_[il];
    slots_.HostKv(il)[slot] = {active ? a.k_cache : nullptr,
                               active ? a.v_cache : nullptr, a.swa_rows, 0,
                               active ? a.k_cache_q : nullptr,
                               active ? a.v_cache_q : nullptr,
                               active ? a.k_scales : nullptr,
                               active ? a.v_scales : nullptr};
  }
  slots_.HostMtpKv()[slot] = {active && mtp != nullptr ? mtp->k_cache : nullptr,
                              active && mtp != nullptr ? mtp->v_cache
                                                        : nullptr,
                              mtp != nullptr ? mtp->swa_rows : 0, 0,
                              nullptr, nullptr, nullptr, nullptr};
}

bool Executor::GraphCaptureEnabled() {
  // Host-driven communication is graph-safe only under a transport that
  // implements the capture protocol. Any debug probe that synchronizes or
  // dumps inside the body (SYNC_TRACE, DUMP_DIR, LAYER/MX/RT/MOE dumps,
  // FFN_TRACE phase numbers) keeps the body eager; GUFO_NO_GRAPH is the
  // escape hatch.
  static const bool kCaptureOk = [] {
    return getenv("GUFO_NO_GRAPH") == nullptr &&
           getenv("GUFO_SYNC_TRACE") == nullptr &&
           getenv("GUFO_DUMP_DIR") == nullptr &&
           getenv("GUFO_LAYER_TRACE") == nullptr &&
           getenv("GUFO_MX_DUMP") == nullptr &&
           getenv("GUFO_RT_DUMP") == nullptr &&
           getenv("GUFO_MOE_OUT") == nullptr &&
           getenv("GUFO_FFN_TRACE") == nullptr;
  }();
  return kCaptureOk;
}

bool Executor::RunCaptured(std::uint64_t key, const std::function<bool()>& body,
                           std::string* error_msg) const {
  const bool kCaptureOk = GraphCaptureEnabled();
  if (!kCaptureOk ||
      (model_->tp_world_size() > 1 &&
       !(options_.communicator && options_.communicator->CaptureSafe()))) {
    if (getenv("GUFO_GRAPH_DEBUG") != nullptr) {
      fprintf(stderr, "GRAPHDBG runcaptured gate key=%llu ok=%d comm=%d\n",
              (unsigned long long)key, (int)kCaptureOk,
              options_.communicator ? (int)options_.communicator->CaptureSafe()
                                    : -1);
      fflush(stderr);
    }
    return body();
  }
  // A batch shape runs eagerly once before it is captured: the first pass
  // grows the GEMM tier's arena, which capture forbids. The cache is
  // executor-wide: decode batches share the scratch, so a graph keyed on the
  // batch shape replays correctly across sessions. MTP chain lengths jitter,
  // so cold shapes simply stay eager.
  if (!batch_warmed_.contains(key)) {
    batch_warmed_.insert(key);
    if (getenv("GUFO_GRAPH_DEBUG") != nullptr) {
      fprintf(stderr, "GRAPHDBG batch warm key=%llu\n",
              (unsigned long long)key);
      fflush(stderr);
    }
    return body();
  }
  hipGraphExec_t exec = nullptr;
  if (const auto it = batch_graphs_.find(key); it != batch_graphs_.end()) {
    exec = it->second;
  } else if (batch_graphs_.size() >= kMaxBatchGraphs) {
    return body();
  } else {
    hipGraph_t captured = nullptr;
    mimo2_pool_set_stream(stream_);
    void* capture_session = nullptr;
    if (model_->tp_world_size() > 1 &&
        (capture_session =
             options_.communicator->BeginCaptureSession(error_msg)) == nullptr) {
      return false;
    }
    if (!Check(hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal),
               "batch graph capture", error_msg)) {
      return false;
    }
    const bool ok = body();
    if (!Check(hipStreamEndCapture(stream_, &captured),
               "batch graph capture end", error_msg) || !ok) {
      if (captured != nullptr) {
        (void)hipGraphDestroy(captured);
      }
      return false;
    }
    const bool instantiated =
        Check(hipGraphInstantiate(&exec, captured, nullptr, nullptr, 0),
              "batch graph instantiate", error_msg);
    (void)hipGraphDestroy(captured);
    if (!instantiated) {
      return false;
    }
    fprintf(stderr, "GRAPHCAPTURE key=%llu tp=%u\n",
            (unsigned long long)key, model_->tp_world_size());
    fflush(stderr);
    batch_graphs_.emplace(key, exec);
    if (capture_session != nullptr) {
      std::uint64_t exchange_count = 0;
      void* finished =
          options_.communicator->EndCaptureSession(&exchange_count, error_msg);
      if (finished == nullptr) {
        return false;
      }
      batch_graph_sessions_.emplace(key, finished);
    }
  }
  // The slots' values for this replay are assigned from the transport's
  // global operation ledger before the launch (the graph's kernels and host
  // functions read them as they run), keeping replayed and eager exchanges
  // on one sequence.
  const auto session_it = batch_graph_sessions_.find(key);
  if (session_it != batch_graph_sessions_.end() &&
      !options_.communicator->AdvanceCapturedSession(session_it->second,
                                                     error_msg)) {
    return false;
  }
  if (!Check(hipGraphLaunch(exec, stream_), "batch graph launch", error_msg)) {
    return false;
  }
  return true;
}

bool Executor::Run(Session& session, std::uint64_t key, bool graph,
                   const std::function<bool()>& body, std::string* error_msg,
                   bool synchronize) const {
  // Host-driven communication is not graph-safe unless the transport
  // implements the capture protocol (the exchange host half replays from a
  // stream host function; see TwoRankAllReduce). GUFO_NO_GRAPH is the same
  // escape hatch RunCaptured honours.
  static const bool kNoGraph = getenv("GUFO_NO_GRAPH") != nullptr;
  if (kNoGraph ||
      (model_->tp_world_size() > 1 &&
       !(options_.communicator && options_.communicator->CaptureSafe()))) {
    graph = false;
  }

  // A batch shape runs eagerly once before it is captured: the first pass
  // grows the GEMM tier's arena, which capture forbids.
  if (graph && session.warmed_.contains(key)) {
    hipGraphExec_t exec = nullptr;
    if (const auto it = session.graphs_.find(key);
        it != session.graphs_.end()) {
      exec = it->second;
    } else {
      hipGraph_t captured = nullptr;
      mimo2_pool_set_stream(stream_);
      void* capture_session = nullptr;
      if (model_->tp_world_size() > 1 &&
          (capture_session =
               options_.communicator->BeginCaptureSession(error_msg)) ==
              nullptr) {
        return false;
      }
      if (!Check(
              hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal),
              "graph capture", error_msg)) {
        return false;
      }
      const bool ok = body();
      if (!Check(hipStreamEndCapture(stream_, &captured), "graph capture end",
                 error_msg) ||
          !ok) {
        if (captured != nullptr) {
          (void)hipGraphDestroy(captured);
        }
        return false;
      }
      const bool instantiated =
          Check(hipGraphInstantiate(&exec, captured, nullptr, nullptr, 0),
                "graph instantiate", error_msg);
      (void)hipGraphDestroy(captured);
      if (!instantiated) {
    if (kEntryTrace) { fprintf(stderr, "ENTRY exec:Forward\n"); fflush(stderr); }
        return false;
      }
      session.graphs_.emplace(key, exec);
      if (capture_session != nullptr) {
        std::uint64_t exchange_count = 0;
        void* finished = options_.communicator->EndCaptureSession(
            &exchange_count, error_msg);
        if (finished == nullptr) {
          return false;
        }
        session.graph_capture_sessions_.emplace(key, finished);
      }
    }
    // Assign this replay's exchange ids from the global ledger before the
    // launch (see RunCaptured).
    const auto session_it = session.graph_capture_sessions_.find(key);
    if (session_it != session.graph_capture_sessions_.end() &&
        !options_.communicator->AdvanceCapturedSession(session_it->second,
                                                       error_msg)) {
      return false;
    }
    if (!Check(hipGraphLaunch(exec, stream_), "graph launch", error_msg)) {
      return false;
    }
  } else {
    if (!body()) {
      return false;
    }
    session.warmed_.insert(key);
  }
  return !synchronize || Drain("forward", error_msg);
}

bool Executor::Forward(Session& session, std::span<const std::int32_t> tokens,
                       std::uint32_t n_logits, float* logits, ForwardMode mode,
                       std::string* error_msg) const {
  const bool speculative = mode == ForwardMode::kVerify;
  PrefillPhase phase(mode == ForwardMode::kPrefill);
  selected_logits_ = nullptr;
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
  const Config& c = config();
  const auto n = static_cast<std::uint32_t>(tokens.size());
  const bool prefill = mode == ForwardMode::kPrefill;
  if (kDumpOnce && prefill) {
    static bool dumped_tokens = false;
    if (!dumped_tokens) {
      dumped_tokens = true;
      std::string path = std::string(getenv("GUFO_DUMP_DIR")) + "/tokens.txt";
      FILE* f = fopen(path.c_str(), "w");
      for (std::size_t i = 0; i < tokens.size() && i < 64; ++i) {
        fprintf(f, "%d\n", static_cast<int>(tokens[i]));
      }
      fclose(f);
      fprintf(stderr, "DUMPED %zu tokens\n", tokens.size());
      fflush(stderr);
    }
  }
  const std::uint32_t lead =
      prefill && session.tap_layers_.empty() ? PairLead(n) : 0;
  const bool pair = lead != 0;
  if (n == 0 ||
      n > (prefill ? PrefillChunk(session) : options_.max_batch) ||
      (speculative && n > options_.max_speculative)) {
    AssignError(error_msg, "token batch is empty or exceeds the batch limit");
    return false;
  }
  // A pair's logits come from its second batch.
  if (n_logits > (pair ? n - lead : n) || n_logits > options_.max_logit_rows) {
    AssignError(error_msg, "requested logit rows exceed the batch or limit");
    return false;
  }
  if (session.owner_ != this || session.position_ + n > session.max_context_) {
    AssignError(error_msg, "session context is full");
    return false;
  }
  for (auto t : tokens) {
    if (t < 0 || static_cast<std::uint32_t>(t) >= c.vocab_size) {
      AssignError(error_msg, "token out of range");
      return false;
    }
  }
  const std::uint32_t start_pos = session.position_;
  if (!session.CheckCancellation(error_msg))
    return false;
  TAPTRACE("forward enter mode=%d n=%u pos=%u taps=%zu", static_cast<int>(mode),
           n, start_pos, session.tap_layers_.size());
  // Tap capture follows the phase: prefill appends host rows per chunk,
  // verification keeps its rows on the host, decode keeps only the frontier
  // on the device.
  session.tap_phase_ = session.tap_layers_.empty()
                           ? Session::TapPhase::kNone
                           : prefill ? Session::TapPhase::kPrompt
                           : speculative ? Session::TapPhase::kVerify
                                         : Session::TapPhase::kLastRow;
  if (session.tap_phase_ == Session::TapPhase::kPrompt) {
    session.tap_batch_offset_ = session.prompt_taps_.size();
    session.prompt_taps_.resize(session.tap_batch_offset_ +
                                static_cast<std::size_t>(n) *
                                    session.tap_width_);
  } else if (session.tap_phase_ == Session::TapPhase::kVerify) {
    session.verify_taps_.resize(static_cast<std::size_t>(n) *
                                session.tap_width_);
  }
  if (session.mtp_enabled_ && prefill_phase &&
      session.mtp_().position != start_pos) {
    AssignError(error_msg,
                "MTP must consume the preceding frontier before prefill");
    return false;
  }
  ++session.mutation_epoch_;
  session.spec_base_ = start_pos;
  session.spec_tokens_ = speculative ? n : 0;
  // Everything the launched work reads from the host sits in pinned
  // buffers the graph nodes point at: tokens and the control block (the
  // positions the kernels read on the device).
  std::copy(tokens.begin(), tokens.end(), tokens_host_);
  control_host_->position = start_pos;
  control_host_->mtp_position = session.mtp_().position;
  control_host_->hidden_row = -1;
  if (pair) {
    return ForwardPair(session, tokens, lead, n_logits, logits, error_msg);
  }
  // Decode-sized batches replay as graphs: a verification's tap rows stage in
  // device memory, so the armed path is as graph-safe as a plain decode.
  const bool graph = !prefill_phase && n <= kVecBatch;
  const std::uint64_t key = static_cast<std::uint64_t>(n) |
                            (static_cast<std::uint64_t>(n_logits) << 16) |
                            (static_cast<std::uint64_t>(speculative) << 32) |
                            (std::uint64_t{logits != nullptr} << 34);
  const auto body = [&] {
    return ForwardBody(session, n, n_logits, logits != nullptr, speculative,
                       start_pos, 0, c.num_layers, error_msg);
  };
  TAPTRACE("forward validated, dispatching n=%u", n);
  if (!Run(session, key, graph, body, error_msg)) {
    return false;
  }
  TAPTRACE("forward body done");
  if (n_logits > 0 && getenv("GUFO_GRAPH_DEBUG") != nullptr) {
    double sum = 0;
    for (std::size_t i = 0;
         i < static_cast<std::size_t>(n_logits) * c.vocab_size; ++i) {
      sum += logits_host_[i];
    }
    fprintf(stderr, "LOGITS fwd n=%u pos=%u first=%.6f sum=%.3f\n", n_logits,
            start_pos, logits_host_[0], sum);
    fflush(stderr);
  }
  if (n_logits > 0 && logits != nullptr) {
    std::copy_n(logits_host_, static_cast<std::size_t>(n_logits) * c.vocab_size,
                logits);
  }
  session.position_ += n;
  if (session.mtp_enabled_ && prefill_phase && n > 1) {
    if (kEntryTrace) { fprintf(stderr, "ENTRY exec:ForwardBody\n"); fflush(stderr); }
    // Every successor except the last one is already known. Consume it while
    // the trunk residual is still in shared scratch: MtpBody reads those
    // rows before it reuses the residual for the draft output.
    PrefillPhase draft_phase(false);
    if (!MtpForward(session, tokens.subspan(1), 0, {}, error_msg, s_.res))
      return false;
  }
  // The host tap rows a prompt chunk or verification batch captured must be
  // readable once Forward returns. A verification batch staged its rows in
  // device memory (a graph node wrote them), so the host copy happens here,
  // after the launch.
  if (session.tap_phase_ == Session::TapPhase::kPrompt) {
    if (!Drain("tap capture", error_msg)) {
      return false;
    }
  } else if (session.tap_phase_ == Session::TapPhase::kVerify) {
    const std::size_t rows = static_cast<std::size_t>(n) * session.tap_width_;
    session.verify_taps_.resize(rows);
    if (rows > 0 &&
        !Check(hipMemcpyAsync(session.verify_taps_.data(), session.tap_device_,
                              rows * sizeof(float), hipMemcpyDeviceToHost,
                              stream_),
               "verification tap readback", error_msg)) {
      return false;
    }
    if (!Drain("tap capture", error_msg)) {
      return false;
    }
  }
  return true;
}

bool Executor::ForwardBody(Session& session, std::uint32_t n,
                           std::uint32_t n_logits, bool download_logits,
                           bool speculative, std::uint32_t start_pos,
                           std::uint32_t first_layer, std::uint32_t end_layer,
                           std::string* error_msg) const {
  const Config& c = config();
  if (first_layer == 0) {
    if (!Check(hipMemcpyAsync(session.control_, control_host_,
                              sizeof(Session::Control), hipMemcpyHostToDevice,
                              stream_),
               "control upload", error_msg) ||
        !Check(hipMemcpyAsync(s_.tokens, tokens_host_, n * sizeof(std::int32_t),
                              hipMemcpyHostToDevice, stream_),
               "token upload", error_msg)) {
      return false;
    }
    EmbedTokens(model_->token_embd().data, SmallType(model_->token_embd().type),
                s_.tokens, s_.res, n, c.hidden_size, 1, stream_);
  }
  TAPTRACE("body start n=%u first=%u end=%u control=%p tokens_host=%p", n,
           first_layer, end_layer, static_cast<void*>(session.control_),
           static_cast<void*>(tokens_host_));
  const TrunkBatch batch{.n = n,
                         .start_pos = start_pos,
                         .speculative = speculative,
                         .control = session.control_};
  // GUFO_LAYER_SYNC drains between layers: with dispatch left asynchronous
  // this bisects the corruption HIP_LAUNCH_BLOCKING used to hide (a racing
  // kernel pair straddling a layer boundary goes quiet when the boundary
  // drains).
  static const bool kLayerSync = getenv("GUFO_LAYER_SYNC") != nullptr;
  for (std::uint32_t il = first_layer; il < end_layer; ++il) {
    if (il < 2 || il + 2 >= end_layer) {
      TAPTRACE("layer %u enter", il);
    }
    if (!session.CheckCancellation(error_msg) ||
        !AttentionHalf(session, il, batch, /*reduce=*/true, error_msg)) {
      return false;
    }
    if (kLayerSync && !Check(hipStreamSynchronize(stream_),
                             "layer sync (attn)", error_msg)) {
      return false;
    }
    if (il < 6) {
      DumpRow("post-attn", s_.block_out, c.hidden_size, stream_);
    }
    AddResidual(s_.res, s_.block_out, n);
    if (!session.CheckCancellation(error_msg) ||
        !FfnHalf(il, n, /*reduce=*/true, error_msg)) {
      return false;
    }
    if (kLayerSync && !Check(hipStreamSynchronize(stream_),
                             "layer sync (ffn)", error_msg)) {
      return false;
    }
    AddResidual(s_.res, s_.block_out, n);
    // The layer output (post-FFN residual) is what a DFlash tap reads.
    if (session.tap_phase_ != Session::TapPhase::kNone &&
        !CaptureTaps(session, il, n, error_msg)) {
      return false;
    }
    if (il < 6) {
      DumpRow("post-ffn", s_.res, c.hidden_size, stream_);
    }
  }
  TAPTRACE("layers done");
  if (end_layer < c.num_layers) {
    return true;
  }
  // Prefill catches up immediately from shared scratch. Only the short tail
  // needed across calls belongs to this session.
  const auto kept = std::min(n, options_.max_speculative);
  if (session.mtp_enabled_ &&
      !Check(hipMemcpyAsync(
                 session.target_hidden_,
                 s_.res + static_cast<std::size_t>(n - kept) * c.hidden_size,
                 static_cast<std::size_t>(kept) * c.hidden_size * sizeof(float),
                 hipMemcpyDeviceToDevice, stream_),
             "hidden keep", error_msg)) {
    return false;
  }
  if (n_logits > 0) {
    TAPTRACE("head enter");
    PrefillPhase head_phase(false);
    // The head norms its tail rows itself.
    const std::size_t skip = static_cast<std::size_t>(n - n_logits);
    RmsNormRows(s_.res + skip * c.hidden_size, model_->output_norm().f32(),
                s_.xn, n_logits, c.hidden_size, 1, c.rms_eps, stream_);
    if (!Dense(model_->output(), s_.xn, s_.logits, n_logits, error_msg)) {
      return false;
    }
    if (download_logits &&
        !Check(hipMemcpyAsync(logits_host_, s_.logits,
                              static_cast<std::size_t>(n_logits) *
                                  c.vocab_size * sizeof(float),
                              hipMemcpyDeviceToHost, stream_),
               "logits download", error_msg)) {
      return false;
    }
  }
  TAPTRACE("body return");
  return true;
}

Executor::Scratch Executor::PairScratch(const Scratch& base) const {
  auto s = base;
  s.res = pair_res_;
  s.block_out = pair_block_out_;
  return s;
}

bool Executor::ForwardPair(Session& session,
                           std::span<const std::int32_t> tokens,
                           std::uint32_t lead, std::uint32_t n_logits,
                           float* logits, std::string* error_msg) const {
  if (!session.tap_layers_.empty()) {
    AssignError(error_msg,
                "target tap capture requires the single-batch prefill path");
    return false;
  }
  // Forward validated the chunk, bumped the mutation epoch and staged the
  // tokens and the first batch's control. What follows computes what two
  // Forward calls of `lead` rows and the rest compute, with the draft
  // block's catch-up between them. Each batch runs every kernel the same
  // way; only their order changes:
  //
  //   A0 B0 | A1 B1 | ... where Ai is batch A's layer i up to its attention
  //   partial, then the sum of that partial and the layer's FFN half.
  //
  // A's partial crosses the link while B computes the same layer, and B's
  // while A computes the next one. B's layer reads A's KV rows of that
  // layer, which A has written by then.
  const Config& c = config();
  const auto n = static_cast<std::uint32_t>(tokens.size());
  const std::uint32_t n_a = lead;
  const std::uint32_t n_b = n - n_a;
  const std::uint32_t start_a = session.position_;
  const std::uint32_t start_b = start_a + n_a;
  control_host_[1] = control_host_[0];
  control_host_[1].position = start_b;

  // Each lane runs its layers as halves: a layer's attention block, then
  // its feed-forward. A half that ends in a partial starts its sum and
  // yields to the other lane.
  struct Lane {
    Scratch scratch;
    TrunkBatch batch;
    const std::int32_t* tokens;
    std::uint32_t part;  ///< the next half to run
    bool pending;        ///< its last half's sum is in flight
  };
  const Scratch base = s_;
  std::array<Lane, 2> lanes{
      Lane{.scratch = base,
           .batch = {.n = n_a,
                     .start_pos = start_a,
                     .speculative = false,
                     .control = session.control_},
           .tokens = s_.tokens,
           .part = 0,
           .pending = false},
      Lane{.scratch = PairScratch(base),
           .batch = {.n = n_b,
                     .start_pos = start_b,
                     .speculative = false,
                     .control = pair_control_},
           .tokens = s_.tokens + n_a,
           .part = 0,
           .pending = false}};
  // Sums finish in the order they started. A failed forward still waits for
  // the exchanges it started, so both ranks' sequences stay aligned.
  std::array<const Lane*, 2> in_flight{};
  std::size_t flying = 0;
  struct Settle {
    const Executor* executor;
    const Scratch& base;
    std::array<const Lane*, 2>& in_flight;
    std::size_t& flying;
    ~Settle() {
      const auto& reduce = executor->options_.split_reduce;
      for (std::size_t i = 0; i < flying; ++i) {
        const Lane& lane = *in_flight[i];
        std::string ignored;
        (void)reduce.finish(lane.scratch.block_out,
                            static_cast<std::size_t>(lane.batch.n) *
                                executor->config().hidden_size * sizeof(float),
                            executor->stream_, &ignored);
      }
      executor->UseScratch(base);
    }
  } settle{this, base, in_flight, flying};
  const auto bytes = [&](const Lane& lane) {
    return static_cast<std::size_t>(lane.batch.n) * c.hidden_size *
           sizeof(float);
  };
  const auto enter = [&](const Lane& lane) { UseScratch(lane.scratch); };
  const std::uint32_t parts = 2 * c.num_layers;
  // Runs the lane's halves until one ends in a partial, whose sum it starts.
  const auto advance = [&](Lane& lane) {
    while (lane.part < parts) {
      const std::uint32_t part = lane.part++;
      const std::uint32_t il = part / 2;
      if (part % 2 == 0) {
        if (!session.CheckCancellation(error_msg) ||
            !AttentionHalf(session, il, lane.batch, /*reduce=*/false,
                           error_msg)) {
          return false;
        }
      } else if (!FfnHalf(il, lane.batch.n, /*reduce=*/false, error_msg)) {
        return false;
      }
      if (!options_.split_reduce.start(s_.block_out, bytes(lane), stream_,
                                       error_msg)) {
        return false;
      }
      in_flight[flying++] = &lane;
      lane.pending = true;
      return true;
    }
    return true;
  };
  // Adds the peer's partial of the lane's last half; every half's sum joins
  // the residual (the attention half's sum must be in res before the FFN
  // half's norm reads it, exactly as the single-batch path adds both).
  const auto finish = [&](Lane& lane) {
    if (flying == 0 || in_flight[0] != &lane) {
      AssignError(error_msg, "prefill pair sums finished out of order");
      return false;
    }
    in_flight[0] = in_flight[1];
    --flying;
    lane.pending = false;
    if (!options_.split_reduce.finish(s_.block_out, bytes(lane), stream_,
                                      error_msg)) {
      return false;
    }
    if ((lane.part - 1) % 2 == 1 && options_.moe_observer &&
        !model_->layers()[(lane.part - 1) / 2].dense_ffn) {
      options_.moe_observer(s_.block_out, bytes(lane), stream_);
    }
    AddResidual(s_.res, s_.block_out, lane.batch.n);
    return true;
  };

  if (!Check(hipMemcpyAsync(session.control_, &control_host_[0],
                            sizeof(Session::Control), hipMemcpyHostToDevice,
                            stream_),
             "control upload", error_msg) ||
      !Check(hipMemcpyAsync(pair_control_, &control_host_[1],
                            sizeof(Session::Control), hipMemcpyHostToDevice,
                            stream_),
             "control upload", error_msg) ||
      !Check(hipMemcpyAsync(s_.tokens, tokens_host_, n * sizeof(std::int32_t),
                            hipMemcpyHostToDevice, stream_),
             "token upload", error_msg)) {
    return false;
  }
  for (Lane& lane : lanes) {
    enter(lane);
    EmbedTokens(model_->token_embd().data, SmallType(model_->token_embd().type),
                lane.tokens, s_.res, lane.batch.n, c.hidden_size, 1, stream_);
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

  // Batch B's logits, as its own Forward would compute them.
  if (n_logits > 0) {
    PrefillPhase head_phase(false);
    enter(lanes[1]);
    const std::size_t skip = static_cast<std::size_t>(n_b - n_logits);
    RmsNormRows(s_.res + skip * c.hidden_size, model_->output_norm().f32(),
                s_.xn, n_logits, c.hidden_size, 1, c.rms_eps, stream_);
    if (!Dense(model_->output(), s_.xn, s_.logits, n_logits, error_msg) ||
        (logits != nullptr &&
         !Check(hipMemcpyAsync(logits_host_, s_.logits,
                               static_cast<std::size_t>(n_logits) *
                                   c.vocab_size * sizeof(float),
                               hipMemcpyDeviceToHost, stream_),
                "logits download", error_msg))) {
      return false;
    }
  }
  // The draft block keeps each batch's final rows in turn: A's for the
  // step that bridges the two batches, then B's.
  const auto keep = [&](const Lane& lane) {
    const std::uint32_t kept = std::min(lane.batch.n, options_.max_speculative);
    return Check(hipMemcpyAsync(
                     session.target_hidden_,
                     lane.scratch.res +
                         static_cast<std::size_t>(lane.batch.n - kept) *
                             c.hidden_size,
                     static_cast<std::size_t>(kept) * c.hidden_size *
                         sizeof(float),
                     hipMemcpyDeviceToDevice, stream_),
                 "hidden keep", error_msg);
  };
  if ((session.mtp_enabled_ && !keep(lanes[0])) ||
      !Drain("forward", error_msg)) {
    return false;
  }
  session.position_ = start_b;
  if (session.mtp_enabled_) {
    // Catch up over A (hidden rows from A's residual), bridge to B's first
    // token from A's kept final row, keep B's rows, then catch up over B.
    PrefillPhase draft_phase(false);
    enter(lanes[0]);
    const std::uint32_t kept_a = std::min(n_a, options_.max_speculative);
    if (!MtpForward(session, tokens.subspan(1, n_a - 1), 0, {}, error_msg,
                    s_.res) ||
        !MtpForward(session, tokens.subspan(n_a, 1),
                    static_cast<std::int32_t>(kept_a - 1), {}, error_msg) ||
        !keep(lanes[1])) {
      return false;
    }
  }
  ++session.mutation_epoch_;
  session.spec_base_ = start_b;
  session.position_ = start_b + n_b;
  if (session.mtp_enabled_ && n_b > 1) {
    if (kEntryTrace) { fprintf(stderr, "ENTRY exec:AttentionHalf\n"); fflush(stderr); }
    PrefillPhase draft_phase(false);
    enter(lanes[1]);
    if (!MtpForward(session, tokens.subspan(n_a + 1), 0, {}, error_msg,
                    s_.res)) {
      return false;
    }
  }
  if (n_logits > 0 && logits != nullptr) {
    // The last MtpForward (or the forward itself) synchronized the stream.
    std::copy_n(logits_host_, static_cast<std::size_t>(n_logits) * c.vocab_size,
                logits);
  }
  return true;
}

bool Executor::AttentionHalf(Session& session, std::uint32_t il,
                             const TrunkBatch& batch, bool reduce,
                             std::string* error_msg) const {
  const Config& c = config();
  const DeviceLayer& l = model_->layers()[il];
  RmsNormRows(s_.res, l.attn_norm.f32(), s_.xn, batch.n, c.hidden_size, 1,
              c.rms_eps, stream_);
  Session::AttentionState& state = session.attention_[il];
  if (!AttentionPart(l, state, s_.xn, s_.block_out, batch.n,
                     &batch.control->position, error_msg,
                     /*projections_ready=*/false,
                     /*project_output=*/true,
                     /*slot_table=*/nullptr, /*slot=*/0,
                     batch.start_pos + batch.n)) {
    if (kEntryTrace) { fprintf(stderr, "ENTRY AttentionHalf\n"); fflush(stderr); }
    return false;
  }
  if (!reduce) {
    return true;
  }
  // An unsplit replicated block belongs to rank zero; peers still ran the
  // projections (their staging side effects keep the ranks in step) and
  // contribute a zero partial.
  if (model_->tp_world_size() > 1 && !l.attn_split &&
      model_->tp_rank() != 0 &&
      !Check(hipMemsetAsync(
                 s_.block_out, 0,
                 static_cast<std::size_t>(batch.n) * c.hidden_size *
                     sizeof(float),
                 stream_),
             "non-owner attention clear", error_msg)) {
    return false;
  }
  if (kEntryTrace) {
    hipStreamSynchronize(stream_);
    const hipError_t e = hipGetLastError();
    fprintf(stderr, "TRACE pre-AttnAllReduce rc=%d\n", static_cast<int>(e));
    fflush(stderr);
    if (e != hipSuccess) {
      AssignError(error_msg, hipGetErrorString(e));
      return false;
    }
  }
  const bool ok = AllReduce(s_.block_out, batch.n, error_msg);
  if (kEntryTrace) {
    hipStreamSynchronize(stream_);
    const hipError_t e = hipGetLastError();
    fprintf(stderr, "TRACE post-AttnAllReduce rc=%d\n", static_cast<int>(e));
    fflush(stderr);
    if (e != hipSuccess) {
      AssignError(error_msg, hipGetErrorString(e));
      return false;
    }
  }
  return ok;
}

bool Executor::FfnHalf(std::uint32_t il, std::uint32_t n_tokens, bool reduce,
                       std::string* error_msg) const {
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
  const DeviceLayer& l = model_->layers()[il];
  RmsNormRows(s_.res, l.ffn_norm.f32(), s_.xn, n_tokens, c.hidden_size, 1,
              c.rms_eps, stream_);
  const bool dump = kDumpOnce && il == 1;
  if (dump) {
    DumpVec("ffn-res-in", s_.res, 4096, stream_);
    DumpVec("ffn-xn", s_.xn, 4096, stream_);
  }
  const bool ok =
      FeedForward(l, s_.xn, s_.block_out, n_tokens, error_msg, reduce);
  if (dump && ok) {
    DumpVec("ffn-out", s_.block_out, 4096, stream_);
    fprintf(stderr, "DUMPED layer1 ffn buffers\n");
    fflush(stderr);
  }
  return ok;
}

void Executor::AddResidual(float* res, const float* block_out,
                           std::uint32_t n_tokens) const {
  AddRowsBroadcast(block_out, res, n_tokens, config().hidden_size, 1, stream_);
}

bool Executor::Rollback(Session& session, std::uint32_t keep,
                        std::string* error_msg, float* logits) const {
  const Config& c = config();
  const std::uint32_t n = session.spec_tokens_;
  if (n == 0 || keep == 0 || keep > n) {
    AssignError(error_msg, "rollback outside the pending speculative batch");
    return false;
  }
  session.spec_tokens_ = 0;
  if (logits != nullptr &&
      !Check(hipMemcpyAsync(
                 logits_host_,
                 VerificationLogits() +
                     static_cast<std::size_t>(keep - 1) * c.vocab_size,
                 c.vocab_size * sizeof(float), hipMemcpyDeviceToHost, stream_),
             "frontier download", error_msg)) {
    return false;
  }
  // KV rows past the kept prefix are stale; the next replay overwrites
  // them before any query can see them.
  session.position_ = session.spec_base_ + keep;
  if (!Check(hipStreamSynchronize(stream_), "rollback", error_msg)) {
    return false;
  }
  if (logits != nullptr) {
    std::copy_n(logits_host_, c.vocab_size, logits);
  }
  return true;
}

bool Executor::Rewind(Session& session, std::uint32_t position,
                      std::string* error_msg) const {
  if (position > session.position_ || position < session.spec_base_) {
    AssignError(error_msg, "rewind outside the pending batch");
    return false;
  }
  // KV rows past the rewound position are stale; the next replay overwrites
  // them before any query can see them.
  session.spec_tokens_ = 0;
  session.position_ = position;
  return true;
}

constexpr std::array<char, 8> kSnapshotMagic{'M', 'I', 'M', 'O', '2', 'S',
                                            'N', '1'};

/// Fixed header ahead of the section bytes. It carries every geometry value
/// the section sizes derive from, so a payload of another artifact or
/// executor configuration is rejected before anything is copied.
struct SnapshotHeader {
  std::array<char, 8> magic;
  std::uint32_t num_layers;
  std::uint32_t nextn_blocks;
  std::uint32_t hidden_size;
  std::uint32_t head_dim_qk;
  std::uint32_t head_dim_v;
  std::uint32_t tp_world_size;
  std::uint32_t has_mtp;
  std::uint32_t position;
  std::uint32_t mtp_position;
  std::uint32_t hidden_rows;
  std::uint64_t payload_bytes;
};
static_assert(std::is_trivially_copyable_v<SnapshotHeader>);

namespace {

SnapshotHeader MakeSnapshotHeader(const Config& c, const DeviceModel& model,
                                  bool has_mtp, const Session& session) {
  SnapshotHeader h{};
  h.magic = kSnapshotMagic;
  h.num_layers = c.num_layers;
  h.nextn_blocks = static_cast<std::uint32_t>(model.nextn_layers().size());
  h.hidden_size = c.hidden_size;
  h.head_dim_qk = c.head_dim_qk;
  h.head_dim_v = c.head_dim_v;
  h.tp_world_size = model.tp_world_size();
  h.has_mtp = has_mtp ? 1 : 0;
  h.position = session.position();
  return h;
}

/// Whether `h` describes this executor's geometry (positions aside).
bool SameGeometry(const SnapshotHeader& h, const SnapshotHeader& mine) {
  return h.magic == mine.magic && h.num_layers == mine.num_layers &&
         h.nextn_blocks == mine.nextn_blocks &&
         h.hidden_size == mine.hidden_size &&
         h.head_dim_qk == mine.head_dim_qk &&
         h.head_dim_v == mine.head_dim_v &&
         h.tp_world_size == mine.tp_world_size && h.has_mtp == mine.has_mtp;
}

}  // namespace

template<typename Visit>
std::uint64_t Executor::WalkSnapshot(const SnapshotHeader& h,
                                     const Session* session,
                                     Visit&& visit) const {
  std::uint64_t offset = sizeof(SnapshotHeader);
  const auto region = [&](void* device, std::uint64_t bytes, const char* what) {
    if (bytes != 0 && !visit(device, offset, bytes, what)) {
      return false;
    }
    offset += bytes;
    return true;
  };
  const Config& c = model_->config();
  // A ring layer only ever reads the window before its frontier again, so its
  // snapshot carries those rows and nothing else: the dead part of the ring
  // made every snapshot hundreds of megabytes (the ring is sized for
  // window + max batch) and, at 64K, seconds of copying. The live rows are
  // [position - window, position) in ring order: at most two contiguous runs
  // when the window wraps.
  const auto emit_cache = [&](const void* base, std::uint64_t row_bytes,
                              std::uint32_t position, bool ring,
                              std::uint32_t ring_rows) -> bool {
    if (row_bytes == 0 || position == 0) {
      return true;
    }
    if (!ring) {
      return region(const_cast<void*>(base), row_bytes * position, "cache");
    }
    const std::uint32_t rows = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(position, c.sliding_window));
    const auto* bytes = static_cast<const std::uint8_t*>(base);
    if (ring_rows == 0) {
      // Sizing walk: nothing is copied, and the window split does not change
      // the byte count, so one run describes it.
      return region(const_cast<void*>(base),
                    std::uint64_t{rows} * row_bytes, "cache window");
    }
    const std::uint32_t start = (position - rows) & (ring_rows - 1);
    const std::uint32_t first = std::min(rows, ring_rows - start);
    if (!region(const_cast<void*>(static_cast<const void*>(
                    bytes + std::uint64_t{start} * row_bytes)),
                std::uint64_t{first} * row_bytes, "cache window")) {
      return false;
    }
    const std::uint32_t second = rows - first;
    if (second != 0 &&
        !region(const_cast<void*>(static_cast<const void*>(bytes)),
                std::uint64_t{second} * row_bytes, "cache window")) {
      return false;
    }
    return true;
  };
  // Per-layer KV widths follow this rank's shares of the heads.
  for (std::uint32_t il = 0; il < h.num_layers; ++il) {
    const DeviceLayer& l = model_->layers()[il];
    const auto* at = session != nullptr ? &session->attention_[il] : nullptr;
    const std::uint32_t ring_rows = at != nullptr ? at->swa_rows : 0;
    if (!emit_cache(at != nullptr ? at->k_cache : nullptr,
                    std::uint64_t{l.kv_heads} * c.head_dim_qk * sizeof(__half),
                    h.position, l.swa, ring_rows) ||
        !emit_cache(at != nullptr ? at->v_cache : nullptr,
                    std::uint64_t{l.kv_heads} * c.head_dim_v * sizeof(__half),
                    h.position, l.swa, ring_rows)) {
      return 0;
    }
  }
  if (h.has_mtp != 0) {
    // The draft chain runs the first block; the dormant ones hold position
    // zero and contribute no bytes.
    for (std::uint32_t j = 0; j < h.nextn_blocks; ++j) {
      const DeviceLayer& l = model_->nextn_layers()[j];
      const std::uint32_t position = j == 0 ? h.mtp_position : 0;
      const auto* mtp =
          session != nullptr ? &session->mtp_blocks_[j] : nullptr;
      const std::uint32_t ring_rows = mtp != nullptr ? mtp->swa_rows : 0;
      if (!emit_cache(mtp != nullptr ? mtp->k_cache : nullptr,
                      std::uint64_t{l.kv_heads} * c.head_dim_qk * sizeof(__half),
                      position, l.swa, ring_rows) ||
          !emit_cache(mtp != nullptr ? mtp->v_cache : nullptr,
                      std::uint64_t{l.kv_heads} * c.head_dim_v * sizeof(__half),
                      position, l.swa, ring_rows) ||
          !region(mtp != nullptr ? mtp->h : nullptr,
                  std::uint64_t{h.hidden_size} * sizeof(float),
                  "draft residual")) {
        return 0;
      }
    }
    if (!region(session != nullptr ? session->target_hidden_ : nullptr,
                std::uint64_t{h.hidden_rows} * h.hidden_size * sizeof(float),
                "kept trunk rows")) {
      return 0;
    }
  }
  return offset;
}

std::uint64_t Executor::SnapshotBytes(const Session& session,
                                      std::uint32_t hidden_rows) const {
  SnapshotHeader h = MakeSnapshotHeader(config(), *model_,
                                        session.mtp_enabled_, session);
  h.mtp_position = session.mtp_().position;
  h.hidden_rows = hidden_rows;
  return WalkSnapshot(
      h, nullptr,
      [](void*, std::uint64_t, std::uint64_t, const char*) { return true; });
}

bool Executor::SaveSnapshot(const Session& session, std::uint32_t hidden_rows,
                            std::span<std::uint8_t> payload,
                            std::string* error_msg) const {
  if (session.owner_ != this) {
    AssignError(error_msg, "session belongs to another executor");
    return false;
  }
  if (session.spec_tokens_ != 0) {
    AssignError(error_msg,
                "snapshot with a pending speculative batch; roll back first");
    return false;
  }
  if (hidden_rows > session.position_ ||
      hidden_rows > options_.max_speculative) {
    AssignError(error_msg, "kept trunk rows exceed the position or batch");
    return false;
  }
  SnapshotHeader h = MakeSnapshotHeader(config(), *model_,
                                        session.mtp_enabled_, session);
  h.mtp_position = session.mtp_().position;
  h.hidden_rows = hidden_rows;
  h.payload_bytes = SnapshotBytes(session, hidden_rows);
  if (payload.size() != h.payload_bytes) {
    AssignError(error_msg, "snapshot buffer size does not match the payload");
    return false;
  }
  gufo::hip::SnapshotTransfer transfer;
  std::memcpy(payload.data(), &h, sizeof(h));
  const bool walked =
      WalkSnapshot(h, &session,
                   [&](void* device, std::uint64_t offset,
                       std::uint64_t bytes, const char*) {
                     if (!session.CheckCancellation(error_msg))
                       return false;
                     transfer.Copy(payload.data() + offset, device, bytes);
                     return true;
                   }) != 0;
  // One wait for the whole walk; a queued copy's error only surfaces here.
  return transfer.Finish(error_msg) && walked;
}

std::size_t Executor::SnapshotHeaderBytes() noexcept {
  return sizeof(SnapshotHeader);
}

bool Executor::VisitSnapshotLayout(std::uint32_t position,
                                   std::uint32_t hidden_rows,
                                   std::uint32_t mtp_position,
                                   bool has_mtp,
                                   const SnapshotRegionVisitor& visit) const {
  if (!visit || position == 0 || hidden_rows > position ||
      hidden_rows > options_.max_speculative || mtp_position > position) {
    return false;
  }
  SnapshotHeader h{};
  h.magic = kSnapshotMagic;
  h.num_layers = config().num_layers;
  h.nextn_blocks = static_cast<std::uint32_t>(model_->nextn_layers().size());
  h.hidden_size = config().hidden_size;
  h.head_dim_qk = config().head_dim_qk;
  h.head_dim_v = config().head_dim_v;
  h.tp_world_size = model_->tp_world_size();
  h.has_mtp = has_mtp ? 1 : 0;
  h.position = position;
  h.hidden_rows = hidden_rows;
  h.mtp_position = mtp_position;
  if (!visit(nullptr, 0, sizeof(SnapshotHeader), "payload header")) {
    return false;
  }
  return WalkSnapshot(h, nullptr, visit) != 0;
}

bool Executor::VisitSnapshotRegions(const Session& session,
                                    std::uint32_t hidden_rows,
                                    const SnapshotRegionVisitor& visit) const {
  if (!visit || session.owner_ != this || session.spec_tokens_ != 0 ||
      hidden_rows > session.position_ ||
      hidden_rows > options_.max_speculative) {
    return false;
  }
  SnapshotHeader h =
      MakeSnapshotHeader(config(), *model_, session.mtp_enabled_, session);
  h.mtp_position = session.mtp_().position;
  h.hidden_rows = hidden_rows;
  return WalkSnapshot(h, &session, visit) != 0;
}

bool Executor::RestoreSnapshot(Session& session,
                               std::span<const std::uint8_t> payload,
                               SnapshotInfo* info, std::string* error_msg,
                               std::uint32_t next_drafts) const {
  (void)next_drafts;  // no rollback arena to retain
  if (session.owner_ != this) {
    AssignError(error_msg, "session belongs to another executor");
    return false;
  }
  if (payload.size() < sizeof(SnapshotHeader)) {
    AssignError(error_msg, "snapshot payload is truncated");
    return false;
  }
  SnapshotHeader h{};
  std::memcpy(&h, payload.data(), sizeof(h));
  const SnapshotHeader mine =
      MakeSnapshotHeader(config(), *model_, session.mtp_enabled_, session);
  if (!SameGeometry(h, mine)) {
    AssignError(error_msg, "snapshot was taken with another model geometry");
    return false;
  }
  if (h.position == 0 || h.position > session.max_context_ ||
      h.mtp_position > h.position || h.hidden_rows > h.position ||
      h.hidden_rows > options_.max_speculative ||
      (h.has_mtp == 0 && (h.mtp_position != 0 || h.hidden_rows != 0))) {
    AssignError(error_msg, "snapshot positions do not fit this session");
    return false;
  }
  if (h.payload_bytes != payload.size() ||
      WalkSnapshot(h, nullptr,
                   [](void*, std::uint64_t, std::uint64_t, const char*) {
                     return true;
                   }) != payload.size()) {
    AssignError(error_msg, "snapshot payload size does not match its header");
    return false;
  }
  // The session's queued work targets buffers the copies overwrite.
  if (!Check(hipStreamSynchronize(stream_), "restore drain", error_msg)) {
    return false;
  }
  if (WalkSnapshot(h, &session,
                   [&](void* device, std::uint64_t offset,
                       std::uint64_t bytes, const char* what) {
    if (kEntryTrace) { fprintf(stderr, "ENTRY exec:GreedyMtpPredictions\n"); fflush(stderr); }
                     return session.CheckCancellation(error_msg) &&
                            Check(hipMemcpy(device, payload.data() + offset,
                                            bytes, hipMemcpyHostToDevice),
                                  what, error_msg);
                   }) == 0) {
    session.Reset();
    return false;
  }
  session.position_ = h.position;
  for (auto& block : session.mtp_blocks_) {
    block.position = 0;
  }
  session.mtp_().position = h.mtp_position;
  session.spec_base_ = h.position;
  session.spec_tokens_ = 0;
  if (KvInt8Enabled()) {
    // Snapshots carry the authoritative f16 rows only, so a restored
    // session's int8 twin is rebuilt here instead of being serialized.
    const Config& c = config();
    for (std::uint32_t il = 0; il < session.attention_.size(); ++il) {
      auto& at = session.attention_[il];
      if (at.k_cache_q == nullptr) {
        continue;
      }
      const DeviceLayer& l = model_->layers()[il];
      const auto rows = l.swa ? at.swa_rows
                              : static_cast<std::uint32_t>(h.position);
      KvQuantizeFromHalf(at.k_cache, at.k_cache_q, at.k_scales, rows,
                         l.kv_heads * c.head_dim_qk, stream_);
      KvQuantizeFromHalf(at.v_cache, at.v_cache_q, at.v_scales, rows,
                         l.kv_heads * c.head_dim_v, stream_);
    }
  }
  if (info != nullptr) {
    info->position = h.position;
    info->hidden_rows = h.hidden_rows;
  }
  return true;
}

void Executor::RouterSelfCheck(const __half* router_f16,
                               std::uint32_t n_tokens,
                               std::uint32_t splits) const {
  const Config& c = config();
  const std::uint32_t m = c.num_experts;
  const std::uint32_t count = n_tokens * m;
  RouterGemmWmma(router_f16, static_cast<const __half*>(s_.x_half),
                 s_.router_alt, n_tokens, m, c.hidden_size, stream_);
  std::vector<float> split(count);
  std::vector<float> tiled(count);
  if (hipMemcpy(split.data(), s_.router, count * sizeof(float),
                hipMemcpyDeviceToHost) != hipSuccess ||
      hipMemcpy(tiled.data(), s_.router_alt, count * sizeof(float),
                hipMemcpyDeviceToHost) != hipSuccess) {
    fprintf(stderr, "ROUTERCHK readback failed: %s\n",
            hipGetErrorString(hipGetLastError()));
    return;
  }
  float max_delta = 0.0F;
  std::uint32_t flips = 0;
  std::vector<std::uint32_t> order(m);
  for (std::uint32_t t = 0; t < n_tokens; ++t) {
    const float* a = split.data() + static_cast<std::size_t>(t) * m;
    const float* b = tiled.data() + static_cast<std::size_t>(t) * m;
    for (std::uint32_t e = 0; e < m; ++e) {
      max_delta = std::max(max_delta, std::fabs(a[e] - b[e]));
    }
    const auto top8 = [&](const float* row) {
      std::iota(order.begin(), order.end(), 0U);
      std::partial_sort(
          order.begin(), order.begin() + 8, order.end(),
          [&](std::uint32_t x, std::uint32_t y) {
            return row[x] != row[y] ? row[x] > row[y] : x < y;
          });
      return std::vector<std::uint32_t>(order.begin(), order.begin() + 8);
    };
    if (top8(a) != top8(b)) {
      ++flips;
    }
  }
  fprintf(stderr,
          "ROUTERCHK rows=%u splits=%u max|d|=%.6g top8_flips=%u/%u\n",
          n_tokens, splits, static_cast<double>(max_delta), flips, n_tokens);
  fflush(stderr);
}

bool Executor::GreedyMtpPredictions(std::span<ArgmaxCandidate> predictions,
                                    std::string* error_msg) const {
  const auto rows = predictions.size();
  if (rows == 0 || rows > options_.max_logit_rows || s_.mtp_ids == nullptr) {
    AssignError(error_msg, "invalid greedy MTP verification request");
    return false;
  }
  // Proposal selection has finished. The ID buffers and argmax scratch are
  // reusable until the next draft head overwrites them.
  const auto vocab = config().vocab_size;
  auto* ids = reinterpret_cast<std::int32_t*>(s_.mtp_scratch_ids);
  for (std::size_t r = 0; r < rows; ++r) {
    if (kEntryTrace) { fprintf(stderr, "ENTRY exec:MtpForward\n"); fflush(stderr); }
    Argmax(VerificationLogits() + r * vocab, s_.mtp_argmax, ids + r, 1, vocab,
           stream_);
  }
  GatherArgmaxCandidates(VerificationLogits(), s_.mtp_scratch_ids,
                         s_.mtp_argmax, static_cast<std::uint32_t>(rows), vocab,
                         stream_);
  return Check(hipMemcpyAsync(predictions.data(), s_.mtp_argmax,
                              predictions.size_bytes(), hipMemcpyDeviceToHost,
                              stream_),
               "greedy MTP predictions download", error_msg) &&
         Check(hipStreamSynchronize(stream_), "greedy MTP verification",
               error_msg);
}

bool Executor::MtpForward(Session& session,
                          std::span<const std::int32_t> tokens,
                          std::int32_t hidden_row, MtpOutput output,
                          std::string* error_msg,
                          const float* hidden_source) const {
  const PhaseScope _ps(&kPhases.mtp_ms, &kPhases.mtp_n);
  const auto n = static_cast<std::uint32_t>(tokens.size());
  if (session.owner_ != this ||
      std::any_of(tokens.begin(), tokens.end(), [&](auto t) {
        return t < 0 || static_cast<std::uint32_t>(t) >= config().vocab_size;
      })) {
    AssignError(error_msg, "invalid MTP session or token");
    return false;
  }
  if (!session.CheckCancellation(error_msg))
    return false;
  if (!session.mtp_enabled_) {
    AssignError(error_msg, "no MTP block loaded");
    return false;
  }
  if (n == 0 || n > options_.max_batch || (hidden_row < 0 && n != 1) ||
      (hidden_row >= 0 &&
       static_cast<std::uint32_t>(hidden_row) + n >
           (hidden_source ? options_.max_batch : options_.max_speculative))) {
    AssignError(error_msg, "MTP batch outside the kept hidden rows");
    return false;
  }
  const std::uint32_t pos = session.mtp_().position;
  if (pos + n > session.max_context_) {
    AssignError(error_msg, "MTP context is full");
    return false;
  }
  ++session.mutation_epoch_;
  std::copy(tokens.begin(), tokens.end(), tokens_host_);
  control_host_->position = session.position_;
  control_host_->mtp_position = pos;
  control_host_->hidden_row = hidden_row;
  const bool graph = hidden_source == nullptr && n <= kVecBatch;
  const std::uint64_t key =
      static_cast<std::uint64_t>(n) |
      (static_cast<std::uint64_t>(hidden_row < 0) << 32) |
      (std::uint64_t{1} << 40) |
      (std::uint64_t{output.token != nullptr} << 41) |
      (std::uint64_t{output.candidates != nullptr} << 43);
  const auto body = [&] {
    return MtpBody(session, n, pos, output.token != nullptr,
                   output.candidates != nullptr, error_msg, hidden_source);
  };
  TAPTRACE("forward validated, dispatching n=%u", n);
  if (!Run(session, key, graph, body, error_msg)) {
    return false;
  }
  TAPTRACE("forward body done");
  if (output.token != nullptr) {
    *output.token = *mtp_token_host_;
  }
  if (output.candidates != nullptr) {
    *output.candidates = *mtp_candidates_host_;
  }
  session.mtp_().position = pos + n;
  return true;
}

bool Executor::MtpBody(Session& session, std::uint32_t n, std::uint32_t pos,
                       bool token, bool candidates, std::string* error_msg,
                       const float* hidden_source) const {
  const Config& c = config();
  const DeviceLayer& block = model_->nextn_layers().front();
  if (!Check(hipMemcpyAsync(session.control_, control_host_,
                            sizeof(Session::Control), hipMemcpyHostToDevice,
                            stream_),
                 "control upload", error_msg) ||
      !Check(hipMemcpyAsync(s_.tokens, tokens_host_, n * sizeof(std::int32_t),
                            hipMemcpyHostToDevice, stream_),
             "token upload", error_msg)) {
    return false;
  }
  // e = norm(embed(tokens)): the embedding table is private when the
  // sidecar carries one, else the trunk's.
  const DeviceTensor& table = !block.nextn_embed.empty() ? block.nextn_embed
                                                         : model_->token_embd();
  EmbedTokens(table.data, SmallType(table.type), s_.tokens, s_.xn, n,
              c.hidden_size, 1, stream_);
  RmsNormRows(s_.xn, block.nextn_enorm.f32(), s_.xn, n, c.hidden_size, 1,
              c.rms_eps, stream_);
  MtpHidden(hidden_source ? hidden_source : session.target_hidden_,
            session.mtp_().h, &session.control_->hidden_row, s_.mtp_h, n,
            c.hidden_size, stream_);
  RmsNormRows(s_.mtp_h, block.nextn_hnorm.f32(), s_.mtp_h, n, c.hidden_size, 1,
              c.rms_eps, stream_);
  // eh_proj reads [e_norm | h_norm]; the two halves gather into one row via
  // strided device copies (there is no concat kernel).
  if (!Check(hipMemcpy2DAsync(s_.mtp_eh, 2 * c.hidden_size * sizeof(float),
                              s_.xn, c.hidden_size * sizeof(float),
                              c.hidden_size * sizeof(float), n,
                              hipMemcpyDeviceToDevice, stream_),
             "draft embedding concat", error_msg) ||
      !Check(hipMemcpy2DAsync(
                 s_.mtp_eh + c.hidden_size, 2 * c.hidden_size * sizeof(float),
                 s_.mtp_h, c.hidden_size * sizeof(float),
                 c.hidden_size * sizeof(float), n, hipMemcpyDeviceToDevice,
                 stream_),
             "draft hidden concat", error_msg) ||
      !Dense(block.nextn_eh_proj, s_.mtp_eh, s_.res, n, error_msg)) {
    return false;
  }
  // One decoder block at the draft's own position, over its own caches.
  // The block's weights are whole on every rank, so its attention needs no
  // boundary sum; its dense FFN stays split under TP and reduces.
  Session::AttentionState attn;
  attn.k_cache = session.mtp_().k_cache;
  attn.v_cache = session.mtp_().v_cache;
  attn.swa_rows = session.mtp_().swa_rows;
  RmsNormRows(s_.res, block.attn_norm.f32(), s_.xn, n, c.hidden_size, 1,
              c.rms_eps, stream_);
  if (!AttentionPart(block, attn, s_.xn, s_.block_out, n,
                     &session.control_->mtp_position, error_msg,
                     /*projections_ready=*/false,
                     /*project_output=*/true,
                     /*slot_table=*/nullptr, /*slot=*/0,
                     session.mtp_().position + n)) {
    return false;
  }
  AddResidual(s_.res, s_.block_out, n);
  RmsNormRows(s_.res, block.ffn_norm.f32(), s_.xn, n, c.hidden_size, 1,
              c.rms_eps, stream_);
  if (!FeedForward(block, s_.xn, s_.block_out, n, error_msg,
                   model_->tp_world_size() > 1)) {
    return false;
  }
  AddResidual(s_.res, s_.block_out, n);
  const float* last = s_.res + static_cast<std::size_t>(n - 1) * c.hidden_size;
  if (!Check(hipMemcpyAsync(session.mtp_().h, last, c.hidden_size * sizeof(float),
                            hipMemcpyDeviceToDevice, stream_),
             "MTP hidden carry", error_msg)) {
    return false;
  }
  return (!token && !candidates) ||
         MtpHead(block, s_.res, n - 1, token, candidates, error_msg);
}

}  // namespace gufo::models::mimo2::rocm
