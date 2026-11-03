#include "src/models/mimo2/kernels/rocm/device_model.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "src/core/hip/weight_upload.hpp"
#include "src/core/quant/ggml_dequant.hpp"

namespace gufo::models::mimo2::rocm {
namespace {

/// The quantized GEMM tier reads whole 256-element k-iterations, so a row
/// whose length is only a multiple of 32 over-reads into the next row and,
/// on the last row, past the tensor. Every upload carries this tail so the
/// over-read stays inside the allocation (the extra bytes meet zeroed
/// activation padding and contribute nothing).
constexpr std::size_t kTailMargin = 4096;

struct Uploader {
  hip::WeightUpload& stager;
  std::vector<void*>& allocations;
  std::size_t& bytes;
  std::size_t& max_half_cols;
  std::size_t& max_q8_cols;
  std::string* error;
  bool ok{true};
  std::uint32_t shard_base{0};
  const distributed::TpPartition* partition{nullptr};
  const Config* config{nullptr};

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  DeviceTensor CopyRange(const TensorRef& t, std::uint64_t relative_offset,
                         std::size_t size, std::uint32_t experts) {
    DeviceTensor d;
    if (t.empty() || !ok) {
      return d;
    }
    const std::size_t tensor_bytes = t.SizeBytes();
    if (size > tensor_bytes || relative_offset > tensor_bytes - size ||
        size > std::numeric_limits<std::size_t>::max() - kTailMargin ||
        t.cols > std::numeric_limits<std::uint32_t>::max() ||
        t.rows > std::numeric_limits<std::uint32_t>::max()) {
      Fail("weight range geometry is invalid for " + std::string(t.name));
      return d;
    }
    if (relative_offset >
        std::numeric_limits<std::uint64_t>::max() - t.file_offset) {
      Fail("weight range offset overflow for " + std::string(t.name));
      return d;
    }
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for " + std::string(t.name) + " (" +
           std::to_string(size) + " bytes)");
      return d;
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    if (!stager.Copy(shard_base + t.shard, t.file_offset + relative_offset,
                     size, ptr, error)) {
      Fail("upload failed for " + std::string(t.name) +
           (error != nullptr ? ": " + *error : std::string()));
      return d;
    }
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0, kTailMargin,
                         nullptr);
    d.data = ptr;
    d.type = t.type;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = static_cast<std::uint32_t>(t.rows);
    d.experts = experts;
    if (t.type == core::GgmlType::kBF16 || t.type == core::GgmlType::kF16) {
      max_half_cols = std::max<std::size_t>(max_half_cols, t.cols);
    }
    if (t.type == core::GgmlType::kQ8_0 && experts == 1) {
      max_q8_cols = std::max<std::size_t>(max_q8_cols, t.cols);
    }
    return d;
  }

  DeviceTensor Copy(const TensorRef& t) {
    if (t.empty() || !ok) {
      return {};
    }
    return CopyRange(t, 0, t.SizeBytes(),
                     static_cast<std::uint32_t>(t.experts));
  }

  /// A gate or up projection: this rank's rows of every expert. Each share is
  /// a contiguous range of the file, so only it is read.
  DeviceTensor CopyExpertRows(const TensorRef& t) {
    if (t.empty() || !ok || partition == nullptr) {
      return Copy(t);
    }
    const std::size_t row_bytes = t.RowBytes();
    if (t.rows != partition->expert_ff || row_bytes == 0 ||
        t.experts > std::numeric_limits<std::uint32_t>::max() ||
        t.cols > std::numeric_limits<std::uint32_t>::max()) {
      Fail("routed tensor does not match the TP split: " + std::string(t.name));
      return {};
    }
    const std::size_t share = partition->ff_count * row_bytes;
    const std::size_t size = share * t.experts;
    void* ptr = nullptr;
    if (hipMalloc(&ptr, size + kTailMargin) != hipSuccess) {
      Fail("hipMalloc failed for " + std::string(t.name) + " (" +
           std::to_string(size) + " bytes)");
      return {};
    }
    allocations.push_back(ptr);
    bytes += size + kTailMargin;
    for (std::uint64_t e = 0; e < t.experts; ++e) {
      const std::uint64_t first_row = e * t.rows + partition->ff_begin;
      if (!stager.Copy(shard_base + t.shard,
                       t.file_offset + first_row * row_bytes, share,
                       static_cast<std::uint8_t*>(ptr) + e * share, error)) {
        Fail("upload failed for " + std::string(t.name) +
             (error != nullptr ? ": " + *error : std::string()));
        return {};
      }
    }
    (void)hipMemsetAsync(static_cast<std::uint8_t*>(ptr) + size, 0, kTailMargin,
                         nullptr);
    DeviceTensor d;
    d.data = ptr;
    d.type = t.type;
    d.cols = static_cast<std::uint32_t>(t.cols);
    d.rows = partition->ff_count;
    d.experts = static_cast<std::uint32_t>(t.experts);
    return d;
  }

  /// A down projection: this rank's columns of every expert's rows.
  DeviceTensor CopyExpertColumns(const TensorRef& t) {
    if (t.empty() || !ok || partition == nullptr) {
      return Copy(t);
    }
    if (t.cols != partition->expert_ff) {
      Fail("routed tensor does not match the TP split: " + std::string(t.name));
      return {};
    }
    if (SpanBytes(t.type, partition->ff_count) == 0) {
      Fail("routed tensor share is not whole quantization blocks: " +
           std::string(t.name) + ": " + std::to_string(partition->ff_count) +
           " columns must be a multiple "
           "of the quantization block size (32 for Q8_0, 64 for MXFP4)");
      return {};
    }
    return CopyColumns(t, partition->ff_begin, partition->ff_count);
  }

  /// Bytes of `elements` consecutive values of one row, or zero when the
  /// count is not a whole number of quantization blocks.
  static std::size_t SpanBytes(core::GgmlType type, std::size_t elements) {
    switch (type) {
      case core::GgmlType::kF32:
        return elements * 4;
      case core::GgmlType::kF16:
      case core::GgmlType::kBF16:
        return elements * 2;
      default:
        return gufo::quant::QuantizedRowBytes(type, elements);
    }
  }

  /// Rows [begin, begin + count) of a matrix (of every stacked expert).
  DeviceTensor CopyRows(const TensorRef& t, std::uint32_t begin,
                        std::uint32_t count) {
    if (t.empty() || !ok) {
      return {};
    }
    const std::size_t row_bytes = t.RowBytes();
    DeviceTensor d = CopyRange(t, static_cast<std::uint64_t>(begin) * row_bytes,
                               static_cast<std::size_t>(count) * row_bytes,
                               static_cast<std::uint32_t>(t.experts));
    d.rows = count;
    return d;
  }

  /// Columns [begin, begin + count) of every row, split on a
  /// quantization-block boundary without dequantizing or changing any
  /// weight: the whole tensor lands first, the column range gathers into a
  /// fresh allocation, and the staging copy is released.
  DeviceTensor CopyColumns(const TensorRef& t, std::uint32_t begin,
                           std::uint32_t count) {
    const std::size_t row_bytes = t.RowBytes();
    const std::size_t begin_bytes = begin == 0 ? 0 : SpanBytes(t.type, begin);
    const std::size_t span = SpanBytes(t.type, count);
    if (begin_bytes == 0 && begin != 0) {
      Fail("column split start is not a whole quantization block: " +
           std::string(t.name));
      return {};
    }
    DeviceTensor full = Copy(t);
    if (!ok || !stager.Finish(error)) {
      Fail("column split upload failed for " + std::string(t.name));
      return {};
    }
    DeviceTensor d = full;
    d.cols = count;
    const std::size_t rows = t.rows * t.experts;
    const std::size_t part_bytes = span * rows;
    if (hipMalloc(&d.data, part_bytes + kTailMargin) != hipSuccess) {
      Fail("column split allocation failed for " + std::string(t.name));
      return {};
    }
    allocations.push_back(d.data);
    bytes += part_bytes + kTailMargin;
    if (hipMemcpy2D(d.data, span,
                    static_cast<const std::uint8_t*>(full.data) + begin_bytes,
                    row_bytes, span, rows,
                    hipMemcpyDeviceToDevice) != hipSuccess ||
        hipMemset(static_cast<std::uint8_t*>(d.data) + part_bytes, 0,
                  kTailMargin) != hipSuccess) {
      Fail("column split copy failed for " + std::string(t.name));
      return {};
    }
    std::erase(allocations, full.data);
    (void)hipFree(full.data);
    bytes -= t.SizeBytes() + kTailMargin;
    return d;
  }

  /// This rank's share of one attention block. Query head h reads KV head
  /// h / (heads / kv_heads), so each rank keeps whole query groups: its
  /// kv_heads/world KV heads and the query heads they serve. An odd KV
  /// count leaves the block whole (attn_split false); the layer boundary
  /// then zeroes a peer's output before the sum.
  void SplitAttention(const LayerWeights& l, const Config& c,
                      std::uint32_t il, DeviceLayer& d) {
    const std::uint64_t kv = c.kv_heads[il];
    const std::uint64_t world = partition->world_size;
    const std::uint64_t kv_local = kv / world;
    const std::uint64_t q_local = kv_local * (c.num_heads / kv);
    const std::uint64_t kv_first = kv_local * partition->rank;
    const std::uint64_t q_first = q_local * partition->rank;
    d.attn_split = true;
    d.q_heads = static_cast<std::uint32_t>(q_local);
    d.kv_heads = static_cast<std::uint32_t>(kv_local);
    if (!l.attn_qkv.empty()) {
      // Rows of the fused matrix are [Q | K | V]; the three blocks of this
      // rank's heads stay contiguous inside their section.
      const std::uint64_t q_rows = static_cast<std::uint64_t>(c.num_heads) *
                                   c.head_dim_qk;
      const std::uint64_t k_rows = kv * c.head_dim_qk;
      d.attn_q = CopyRows(l.attn_qkv, static_cast<std::uint32_t>(q_first *
                                                                 c.head_dim_qk),
                          static_cast<std::uint32_t>(q_local * c.head_dim_qk));
      d.attn_k = CopyRows(
          l.attn_qkv,
          static_cast<std::uint32_t>(q_rows + kv_first * c.head_dim_qk),
          static_cast<std::uint32_t>(kv_local * c.head_dim_qk));
      d.attn_v = CopyRows(
          l.attn_qkv,
          static_cast<std::uint32_t>(q_rows + k_rows + kv_first *
                                                        c.head_dim_v),
          static_cast<std::uint32_t>(kv_local * c.head_dim_v));
    } else {
      d.attn_q = CopyRows(l.attn_q,
                          static_cast<std::uint32_t>(q_first * c.head_dim_qk),
                          static_cast<std::uint32_t>(q_local * c.head_dim_qk));
      d.attn_k = CopyRows(l.attn_k,
                          static_cast<std::uint32_t>(kv_first * c.head_dim_qk),
                          static_cast<std::uint32_t>(kv_local * c.head_dim_qk));
      d.attn_v = CopyRows(l.attn_v,
                          static_cast<std::uint32_t>(kv_first * c.head_dim_v),
                          static_cast<std::uint32_t>(kv_local * c.head_dim_v));
    }
    // The output projection reads the context of this rank's query heads:
    // their slice of the value-width input columns.
    d.attn_out =
        CopyColumns(l.attn_out, static_cast<std::uint32_t>(q_first *
                                                           c.head_dim_v),
                    static_cast<std::uint32_t>(q_local * c.head_dim_v));
    if (!l.attn_sinks.empty()) {
      d.attn_sinks = Copy(l.attn_sinks);
      // One float per query head; advance to this rank's first head.
      d.attn_sinks.data = static_cast<std::uint8_t*>(d.attn_sinks.data) +
                          q_first * sizeof(float);
      d.attn_sinks.cols = d.q_heads;
    }
  }

  /// Splits a dense FFN's intermediate rows across ranks when every piece
  /// falls on a block boundary; returns false to keep it whole.
  bool SplitDenseFfn(const LayerWeights& l, DeviceLayer& d) {
    if (partition == nullptr || !partition->distributed() ||
        l.ffn_gate.empty() || l.ffn_up.empty() || l.ffn_down.empty()) {
      return false;
    }
    const std::uint64_t ff = l.ffn_gate.rows;
    const std::uint32_t world = partition->world_size;
    if (ff == 0 || ff % world != 0 || l.ffn_up.rows != ff ||
        l.ffn_down.cols != ff) {
      return false;
    }
    const auto share = static_cast<std::uint32_t>(ff / world);
    const auto begin = share * partition->rank;
    if (SpanBytes(l.ffn_down.type, share) == 0) {
      return false;
    }
    d.ffn_gate = CopyRows(l.ffn_gate, begin, share);
    d.ffn_up = CopyRows(l.ffn_up, begin, share);
    d.ffn_down = CopyColumns(l.ffn_down, begin, share);
    d.ffn_split = true;
    return true;
  }

  /// Uploads one block. `il` indexes the geometry tables (per-layer KV
  /// heads and window) whether the block is a trunk layer or a NextN one.
  /// `split` divides the attention heads across ranks when the per-layer KV
  /// count divides evenly; the draft blocks are never split.
  DeviceLayer Layer(const LayerWeights& l, const Config& c,
                    std::uint32_t il, bool split) {
    DeviceLayer d;
    d.dense_ffn = l.dense_ffn;
    d.swa = c.LayerIsSwa(il);
    d.rope_theta = c.RopeThetaFor(il);
    d.attn_norm = Copy(l.attn_norm);
    d.ffn_norm = Copy(l.ffn_norm);
    const bool split_heads =
        split && c.kv_heads[il] % partition->world_size == 0;
    if (!split_heads) {
      d.q_heads = c.num_heads;
      d.kv_heads = c.kv_heads[il];
      if (!l.attn_qkv.empty()) {
        // Split the fused [Q | K | V] rows once, so the executor projects
        // the three matrices separately on every path.
        const std::uint64_t q_rows =
            static_cast<std::uint64_t>(c.num_heads) * c.head_dim_qk;
        const std::uint64_t k_rows =
            static_cast<std::uint64_t>(c.kv_heads[il]) * c.head_dim_qk;
        d.attn_q = CopyRows(l.attn_qkv, 0,
                            static_cast<std::uint32_t>(q_rows));
        d.attn_k = CopyRows(l.attn_qkv, static_cast<std::uint32_t>(q_rows),
                            static_cast<std::uint32_t>(k_rows));
        d.attn_v = CopyRows(
            l.attn_qkv, static_cast<std::uint32_t>(q_rows + k_rows),
            static_cast<std::uint32_t>(static_cast<std::uint64_t>(
                                           c.kv_heads[il]) *
                                       c.head_dim_v));
      } else {
        d.attn_q = Copy(l.attn_q);
        d.attn_k = Copy(l.attn_k);
        d.attn_v = Copy(l.attn_v);
      }
      d.attn_out = Copy(l.attn_out);
      if (!l.attn_sinks.empty()) {
        d.attn_sinks = Copy(l.attn_sinks);
      }
    } else {
      SplitAttention(l, c, il, d);
    }
    if (d.attn_out.cols != d.q_heads * c.head_dim_v) {
      Fail("attn_output expects " +
           std::to_string(static_cast<std::uint64_t>(d.q_heads) *
                          c.head_dim_v) +
           " input columns (its query heads at the value width) but the "
           "bound tensor has " +
           std::to_string(d.attn_out.cols));
      return d;
    }
    if (l.dense_ffn) {
      if (!SplitDenseFfn(l, d)) {
        d.ffn_gate = Copy(l.ffn_gate);
        d.ffn_up = Copy(l.ffn_up);
        d.ffn_down = Copy(l.ffn_down);
      }
    } else {
      d.router = Copy(l.router);
      d.router_bias = Copy(l.router_bias);
      d.ffn_gate_exps = CopyExpertRows(l.ffn_gate_exps);
      d.ffn_up_exps = CopyExpertRows(l.ffn_up_exps);
      d.ffn_down_exps = CopyExpertColumns(l.ffn_down_exps);
    }
    d.nextn_eh_proj = Copy(l.nextn_eh_proj);
    d.nextn_enorm = Copy(l.nextn_enorm);
    d.nextn_hnorm = Copy(l.nextn_hnorm);
    d.nextn_embed = Copy(l.nextn_embed_tokens);
    d.nextn_head_head = Copy(l.nextn_shared_head_head);
    d.nextn_head_norm = Copy(l.nextn_shared_head_norm);
    d.layer_out_norm = Copy(l.layer_out_norm);
    return d;
  }
};

/// Replaces the dense attention projections, the lm head, the MTP draft
/// network's own dense tensors and (when the sidecar carries them) the routed
/// expert tensors with a requantized sidecar's copies -- see
/// tools/mimo2_requant_dense.py. The sidecar repeats each tensor's name and
/// logical shape at a smaller type (Q4_0/Q4_K), cutting the bytes those
/// tensors cost per token (and per draft pass) roughly in half; every
/// consumer stays untouched because the uploader and the TP row splits
/// address tensors through RowBytes(), which is type-generic.
/// Tensors the sidecar does not carry keep the artifact's own copy.
bool GraftDenseSidecar(ModelWeights& w, MtpWeights* mtp,
                       const core::GgufReader& sidecar,
                       std::uint32_t trunk_shard_base,
                       std::uint32_t mtp_shard_base, std::string* error_msg) {
  const auto regions = sidecar.GetMappedRegions();
  if (std::getenv("GUFO_DENSE_SC_DEBUG") != nullptr) {
    fprintf(stderr, "GRAFT enter: sidecar tensors=%zu regions=%zu\n",
            sidecar.GetTensors().size(),
            static_cast<std::size_t>(regions.size()));
    fflush(stderr);
  }
  std::uint32_t grafted = 0;
  std::size_t bytes_before = 0;
  std::size_t bytes_after = 0;
  const auto fail = [&](const std::string& message) {
    if (error_msg != nullptr) {
      *error_msg = message;
    }
    return false;
  };
  // The uploader adds its own per-section shard base (0 for the trunk, the
  // trunk's shard count for the MTP blocks), so each section's refs carry the
  // index relative to that base.
  const auto graft = [&](TensorRef& t, std::uint32_t base) {
    if (t.empty() || t.name.empty()) {
      return true;
    }
    const core::GgufTensorInfo* info = sidecar.FindTensor(t.name);
    if (info == nullptr) {
      return true;
    }
    const bool stacked = t.experts > 1;
    if (info->dimensions.size() != (stacked ? 3u : 2u) ||
        info->dimensions[0] != t.cols || info->dimensions[1] != t.rows ||
        (stacked && info->dimensions[2] != t.experts)) {
      return fail("dense sidecar tensor " + std::string(t.name) +
                  " does not repeat the artifact's logical shape");
    }
    if (gufo::quant::QuantizedRowBytes(info->type, t.cols) == 0) {
      return fail("dense sidecar tensor " + std::string(t.name) +
                  " has an unusable type for its row length");
    }
    const auto address = reinterpret_cast<std::uintptr_t>(info->data);
    std::uint32_t shard = 0;
    std::uint64_t file_offset = 0;
    bool inside = false;
    for (std::uint32_t i = 0; i < regions.size(); ++i) {
      // `region_base` must not be named `base`: that would shadow the shard
      // base parameter and put a mapped address into `shard`.
      const auto region_base =
          reinterpret_cast<std::uintptr_t>(regions[i].data);
      if (address >= region_base && address < region_base + regions[i].size) {
        shard = base + i;
        file_offset = address - region_base;
        inside = true;
        break;
      }
    }
    if (!inside) {
      return fail("dense sidecar tensor " + std::string(t.name) +
                  " is outside its file");
    }
    bytes_before += t.SizeBytes();
    t.data = info->data;
    t.type = info->type;
    t.shard = shard;
    t.file_offset = file_offset;
    t.name = info->name;
    // The uploader adds its own shard base and validates the range against the
    // region; a mismatch would otherwise surface as an opaque upload error.
    // The size must come from the *grafted* type (Q4 rows are ~1.9x shorter
    // than the Q8 original, and the last tensors of the sidecar end flush
    // with the file).
    {
      const std::size_t size = t.SizeBytes();
      const std::size_t region_bytes = regions[shard - base].size;
      if (t.file_offset > region_bytes ||
          size > region_bytes - t.file_offset) {
        return fail("dense sidecar tensor " + std::string(t.name) +
                    " payload would not fit its region: offset=" +
                    std::to_string(t.file_offset) + " size=" +
                    std::to_string(size) + " region=" +
                    std::to_string(region_bytes) + " shard_base=" +
                    std::to_string(base) + " shard=" +
                    std::to_string(t.shard));
      }
    }
    if (std::getenv("GUFO_DENSE_SC_DEBUG") != nullptr) {
      fprintf(stderr,
              "GRAFT %s type=%d shard=%u offset=%llu size=%zu region=%zu\n",
              std::string(t.name).c_str(), static_cast<int>(t.type), t.shard,
              static_cast<unsigned long long>(t.file_offset),
              static_cast<std::size_t>(t.SizeBytes()),
              static_cast<std::size_t>(regions[shard - base].size));
      fflush(stderr);
    }
    bytes_after += t.SizeBytes();
    ++grafted;
    return true;
  };
  if (std::getenv("GUFO_DENSE_SC_DEBUG") != nullptr) {
    fprintf(stderr, "GRAFT trunk: layers=%zu\n", w.layers.size());
    fflush(stderr);
  }
  const auto graft_block = [&](LayerWeights& l, std::uint32_t base) {
    if (!graft(l.attn_qkv, base) || !graft(l.attn_q, base) ||
        !graft(l.attn_k, base) || !graft(l.attn_v, base) ||
        !graft(l.attn_out, base)) {
      return false;
    }
    if (!graft(l.ffn_gate_exps, base) || !graft(l.ffn_up_exps, base) ||
        !graft(l.ffn_down_exps, base)) {
      return false;
    }
    return !l.dense_ffn || (graft(l.ffn_gate, base) &&
                            graft(l.ffn_up, base) &&
                            graft(l.ffn_down, base));
  };
  for (LayerWeights& l : w.layers) {
    if (!graft_block(l, trunk_shard_base)) {
      return false;
    }
  }
  // The lm head drives both the trunk's logits and (through MtpHead's
  // fallback) every draft pass.
  if (w.output.data != w.token_embd.data &&
      !graft(w.output, trunk_shard_base)) {
    return false;
  }
  if (std::getenv("GUFO_DENSE_SC_DEBUG") != nullptr) {
    fprintf(stderr, "GRAFT trunk done: grafted=%u\n", grafted);
    fflush(stderr);
  }
  if (mtp != nullptr) {
    for (LayerWeights& l : mtp->blocks) {
      if (!graft_block(l, mtp_shard_base)) {
        return false;
      }
    }
  }
  fprintf(stderr,
          "DENSE_SIDECAR grafted %u tensors: dense weights %.0f MB -> %.0f MB "
          "(sidecar regions=%zu trunk_base=%u mtp_base=%u)\n",
          grafted, static_cast<double>(bytes_before) / 1e6,
          static_cast<double>(bytes_after) / 1e6, regions.size(),
          trunk_shard_base, mtp_shard_base);
  fflush(stderr);
  return grafted != 0;
}

}  // namespace

DeviceModel::~DeviceModel() {
  for (void* p : allocations_) {
    (void)hipFree(p);
  }
}

std::unique_ptr<DeviceModel> DeviceModel::Upload(
    const ModelWeights& w, const core::GgufReader& reader,
    const MtpWeights* mtp, const core::GgufReader* mtp_reader,
    std::string* error_msg, const distributed::TpPartition* partition,
    const core::GgufReader* dense_sidecar,
    const core::GgufReader* expert_sidecar) {
  // Q6_K decodes on the CPU reference, but neither the model-private kernels
  // nor the vendored GEMM tier accept it; reject it before allocating.
  const auto supported = [&](const TensorRef& t) {
    if (t.empty() || t.type != core::GgmlType::kQ6_K)
      return true;
    if (error_msg != nullptr)
      *error_msg = "unsupported HIP tensor format Q6_K: " + std::string(t.name);
    return false;
  };
  const auto layer_supported = [&](const LayerWeights& l) {
    return supported(l.attn_qkv) && supported(l.attn_q) &&
           supported(l.attn_k) && supported(l.attn_v) &&
           supported(l.attn_out) && supported(l.ffn_gate) &&
           supported(l.ffn_up) && supported(l.ffn_down) &&
           supported(l.ffn_gate_exps) && supported(l.ffn_up_exps) &&
           supported(l.ffn_down_exps);
  };
  if (!supported(w.token_embd) || !supported(w.output) ||
      !std::all_of(w.layers.begin(), w.layers.end(), layer_supported) ||
      (mtp != nullptr &&
       !std::all_of(mtp->blocks.begin(), mtp->blocks.end(),
                    layer_supported))) {
    return nullptr;
  }
  std::unique_ptr<DeviceModel> m(new DeviceModel());
  m->config_ = w.config;
  if (partition != nullptr) {
    if (!partition->Valid() || partition->world_size > 2 ||
        partition->expert_ff != w.config.expert_ff) {
      if (error_msg != nullptr) {
        *error_msg = "TP split does not match the model's expert size";
      }
      return nullptr;
    }
    m->tp_rank_ = partition->rank;
    m->tp_world_size_ = partition->world_size;
  }
  const auto regions = reader.GetMappedRegions();
  std::vector<core::GgufMappedRegion> shards(regions.begin(), regions.end());
  const auto shard_count = static_cast<std::uint32_t>(shards.size());
  if (mtp != nullptr) {
    if (mtp_reader == nullptr) {
      if (error_msg)
        *error_msg = "MTP weights require their bound reader";
      return nullptr;
    }
    const auto extra = mtp_reader->GetMappedRegions();
    shards.insert(shards.end(), extra.begin(), extra.end());
  }
  // The dense sidecar's own regions join the shard table last, so the grafted
  // refs address them by (shard_base + index) with no other change.
  ModelWeights wlocal = w;
  std::optional<MtpWeights> mtp_local;
  if (mtp != nullptr) {
    mtp_local = *mtp;
  }
  if (dense_sidecar != nullptr) {
    const auto sidecar_base = static_cast<std::uint32_t>(shards.size());
    const auto extra = dense_sidecar->GetMappedRegions();
    shards.insert(shards.end(), extra.begin(), extra.end());
    if (!GraftDenseSidecar(
            wlocal, mtp_local.has_value() ? &*mtp_local : nullptr,
            *dense_sidecar, sidecar_base,
            mtp_reader != nullptr
                ? static_cast<std::uint32_t>(
                      mtp_reader->GetMappedRegions().size())
                : 0U,
            error_msg)) {
      return nullptr;
    }
  }
  // The expert sidecar carries only the routed expert tensors; running it
  // through the same graft leaves the dense tensors it does not carry alone.
  if (expert_sidecar != nullptr) {
    const auto sidecar_base = static_cast<std::uint32_t>(shards.size());
    const auto extra = expert_sidecar->GetMappedRegions();
    shards.insert(shards.end(), extra.begin(), extra.end());
    if (!GraftDenseSidecar(
            wlocal, mtp_local.has_value() ? &*mtp_local : nullptr,
            *expert_sidecar, sidecar_base,
            mtp_reader != nullptr
                ? static_cast<std::uint32_t>(
                      mtp_reader->GetMappedRegions().size())
                : 0U,
            error_msg)) {
      return nullptr;
    }
  }
  const MtpWeights* mtp_use = mtp_local.has_value() ? &*mtp_local : mtp;
  auto stager = hip::WeightUpload::Create(shards, error_msg);
  if (!stager) {
    return nullptr;
  }
  Uploader up{*stager,          m->allocations_, m->bytes_,
              m->max_half_cols_, m->max_q8_cols_, error_msg,
              true,             0,               partition,
              &m->config_};
  const Config& c = m->config_;
  const bool split = partition != nullptr && partition->distributed();
  m->token_embd_ = up.Copy(wlocal.token_embd);
  m->output_ = wlocal.output.data == wlocal.token_embd.data
                   ? m->token_embd_
                   : up.Copy(wlocal.output);
  m->output_norm_ = up.Copy(wlocal.output_norm);
  m->layers_.reserve(c.num_layers);
  for (std::uint32_t il = 0; il < c.num_layers; ++il) {
    m->layers_.push_back(up.Layer(wlocal.layers[il], c, il, split));
    if (!up.ok) {
      return nullptr;
    }
  }
  if (mtp_use != nullptr && !mtp_use->blocks.empty()) {
    // The sidecar repeats the trunk geometry; its blocks sit at the trunk's
    // trailing layer indices. A sidecar grafted onto the trunk's numbering
    // (MtpWeights::Bind) carries the trunk's shorter tables, so clamp the
    // lookup index instead of reading past them.
    up.shard_base = shard_count;
    m->nextn_.reserve(mtp_use->blocks.size());
    for (std::size_t j = 0; j < mtp_use->blocks.size(); ++j) {
      const std::uint32_t mtp_il = std::min(
          c.num_layers + static_cast<std::uint32_t>(j),
          mtp_use->config.num_layers_all - 1);
      m->nextn_.push_back(
          up.Layer(mtp_use->blocks[j], mtp_use->config, mtp_il,
                   /*split=*/false));
      if (!up.ok) {
        return nullptr;
      }
    }
  }
  if (!up.ok || !stager->Finish(error_msg)) {
    return nullptr;
  }
  // Bring-up self-test: every resident tensor must be readable. Names the
  // exact tensors whose device pointer never got mapped.
  {
    const auto probe = [&](const char* what, const DeviceTensor& t) {
      if (t.empty()) {
        return;
      }
      float sink = 0.0F;
      if (hipMemcpy(&sink, t.data, sizeof(sink),
                    hipMemcpyDeviceToHost) != hipSuccess) {
        fprintf(stderr, "DEVICETENSOR FAULT %s ptr=%p rows=%u\n", what,
                t.data, t.rows);
        fflush(stderr);
      }
    };
    for (std::size_t i = 0; i < m->layers_.size(); ++i) {
      const DeviceLayer& l = m->layers_[i];
      probe("attn_norm", l.attn_norm);
      probe("ffn_norm", l.ffn_norm);
    }
    hipDeviceSynchronize();
  }
  return m;
}

}  // namespace gufo::models::mimo2::rocm
