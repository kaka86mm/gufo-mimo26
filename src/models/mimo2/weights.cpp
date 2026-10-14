#include "src/models/mimo2/weights.hpp"

#include <initializer_list>
#include <string>
#include <utility>

namespace gufo::models::mimo2 {
namespace {

using core::GgmlType;

struct Format {
  std::uint32_t block;
  std::uint32_t bytes;
};

/// Storage geometry of the formats this runtime decodes. Unknown types get a
/// zero block, which every validation below rejects. MXFP4 is the native
/// expert format of the official artifacts (one E8M0 scale per 32 values).
[[nodiscard]] Format FormatOf(GgmlType type) noexcept {
  switch (type) {
    case GgmlType::kF32:
      return {1, 4};
    case GgmlType::kF16:
    case GgmlType::kBF16:
      return {1, 2};
    case GgmlType::kQ8_0:
      return {32, 34};
    case GgmlType::kQ4_0:
      return {32, 18};
    case GgmlType::kQ5_1:
      return {32, 24};
    case GgmlType::kQ2_K:
      return {256, 84};
    case GgmlType::kQ3_K:
      return {256, 110};
    case GgmlType::kQ4_K:
      return {256, 144};
    case GgmlType::kQ5_K:
      return {256, 176};
    case GgmlType::kQ6_K:
      return {256, 210};
    case GgmlType::kMXFP4:
      return {32, 17};
    default:
      return {0, 0};
  }
}

struct Binder {
  const core::GgufReader& reader;
  std::string* error;
  bool ok{true};
  /// Per-layer KV head counts derived from fused tensor geometry (nonzero
  /// overrides the compressed metadata); applied by the Bind epilogues.
  std::vector<std::uint32_t> kv_override;

  void Fail(const std::string& message) {
    if (ok && error != nullptr) {
      *error = message;
    }
    ok = false;
  }

  /// Binds `name` with the exact shape and one of the accepted formats.
  TensorRef Get(const std::string& name, std::uint64_t cols, std::uint64_t rows,
                std::uint64_t experts, std::initializer_list<GgmlType> types,
                bool required = true) {
    TensorRef t;
    const auto* info = reader.FindTensor(name);
    if (info == nullptr) {
      if (required) {
        Fail("missing tensor " + name);
      }
      return t;
    }
    const auto& d = info->dimensions;
    const std::uint64_t got_cols = d.size() > 0 ? d[0] : 1;
    const std::uint64_t got_rows = d.size() > 1 ? d[1] : 1;
    const std::uint64_t got_experts = d.size() > 2 ? d[2] : 1;
    if (got_cols != cols || got_rows != rows || got_experts != experts ||
        d.size() > 3) {
      Fail("tensor " + name + " has shape [" + std::to_string(got_cols) + ", " +
           std::to_string(got_rows) + ", " + std::to_string(got_experts) +
           "], expected [" + std::to_string(cols) + ", " +
           std::to_string(rows) + ", " + std::to_string(experts) + "]");
      return t;
    }
    bool type_ok = false;
    for (auto type : types) {
      type_ok = type_ok || info->type == type;
    }
    const Format format = FormatOf(info->type);
    if (!type_ok || format.block == 0 || cols % format.block != 0) {
      Fail("tensor " + name + " has unsupported format " +
           std::string(core::ToString(info->type)));
      return t;
    }
    t.data = info->data;
    t.type = info->type;
    t.cols = cols;
    t.rows = rows;
    t.experts = experts;
    t.name = info->name;
    // Locate the shard so disk readers can address the payload directly, and
    // so a payload running past its shard is caught here rather than as a
    // fault later.
    const auto regions = reader.GetMappedRegions();
    const auto address = reinterpret_cast<std::uintptr_t>(info->data);
    bool inside = false;
    for (std::uint32_t i = 0; i < regions.size(); ++i) {
      const auto base = reinterpret_cast<std::uintptr_t>(regions[i].data);
      if (address >= base && address < base + regions[i].size) {
        t.shard = i;
        t.file_offset = address - base;
        inside = t.SizeBytes() <= regions[i].size - t.file_offset;
        break;
      }
    }
    if (!inside) {
      Fail("tensor " + name + " is truncated");
      return TensorRef{};
    }
    return t;
  }

  /// Binds one block's tensors against `c`'s per-layer geometry. `il` is the
  /// artifact's own layer index; `is_nextn` selects the MTP tensor set.
  LayerWeights Layer(const Config& c, std::uint32_t il, bool is_nextn) {
    LayerWeights l;
    l.is_nextn = is_nextn;
    l.dense_ffn = is_nextn || il == 0;
    const std::string p = "blk." + std::to_string(il) + ".";
    const std::uint64_t hidden = c.hidden_size;
    const std::uint64_t q_rows =
        static_cast<std::uint64_t>(c.num_heads) * c.head_dim_qk;
    const std::uint64_t kv = c.kv_heads[il];
    const std::uint64_t k_rows = kv * c.head_dim_qk;
    const std::uint64_t v_rows = kv * c.head_dim_v;
    // Q4_K is in the dense set for the requantized attn projections: the same
    // 4-bit super-block format the experts already use, at half the Q8_0
    // bytes per token.
    const auto dense = {GgmlType::kQ8_0, GgmlType::kQ6_K, GgmlType::kQ4_K,
                        GgmlType::kQ4_0, GgmlType::kQ5_1, GgmlType::kBF16,
                        GgmlType::kF16, GgmlType::kF32};
    const auto experts = {GgmlType::kMXFP4, GgmlType::kQ2_K, GgmlType::kQ3_K,
                          GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K,
                          GgmlType::kQ5_1, GgmlType::kQ8_0};

    l.attn_norm = Get(p + "attn_norm.weight", hidden, 1, 1, {GgmlType::kF32});
    l.ffn_norm = Get(p + "ffn_norm.weight", hidden, 1, 1, {GgmlType::kF32});
    l.attn_sinks = Get(p + "attn_sinks.weight", c.num_heads, 1, 1,
                       {GgmlType::kF32}, /*required=*/false);

    // The fused projection packs [Q | K | V] rows with K padded to the key
    // head width; the split form is bound otherwise.
    std::uint64_t fused_rows = q_rows + k_rows + v_rows;
    {
      // Compressed per-layer metadata can disagree with the artifact (the
      // official MTP sidecar declares head_count_kv=4 while its fused rows
      // carry eight KV heads). The tensor is the truth: 64*192 + n*320
      // pins n, so derive the layer's KV count from the file and re-check
      // the geometry against the config only through the cache layout.
      const auto* info = reader.FindTensor(p + "attn_qkv.weight");
      if (info != nullptr && info->dimensions.size() == 2 &&
          info->dimensions[0] == hidden &&
          info->dimensions[1] > q_rows &&
          (info->dimensions[1] - q_rows) % (c.head_dim_qk + c.head_dim_v) ==
              0) {
        const std::uint64_t file_kv =
            (info->dimensions[1] - q_rows) /
            (c.head_dim_qk + c.head_dim_v);
        if (file_kv != kv && file_kv <= c.num_heads &&
            c.num_heads % file_kv == 0) {
          if (kv_override.size() <= il) {
            kv_override.resize(il + 1, 0);
          }
          kv_override[il] = static_cast<std::uint32_t>(file_kv);
          fused_rows = info->dimensions[1];
        }
      }
    }
    l.attn_qkv = Get(p + "attn_qkv.weight", hidden, fused_rows, 1, dense,
                     /*required=*/false);
    if (l.attn_qkv.empty()) {
      l.attn_q = Get(p + "attn_q.weight", hidden, q_rows, 1, dense);
      l.attn_k = Get(p + "attn_k.weight", hidden, k_rows, 1, dense);
      l.attn_v = Get(p + "attn_v.weight", hidden, v_rows, 1, dense);
    }
    l.attn_out = Get(p + "attn_output.weight",
                     static_cast<std::uint64_t>(c.num_heads) * c.head_dim_v,
                     hidden, 1, dense);

    if (l.dense_ffn) {
      l.ffn_gate = Get(p + "ffn_gate.weight", hidden, c.dense_ff, 1, dense);
      l.ffn_up = Get(p + "ffn_up.weight", hidden, c.dense_ff, 1, dense);
      l.ffn_down = Get(p + "ffn_down.weight", c.dense_ff, hidden, 1, dense);
    } else {
      l.router = Get(p + "ffn_gate_inp.weight", hidden, c.num_experts, 1,
                     {GgmlType::kF32});
      l.router_bias =
          Get(p + "ffn_exp_probs_b.bias", c.num_experts, 1, 1,
              {GgmlType::kF32}, /*required=*/false);
      l.ffn_gate_exps = Get(p + "ffn_gate_exps.weight", hidden, c.expert_ff,
                            c.num_experts, experts);
      l.ffn_up_exps = Get(p + "ffn_up_exps.weight", hidden, c.expert_ff,
                          c.num_experts, experts);
      l.ffn_down_exps = Get(p + "ffn_down_exps.weight", c.expert_ff, hidden,
                            c.num_experts, experts);
    }

    if (is_nextn) {
      l.nextn_enorm = Get(p + "nextn.enorm.weight", hidden, 1, 1,
                          {GgmlType::kF32});
      l.nextn_hnorm = Get(p + "nextn.hnorm.weight", hidden, 1, 1,
                          {GgmlType::kF32});
      l.nextn_eh_proj =
          Get(p + "nextn.eh_proj.weight", 2 * hidden, hidden, 1, dense);
      l.nextn_embed_tokens =
          Get(p + "nextn.embed_tokens.weight", hidden, c.vocab_size, 1, dense,
              /*required=*/false);
      l.nextn_shared_head_norm =
          Get(p + "nextn.shared_head_norm.weight", hidden, 1, 1,
              {GgmlType::kF32}, /*required=*/false);
      l.nextn_shared_head_head =
          Get(p + "nextn.shared_head_head.weight", hidden, c.vocab_size, 1,
              dense, /*required=*/false);
      l.layer_out_norm = Get(p + "layer_out_norm.weight", hidden, 1, 1,
                             {GgmlType::kF32}, /*required=*/false);
    }
    return l;
  }
};

}  // namespace

std::size_t TensorRef::RowBytes() const noexcept {
  const Format f = FormatOf(type);
  if (f.block == 0 || cols % f.block != 0) {
    return 0;
  }
  return static_cast<std::size_t>(cols / f.block) * f.bytes;
}

std::optional<ModelWeights> ModelWeights::Bind(const core::GgufReader& reader,
                                               std::string* error_msg) {
  auto config = Config::FromGguf(reader, true, error_msg);
  if (!config.has_value()) {
    return std::nullopt;
  }
  Binder b{reader, error_msg};
  b.kv_override.resize(config->num_layers_all, 0);
  ModelWeights w;
  w.config = *config;
  const Config& c = w.config;
  const auto dense = {GgmlType::kQ8_0, GgmlType::kQ6_K, GgmlType::kBF16,
                      GgmlType::kF16, GgmlType::kF32};

  {
    // The vocabulary is the embedding row count; no metadata key carries it.
    const auto* info = reader.FindTensor("token_embd.weight");
    if (info == nullptr || info->dimensions.size() != 2) {
      b.Fail("missing token_embd.weight");
      return std::nullopt;
    }
    w.config.vocab_size = static_cast<std::uint32_t>(info->dimensions[1]);
  }
  w.token_embd =
      b.Get("token_embd.weight", c.hidden_size, c.vocab_size, 1, dense);
  w.output =
      b.Get("output.weight", c.hidden_size, c.vocab_size, 1, dense, false);
  if (w.output.empty()) {
    w.output = w.token_embd;
  }
  w.output_norm =
      b.Get("output_norm.weight", c.hidden_size, 1, 1, {GgmlType::kF32});

  w.layers.reserve(c.num_layers_all);
  for (std::uint32_t il = 0; il < c.num_layers_all; ++il) {
    w.layers.push_back(b.Layer(c, il, il >= c.num_layers));
  }
  // The head must not carry a tied embedding while the trunk also stores one
  // of a different width; nothing else cross-checks that fallback.
  if (!b.ok) {
    return std::nullopt;
  }
  for (std::uint32_t il = 0; il < b.kv_override.size() &&
                            il < w.config.kv_heads.size();
       ++il) {
    if (b.kv_override[il] != 0) {
      w.config.kv_heads[il] = b.kv_override[il];
    }
  }
  w.config.router_bias = false;
  for (std::uint32_t il = 1; il < c.num_layers; ++il) {
    if (!w.layers[il].router_bias.empty()) {
      w.config.router_bias = true;
      break;
    }
  }
  // The official artifact compresses attention.sliding_window.pattern to a
  // single zero entry, which broadcasts to "no sliding layers". The original
  // model puts attention sinks exactly on its 39 sliding layers (HF config:
  // add_swa_attention_sink_bias=true, add_full=false), so the bound sink
  // tensors recover the per-layer window layout. Only take this inference
  // when the metadata claims no sliding layer at all.
  bool any_swa = false;
  for (std::uint32_t il = 0; il < c.num_layers_all; ++il) {
    any_swa = any_swa || w.config.LayerIsSwa(il);
  }
  if (!any_swa) {
    std::uint32_t sink_layers = 0;
    for (std::uint32_t il = 0; il < c.num_layers_all; ++il) {
      sink_layers += w.layers[il].attn_sinks.empty() ? 0 : 1;
    }
    if (sink_layers > 0 && sink_layers < c.num_layers) {
      for (std::uint32_t il = 0; il < c.num_layers_all; ++il) {
        w.config.is_swa[il] = w.layers[il].attn_sinks.empty() ? 0 : 1;
      }
    }
  }
  return w;
}

std::optional<MtpWeights> MtpWeights::Bind(const core::GgufReader& reader,
                                           const Config& trunk,
                                           std::string* error_msg) {
  auto config = Config::FromGguf(reader, false, error_msg);
  if (!config.has_value()) {
    return std::nullopt;
  }
  Binder b{reader, error_msg};
  MtpWeights m;
  m.config = *config;
  m.config.vocab_size = trunk.vocab_size;
  b.kv_override.resize(m.config.num_layers_all, 0);

  // The sidecar either repeats the trunk's layer numbering or numbers its
  // blocks from zero; locate every block that carries an eh_proj tensor.
  std::vector<std::uint32_t> indices;
  for (std::uint32_t il = 0; il < config->num_layers_all; ++il) {
    const std::string probe =
        "blk." + std::to_string(il) + ".nextn.eh_proj.weight";
    if (reader.FindTensor(probe) != nullptr) {
      indices.push_back(il);
    }
  }
  if (indices.empty()) {
    b.Fail("MTP sidecar carries no nextn.eh_proj block");
  }
  // The trunk may declare no NextN blocks while the sidecar carries the
  // draft network; the sidecar's own list is the authoritative capability.
  (void)trunk.nextn_layers;
  // Per-layer geometry comes from the trunk when the numbering differs.
  const Config& geometry =
      config->num_layers_all == trunk.num_layers_all ? trunk : *config;
  if (geometry.num_layers_all != trunk.num_layers_all) {
    // Zero-numbered sidecar: graft its block types onto the trunk's MTP
    // layers so kv_heads/window lookups use trunk values.
    m.config = trunk;
    m.config.vocab_size = trunk.vocab_size;
  }
  m.blocks.reserve(indices.size());
  for (const std::uint32_t il : indices) {
    m.blocks.push_back(b.Layer(m.config, il, true));
  }
  if (!b.ok) {
    return std::nullopt;
  }
  for (std::uint32_t il = 0; il < b.kv_override.size() &&
                            il < m.config.kv_heads.size();
       ++il) {
    if (b.kv_override[il] != 0) {
      m.config.kv_heads[il] = b.kv_override[il];
    }
  }
  return m;
}

}  // namespace gufo::models::mimo2
