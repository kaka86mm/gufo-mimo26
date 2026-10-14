#include "src/models/mimo2/config.hpp"

#include <cmath>

#include "src/core/gguf_reader.hpp"
#include <string_view>
#include <utility>
#include <variant>

namespace gufo::models::mimo2 {
namespace {

constexpr std::string_view kArchitecture = "mimo2";

struct Reader {
  const core::GgufReader& gguf;
  std::string* error;
  bool ok{true};

  void Fail(std::string message) {
    if (ok && error != nullptr) {
      *error = std::move(message);
    }
    ok = false;
  }

  std::uint32_t U32(std::string_view key, bool required = true,
                    std::uint32_t fallback = 0) {
    const auto value = gguf.GetMetadataUint32(key);
    if (value.has_value()) {
      return *value;
    }
    if (required || gguf.FindMetadata(key) != nullptr) {
      Fail("missing or invalid GGUF integer " + std::string(key));
    }
    return fallback;
  }

  float F32(std::string_view key, float fallback, bool required = false) {
    const auto value = gguf.GetMetadataFloat32(key);
    if (!value) {
      if (required || gguf.FindMetadata(key) != nullptr) {
        Fail("missing or invalid GGUF float " + std::string(key));
      }
      return fallback;
    }
    if (!std::isfinite(*value)) {
      Fail("non-finite GGUF float " + std::string(key));
    }
    return *value;
  }

  /// Integer arrays of any width; GGUF writers pick the narrowest type.
  std::vector<std::uint64_t> U64Array(std::string_view key,
                                      bool required = true) {
    const auto* meta = gguf.FindMetadata(key);
    std::vector<std::uint64_t> out;
    if (meta == nullptr) {
      if (required) {
        Fail("missing GGUF array " + std::string(key));
      }
      return out;
    }
    if (const auto* u = std::get_if<std::vector<std::uint64_t>>(&meta->value)) {
      out = *u;
    } else if (const auto* d =
                   std::get_if<std::vector<double>>(&meta->value)) {
      // Some writers emit integer arrays in floating form (this artifact's
      // expert_feed_forward_length); accept whole values.
      for (double v : *d) {
        if (v < 0 || v != std::floor(v) || v > 4e18) {
          Fail("non-integral GGUF float array value " + std::string(key));
          return {};
        }
        out.push_back(static_cast<std::uint64_t>(v));
      }
    } else if (const auto* s =
                   std::get_if<std::vector<std::int64_t>>(&meta->value)) {
      for (auto v : *s) {
        if (v < 0) {
          Fail("negative GGUF array value " + std::string(key));
          return {};
        }
        out.push_back(static_cast<std::uint64_t>(v));
      }
    } else {
      Fail("GGUF array must contain integers: " + std::string(key));
    }
    return out;
  }
};

}  // namespace

std::optional<Config> Config::FromGguf(const core::GgufReader& gguf,
                                       bool require_trunk,
                                       std::string* error_msg) {
  Reader r{gguf, error_msg};
  if (gguf.GetMetadataString("general.architecture") != kArchitecture) {
    r.Fail("GGUF architecture is not mimo2");
    return std::nullopt;
  }

  Config c;
  const std::string_view p = "mimo2.";
  auto key = [&](std::string_view suffix) {
    return std::string(p) + std::string(suffix);
  };

  c.num_layers_all = r.U32(key("block_count"));
  c.nextn_layers = r.U32(key("nextn_predict_layers"), false, 0);
  if (c.nextn_layers > 3) {
    r.Fail("mimo2 runtime supports at most three NextN blocks");
    return std::nullopt;
  }
  if (c.nextn_layers >= c.num_layers_all) {
    r.Fail("nextn_predict_layers must be below block_count");
    return std::nullopt;
  }
  c.num_layers = c.num_layers_all - c.nextn_layers;
  c.hidden_size = r.U32(key("embedding_length"));
  c.context_length = r.U32(key("context_length"));
  c.rms_eps = r.F32(key("attention.layer_norm_rms_epsilon"), 1e-6F);

  c.num_heads = r.U32(key("attention.head_count"));
  c.head_dim_qk = r.U32(key("attention.key_length"));
  c.head_dim_v = r.U32(key("attention.value_length"));
  if (c.head_dim_v != c.head_dim_qk && c.head_dim_v * 3 / 2 != c.head_dim_qk) {
    // The known artifact is 192/128; keep the guard honest about anything
    // else reaching this runtime untested.
    r.Fail("unsupported head geometry: key " + std::to_string(c.head_dim_qk) +
           " vs value " + std::to_string(c.head_dim_v));
    return std::nullopt;
  }
  c.rotary_dim = r.U32(key("rope.dimension_count"));
  if (c.rotary_dim == 0 || c.rotary_dim > c.head_dim_qk ||
      c.rotary_dim % 2 != 0) {
    r.Fail("rope.dimension_count must be even and within the key head");
    return std::nullopt;
  }
  c.rope_theta = r.F32(key("rope.freq_base"), 10000.0F);
  c.rope_theta_swa = r.F32(key("rope.freq_base_swa"), 0.0F);
  c.value_scale = r.F32(key("attention.value_scale"), 1.0F);
  if (c.value_scale <= 0.0F) {
    r.Fail("attention.value_scale must be positive");
    return std::nullopt;
  }
  c.sliding_window = r.U32(key("attention.sliding_window"));
  if (c.sliding_window == 0) {
    r.Fail("mimo2 requires a nonzero attention.sliding_window");
    return std::nullopt;
  }

  // Converters compress a per-layer array whose values are all equal into a
  // single entry; broadcast that to every layer (llama.cpp get_key_or_arr).
  const auto per_layer = [&](std::string_view k, std::vector<std::uint64_t>* out) {
    if (gguf.FindMetadata(k) != nullptr &&
        gguf.FindMetadata(k)->value.index() != 12 /* no array variant */) {
      // Uniform writers may store a bare scalar instead of an array.
      const auto scalar = gguf.GetMetadataUint64(k);
      if (scalar.has_value()) {
        out->assign(c.num_layers_all, *scalar);
        return;
      }
    }
    *out = r.U64Array(k);
    if (r.ok && out->size() == 1 && c.num_layers_all > 1) {
      out->resize(c.num_layers_all, (*out)[0]);
    }
  };
  {
    std::vector<std::uint64_t> pattern;
    per_layer(key("attention.sliding_window_pattern"), &pattern);
    if (!r.ok || pattern.size() != c.num_layers_all) {
      r.Fail("attention.sliding_window.pattern must describe every layer");
      return std::nullopt;
    }
    c.is_swa.resize(c.num_layers_all);
    for (std::uint32_t i = 0; i < c.num_layers_all; ++i) {
      if (pattern[i] > 1) {
        r.Fail("attention.sliding_window.pattern entries must be 0 or 1");
        return std::nullopt;
      }
      c.is_swa[i] = static_cast<std::uint8_t>(pattern[i]);
    }
  }
  {
    std::vector<std::uint64_t> kv;
    per_layer(key("attention.head_count_kv"), &kv);
    if (!r.ok || kv.size() != c.num_layers_all) {
      r.Fail("attention.head_count_kv must describe every layer");
      return std::nullopt;
    }
    c.kv_heads.resize(c.num_layers_all);
    for (std::uint32_t i = 0; i < c.num_layers_all; ++i) {
      if (kv[i] == 0 || kv[i] > c.num_heads ||
          c.num_heads % kv[i] != 0) {
        r.Fail("attention.head_count_kv entry does not divide the heads");
        return std::nullopt;
      }
      c.kv_heads[i] = static_cast<std::uint32_t>(kv[i]);
    }
  }

  c.dense_ff = r.U32(key("feed_forward_length"));
  c.num_experts = r.U32(key("expert_count"));
  c.num_experts_used = r.U32(key("expert_used_count"));
  {
    std::vector<std::uint64_t> ff;
    per_layer(key("expert_feed_forward_length"), &ff);
    if (ff.size() != c.num_layers_all) {
      r.Fail("expert_feed_forward_length must describe every layer");
      return std::nullopt;
    }
    for (std::uint32_t i = 0; i < c.num_layers_all; ++i) {
      if (ff[i] == 0 || ff[i] > (1u << 24)) {
        r.Fail("expert_feed_forward_length entry out of range");
        return std::nullopt;
      }
      if (i > 0 && ff[i] != ff[0]) {
        r.Fail("expert_feed_forward_length must be uniform across layers");
        return std::nullopt;
      }
    }
    c.expert_ff = static_cast<std::uint32_t>(ff[0]);
  }
  if (c.num_experts == 0 || c.num_experts_used == 0 ||
      c.num_experts_used > c.num_experts) {
    r.Fail("invalid expert geometry");
    return std::nullopt;
  }
  if (c.num_layers < 2 || c.hidden_size == 0 || c.hidden_size % 32 != 0) {
    r.Fail("unsupported trunk geometry");
    return std::nullopt;
  }
  if (!r.ok) {
    return std::nullopt;
  }
  if (require_trunk && c.num_layers < 8) {
    // Guard against binding a sidecar as a trunk; the real trunk is 48.
    r.Fail("mimo2 trunk requires at least eight layers");
    return std::nullopt;
  }
  return c;
}

bool Config::MtpMatches(const Config& trunk) const noexcept {
  return num_layers_all == trunk.num_layers_all &&
         nextn_layers == trunk.nextn_layers && hidden_size == trunk.hidden_size &&
         num_heads == trunk.num_heads && head_dim_qk == trunk.head_dim_qk &&
         head_dim_v == trunk.head_dim_v && rotary_dim == trunk.rotary_dim &&
         rope_theta == trunk.rope_theta &&
         rope_theta_swa == trunk.rope_theta_swa &&
         value_scale == trunk.value_scale &&
         sliding_window == trunk.sliding_window && is_swa == trunk.is_swa &&
         kv_heads == trunk.kv_heads && dense_ff == trunk.dense_ff &&
         num_experts == trunk.num_experts &&
         num_experts_used == trunk.num_experts_used &&
         expert_ff == trunk.expert_ff;
}

}  // namespace gufo::models::mimo2
