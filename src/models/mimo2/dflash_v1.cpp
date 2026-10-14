#if defined(ENGINE_ENABLE_HIP)
#include "src/models/mimo2/dflash_v1.hpp"

#include <sys/mman.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "src/core/mapped_prefetch.hpp"
#include "src/models/mimo2/kernels/rocm/dflash_v1_kernels.hpp"
#include "src/models/mimo2/kernels/rocm/kernels.hpp"
#include "src/models/mimo2/mtp_costs.hpp"

namespace gufo::models::mimo2 {
namespace {

/// Reads `partial_rotary_factor` from a `config.json` next to the sidecar (the
/// HF drafter config the checkpoint ships with). MiMo's drafter rotates only
/// the leading half of every head; the sidecar GGUF never carries the factor,
/// so the sibling file is the source of truth.
bool ReadPartialRotaryFactor(std::string_view sidecar_path, double* factor) {
  const std::size_t slash = sidecar_path.find_last_of('/');
  const std::string dir =
      slash == std::string_view::npos
          ? std::string()
          : std::string(sidecar_path.substr(0, slash + 1));
  for (const char* name : {"config.json", "dflash-hf/config.json"}) {
    std::FILE* file = std::fopen((dir + name).c_str(), "rb");
    if (file == nullptr) {
      continue;
    }
    std::string text;
    char buffer[4096];
    std::size_t read = 0;
    while ((read = std::fread(buffer, 1, sizeof(buffer), file)) > 0) {
      text.append(buffer, read);
    }
    std::fclose(file);
    const std::size_t key = text.find("\"partial_rotary_factor\"");
    if (key == std::string::npos) {
      continue;
    }
    const std::size_t colon = text.find(':', key);
    if (colon == std::string::npos) {
      continue;
    }
    const double value = std::strtod(text.c_str() + colon + 1, nullptr);
    if (value > 0.0 && value <= 1.0) {
      *factor = value;
      return true;
    }
  }
  return false;
}

/// GUFO_DFLASH_DUMP=<dir> writes the primed target features and, per
/// verification chunk, the target's own row logits so a captured run can be
/// replayed offline (probe --features) and compared against the draft logits.
const char* dump_dir() {
  static const char* dir = std::getenv("GUFO_DFLASH_DUMP");
  return dir;
}

bool WriteBinaryFile(const std::string& path, const void* data,
                     std::size_t bytes) {
  std::FILE* file = std::fopen(path.c_str(), "wb");
  if (file == nullptr) {
    return false;
  }
  const std::size_t written = std::fwrite(data, 1, bytes, file);
  std::fclose(file);
  return written == bytes;
}

double NowMs() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

bool injected_trace() {
  static const bool enabled = std::getenv("GUFO_DFLASH_TRACE") != nullptr;
  return enabled;
}

using rocm::WeightType;

void SetError(std::string* error_msg, std::string message) {
  if (error_msg != nullptr) {
    *error_msg = std::move(message);
  }
}

bool Fail(std::string* error_msg, std::string message) {
  SetError(error_msg, std::move(message));
  return false;
}

bool RequiredUint32(const core::GgufReader& reader, std::string_view key,
                    std::uint32_t* out, std::string* error_msg) {
  const auto value = reader.GetMetadataUint32(key);
  if (!value.has_value()) {
    return Fail(error_msg, "DFlash v1 GGUF is missing required metadata '" +
                               std::string(key) + "'");
  }
  *out = *value;
  return true;
}

bool RequiredFloat32(const core::GgufReader& reader, std::string_view key,
                     float* out, std::string* error_msg) {
  const auto value = reader.GetMetadataFloat32(key);
  if (!value.has_value()) {
    return Fail(error_msg, "DFlash v1 GGUF is missing required metadata '" +
                               std::string(key) + "'");
  }
  *out = *value;
  return true;
}

bool RequiredBool(const core::GgufReader& reader, std::string_view key,
                  bool* out, std::string* error_msg) {
  const auto value = reader.GetMetadataBool(key);
  if (!value.has_value()) {
    return Fail(error_msg, "DFlash v1 GGUF is missing required metadata '" +
                               std::string(key) + "'");
  }
  *out = *value;
  return true;
}

/// `dflash.target_layers` is an integer array of 1-based layer inputs; gufo
/// taps the zero-based layer outputs, so N normalizes to N - 1.
bool RequiredTargetLayers(const core::GgufReader& reader,
                          std::vector<std::uint32_t>* out,
                          std::string* error_msg) {
  const auto* metadata = reader.FindMetadata("dflash.target_layers");
  if (metadata == nullptr) {
    return Fail(error_msg,
                "DFlash v1 GGUF is missing required metadata "
                "'dflash.target_layers'");
  }
  std::vector<std::uint64_t> values;
  if (const auto* unsigned_values =
          std::get_if<std::vector<std::uint64_t>>(&metadata->value)) {
    values = *unsigned_values;
  } else if (const auto* signed_values =
                 std::get_if<std::vector<std::int64_t>>(&metadata->value)) {
    for (const auto value : *signed_values) {
      if (value < 0) {
        return Fail(error_msg, "DFlash v1 target layer indices must be >= 1");
      }
      values.push_back(static_cast<std::uint64_t>(value));
    }
  } else {
    return Fail(error_msg,
                "DFlash v1 metadata 'dflash.target_layers' is not an integer "
                "array");
  }
  if (values.empty()) {
    return Fail(error_msg, "DFlash v1 target layer list must not be empty");
  }
  // `dflash.target_layers` uses HF hidden_states numbering (the EAGLE3
  // convention): the entry N names hidden_states[N] = the OUTPUT of layer
  // N - 1, so gufo taps layer N - 1. GUFO_DFLASH_TAP_0BASED switches to the
  // bare zero-based layer reading for A/B measurements.
  static const bool one_based = std::getenv("GUFO_DFLASH_TAP_0BASED") == nullptr;
  out->clear();
  for (const auto encoded : values) {
    if (one_based && encoded == 0) {
      return Fail(error_msg,
                  "DFlash v1 target layer indices must use positive 1-based "
                  "layer-input numbering");
    }
    if (encoded > static_cast<std::uint64_t>(
                      std::numeric_limits<std::uint32_t>::max())) {
      return Fail(error_msg, "DFlash v1 target layer index exceeds range");
    }
    const auto normalized =
        static_cast<std::uint32_t>(one_based ? encoded - 1 : encoded);
    if (!out->empty() && normalized <= out->back()) {
      return Fail(error_msg,
                  "DFlash v1 target layer indices must be strictly increasing");
    }
    out->push_back(normalized);
  }
  return true;
}

bool Bind(const core::GgufReader& reader, std::string_view name,
          std::uint64_t cols, std::uint64_t rows,
          std::span<const core::GgmlType> types, DFlashV1Tensor* out,
          std::string* error_msg) {
  const auto* info = reader.FindTensor(name);
  if (info == nullptr) {
    return Fail(error_msg,
                "DFlash v1 GGUF is missing tensor '" + std::string(name) + "'");
  }
  const bool vector = rows == 1;
  const std::size_t expected_dims = vector ? 1 : 2;
  if (info->dimensions.size() != expected_dims ||
      info->dimensions[0] != cols || (!vector && info->dimensions[1] != rows)) {
    return Fail(error_msg,
                "DFlash v1 tensor '" + std::string(name) + "' geometry does "
                "not match the configuration");
  }
  bool type_ok = false;
  for (const auto type : types) {
    type_ok = type_ok || info->type == type;
  }
  if (!type_ok) {
    return Fail(error_msg, "DFlash v1 tensor '" + std::string(name) +
                               "' has unsupported type " +
                               std::string(core::ToString(info->type)));
  }
  *out = {.data = info->data,
          .type = info->type,
          .cols = static_cast<std::uint32_t>(cols),
          .rows = static_cast<std::uint32_t>(rows)};
  return true;
}

WeightType KernelType(core::GgmlType type) {
  switch (type) {
    case core::GgmlType::kF32:
      return WeightType::kF32;
    case core::GgmlType::kF16:
      return WeightType::kF16;
    case core::GgmlType::kBF16:
      return WeightType::kBF16;
    case core::GgmlType::kQ8_0:
      return WeightType::kQ8_0;
    default:
      throw std::logic_error("DFlash v1 tensor format is unsupported");
  }
}

template<typename T>
T* Alloc(std::vector<void*>& allocations, std::size_t count,
         std::string* error_msg, std::size_t* bytes) {
  void* pointer = nullptr;
  const std::size_t size = std::max<std::size_t>(1, count) * sizeof(T);
  if (hipMalloc(&pointer, size) != hipSuccess) {
    Fail(error_msg, "DFlash v1 hipMalloc of " + std::to_string(size) +
                        " bytes failed");
    allocations.push_back(nullptr);
    return nullptr;
  }
  if (hipMemset(pointer, 0, size) != hipSuccess) {
    Fail(error_msg, "DFlash v1 hipMemset failed");
    allocations.push_back(pointer);
    return nullptr;
  }
  allocations.push_back(pointer);
  if (bytes != nullptr) {
    *bytes += size;
  }
  return static_cast<T*>(pointer);
}

bool CheckHip(hipError_t status, const char* what, std::string* error_msg) {
  if (status == hipSuccess) {
    return true;
  }
  const std::string message = std::string("DFlash v1 ") + what +
                              " failed: " + hipGetErrorString(status);
  fprintf(stderr, "%s\n", message.c_str());
  return Fail(error_msg, message);
}

/// The GPU cannot read a plain file mapping (the trunk uploads its weights to
/// device memory). The sidecar instead copies each mapped region into
/// anonymous memory and host-registers it, mirroring the Qwen weight loader:
/// the GPU then reads the file-cheap copy in place.
bool RegisterMappedRegions(const core::GgufReader& reader,
                           std::vector<DFlashV1MappedCopy>* copies,
                           std::string* error_msg) {
  const auto regions = reader.GetMappedRegions();
  if (regions.empty()) {
    return Fail(error_msg, "DFlash v1 sidecar has no mapped weight regions");
  }
  for (const auto& region : regions) {
    if (region.size == 0) {
      continue;
    }
    void* host = mmap(nullptr, region.size, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (host == MAP_FAILED) {
      return Fail(error_msg, "DFlash v1 anonymous weight copy failed");
    }
    (void)madvise(host, region.size, MADV_HUGEPAGE);
    try {
      core::PrefaultMappedRange(region.data, region.size);
    } catch (const std::exception& exception) {
      (void)munmap(host, region.size);
      return Fail(error_msg,
                  std::string("DFlash v1 cannot read mapped weights: ") +
                      exception.what());
    }
    std::memcpy(host, region.data, region.size);
    if (hipHostRegister(host, region.size,
                        hipHostRegisterMapped | hipHostRegisterReadOnly) !=
        hipSuccess) {
      (void)munmap(host, region.size);
      return Fail(error_msg, "DFlash v1 weight registration failed");
    }
    void* device = nullptr;
    if (hipHostGetDevicePointer(&device, host, 0) != hipSuccess) {
      (void)hipHostUnregister(host);
      (void)munmap(host, region.size);
      return Fail(error_msg, "DFlash v1 weight device pointer failed");
    }
    copies->push_back({.host = host,
                       .device = device,
                       .source = region.data,
                       .size = region.size});
  }
  return true;
}

/// Translates a pointer into a mapped region to the registered copy's device
/// address; leaves already-device pointers untouched.
void RemapTensor(DFlashV1Tensor* tensor,
                 const std::vector<DFlashV1MappedCopy>& copies) {
  if (tensor->data == nullptr) {
    return;
  }
  const auto* pointer = static_cast<const std::uint8_t*>(tensor->data);
  for (const auto& copy : copies) {
    const auto* source = static_cast<const std::uint8_t*>(copy.source);
    if (pointer >= source && pointer < source + copy.size) {
      tensor->data = static_cast<const std::uint8_t*>(copy.device) +
                     (pointer - source);
      return;
    }
  }
}

}  // namespace

// ─── Model ─────────────────────────────────────────────────────────────────

DFlashV1Model::~DFlashV1Model() {
  for (const auto& copy : mapped_copies_) {
    (void)hipHostUnregister(copy.host);
    (void)munmap(copy.host, copy.size);
  }
  mapped_copies_.clear();
}

std::shared_ptr<const DFlashV1Model> DFlashV1Model::Load(
    std::string_view sidecar_path, const DFlashV1TrunkTensors& trunk,
    std::uint32_t trunk_hidden_size, std::uint32_t trunk_vocab_size,
    float trunk_value_scale, std::string* error_msg) {
  if (sidecar_path.empty() || trunk.token_embedding == nullptr ||
      trunk.output == nullptr || trunk_hidden_size == 0 ||
      trunk_vocab_size == 0) {
    Fail(error_msg, "DFlash v1 sidecar arguments are incomplete");
    return nullptr;
  }
  auto reader = core::GgufReader::OpenFile(std::string(sidecar_path), error_msg);
  if (reader == nullptr) {
    return nullptr;
  }
  std::shared_ptr<DFlashV1Model> model(new DFlashV1Model());
  model->reader_ = std::move(reader);
  if (!model->LoadWeights(*model->reader_, trunk_hidden_size,
                          trunk_vocab_size, error_msg)) {
    return nullptr;
  }
  if (!RegisterMappedRegions(*model->reader_, &model->mapped_copies_,
                             error_msg)) {
    return nullptr;
  }
  const auto remap = [&model](DFlashV1Tensor* tensor) {
    RemapTensor(tensor, model->mapped_copies_);
  };
  remap(&model->fc_);
  remap(&model->enc_norm_);
  remap(&model->output_norm_);
  for (auto& layer : model->layers_) {
    remap(&layer.attn_norm);
    remap(&layer.attn_q);
    remap(&layer.attn_k);
    remap(&layer.attn_v);
    remap(&layer.attn_q_norm);
    remap(&layer.attn_k_norm);
    remap(&layer.attn_sinks);
    remap(&layer.attn_output);
    remap(&layer.ffn_norm);
    remap(&layer.ffn_gate);
    remap(&layer.ffn_up);
    remap(&layer.ffn_down);
  }
  // The sidecar carries no value scale; the HF drafter config's
  // `dflash_config.attention_value_scale` (0.612) is the checkpoint value.
  (void)trunk_value_scale;
  if (const char* override_scale = std::getenv("GUFO_DFLASH_VALUE_SCALE")) {
    model->config_.value_scale = static_cast<float>(std::atof(override_scale));
  }
  // Rotary width: the GGUF key wins, then the environment, then the HF
  // config's partial_rotary_factor, and the full head is the fallback.
  {
    DFlashV1Config& config = model->config_;
    bool resolved = false;
    if (const auto key =
            model->reader_->GetMetadataUint32("dflash.rope.dimension_count")) {
      config.rotary_dim = *key;
      resolved = true;
    }
    if (const char* override_rotary = std::getenv("GUFO_DFLASH_ROTARY_DIM")) {
      config.rotary_dim = static_cast<std::uint32_t>(std::atoi(override_rotary));
      resolved = true;
    }
    if (!resolved) {
      double factor = 1.0;
      if (ReadPartialRotaryFactor(sidecar_path, &factor)) {
        config.rotary_dim = static_cast<std::uint32_t>(
            static_cast<double>(config.head_dim) * factor);
      }
    }
    if (config.rotary_dim == 0 || config.rotary_dim > config.head_dim ||
        config.rotary_dim % 2 != 0) {
      Fail(error_msg, "DFlash v1 rotary dim is invalid");
      return nullptr;
    }
  }
  if (!model->LoadMaskEmbedding(sidecar_path, error_msg)) {
    return nullptr;
  }
  model->token_embedding_ = {.data = trunk.token_embedding,
                             .type = trunk.token_embedding_type,
                             .cols = trunk_hidden_size,
                             .rows = trunk_vocab_size};
  model->output_ = {.data = trunk.output,
                    .type = trunk.output_type,
                    .cols = trunk_hidden_size,
                    .rows = trunk_vocab_size};
  const auto supported_head = [](core::GgmlType type) {
    return type == core::GgmlType::kQ8_0 || type == core::GgmlType::kF16 ||
           type == core::GgmlType::kBF16 || type == core::GgmlType::kF32;
  };
  if (!supported_head(model->token_embedding_.type) ||
      !supported_head(model->output_.type)) {
    Fail(error_msg,
         "DFlash v1 requires a Q8_0/F16/BF16/F32 embedding and output head");
    return nullptr;
  }
  return model;
}

bool DFlashV1Model::LoadWeights(core::GgufReader& reader,
                                std::uint32_t trunk_hidden_size,
                                std::uint32_t trunk_vocab_size,
                                std::string* error_msg) {
  const auto architecture = reader.GetMetadataString("general.architecture");
  if (!architecture.has_value() || *architecture != "dflash") {
    return Fail(error_msg,
                "DFlash v1 sidecar requires general.architecture='dflash'");
  }
  auto& c = config_;
  if (!RequiredUint32(reader, "dflash.block_count", &c.num_layers, error_msg) ||
      !RequiredUint32(reader, "dflash.embedding_length", &c.hidden_size,
                      error_msg) ||
      !RequiredUint32(reader, "dflash.feed_forward_length",
                      &c.intermediate_size, error_msg) ||
      !RequiredUint32(reader, "dflash.attention.head_count", &c.num_heads,
                      error_msg) ||
      !RequiredUint32(reader, "dflash.attention.head_count_kv",
                      &c.num_kv_heads, error_msg) ||
      !RequiredUint32(reader, "dflash.attention.key_length", &c.head_dim,
                      error_msg) ||
      !RequiredUint32(reader, "dflash.block_size", &c.block_size, error_msg) ||
      !RequiredUint32(reader, "dflash.context_length", &c.context_length,
                      error_msg) ||
      !RequiredUint32(reader, "dflash.attention.sliding_window",
                      &c.sliding_window, error_msg) ||
      !RequiredFloat32(reader, "dflash.rope.freq_base", &c.rope_theta,
                       error_msg)) {
    return false;
  }
  const auto eps =
      reader.GetMetadataFloat32("dflash.attention.layer_norm_rms_epsilon");
  if (eps.has_value() && *eps > 0.0F) {
    c.rms_eps = *eps;
  }
  bool causal = true;
  if (!RequiredBool(reader, "dflash.attention.causal", &causal, error_msg)) {
    return false;
  }
  if (causal) {
    return Fail(error_msg, "DFlash v1 block attention must be non-causal");
  }
  // A sliding layer windows its context and keeps its block causal; a full
  // layer sees the whole context with a non-causal block (the v1 reference's
  // `dflash.attention.sliding_window_pattern`).
  if (const auto* pattern =
          reader.FindMetadata("dflash.attention.sliding_window_pattern")) {
    // The reader surfaces GGUF BOOL arrays as 0/1 uint64 vectors.
    if (const auto* bits =
            std::get_if<std::vector<std::uint64_t>>(&pattern->value)) {
      c.layer_swa.reserve(bits->size());
      for (const auto value : *bits) {
        c.layer_swa.push_back(value != 0);
      }
    } else {
      return Fail(error_msg,
                  "DFlash v1 sliding window pattern must be a boolean array");
    }
    if (c.layer_swa.size() != c.num_layers) {
      return Fail(error_msg,
                  "DFlash v1 sliding window pattern does not match the layer "
                  "count");
    }
  } else {
    c.layer_swa.assign(c.num_layers, c.sliding_window != 0);
  }
  if (!RequiredUint32(reader, "tokenizer.ggml.mask_token_id", &c.mask_token_id,
                      error_msg)) {
    return false;
  }
  if (const char* override_id = std::getenv("GUFO_DFLASH_MASK_ID")) {
    // Bring-up A/B: the drafting mask token (padding vs the sidecar's mask).
    c.mask_token_id = static_cast<std::uint32_t>(std::atoi(override_id));
  }
  if (!RequiredTargetLayers(reader, &c.target_layer_ids, error_msg)) {
    return false;
  }
  // Block-row causality: the sidecar records `is_causal=false` for MiMo's
  // drafter, so every slot reads the whole block.
  bool causal_block = false;
  if (const auto value = reader.GetMetadataBool("dflash.attention.causal")) {
    causal_block = *value;
  }
  if (const char* override_causal = std::getenv("GUFO_DFLASH_CAUSAL")) {
    causal_block = std::atoi(override_causal) != 0;
  }
  c.causal_block = causal_block;
  // The sidecar carries no rope width of its own; Load resolves it from the
  // drafter's HF config (partial_rotary_factor) or an override.
  c.rotary_dim = c.head_dim;
  std::uint32_t value_length = 0;
  if (!RequiredUint32(reader, "dflash.attention.value_length", &value_length,
                      error_msg)) {
    return false;
  }
  if (value_length != c.head_dim) {
    return Fail(error_msg,
                "DFlash v1 asymmetric value heads are not supported");
  }
  if (c.hidden_size != trunk_hidden_size) {
    return Fail(error_msg,
                "DFlash v1 hidden size does not match the target model");
  }
  if (c.mask_token_id >= trunk_vocab_size) {
    return Fail(error_msg,
                "DFlash v1 mask token is outside the target vocabulary");
  }
  if (c.num_heads == 0 || c.num_kv_heads == 0 ||
      c.num_heads % c.num_kv_heads != 0 || c.head_dim % 32 != 0 ||
      c.head_dim > 256 || c.block_size < 2) {
    return Fail(error_msg, "DFlash v1 attention geometry is unsupported");
  }

  const std::uint64_t hidden = c.hidden_size;
  const std::uint64_t enc = c.target_layer_ids.size() * hidden;
  const std::uint64_t q_rows = std::uint64_t{c.num_heads} * c.head_dim;
  const std::uint64_t kv_rows = std::uint64_t{c.num_kv_heads} * c.head_dim;
  const auto matrix = {core::GgmlType::kQ8_0, core::GgmlType::kQ6_K,
                       core::GgmlType::kBF16, core::GgmlType::kF16,
                       core::GgmlType::kF32};
  const auto vector = {core::GgmlType::kF32, core::GgmlType::kF16,
                       core::GgmlType::kBF16};

  if (!Bind(reader, "fc.weight", enc, hidden, matrix, &fc_, error_msg) ||
      !Bind(reader, "enc.output_norm.weight", hidden, 1, vector, &enc_norm_,
            error_msg) ||
      !Bind(reader, "output_norm.weight", hidden, 1, vector, &output_norm_,
            error_msg)) {
    return false;
  }
  layers_.resize(c.num_layers);
  for (std::uint32_t i = 0; i < c.num_layers; ++i) {
    const std::string p = "blk." + std::to_string(i) + ".";
    DFlashV1LayerWeights& l = layers_[i];
    if (!Bind(reader, p + "attn_norm.weight", hidden, 1, vector, &l.attn_norm,
              error_msg) ||
        !Bind(reader, p + "attn_q.weight", hidden, q_rows, matrix, &l.attn_q,
              error_msg) ||
        !Bind(reader, p + "attn_k.weight", hidden, kv_rows, matrix, &l.attn_k,
              error_msg) ||
        !Bind(reader, p + "attn_v.weight", hidden, kv_rows, matrix, &l.attn_v,
              error_msg) ||
        !Bind(reader, p + "attn_q_norm.weight", c.head_dim, 1, vector,
              &l.attn_q_norm, error_msg) ||
        !Bind(reader, p + "attn_k_norm.weight", c.head_dim, 1, vector,
              &l.attn_k_norm, error_msg) ||
        !Bind(reader, p + "attn_sinks.weight", c.num_heads, 1, vector,
              &l.attn_sinks, error_msg) ||
        !Bind(reader, p + "attn_output.weight", q_rows, hidden, matrix,
              &l.attn_output, error_msg) ||
        !Bind(reader, p + "ffn_norm.weight", hidden, 1, vector, &l.ffn_norm,
              error_msg) ||
        !Bind(reader, p + "ffn_gate.weight", hidden, c.intermediate_size,
              matrix, &l.ffn_gate, error_msg) ||
        !Bind(reader, p + "ffn_up.weight", hidden, c.intermediate_size, matrix,
              &l.ffn_up, error_msg) ||
        !Bind(reader, p + "ffn_down.weight", c.intermediate_size, hidden,
              matrix, &l.ffn_down, error_msg)) {
      return false;
    }
  }
  weight_bytes_ = 0;
  const auto account = [this](const DFlashV1Tensor& t) {
    if (t.data != nullptr) {
      weight_bytes_ += t.type == core::GgmlType::kQ8_0
                           ? (std::size_t{t.cols} / 32) * 34 * t.rows
                           : std::size_t{t.cols} * t.rows *
                                 (t.type == core::GgmlType::kF32 ? 4 : 2);
    }
  };
  account(fc_);
  for (const auto& l : layers_) {
    account(l.attn_q);
    account(l.attn_k);
    account(l.attn_v);
    account(l.attn_output);
    account(l.ffn_gate);
    account(l.ffn_up);
    account(l.ffn_down);
  }
  return true;
}


bool DFlashV1Model::LoadMaskEmbedding(std::string_view sidecar_path,
                                      std::string* error_msg) {
  // The trunk's embedding table holds zeros at the mask token, so the block's
  // masked slots have to come from the separately trained
  // `dflash/mask_embedding.pt` (converted to raw f32). Look next to the
  // sidecar, or take an explicit path from GUFO_DFLASH_MASK_EMB. When the file
  // is absent the mask slots keep the token embedding, which is what the
  // reference graph does without it, so a missing file is not an error.
  std::string path;
  if (const char* env_path = std::getenv("GUFO_DFLASH_MASK_EMB")) {
    path = env_path;
  } else {
    const std::size_t slash = sidecar_path.find_last_of('/');
    const std::string dir =
        slash == std::string_view::npos
            ? std::string()
            : std::string(sidecar_path.substr(0, slash + 1));
    for (const char* name : {"dflash-mask-embedding.f32", "mask-embedding.f32",
                             "mask_embedding.f32"}) {
      const std::string candidate = dir + name;
      if (std::FILE* probe = std::fopen(candidate.c_str(), "rb")) {
        std::fclose(probe);
        path = candidate;
        break;
      }
    }
  }
  if (path.empty()) {
    return true;
  }
  std::FILE* file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) {
    return Fail(error_msg,
                "DFlash v1 mask embedding is not readable: " + path);
  }
  std::vector<float> values(config_.hidden_size);
  const std::size_t read =
      std::fread(values.data(), sizeof(float), values.size(), file);
  std::fclose(file);
  if (read != values.size()) {
    return Fail(error_msg,
                "DFlash v1 mask embedding has the wrong size: " + path);
  }
  mask_embedding_ = std::move(values);
  return true;
}

// ─── Drafter ───────────────────────────────────────────────────────────────

DFlashV1Drafter::~DFlashV1Drafter() { Release(); }

void DFlashV1Drafter::Release() noexcept {
  if (block_graph_ != nullptr) {
    if (stream_ != nullptr) {
      (void)hipStreamSynchronize(stream_);
    }
    (void)hipGraphExecDestroy(block_graph_);
    block_graph_ = nullptr;
  }
  if (stream_ != nullptr) {
    (void)hipStreamSynchronize(stream_);
  }
  for (void* pointer : allocations_) {
    if (pointer != nullptr) {
      (void)hipFree(pointer);
    }
  }
  allocations_.clear();
  if (stream_ != nullptr) {
    (void)hipStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

std::unique_ptr<DFlashV1Drafter> DFlashV1Drafter::Create(
    std::shared_ptr<const DFlashV1Model> model, std::uint32_t max_context,
    std::string* error_msg) {
  if (model == nullptr || max_context == 0) {
    Fail(error_msg, "DFlash v1 drafter needs a model and a context size");
    return nullptr;
  }
  std::unique_ptr<DFlashV1Drafter> drafter(new DFlashV1Drafter());
  drafter->model_ = std::move(model);
  const std::uint32_t capacity =
      std::min(max_context, drafter->model_->config().sliding_window != 0
                                ? drafter->model_->config().sliding_window
                                : max_context);
  if (capacity == 0) {
    Fail(error_msg, "DFlash v1 history capacity degenerated");
    return nullptr;
  }
  drafter->history_capacity_ = capacity;
  drafter->injection_capacity_ = std::min(max_context, 256U);
  if (!drafter->Allocate(error_msg)) {
    return nullptr;
  }
  return drafter;
}

bool DFlashV1Drafter::Allocate(std::string* error_msg) {
  const auto& c = model_->config();
  if (hipStreamCreate(&stream_) != hipSuccess) {
    return Fail(error_msg, "DFlash v1 stream creation failed");
  }
  const std::size_t hidden = c.hidden_size;
  const std::size_t intermediate = c.intermediate_size;
  const std::size_t kv = std::size_t{c.num_kv_heads} * c.head_dim;
  const std::size_t q_width = std::size_t{c.num_heads} * c.head_dim;
  const std::size_t shared = std::max<std::size_t>(injection_capacity_,
                                                   c.block_size);

  ring_k_.resize(c.num_layers, nullptr);
  ring_v_.resize(c.num_layers, nullptr);
  for (std::uint32_t i = 0; i < c.num_layers; ++i) {
    ring_k_[i] = Alloc<float>(allocations_,
                              std::size_t{history_capacity_} * kv, error_msg,
                              &state_bytes_);
    ring_v_[i] = Alloc<float>(allocations_,
                              std::size_t{history_capacity_} * kv, error_msg,
                              &state_bytes_);
    if (ring_k_[i] == nullptr || ring_v_[i] == nullptr) {
      return false;
    }
  }
  d_features_ = Alloc<float>(
      allocations_, std::size_t{injection_capacity_} * target_features_width(),
      error_msg, &state_bytes_);
  d_fused_ = Alloc<float>(allocations_, shared * hidden, error_msg,
                          &state_bytes_);
  d_normed_ = Alloc<float>(allocations_, shared * hidden, error_msg,
                           &state_bytes_);
  d_k_block_ = Alloc<float>(allocations_, shared * kv, error_msg, &state_bytes_);
  d_v_block_ = Alloc<float>(allocations_, shared * kv, error_msg, &state_bytes_);
  d_block_hidden_ = Alloc<float>(allocations_, std::size_t{c.block_size} * hidden,
                                 error_msg, &state_bytes_);
  d_q_ = Alloc<float>(allocations_, std::size_t{c.block_size} * q_width,
                      error_msg, &state_bytes_);
  d_attn_out_ = Alloc<float>(allocations_, std::size_t{c.block_size} * q_width,
                             error_msg, &state_bytes_);
  d_ffn_gate_ =
      Alloc<float>(allocations_, std::size_t{c.block_size} * intermediate,
                   error_msg, &state_bytes_);
  d_ffn_up_ = Alloc<float>(allocations_, std::size_t{c.block_size} * intermediate,
                           error_msg, &state_bytes_);
  d_ffn_down_ = Alloc<float>(allocations_,
                             std::size_t{c.block_size} * hidden, error_msg,
                             &state_bytes_);
  d_logits_ = Alloc<float>(allocations_,
                           std::size_t{c.block_size} * model_->output().rows,
                           error_msg, &state_bytes_);
  d_block_tokens_ = Alloc<std::int32_t>(allocations_, c.block_size, error_msg,
                                        &state_bytes_);
  d_argmax_scratch_ = Alloc<rocm::ArgmaxCandidate>(
      allocations_, std::size_t{c.block_size} * rocm::kArgmaxParts, error_msg,
      &state_bytes_);
  d_drafts_ = Alloc<std::int32_t>(allocations_, c.block_size, error_msg,
                                  &state_bytes_);
  d_position_ = Alloc<std::uint32_t>(allocations_, 1, error_msg, &state_bytes_);
  if (!model_->mask_embedding().empty()) {
    d_mask_embedding_ =
        Alloc<float>(allocations_, model_->mask_embedding().size(), error_msg,
                     &state_bytes_);
    if (d_mask_embedding_ != nullptr &&
        hipMemcpy(d_mask_embedding_, model_->mask_embedding().data(),
                  model_->mask_embedding().size() * sizeof(float),
                  hipMemcpyHostToDevice) != hipSuccess) {
      d_mask_embedding_ = nullptr;
    }
  }
  w8a8_tiles_bytes_ = rocm::Q8TiledBytes(injection_capacity_,
                                         model_->fc().cols);
  d_w8a8_tiles_ = Alloc<std::uint8_t>(allocations_, w8a8_tiles_bytes_, error_msg,
                                      &state_bytes_);
  if (d_features_ == nullptr || d_fused_ == nullptr || d_normed_ == nullptr ||
      d_k_block_ == nullptr || d_v_block_ == nullptr ||
      d_block_hidden_ == nullptr || d_q_ == nullptr || d_attn_out_ == nullptr ||
      d_ffn_gate_ == nullptr || d_ffn_up_ == nullptr || d_ffn_down_ == nullptr ||
      d_logits_ == nullptr || d_block_tokens_ == nullptr ||
      d_argmax_scratch_ == nullptr || d_drafts_ == nullptr ||
      d_position_ == nullptr || d_w8a8_tiles_ == nullptr) {
    return false;
  }
  // The zero fills above run on the null stream; nothing may read the new
  // buffers until they have landed.
  return CheckHip(hipDeviceSynchronize(), "drafter init", error_msg);
}

void DFlashV1Drafter::Gemm(const DFlashV1Tensor& weight, const float* x,
                           float* out, std::uint32_t tokens) {
  if (tokens == 0 || weight.data == nullptr) {
    return;
  }
  const WeightType type = KernelType(weight.type);
  if (tokens <= 8) {
    rocm::SmallGemm(weight.data, type, x, out, tokens, weight.rows, weight.cols,
                    stream_);
    return;
  }
  if (type != WeightType::kQ8_0) {
    throw std::logic_error("DFlash v1 wide GEMM requires Q8_0 weights");
  }
  if (rocm::Q8TiledBytes(tokens, weight.cols) > w8a8_tiles_bytes_) {
    throw std::logic_error("DFlash v1 W8A8 staging buffer is too small");
  }
  rocm::QuantizeQ8Tiled(x, d_w8a8_tiles_, tokens, weight.cols, stream_);
  rocm::W8A8Gemm(weight.data, d_w8a8_tiles_, out, tokens, weight.rows,
                 weight.cols, stream_);
}

bool DFlashV1Drafter::InjectTargetContext(std::span<const float> features,
                                          std::uint32_t position,
                                          std::uint32_t num_tokens) {
  const std::size_t width = target_features_width();
  if (position != injected_length_ || num_tokens == 0 ||
      position > model_->config().context_length ||
      num_tokens > model_->config().context_length - position ||
      features.size() != std::size_t{num_tokens} * width) {
    return false;
  }
  // Only the final attention window survives this injection; skip complete
  // chunks it overwrites, keeping batch boundaries and absolute positions.
  const std::uint32_t expired =
      num_tokens > history_capacity_ ? num_tokens - history_capacity_ : 0;
  const std::uint32_t first_chunk =
      (expired / injection_capacity_) * injection_capacity_;
  for (std::uint32_t offset = first_chunk; offset < num_tokens;) {
    const std::uint32_t count =
        std::min(injection_capacity_, num_tokens - offset);
    if (!InjectChunk(features.subspan(static_cast<std::size_t>(offset) * width,
                                      static_cast<std::size_t>(count) * width),
                     position + offset, count)) {
      return false;
    }
    offset += count;
  }
  injected_length_ = std::max(injected_length_, position + num_tokens);
  return true;
}

bool DFlashV1Drafter::InjectChunk(std::span<const float> features,
                                  std::uint32_t position, std::uint32_t count) {
  const auto& c = model_->config();
  const std::size_t kv = std::size_t{c.num_kv_heads} * c.head_dim;
  std::string error;
  if (!CheckHip(hipMemcpyAsync(d_features_, features.data(),
                               features.size() * sizeof(float),
                               hipMemcpyHostToDevice, stream_),
                "feature upload", &error) ||
      !CheckHip(hipMemcpyAsync(d_position_, &position, sizeof(position),
                               hipMemcpyHostToDevice, stream_),
                "position upload", &error)) {
    return false;
  }
  // Bring-up A/B: the vLLM drafter normalizes every tap slice to unit RMS
  // before the fc; the HF shim and llama.cpp feed the raw concatenation.
  static const bool slice_norm = std::getenv("GUFO_DFLASH_SLICE_NORM") != nullptr;
  if (slice_norm) {
    rocm::DFlashV1SliceRmsNorm(
        d_features_, count, c.hidden_size,
        static_cast<std::uint32_t>(model_->target_layer_ids().size()),
        c.rms_eps, stream_);
  }
  Gemm(model_->fc(), d_features_, d_fused_, count);
  rocm::RmsNormRows(d_fused_,
                    static_cast<const float*>(model_->enc_norm().data),
                    d_normed_, count, c.hidden_size, 1, c.rms_eps, stream_);
  // Bring-up A/B: the vLLM drafter also re-normalizes the fused context with
  // each layer's input_layernorm before the K/V projections.
  static const bool ctx_norm = std::getenv("GUFO_DFLASH_CTX_NORM") != nullptr;
  for (std::size_t i = 0; i < model_->layers().size(); ++i) {
    const auto& l = model_->layers()[i];
    if (ctx_norm) {
      rocm::RmsNormRows(d_normed_,
                        static_cast<const float*>(l.attn_norm.data), d_normed_,
                        count, c.hidden_size, 1, c.rms_eps, stream_);
    }
    Gemm(l.attn_k, d_normed_, d_k_block_, count);
    Gemm(l.attn_v, d_normed_, d_v_block_, count);
    // The 128-wide k norm weight is shared by every kv head, so each head
    // slice is one norm row here.
    rocm::RmsNormRows(d_k_block_,
                      static_cast<const float*>(l.attn_k_norm.data),
                      d_k_block_, count * c.num_kv_heads, c.head_dim, 1,
                      c.rms_eps, stream_);
    rocm::Rope(d_k_block_, count, c.num_kv_heads, c.head_dim, c.rotary_dim,
               d_position_, c.rope_theta, stream_);
    for (std::uint32_t offset = 0; offset < count;) {
      const std::uint32_t slot = (position + offset) % history_capacity_;
      const std::uint32_t segment =
          std::min(count - offset, history_capacity_ - slot);
      const std::size_t bytes = std::size_t{segment} * kv * sizeof(float);
      if (!CheckHip(hipMemcpyAsync(ring_k_[i] + std::size_t{slot} * kv,
                                   d_k_block_ + std::size_t{offset} * kv, bytes,
                                   hipMemcpyDeviceToDevice, stream_),
                    "ring K store", &error) ||
          !CheckHip(hipMemcpyAsync(ring_v_[i] + std::size_t{slot} * kv,
                                   d_v_block_ + std::size_t{offset} * kv, bytes,
                                   hipMemcpyDeviceToDevice, stream_),
                    "ring V store", &error)) {
        return false;
      }
      offset += segment;
    }
  }
  return CheckHip(hipStreamSynchronize(stream_), "injection drain", &error);
}

std::vector<std::int32_t> DFlashV1Drafter::ForwardBlock(
    std::int32_t anchor_token, std::uint32_t position,
    std::uint32_t draft_count) {
  const auto& c = model_->config();
  const std::size_t hidden = c.hidden_size;
  const std::size_t kv = std::size_t{c.num_kv_heads} * c.head_dim;
  const std::size_t q_width = std::size_t{c.num_heads} * c.head_dim;
  draft_count = std::min(draft_count, c.block_size - 1);
  if (anchor_token < 0 || position != injected_length_ ||
      static_cast<std::uint32_t>(anchor_token) >= model_->output().rows ||
      draft_count == 0) {
    if (injected_trace()) {
      fprintf(stderr,
              "[DFTRACE] block refused anchor=%d pos=%u injected=%u count=%u\n",
              anchor_token, position, injected_length_, draft_count);
      fflush(stderr);
    }
    return {};
  }
  // The trained block is always `block_size` rows (anchor + mask slots): the
  // non-causal block attention mixes every slot, so a shortened block would
  // condition the drafts on a different mixture than training. Only the
  // returned draft count follows the requested width.
  const std::uint32_t block_count = c.block_size;
  std::string error;
  std::vector<std::int32_t> block_tokens(
      block_count, static_cast<std::int32_t>(c.mask_token_id));
  block_tokens.front() = anchor_token;
  if (!CheckHip(hipMemcpyAsync(d_block_tokens_, block_tokens.data(),
                               block_count * sizeof(std::int32_t),
                               hipMemcpyHostToDevice, stream_),
                "block token upload", &error) ||
      !CheckHip(hipMemcpyAsync(d_position_, &position, sizeof(position),
                               hipMemcpyHostToDevice, stream_),
                "position upload", &error)) {
    return {};
  }
  // The block forward is shape-static (its only inputs are the token row, the
  // device position and the ring contents), so one captured graph replays for
  // every step instead of paying a hundred kernel launches per block.
  const bool graph_enabled =
      !debug_ && std::getenv("GUFO_DFLASH_NO_GRAPH") == nullptr;
  if (!graph_enabled) {
    if (!BlockBody(block_count)) {
      return {};
    }
  } else {
    if (block_graph_ == nullptr) {
      hipGraph_t captured = nullptr;
      if (!CheckHip(hipStreamBeginCapture(stream_,
                                          hipStreamCaptureModeThreadLocal),
                    "block graph capture", &error)) {
        return {};
      }
      const bool built = BlockBody(block_count);
      if (!CheckHip(hipStreamEndCapture(stream_, &captured),
                    "block graph capture end", &error) ||
          !built) {
        if (captured != nullptr) {
          (void)hipGraphDestroy(captured);
        }
        return {};
      }
      const bool instantiated = CheckHip(
          hipGraphInstantiate(&block_graph_, captured, nullptr, nullptr, 0),
          "block graph instantiate", &error);
      (void)hipGraphDestroy(captured);
      if (!instantiated) {
        block_graph_ = nullptr;
        return {};
      }
      if (injected_trace()) {
        fprintf(stderr, "[DFTRACE] block graph captured\n");
        fflush(stderr);
      }
    }
    if (!CheckHip(hipGraphLaunch(block_graph_, stream_), "block graph launch",
                  &error)) {
      return {};
    }
  }
  host_drafts_.assign(block_count - 1, 0);
  if (!CheckHip(hipMemcpyAsync(host_drafts_.data(), d_drafts_,
                               (block_count - 1) * sizeof(std::int32_t),
                               hipMemcpyDeviceToHost, stream_),
                "draft readback", &error) ||
      !CheckHip(hipStreamSynchronize(stream_), "block drain", &error)) {
    return {};
  }
  const std::vector<std::int32_t> drafts(host_drafts_.begin(),
                                         host_drafts_.begin() + draft_count);
  last_block_count_ = block_count;
  last_draft_count_ = draft_count;
  last_drafts_ = drafts;
  return drafts;
}

bool DFlashV1Drafter::BlockBody(std::uint32_t block_count) {
  const auto& c = model_->config();
  const std::size_t hidden = c.hidden_size;
  const std::size_t kv = std::size_t{c.num_kv_heads} * c.head_dim;
  const std::size_t q_width = std::size_t{c.num_heads} * c.head_dim;
  rocm::EmbedTokens(model_->token_embedding().data,
                    KernelType(model_->token_embedding().type), d_block_tokens_,
                    d_block_hidden_, block_count,
                    static_cast<std::uint32_t>(hidden), 1, stream_);
  // The trunk's table is zero at the mask token: without the separately
  // trained mask embedding every masked slot would be an all-zero row.
  if (d_mask_embedding_ != nullptr) {
    rocm::DFlashV1MaskSubstitute(
        d_block_hidden_, d_block_tokens_, block_count,
        static_cast<std::uint32_t>(hidden),
        static_cast<std::int32_t>(c.mask_token_id), d_mask_embedding_, stream_);
  }
  if (debug_) {
    debug_embedding_.resize(std::size_t{block_count} * hidden);
    (void)hipMemcpyAsync(debug_embedding_.data(), d_block_hidden_,
                         debug_embedding_.size() * sizeof(float),
                         hipMemcpyDeviceToHost, stream_);
  }
  const float scale = 1.0F / std::sqrt(static_cast<float>(c.head_dim));
  for (std::size_t i = 0; i < model_->layers().size(); ++i) {
    const auto& l = model_->layers()[i];
    rocm::RmsNormRows(d_block_hidden_,
                      static_cast<const float*>(l.attn_norm.data), d_normed_,
                      block_count, c.hidden_size, 1, c.rms_eps, stream_);
    Gemm(l.attn_q, d_normed_, d_q_, block_count);
    Gemm(l.attn_k, d_normed_, d_k_block_, block_count);
    Gemm(l.attn_v, d_normed_, d_v_block_, block_count);
    if (debug_ && i == 0) {
      debug_raw_query_.resize(std::size_t{block_count} * q_width);
      (void)hipMemcpyAsync(debug_raw_query_.data(), d_q_,
                           debug_raw_query_.size() * sizeof(float),
                           hipMemcpyDeviceToHost, stream_);
    }
    // Per-head q/k RMSNorm with the shared head_dim-wide weight: one norm
    // row per (token, head).
    rocm::RmsNormRows(d_q_, static_cast<const float*>(l.attn_q_norm.data), d_q_,
                      block_count * c.num_heads, c.head_dim, 1, c.rms_eps,
                      stream_);
    rocm::RmsNormRows(d_k_block_,
                      static_cast<const float*>(l.attn_k_norm.data),
                      d_k_block_, block_count * c.num_kv_heads, c.head_dim, 1,
                      c.rms_eps, stream_);
    if (debug_ && i == 0) {
      debug_normed_query_.resize(std::size_t{block_count} * q_width);
      (void)hipMemcpyAsync(debug_normed_query_.data(), d_q_,
                           debug_normed_query_.size() * sizeof(float),
                           hipMemcpyDeviceToHost, stream_);
    }
    rocm::Rope(d_q_, block_count, c.num_heads, c.head_dim, c.rotary_dim,
               d_position_, c.rope_theta, stream_);
    rocm::Rope(d_k_block_, block_count, c.num_kv_heads, c.head_dim, c.rotary_dim,
               d_position_, c.rope_theta, stream_);
    if (debug_ && i == 0) {
      const std::size_t q_floats = std::size_t{block_count} * q_width;
      const std::size_t kv_floats = std::size_t{block_count} * kv;
      debug_query_.resize(q_floats);
      debug_block_key_.resize(kv_floats);
      debug_block_value_.resize(kv_floats);
      debug_ring_key_.resize(kv_floats);
      (void)hipMemcpyAsync(debug_query_.data(), d_q_,
                           q_floats * sizeof(float), hipMemcpyDeviceToHost,
                           stream_);
      (void)hipMemcpyAsync(debug_block_key_.data(), d_k_block_,
                           kv_floats * sizeof(float), hipMemcpyDeviceToHost,
                           stream_);
      (void)hipMemcpyAsync(debug_block_value_.data(), d_v_block_,
                           kv_floats * sizeof(float), hipMemcpyDeviceToHost,
                           stream_);
      (void)hipMemcpyAsync(debug_ring_key_.data(), ring_k_[i],
                           kv_floats * sizeof(float), hipMemcpyDeviceToHost,
                           stream_);
    }
    // The v1 reference graph never reads `attn_sinks`; MiMo's sidecar still
    // carries per-head sinks, so GUFO_DFLASH_SINKS=1 re-enables them for A/B.
    static const bool use_sinks = std::getenv("GUFO_DFLASH_NO_SINKS") == nullptr;
    const bool layer_swa =
        i < c.layer_swa.size() ? c.layer_swa[i] : c.sliding_window != 0;
    rocm::DFlashV1BlockAttention(
        d_q_, ring_k_[i], ring_v_[i], d_k_block_, d_v_block_,
        use_sinks ? static_cast<const float*>(l.attn_sinks.data) : nullptr,
        d_attn_out_, d_position_, block_count,
        layer_swa ? c.sliding_window : 0U, c.num_heads, c.num_kv_heads,
        c.head_dim, scale, history_capacity_, c.causal_block, stream_);
    Gemm(l.attn_output, d_attn_out_, d_fused_, block_count);
    // MiMo scales the attention output by attention_value_scale (the target
    // applies the same constant after its projection). Scaling here covers the
    // injected context V and the block V alike.
    if (c.value_scale != 1.0F) {
      rocm::ScaleInPlace(d_fused_, c.value_scale,
                         std::size_t{block_count} * hidden, stream_);
    }
    rocm::DFlashV1AddInPlace(d_block_hidden_, d_fused_, block_count * hidden,
                             stream_);
    if (debug_ && i == 0) {
      debug_after_attention_.resize(std::size_t{block_count} * hidden);
      (void)hipMemcpyAsync(debug_after_attention_.data(), d_block_hidden_,
                           debug_after_attention_.size() * sizeof(float),
                           hipMemcpyDeviceToHost, stream_);
    }
    rocm::RmsNormRows(d_block_hidden_,
                      static_cast<const float*>(l.ffn_norm.data), d_normed_,
                      block_count, c.hidden_size, 1, c.rms_eps, stream_);
    Gemm(l.ffn_gate, d_normed_, d_ffn_gate_, block_count);
    Gemm(l.ffn_up, d_normed_, d_ffn_up_, block_count);
    rocm::Swiglu(d_ffn_gate_, d_ffn_up_,
                 std::size_t{block_count} * c.intermediate_size, stream_);
    Gemm(l.ffn_down, d_ffn_gate_, d_ffn_down_, block_count);
    rocm::DFlashV1AddInPlace(d_block_hidden_, d_ffn_down_,
                             block_count * hidden, stream_);
    if (debug_) {
      // Row 0 of the hidden at the end of every layer, so a stage-by-stage
      // comparison can name the first layer that diverges.
      std::vector<float> row0(hidden, 0.0F);
      if (hipMemcpyAsync(row0.data(), d_block_hidden_, hidden * sizeof(float),
                         hipMemcpyDeviceToHost, stream_) == hipSuccess) {
        (void)hipStreamSynchronize(stream_);
        debug_layer_hidden_.insert(debug_layer_hidden_.end(), row0.begin(),
                                   row0.end());
      }
    }
  }
  rocm::RmsNormRows(d_block_hidden_,
                    static_cast<const float*>(model_->output_norm().data),
                    d_normed_, block_count, c.hidden_size, 1, c.rms_eps,
                    stream_);
  // Slots 1..block_size-1 predict the drafts; slot 0 is the anchor. Every
  // slot is computed so one captured graph serves any requested width.
  Gemm(model_->output(), d_normed_ + hidden, d_logits_, block_count - 1);
  rocm::Argmax(d_logits_, static_cast<rocm::ArgmaxCandidate*>(d_argmax_scratch_),
               d_drafts_, block_count - 1, model_->output().rows, stream_);
  return true;
}

std::span<const float> DFlashV1Drafter::CopyLogitsRow(std::size_t row) {
  if (d_logits_ == nullptr || last_draft_count_ == 0 ||
      row >= last_draft_count_) {
    return {};
  }
  const std::size_t vocab = static_cast<std::size_t>(model_->output().rows);
  debug_logits_row_.resize(vocab);
  if (hipMemcpyAsync(debug_logits_row_.data(),
                     d_logits_ + row * vocab, vocab * sizeof(float),
                     hipMemcpyDeviceToHost, stream_) != hipSuccess ||
      hipStreamSynchronize(stream_) != hipSuccess) {
    debug_logits_row_.clear();
    return {};
  }
  return debug_logits_row_;
}

bool DFlashV1Drafter::CopyBlockDebug(BlockDebug* out, std::string* error) {
  if (out == nullptr || last_block_count_ == 0 ||
      d_block_hidden_ == nullptr || d_normed_ == nullptr ||
      d_logits_ == nullptr) {
    return false;
  }
  const auto& c = model_->config();
  const auto width = static_cast<std::size_t>(c.hidden_size);
  const auto vocab = static_cast<std::size_t>(model_->output().rows);
  const std::size_t block = last_block_count_;
  const std::size_t drafts = last_draft_count_;
  out->block_count = static_cast<std::uint32_t>(block);
  out->draft_count = static_cast<std::uint32_t>(drafts);
  out->hidden.resize(block * width);
  out->normed.resize(block * width);
  out->logits.resize(drafts * vocab);
  out->drafts = last_drafts_;
  out->embedding = debug_embedding_;
  out->after_attention = debug_after_attention_;
  out->query = debug_query_;
  out->raw_query = debug_raw_query_;
  out->normed_query = debug_normed_query_;
  out->layer_hidden = debug_layer_hidden_;
  out->block_key = debug_block_key_;
  out->block_value = debug_block_value_;
  out->ring_key = debug_ring_key_;
  if (!CheckHip(hipMemcpyAsync(out->hidden.data(), d_block_hidden_,
                               out->hidden.size() * sizeof(float),
                               hipMemcpyDeviceToHost, stream_),
                "block hidden dump", error) ||
      !CheckHip(hipMemcpyAsync(out->normed.data(), d_normed_,
                               out->normed.size() * sizeof(float),
                               hipMemcpyDeviceToHost, stream_),
                "block normed dump", error) ||
      !CheckHip(hipMemcpyAsync(out->logits.data(), d_logits_,
                               out->logits.size() * sizeof(float),
                               hipMemcpyDeviceToHost, stream_),
                "block logits dump", error) ||
      !CheckHip(hipStreamSynchronize(stream_), "block debug drain", error)) {
    return false;
  }
  return true;
}

// ─── Draft width policy ────────────────────────────────────────────────────

std::uint32_t DFlashV1LengthController::Choose(
    std::uint32_t budget) const noexcept {
  const std::uint32_t cap = std::min(budget, limit_);
  if (cap == 0) {
    return 0;
  }
  // Bring-up diagnostics: GUFO_DFLASH_FORCE_WIDTH pins the draft width so
  // acceptance and per-cycle costs can be measured without the cost
  // controller backing off. A comma-separated list rotates one width per
  // cycle; a zero entry pins plain decode, which yields the baseline a
  // proposal has to beat. The list is read once per process.
  static const std::vector<std::uint32_t> forced = [] {
    std::vector<std::uint32_t> list;
    const char* env = std::getenv("GUFO_DFLASH_FORCE_WIDTH");
    for (const char* p = env; p != nullptr && *p != '\0';) {
      char* end = nullptr;
      const long value = std::strtol(p, &end, 10);
      if (end == p) {
        ++p;
        continue;
      }
      list.push_back(static_cast<std::uint32_t>(value));
      p = end;
    }
    return list;
  }();
  if (!forced.empty()) {
    static std::size_t rotation = 0;
    return std::min(cap, forced[rotation++ % forced.size()]);
  }
  if (warmup_steps_ > 0) {
    // Nothing has been measured yet: draft a small width so the cost model
    // below starts from this trunk's real numbers instead of guesses.
    return std::min(cap, 2u);
  }
  // Cost in milliseconds, from the TB4 pair's measured cycle: one draft block
  // is ~16 ms (five layers of Q8 sidecar over eight rows), one undrafted step
  // ~45 ms. A verification row costs 22 ms on repetitive content (the drafts
  // and the anchor then share experts) up to 41 ms on mixed prose, so these
  // are the averages a fixed policy has to live with; the width/keep ratio is
  // what decides, and this pair of numbers keeps wide drafting enabled on
  // high-acceptance content, where rows are the cheap end of that range. The
  // constants have to be shared by both ranks: the two processes run the same
  // draft decisions, and a width that followed each rank's own timings would
  // desync the collectives.
  constexpr float kBlockMs = 16.2F;
  constexpr float kRowMs = 22.0F;
  constexpr float kPlainMs = 45.4F;
  constexpr float kDeeperDecay = 0.84F;
  float survival = 1.0F;
  float expected = 1.0F;  // the sampled target anchor
  float deeper_ = p2_;   // walks the depth-decayed conditional rate
  // A plain step is one token for kPlainMs: that is the value to beat.
  float best_score = 1.0F / kPlainMs;
  std::uint32_t best = 0;
  for (std::uint32_t width = 1; width <= cap; ++width) {
    const std::uint32_t depth = width - 1;
    // An explored depth with no recorded rejection stays optimistic: a fully
    // accepted chain must be allowed to probe wider.
    const bool unproven = depth < explored_depth_ &&
                          (failed_depths_ & (1U << depth)) == 0;
    // Conditional acceptance decays with depth: measured on the TB4 pair at
    // 0.64 / 0.52 / 0.47 / 0.39 / 0.33 / 0.28 for slots two to seven (ratios
    // ~0.84). Using one rate for every deeper slot over-drafts wide.
    const float probability =
        unproven ? 1.0F : (depth == 0 ? p1_ : deeper_);
    deeper_ *= kDeeperDecay;
    survival *= probability;
    expected += survival;
    const float cost = kBlockMs + kRowMs * static_cast<float>(width + 1);
    const float score = expected / cost;
    if (score > best_score) {
      best_score = score;
      best = width;
    }
  }
  if (best == 0 && plain_steps_since_probe_ >= probe_interval_) {
    // Exploration: re-sample a small width after a long undrafted stretch.
    return std::min(cap, 2u);
  }
  return std::min(cap, std::max(best, probe_depth_));
}

void DFlashV1LengthController::Observe(std::size_t accepted,
                                       std::size_t drafted) noexcept {
  if (warmup_steps_ > 0) {
    --warmup_steps_;
  }
  if (drafted == 0) {
    return;
  }
  if (injected_trace()) {
    fprintf(stderr, "[DFTRACE] observe accepted=%zu drafted=%zu\n", accepted,
            drafted);
    fflush(stderr);
  }
  drafted = std::min<std::size_t>(drafted, limit_);
  accepted = std::min(accepted, drafted);
  const std::size_t observed = accepted + (accepted < drafted ? 1 : 0);
  // A convex EWMA of the 0/1 acceptance at each depth: `0.75x + 1` would sit
  // at a fixed point of 4, so the estimates have to move toward the
  // observation by the deficit, not by a constant.
  constexpr float kAlpha = 0.25F;
  for (std::size_t depth = 0; depth < observed; ++depth) {
    float& estimate = depth == 0 ? p1_ : p2_;
    const float observation = depth < accepted ? 1.0F : 0.0F;
    estimate += kAlpha * (observation - estimate);
  }
  explored_depth_ =
      std::max<std::uint32_t>(explored_depth_,
                              static_cast<std::uint32_t>(observed));
  if (accepted < drafted &&
      accepted < std::numeric_limits<std::uint32_t>::digits) {
    failed_depths_ |= 1U << accepted;
  }
  probe_depth_ = accepted == drafted && drafted < limit_
                     ? std::min<std::uint32_t>(2 * static_cast<std::uint32_t>(drafted), limit_)
                     : 0;
  // A probe that finds nothing doubles the next interval: on content that
  // never accepts drafts the exploration is pure overhead. Any accepted draft
  // resets it, so a workload shift is picked up on the next probe.
  if (accepted > 0) {
    probe_interval_ = 24;
  } else if (probe_interval_ < 512) {
    probe_interval_ *= 2;
  }
  observed_ = true;
}

void DFlashV1LengthController::ObservePlainStep(double ms) noexcept {
  if (ms <= 0.0) {
    return;
  }
  constexpr float kAlpha = 0.25F;
  plain_ms_ = plain_ms_ <= 0.0F
                  ? static_cast<float>(ms)
                  : plain_ms_ + kAlpha * (static_cast<float>(ms) - plain_ms_);
}

void DFlashV1LengthController::ObserveCosts(double block_ms, double verify_ms,
                                            std::uint32_t rows) noexcept {
  const auto blend = [](float current, double sample) {
    constexpr float kAlpha = 0.25F;
    if (current <= 0.0F) {
      return static_cast<float>(sample);
    }
    return current + kAlpha * (static_cast<float>(sample) - current);
  };
  if (block_ms > 0.0) {
    block_ms_ = blend(block_ms_, block_ms);
  }
  if (verify_ms > 0.0 && rows > 0) {
    row_ms_ = blend(row_ms_, verify_ms / static_cast<double>(rows));
    if (rows == 1) {
      // An undrafted step is exactly one row: that is the value a draft must
      // beat, measured rather than assumed.
      plain_ms_ = blend(plain_ms_, verify_ms);
    }
  }
}

// ─── Draft backend ─────────────────────────────────────────────────────────

std::size_t DFlashV1DraftBackend::StateBytesFor(const DFlashV1Model& model,
                                                std::uint32_t max_context) noexcept {
  // Mirrors DFlashV1Drafter::Allocate(). Every buffer is f32 except the
  // token/argmax scratch and the W8A8 tile staging.
  const auto& c = model.config();
  const std::size_t kv = std::size_t{c.num_kv_heads} * c.head_dim;
  const std::size_t q_width = std::size_t{c.num_heads} * c.head_dim;
  const std::uint32_t sliding = c.sliding_window != 0 ? c.sliding_window
                                                      : max_context;
  const std::size_t history = std::min<std::uint32_t>(max_context, sliding);
  const std::size_t injection = std::min<std::uint32_t>(max_context, 256U);
  const std::size_t shared = std::max<std::size_t>(injection, c.block_size);
  const std::size_t hidden = c.hidden_size;
  const std::size_t block = c.block_size;
  const std::size_t f32 = sizeof(float);
  std::size_t bytes = 0;
  bytes += 2 * std::size_t{c.num_layers} * history * kv * f32;   // ring K/V
  bytes += injection * model.target_features_width() * f32;      // d_features
  bytes += 2 * shared * hidden * f32;                            // fused/normed
  bytes += 2 * shared * kv * f32;                                // k/v block
  bytes += block * hidden * f32;                                 // block_hidden
  bytes += 2 * block * q_width * f32;                            // q/attn_out
  bytes += 2 * block * c.intermediate_size * f32;                // ffn gate/up
  bytes += block * hidden * f32;                                 // ffn_down
  bytes += block * model.output().rows * f32;                    // logits
  bytes += 2 * block * sizeof(std::int32_t);                     // tokens/drafts
  bytes += block * rocm::kArgmaxParts * sizeof(rocm::ArgmaxCandidate);
  bytes += sizeof(std::uint32_t);                                // position
  bytes += model.mask_embedding().size() * f32;
  bytes += rocm::Q8TiledBytes(injection, model.fc().cols);       // w8a8 tiles
  // Headroom for allocator alignment and any drift from Allocate().
  return bytes + bytes / 10;
}

std::unique_ptr<DFlashV1DraftBackend> DFlashV1DraftBackend::Create(
    std::shared_ptr<const DFlashV1Model> model, std::uint32_t max_context,
    std::uint32_t max_draft_tokens, std::string* error_msg) {
  if (max_draft_tokens == 0) {
    Fail(error_msg, "DFlash v1 draft limit is zero");
    return nullptr;
  }
  auto drafter = DFlashV1Drafter::Create(std::move(model), max_context,
                                         error_msg);
  if (drafter == nullptr) {
    return nullptr;
  }
  const std::uint32_t block = drafter->config().block_size;
  if (block < 2) {
    Fail(error_msg, "DFlash v1 block size must be at least two");
    return nullptr;
  }
  const std::uint32_t limit = std::min(
      {max_draft_tokens, block - 1, DFlashV1LengthController::kMaxDraftTokens});
  std::unique_ptr<DFlashV1DraftBackend> backend(
      new DFlashV1DraftBackend(std::move(drafter), limit));
  backend->max_context_ = max_context;
  return backend;
}

std::uint32_t DFlashV1DraftBackend::PendingFeatureCount(
    std::uint32_t current_pos) const {
  const std::uint32_t injected = drafter_->injected_length();
  if (current_pos < injected) {
    throw std::logic_error(
        "DFlash v1 committed position precedes injected history");
  }
  if (current_pos > injected) {
    const std::uint32_t count = current_pos - injected;
    const std::size_t expected =
        std::size_t{count} * drafter_->target_features_width();
    if (pending_target_features_.size() != expected) {
      throw std::logic_error(
          "DFlash v1 committed target feature history is incomplete");
    }
    return count;
  }
  if (!pending_target_features_.empty()) {
    throw std::logic_error(
        "DFlash v1 has target features without a matching committed position");
  }
  return 0;
}

void DFlashV1DraftBackend::InjectPendingFeatures(std::uint32_t current_pos) {
  const std::uint32_t count = PendingFeatureCount(current_pos);
  if (count == 0) {
    return;
  }
  if (!drafter_->InjectTargetContext(pending_target_features_,
                                     drafter_->injected_length(), count)) {
    throw std::runtime_error(
        "DFlash v1 committed target feature injection failed");
  }
  pending_target_features_.clear();
}

bool DFlashV1DraftBackend::PrimeTargetContext(
    const speculative::DraftTargetContext& context) {
  if (injected_trace()) {
    double sum = 0.0;
    double square = 0.0;
    for (const float value : context.prompt_hidden_states) {
      sum += value;
      square += static_cast<double>(value) * value;
    }
    const std::size_t rows = context.prompt_tokens.size();
    const auto sample = [&](std::size_t row, std::size_t offset) {
      const std::size_t index = row * context.hidden_size + offset;
      return index < context.prompt_hidden_states.size()
                 ? context.prompt_hidden_states[index]
                 : 0.0F;
    };
    fprintf(stderr,
            "[DFTRACE] backend prime tokens=%zu width=%zu sum=%.4f rms=%.5f "
            "rows(first/mid/last)=[%.5f %.5f %.5f]\n",
            rows, drafter_->target_features_width(), sum,
            std::sqrt(square /
                      static_cast<double>(std::max<std::size_t>(
                          1, context.prompt_hidden_states.size()))),
            sample(0, 0), sample(rows / 2, 0), sample(rows - 1, 0));
    fflush(stderr);
  }
  Reset();
  if (context.prompt_tokens.empty() ||
      context.prompt_tokens.size() > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  const std::size_t width = drafter_->target_features_width();
  const std::size_t tokens = context.prompt_tokens.size();
  if (context.hidden_size != width ||
      context.prompt_hidden_states.size() != tokens * width) {
    return false;
  }
  if (const char* dir = dump_dir()) {
    static std::uint32_t dump_index = 0;
    const std::string base =
        std::string(dir) + "/feat_" + std::to_string(dump_index++);
    if (WriteBinaryFile(base + ".f32", context.prompt_hidden_states.data(),
                        context.prompt_hidden_states.size() * sizeof(float))) {
      std::string tokens_text;
      for (const auto token : context.prompt_tokens) {
        tokens_text += std::to_string(token) + "\n";
      }
      (void)WriteBinaryFile(base + ".tokens", tokens_text.data(),
                            tokens_text.size());
      fprintf(stderr, "[DFDUMP] features -> %s.f32 (%zu rows x %zu)\n",
              base.c_str(), tokens, width);
      fflush(stderr);
    }
  }
  if (!drafter_->InjectTargetContext(context.prompt_hidden_states, 0,
                                     static_cast<std::uint32_t>(tokens))) {
    Reset();
    return false;
  }
  primed_ = true;
  return true;
}

bool DFlashV1DraftBackend::AppendTargetContext(
    const speculative::DraftTargetContext& context, std::uint32_t position) {
  const std::size_t width = drafter_->target_features_width();
  if (!primed_ || proposal_active_ || context.hidden_size != width ||
      context.prompt_hidden_states.size() !=
          context.prompt_tokens.size() * width ||
      context.prompt_tokens.empty() ||
      context.prompt_tokens.size() > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  InjectPendingFeatures(position);
  if (injected_trace()) {
    double sum = 0.0;
    for (const float value : context.prompt_hidden_states) {
      sum += value;
    }
    fprintf(stderr,
            "[DFTRACE] append pos=%u rows=%zu sum=%.4f first=%.5f last=%.5f\n",
            position, context.prompt_tokens.size(), sum,
            context.prompt_hidden_states.front(),
            context.prompt_hidden_states[(context.prompt_tokens.size() - 1) *
                                             width]);
    fflush(stderr);
  }
  return drafter_->InjectTargetContext(
      context.prompt_hidden_states, position,
      static_cast<std::uint32_t>(context.prompt_tokens.size()));
}

void DFlashV1DraftBackend::UpdateTargetHidden(std::span<const float> hidden) {
  if (hidden.size() != drafter_->target_features_width()) {
    throw std::invalid_argument(
        "DFlash v1 committed target feature width is invalid");
  }
  pending_target_features_.insert(pending_target_features_.end(), hidden.begin(),
                                  hidden.end());
}

speculative::DraftProposal DFlashV1DraftBackend::Propose(
    std::span<const tokenization::TokenId> prompt_tokens,
    std::uint32_t current_pos, std::uint32_t max_tokens) {

  if (injected_trace()) {
    fprintf(stderr, "[DFTRACE] propose pos=%u max=%u anchor=%u\n", current_pos,
            max_tokens, prompt_tokens.empty() ? 0u : prompt_tokens.back());
    fflush(stderr);
  }
  if (!primed_ || prompt_tokens.empty() || current_pos == 0) {
    throw std::logic_error("DFlash v1 draft backend is not primed");
  }
  if (proposal_active_) {
    throw std::logic_error("DFlash v1 proposal feedback is pending");
  }
  InjectPendingFeatures(current_pos);

  speculative::DraftProposal proposal;
  proposal.start_pos = current_pos;
  const std::uint32_t context_budget =
      current_pos < max_context_ ? max_context_ - current_pos - 1 : 0;
  const std::uint32_t count =
      controller_.Choose(std::min({max_tokens, max_draft_tokens_,
                                   context_budget}));
  controller_.NoteStep(count);
  if (injected_trace()) {
    fprintf(stderr,
            "[DFTRACE] propose chosen=%u injected=%u pending=%zu p1=%.3f "
            "p2=%.3f probe=%u expl=%u fails=0x%x blk=%.1f row=%.1f plain=%.1f\n",
            count, drafter_->injected_length(),
            pending_target_features_.size() / drafter_->target_features_width(),
            controller_.DepthOneRate(), controller_.DeeperRate(),
            controller_.ProbeDepth(), controller_.ExploredDepth(),
            controller_.FailedDepths(), controller_.BlockMs(),
            controller_.RowMs(), controller_.PlainMs());
    fflush(stderr);
  }
  if (count == 0) {
    return proposal;
  }
  const double block_start = NowMs();
  const std::vector<std::int32_t> drafts = drafter_->ForwardBlock(
      prompt_tokens.back(), current_pos, count);
  last_block_ms_ = NowMs() - block_start;
  propose_end_ms_ = NowMs();
  if (injected_trace()) {
    fprintf(stderr, "[DFTRACE] block_ms=%.1f\n", last_block_ms_);
    fflush(stderr);
  }
  proposed_tokens_.assign(drafts.begin(), drafts.end());
  if (injected_trace()) {
    fprintf(stderr, "[DFTRACE] drafts");
    for (const auto token : proposed_tokens_) {
      fprintf(stderr, " %d", token);
    }
    fprintf(stderr, "\n");
    fflush(stderr);
  }
  proposal.tokens = proposed_tokens_;
  proposal_active_ = !proposed_tokens_.empty();
  return proposal;
}

std::span<const float> DFlashV1DraftBackend::CopyProposalLogits(
    std::size_t row) {
  // The verifier clears the proposal before the step returns, so this reads
  // the drafter's retained logits rather than the pending proposal state.
  return drafter_->CopyLogitsRow(row);
}

void DFlashV1DraftBackend::AcceptFeedback(
    std::span<const tokenization::TokenId> accepted,
    tokenization::TokenId correction_token) {
  (void)correction_token;
  if (!proposal_active_ || accepted.size() > proposed_tokens_.size()) {
    throw std::logic_error("DFlash v1 proposal feedback is invalid");
  }
  controller_.Observe(accepted.size(), proposed_tokens_.size());
  // Charge the width controller the verification that followed the block: its
  // rows are what a wider draft costs on this trunk.
  const double verify_ms = NowMs() - propose_end_ms_;
  controller_.ObserveCosts(
      last_block_ms_, verify_ms,
      static_cast<std::uint32_t>(proposed_tokens_.size() + 1));
  if (injected_trace()) {
    // One line per drafted cycle: the block cost predictions, the
    // verification that followed and what it accepted. Per-width marginal
    // costs come from regressing these rows.
    fprintf(stderr,
            "[DFTRACE] cycle width=%zu block_ms=%.1f verify_ms=%.1f "
            "accepted=%zu\n",
            proposed_tokens_.size(), last_block_ms_, verify_ms,
            accepted.size());
    fflush(stderr);
  }
  last_feedback_ms_ = NowMs();
  feedback_seen_ = true;
  proposal_active_ = false;
  proposed_tokens_.clear();
}

void DFlashV1DraftBackend::DiscardPendingTargetContext(
    std::uint32_t position) {
  if (proposal_active_ || drafter_->injected_length() != position) {
    throw std::logic_error("DFlash v1 cancellation frontier is not retained");
  }
  pending_target_features_.clear();
}

void DFlashV1DraftBackend::Reset() noexcept {
  drafter_->Reset();
  controller_.Reset();
  pending_target_features_.clear();
  proposed_tokens_.clear();
  primed_ = false;
  proposal_active_ = false;
}

}  // namespace gufo::models::mimo2
#endif  // defined(ENGINE_ENABLE_HIP)