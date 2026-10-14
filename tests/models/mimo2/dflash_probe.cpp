// DFlash v1 drafter bring-up probe. Loads the real sidecar GGUF, binds stub
// device tensors for the trunk's embedding/output head (a full-size vocabulary
// so the head GEMM runs at production shape), injects random target features
// and drafts blocks. It reports configuration, timings and structural sanity
// only; draft quality is judged end-to-end by the speculative verifier.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <span>
#include <string>
#include <tuple>
#include <vector>

#include <hip/hip_fp16.h>

#include "src/models/mimo2/dflash_v1.hpp"

namespace {

using gufo::core::GgmlType;
using gufo::models::mimo2::DFlashV1Drafter;
using gufo::models::mimo2::DFlashV1Model;
using gufo::models::mimo2::DFlashV1TrunkTensors;

double NowMs() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

/// A Q8_0 stand-in matrix on the device: rows of `cols` elements, block scale
/// 0.01, small random codes.
void* MakeStubQ8(std::uint32_t cols, std::uint32_t rows, std::mt19937* rng) {
  const std::size_t blocks = std::size_t{cols / 32} * rows;
  std::vector<std::uint8_t> host(blocks * 34);
  std::uniform_int_distribution<int> code(-42, 42);
  const __half scale = __float2half(0.01F);
  for (std::size_t b = 0; b < blocks; ++b) {
    std::memcpy(host.data() + b * 34, &scale, 2);
    for (std::size_t i = 0; i < 32; ++i) {
      host[b * 34 + 2 + i] =
          static_cast<std::uint8_t>(static_cast<std::int8_t>(code(*rng)));
    }
  }
  void* device = nullptr;
  if (hipMalloc(&device, host.size()) != hipSuccess ||
      hipMemcpy(device, host.data(), host.size(), hipMemcpyHostToDevice) !=
          hipSuccess) {
    std::fprintf(stderr, "stub allocation failed\n");
    std::exit(1);
  }
  return device;
}


/// Loads one tensor from a trunk GGUF shard onto the device (the real head or
/// embedding table, so probe logits are the production logits).
void* LoadTrunkTensor(const std::string& path, const std::string& name,
                      std::uint32_t* cols, std::uint32_t* rows,
                      gufo::core::GgmlType* type) {
  std::string error;
  auto reader = gufo::core::GgufReader::OpenFile(path, &error);
  if (reader == nullptr) {
    std::fprintf(stderr, "trunk open failed: %s\n", error.c_str());
    return nullptr;
  }
  const auto* info = reader->FindTensor(name);
  if (info == nullptr || info->dimensions.size() != 2) {
    std::fprintf(stderr, "trunk tensor %s not found\n", name.c_str());
    return nullptr;
  }
  *cols = static_cast<std::uint32_t>(info->dimensions[0]);
  *rows = static_cast<std::uint32_t>(info->dimensions[1]);
  *type = info->type;
  const std::size_t bytes = *cols * *rows;  // Q8_0: 34 bytes per 32 values
  const std::size_t q8_bytes = (std::size_t{*cols} / 32) * 34 * *rows;
  void* device = nullptr;
  if (hipMalloc(&device, q8_bytes) != hipSuccess) {
    std::fprintf(stderr, "trunk tensor alloc failed\n");
    return nullptr;
  }
  if (hipMemcpy(device, info->data, q8_bytes, hipMemcpyHostToDevice) !=
      hipSuccess) {
    std::fprintf(stderr, "trunk tensor copy failed (%zu bytes)\n", q8_bytes);
    return nullptr;
  }
  (void)bytes;
  return device;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <dflash-sidecar.gguf> [tokens] [drafts]\n",
                 argv[0]);
    return 2;
  }
  std::uint32_t tokens =
      argc > 2 ? static_cast<std::uint32_t>(std::atoi(argv[2])) : 300;
  const std::uint32_t drafts =
      argc > 3 ? static_cast<std::uint32_t>(std::atoi(argv[3])) : 7;
  constexpr std::uint32_t kHidden = 4096;
  constexpr std::uint32_t kVocab = 152576;

  std::mt19937 rng(1234);
  // --trunk <shard.gguf> points the stub embedding/head at the real tensors;
  // --features <file> replays captured target features and drafts one block,
  // writing the draft logits rows so the quality can be compared offline.
  std::string trunk_path;
  std::string features_path;
  std::uint32_t anchor = 42;
  for (int i = 2; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "--trunk") == 0) {
      trunk_path = argv[++i];
    } else if (std::strcmp(argv[i], "--features") == 0) {
      features_path = argv[++i];
    } else if (std::strcmp(argv[i], "--anchor") == 0) {
      anchor = static_cast<std::uint32_t>(std::atoi(argv[++i]));
    }
  }
  void* embedding = MakeStubQ8(kHidden, kVocab, &rng);
  void* head = MakeStubQ8(kHidden, kVocab, &rng);
  if (!trunk_path.empty()) {
    std::uint32_t cols = 0;
    std::uint32_t rows = 0;
    GgmlType type = GgmlType::kQ8_0;
    if (void* real_embedding = LoadTrunkTensor(
            trunk_path, "token_embd.weight", &cols, &rows, &type)) {
      embedding = real_embedding;
      std::printf("real embedding: cols=%u rows=%u type=%d\n", cols, rows,
                  static_cast<int>(type));
    }
    cols = 0;
    rows = 0;
    if (void* real_head = LoadTrunkTensor(trunk_path, "output.weight", &cols,
                                          &rows, &type)) {
      head = real_head;
      std::printf("real head: cols=%u rows=%u type=%d\n", cols, rows,
                  static_cast<int>(type));
    }
  }
  const DFlashV1TrunkTensors trunk{
      .token_embedding = embedding,
      .token_embedding_type = GgmlType::kQ8_0,
      .output = head,
      .output_type = GgmlType::kQ8_0,
  };

  std::string error;
  auto model = DFlashV1Model::Load(argv[1], trunk, kHidden, kVocab, 1.0F, &error);
  if (model == nullptr) {
    std::fprintf(stderr, "sidecar load failed: %s\n", error.c_str());
    return 1;
  }
  auto drafter = DFlashV1Drafter::Create(model, 131072, &error);
  if (drafter == nullptr) {
    std::fprintf(stderr, "drafter create failed: %s\n", error.c_str());
    return 1;
  }
  const auto& c = drafter->config();
  std::printf("config: layers=%u hidden=%u intermediate=%u heads=%u kv=%u "
              "head_dim=%u block=%u window=%u mask=%u taps=%zu\n",
              c.num_layers, c.hidden_size, c.intermediate_size, c.num_heads,
              c.num_kv_heads, c.head_dim, c.block_size, c.sliding_window,
              c.mask_token_id, c.target_layer_ids.size());
  std::printf("taps:");
  for (const auto id : c.target_layer_ids) {
    std::printf(" %u", id);
  }
  std::printf("\nweights=%.1f MiB state=%.1f MiB history=%u\n",
              model->weight_bytes() / 1048576.0,
              drafter->state_bytes() / 1048576.0,
              drafter->history_capacity());
  drafter->SetDebug(true);
  // The sidecar tensors are mapped host memory: read the first bytes of a
  // few tensors directly and print them for a byte-for-byte GGUF comparison.
  {
    const auto dump = [](const char* name, const gufo::models::mimo2::DFlashV1Tensor& tensor) {
      std::printf("bytes %-18s:", name);
      const auto* bytes = static_cast<const std::uint8_t*>(tensor.data);
      for (int i = 0; i < 32; ++i) {
        std::printf(" %02x", bytes[i]);
      }
      std::printf(" (cols=%u rows=%u)\n", tensor.cols, tensor.rows);
    };
    const auto& first = model->layers().front();
    dump("blk0.attn_q", first.attn_q);
    dump("blk0.attn_k", first.attn_k);
    dump("blk0.attn_norm", first.attn_norm);
    dump("fc", model->fc());
    const auto floats = [](const char* name,
                           const gufo::models::mimo2::DFlashV1Tensor& tensor) {
      std::printf("floats %-17s:", name);
      const auto* values = static_cast<const float*>(tensor.data);
      for (int i = 0; i < 8; ++i) {
        std::printf(" %.4f", values[i]);
      }
      std::printf("\n");
    };
    floats("blk0.attn_q_norm", first.attn_q_norm);
    floats("blk0.attn_k_norm", first.attn_k_norm);
    floats("blk0.ffn_norm", first.ffn_norm);
    floats("output_norm", model->output_norm());
  }

  const std::size_t width = drafter->target_features_width();
  // Deterministic, trivially reproducible in another language: the reference
  // harness recomputes the same values from the same formula.
  const auto deterministic = [](std::size_t index) {
    const std::uint32_t hashed =
        static_cast<std::uint32_t>(index * 2654435761ULL);
    return static_cast<float>(hashed % 2001U) / 1000.0F - 1.0F;
  };
  std::vector<float> features(width * tokens);
  if (features_path.empty()) {
    for (std::size_t i = 0; i < features.size(); ++i) {
      features[i] = deterministic(i);
    }
  } else {
    std::FILE* file = std::fopen(features_path.c_str(), "rb");
    if (file == nullptr) {
      std::fprintf(stderr, "features file missing: %s\n", features_path.c_str());
      return 1;
    }
    std::fseek(file, 0, SEEK_END);
    const long bytes = std::ftell(file);
    std::fseek(file, 0, SEEK_SET);
    const std::size_t count = static_cast<std::size_t>(bytes) / sizeof(float);
    if (count == 0 || count % width != 0) {
      std::fprintf(stderr, "features file size %zu is not a multiple of %zu\n",
                   count, width);
      std::fclose(file);
      return 1;
    }
    tokens = static_cast<std::uint32_t>(count / width);
    features.resize(count);
    if (std::fread(features.data(), sizeof(float), count, file) != count) {
      std::fprintf(stderr, "features read failed\n");
      std::fclose(file);
      return 1;
    }
    std::fclose(file);
    std::printf("replay features: %u rows\n", tokens);
    for (std::uint32_t t = 0; t < tokens; ++t) {
      const float* row = features.data() + std::size_t{t} * width;
      double sum = 0.0;
      for (std::size_t i = 0; i < width; ++i) {
        sum += row[i];
      }
      std::printf("  row %u sum=%.4f first=%.6f\n", t, sum, row[0]);
    }
  }
  const double t0 = NowMs();
  if (!drafter->InjectTargetContext(features, 0, tokens)) {
    std::fprintf(stderr, "injection failed\n");
    return 1;
  }
  const double t1 = NowMs();
  std::printf("inject %u tokens: %.1f ms (injected=%u)\n", tokens, t1 - t0,
              drafter->injected_length());

  double best = 1e9;
  double total = 0;
  std::vector<std::int32_t> last;
  for (int repeat = 0; repeat < 10; ++repeat) {
    const double b0 = NowMs();
    last = drafter->ForwardBlock(static_cast<std::int32_t>(anchor), tokens,
                                 drafts);
    const double b1 = NowMs();
    total += b1 - b0;
    best = std::min(best, b1 - b0);
  }
  std::printf("block draft x10: mean=%.1f best=%.1f ms\n", total / 10.0, best);
  std::printf("drafts:");
  bool sane = last.size() == drafts;
  for (const auto token : last) {
    std::printf(" %d", token);
    sane = sane && token >= 0 && static_cast<std::uint32_t>(token) < kVocab;
  }
  std::printf("\n");

  DFlashV1Drafter::BlockDebug debug;
  std::string debug_error;
  if (drafter->CopyBlockDebug(&debug, &debug_error)) {
    const auto stats = [](std::span<const float> values) {
      float lo = values[0];
      float hi = values[0];
      double sum = 0.0;
      std::size_t zeros = 0;
      for (const float value : values) {
        lo = std::min(lo, value);
        hi = std::max(hi, value);
        sum += value;
        zeros += value == 0.0F ? 1 : 0;
      }
      return std::tuple(lo, hi, sum / static_cast<double>(values.size()), zeros);
    };
    const auto lead = [](const char* name, std::span<const float> values) {
      std::printf("lead %-20s:", name);
      for (int i = 0; i < 8 && i < static_cast<int>(values.size()); ++i) {
        std::printf(" %.9g", values[i]);
      }
      std::printf("\n");
    };
    lead("embedding.row0", debug.embedding);
    // Full block embedding for the numeric reference harness.
    if (FILE* dump = std::fopen("/tmp/dflash-embedding.f32", "wb")) {
      std::fwrite(debug.embedding.data(), sizeof(float),
                  debug.embedding.size(), dump);
      std::fclose(dump);
      std::printf("wrote /tmp/dflash-embedding.f32 (%zu floats)\n",
                  debug.embedding.size());
    }
    lead("q_raw.row0", debug.raw_query);
    lead("q_norm.row0", debug.normed_query);
    lead("q_rope.row0", debug.query);
    lead("block_k.row0", debug.block_key);
    lead("block_v.row0", debug.block_value);
    lead("ring_k.row0", debug.ring_key);
    lead("attn_row0_ffn", debug.after_attention);
    for (std::size_t i = 0; i + 8 <= debug.layer_hidden.size(); i += 4096) {
      char name[32];
      std::snprintf(name, sizeof(name), "layer%zu.row0", i / 4096);
      lead(name, std::span<const float>(debug.layer_hidden.data() + i, 8));
    }
    lead("hidden.row0", debug.hidden);
    lead("normed.row0", debug.normed);
    lead("logits.row0", debug.logits);
    if (!features_path.empty()) {
      for (std::size_t r = 0; r < 7; ++r) {
        const auto row = drafter->CopyLogitsRow(r);
        if (row.empty()) {
          break;
        }
        char name[64];
        std::snprintf(name, sizeof(name), "/tmp/dflash-probe-logits-%zu.f32", r);
        if (FILE* dump = std::fopen(name, "wb")) {
          std::fwrite(row.data(), sizeof(float), row.size(), dump);
          std::fclose(dump);
        }
        std::size_t best = 0;
        for (std::size_t i = 1; i < row.size(); ++i) {
          if (row[i] > row[best]) {
            best = i;
          }
        }
        std::printf("draft%zu top=%zu logit=%.4f\n", r, best, row[best]);
      }
      const auto row = drafter->CopyLogitsRow(0);
      if (!row.empty()) {
        if (FILE* dump = std::fopen("/tmp/dflash-probe-logits.f32", "wb")) {
          std::fwrite(row.data(), sizeof(float), row.size(), dump);
          std::fclose(dump);
        }
        std::size_t best = 0;
        for (std::size_t i = 1; i < row.size(); ++i) {
          if (row[i] > row[best]) {
            best = i;
          }
        }
        std::printf("draft0 top=%zu logit=%.4f\n", best, row[best]);
        std::printf("wrote /tmp/dflash-probe-logits.f32 (%zu floats)\n",
                    row.size());
      }
    }
    lead("features.row0", features);
    const auto hidden = stats(debug.hidden);
    const auto normed = stats(debug.normed);
    const auto logits = stats(debug.logits);
    if (!debug.embedding.empty()) {
      const auto embed = stats(debug.embedding);
      std::printf("embedding: min=%.4g max=%.4g mean=%.4g zeros=%zu of %zu\n",
                  std::get<0>(embed), std::get<1>(embed), std::get<2>(embed),
                  std::get<3>(embed), debug.embedding.size());
    }
    if (!debug.after_attention.empty()) {
      const auto attn = stats(debug.after_attention);
      std::printf("attn out: min=%.4g max=%.4g mean=%.4g zeros=%zu of %zu\n",
                  std::get<0>(attn), std::get<1>(attn), std::get<2>(attn),
                  std::get<3>(attn), debug.after_attention.size());
    }
    for (const auto& [name, values] :
         std::initializer_list<std::pair<const char*, std::span<const float>>>{
             {"layer0 q(raw)", debug.raw_query},
             {"layer0 q(norm)", debug.normed_query},
             {"layer0 q(rope)", debug.query},
             {"layer0 k(block)", debug.block_key},
             {"layer0 v(block)", debug.block_value},
             {"layer0 ring K head", debug.ring_key},
         }) {
      if (values.empty()) {
        continue;
      }
      const auto stage = stats(values);
      std::printf("%s: min=%.4g max=%.4g mean=%.4g zeros=%zu of %zu\n", name,
                  std::get<0>(stage), std::get<1>(stage), std::get<2>(stage),
                  std::get<3>(stage), values.size());
    }
    std::printf("hidden: min=%.4g max=%.4g mean=%.4g zeros=%zu of %zu\n",
                std::get<0>(hidden), std::get<1>(hidden), std::get<2>(hidden),
                std::get<3>(hidden), debug.hidden.size());
    std::printf("normed: min=%.4g max=%.4g mean=%.4g zeros=%zu of %zu\n",
                std::get<0>(normed), std::get<1>(normed), std::get<2>(normed),
                std::get<3>(normed), debug.normed.size());
    for (std::uint32_t row = 0; row < debug.draft_count; ++row) {
      const auto row_stats =
          stats(std::span(debug.logits).subspan(
              static_cast<std::size_t>(row) * kVocab, kVocab));
      std::printf("logits row %u: min=%.4g max=%.4g mean=%.4g zeros=%zu\n",
                  row, std::get<0>(row_stats), std::get<1>(row_stats),
                  std::get<2>(row_stats), std::get<3>(row_stats));
    }
  } else {
    std::printf("block debug unavailable: %s\n", debug_error.c_str());
  }

  // Extension: inject further positions and draft again; a second context
  // beyond the window exercises the ring wrap.
  const std::uint32_t wrap = drafter->history_capacity() + 40;
  std::vector<float> more(width * wrap);
  for (std::size_t i = 0; i < more.size(); ++i) {
    more[i] = deterministic(features.size() + i);
  }
  const bool extended =
      drafter->InjectTargetContext(more, tokens, wrap) &&
      !drafter->ForwardBlock(7, tokens + wrap, 3).empty();
  std::printf("extension(+%u): %s injected=%u\n", wrap,
              extended ? "ok" : "FAILED", drafter->injected_length());
  drafter->Reset();
  std::printf("reset: injected=%u\n", drafter->injected_length());
  return sane && extended ? 0 : 1;
}