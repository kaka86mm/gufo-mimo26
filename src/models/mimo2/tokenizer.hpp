#ifndef GUFO_MODELS_MIMO2_TOKENIZER_HPP_
#define GUFO_MODELS_MIMO2_TOKENIZER_HPP_

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/qwen/tokenizer.hpp"

namespace gufo::core {
class GgufReader;
}

namespace gufo::models::mimo2 {

/// MiMo-V2 BPE tokenizer. The vocabulary, merges, byte-level fallback,
/// special tokens and decode paths are the generic GGUF machinery shared
/// with `tokenization::QwenTokenizer`; only the pre-tokenizer differs
/// (MiMo registers its own `tokenizer.ggml.pre` value, and the shared
/// tokenizer implements no split for it). Encode therefore re-splits each
/// non-special span with the MiMo piece rules and hands the pieces to the
/// shared byte-level BPE merger.
///
/// The piece rules implemented here follow the Qwen2/GPT-4 family regex
/// (case-insensitive contractions, an optional leading non-letter for word
/// pieces, digits grouped three at a time, punctuation runs with trailing
/// newlines, whitespace with the `\s+(?!\S)` backtrack). Verify against
/// llama.cpp's `unicode_regex_from_pre("mimo")` on the Linux box before
/// trusting token identities; the split is the only MiMo-specific logic.
class MimoTokenizer {
public:
  ~MimoTokenizer() = default;
  MimoTokenizer(const MimoTokenizer&) = delete;
  MimoTokenizer& operator=(const MimoTokenizer&) = delete;
  MimoTokenizer(MimoTokenizer&&) noexcept = default;
  MimoTokenizer& operator=(MimoTokenizer&&) noexcept = default;

  /// Rejects artifacts whose tokenizer.ggml.pre is present but not "mimo"
  /// (mis-tokenizing silently is worse than failing loudly).
  [[nodiscard]] static std::unique_ptr<MimoTokenizer> CreateFromGguf(
      const core::GgufReader& reader, std::string* error_msg = nullptr);

  /// Encodes UTF-8 text, splitting on `<|...|>`-shaped special tokens that
  /// exist in the vocabulary.
  [[nodiscard]] std::vector<tokenization::TokenId> Encode(
      std::string_view text) const;
  [[nodiscard]] std::string Decode(
      std::span<const tokenization::TokenId> tokens) const;
  [[nodiscard]] std::string DecodeTokenCopy(
      tokenization::TokenId token_id) const;
  [[nodiscard]] std::string_view DecodeToken(
      tokenization::TokenId token_id) const noexcept {
    return inner_.DecodeToken(token_id);
  }

  [[nodiscard]] std::size_t GetVocabSize() const noexcept {
    return inner_.GetVocabSize();
  }
  [[nodiscard]] tokenization::TokenId GetEosTokenId() const noexcept {
    return inner_.GetEosTokenId();
  }
  [[nodiscard]] tokenization::TokenId GetBosTokenId() const noexcept {
    return inner_.GetBosTokenId();
  }
  [[nodiscard]] bool IsStopToken(tokenization::TokenId token) const noexcept {
    return inner_.IsStopToken(token);
  }
  [[nodiscard]] std::optional<tokenization::TokenId> FindSpecialToken(
      std::string_view token) const noexcept {
    return inner_.FindSpecialToken(token);
  }

private:
  explicit MimoTokenizer(tokenization::QwenTokenizer inner);

  tokenization::QwenTokenizer inner_;
};

}  // namespace gufo::models::mimo2

#endif  // GUFO_MODELS_MIMO2_TOKENIZER_HPP_
