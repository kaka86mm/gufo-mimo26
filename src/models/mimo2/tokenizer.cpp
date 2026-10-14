#include "src/models/mimo2/tokenizer.hpp"

#include <unicode/uchar.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::mimo2 {
namespace {

struct Utf8CodePoint {
  UChar32 value{0};
  std::size_t length{1};
};

Utf8CodePoint DecodeUtf8(std::string_view text, std::size_t offset) noexcept {
  const auto first = static_cast<std::uint8_t>(text[offset]);
  if (first < 0x80U) {
    return {.value = first, .length = 1};
  }
  std::size_t length = 0;
  UChar32 value = 0;
  if ((first & 0xE0U) == 0xC0U) {
    length = 2;
    value = static_cast<UChar32>(first & 0x1FU);
  } else if ((first & 0xF0U) == 0xE0U) {
    length = 3;
    value = static_cast<UChar32>(first & 0x0FU);
  } else if ((first & 0xF8U) == 0xF0U) {
    length = 4;
    value = static_cast<UChar32>(first & 0x07U);
  } else {
    return {.value = first, .length = 1};
  }
  if (offset + length > text.size()) {
    return {.value = first, .length = 1};
  }
  for (std::size_t index = 1; index < length; ++index) {
    const auto byte = static_cast<std::uint8_t>(text[offset + index]);
    if ((byte & 0xC0U) != 0x80U) {
      return {.value = first, .length = 1};
    }
    value = static_cast<UChar32>((value << 6U) | (byte & 0x3FU));
  }
  return {.value = value, .length = length};
}

bool IsUnicodeLetter(UChar32 value) noexcept {
  return (U_GET_GC_MASK(value) & U_GC_L_MASK) != 0;
}

bool IsUnicodeLetterOrMark(UChar32 value) noexcept {
  return (U_GET_GC_MASK(value) & (U_GC_L_MASK | U_GC_M_MASK)) != 0;
}

bool IsUnicodeNumber(UChar32 value) noexcept {
  const auto category = static_cast<UCharCategory>(u_charType(value));
  return category == U_DECIMAL_DIGIT_NUMBER || category == U_LETTER_NUMBER ||
         category == U_OTHER_NUMBER;
}

bool IsUnicodeWhitespace(UChar32 value) noexcept {
  return u_isUWhiteSpace(value) != 0;
}

bool IsNewline(UChar32 value) noexcept {
  return value == '\r' || value == '\n';
}

/// Contraction end for the case-insensitive (?i:'s|'t|'re|'ve|'m|'ll|'d)
/// alternation; returns `offset` when the apostrophe starts no contraction.
std::size_t ContractionEnd(std::string_view text,
                           std::size_t offset) noexcept {
  if (text[offset] != '\'') {
    return offset;
  }
  constexpr std::array<std::string_view, 7> kSuffixes = {
      "s", "t", "re", "ve", "m", "ll", "d",
  };
  for (const std::string_view suffix : kSuffixes) {
    std::size_t end = offset + 1;
    bool matches = true;
    for (const char letter : suffix) {
      if (end == text.size()) {
        matches = false;
        break;
      }
      const auto next = DecodeUtf8(text, end);
      if (u_foldCase(next.value, U_FOLD_CASE_DEFAULT) !=
          static_cast<UChar32>(letter)) {
        matches = false;
        break;
      }
      end += next.length;
    }
    if (matches) {
      return end;
    }
  }
  return offset;
}

/// Digits group three at a time (\p{N}{1,3}), the Qwen2/GPT-4 family rule.
constexpr std::size_t kDigitsPerPiece = 3;

/// End of the pre-token starting at `offset`, following the Qwen2/GPT-4
/// family split: contractions, `[^\r\n\p{L}\p{N}]?\p{L}+`,
/// `\p{N}{1,3}`, ` ?[^\s\p{L}\p{N}]+[\r\n]*`, `\s*[\r\n]+`,
/// `\s+(?!\S)` and `\s+`, in that priority order.
std::size_t MiMoPieceEnd(std::string_view text, std::size_t offset) {
  if (const std::size_t contraction = ContractionEnd(text, offset);
      contraction != offset) {
    return contraction;
  }

  const Utf8CodePoint first = DecodeUtf8(text, offset);
  if (IsUnicodeLetterOrMark(first.value)) {
    std::size_t end = offset + first.length;
    while (end < text.size()) {
      const Utf8CodePoint next = DecodeUtf8(text, end);
      if (!IsUnicodeLetterOrMark(next.value)) {
        break;
      }
      end += next.length;
    }
    return end;
  }

  // A word piece may carry one leading character that is neither a newline
  // nor a letter nor a number (the leading space of " word").
  if (!IsNewline(first.value) && !IsUnicodeLetter(first.value) &&
      !IsUnicodeNumber(first.value) && offset + first.length < text.size()) {
    const Utf8CodePoint next = DecodeUtf8(text, offset + first.length);
    if (IsUnicodeLetterOrMark(next.value)) {
      std::size_t end = offset + first.length + next.length;
      while (end < text.size()) {
        const Utf8CodePoint letter = DecodeUtf8(text, end);
        if (!IsUnicodeLetterOrMark(letter.value)) {
          break;
        }
        end += letter.length;
      }
      return end;
    }
  }

  if (IsUnicodeNumber(first.value)) {
    std::size_t end = offset;
    for (std::size_t i = 0; i < kDigitsPerPiece; ++i) {
      const Utf8CodePoint digit = DecodeUtf8(text, end);
      if (!IsUnicodeNumber(digit.value)) {
        break;
      }
      end += digit.length;
    }
    return end;
  }

  // Punctuation runs, with an optional leading space and trailing newlines.
  std::size_t punctuation_start = offset;
  if (first.value == ' ' && offset + first.length < text.size()) {
    const Utf8CodePoint next = DecodeUtf8(text, offset + first.length);
    if (!IsUnicodeWhitespace(next.value) &&
        !IsUnicodeLetterOrMark(next.value) && !IsUnicodeNumber(next.value)) {
      punctuation_start += first.length;
    }
  }
  const Utf8CodePoint punctuation = DecodeUtf8(text, punctuation_start);
  if (!IsUnicodeWhitespace(punctuation.value) &&
      !IsUnicodeLetterOrMark(punctuation.value) &&
      !IsUnicodeNumber(punctuation.value)) {
    std::size_t end = punctuation_start;
    while (end < text.size()) {
      const Utf8CodePoint next = DecodeUtf8(text, end);
      if (IsUnicodeWhitespace(next.value) ||
          IsUnicodeLetterOrMark(next.value) || IsUnicodeNumber(next.value)) {
        break;
      }
      end += next.length;
    }
    while (end < text.size()) {
      const Utf8CodePoint next = DecodeUtf8(text, end);
      if (!IsNewline(next.value)) {
        break;
      }
      end += next.length;
    }
    return end;
  }

  if (IsUnicodeWhitespace(first.value)) {
    // A newline run stays with its leading spaces and does not feed the
    // next piece; other whitespace runs backtrack one codepoint before a
    // non-space so that piece can own its leading space.
    std::size_t end = offset;
    std::size_t last_start = offset;
    std::size_t last_newline_end = offset;
    while (end < text.size()) {
      const Utf8CodePoint next = DecodeUtf8(text, end);
      if (!IsUnicodeWhitespace(next.value)) {
        break;
      }
      last_start = end;
      end += next.length;
      if (IsNewline(next.value)) {
        last_newline_end = end;
      }
    }
    if (last_newline_end != offset) {
      return last_newline_end;
    }
    // \s+(?!\S): all trailing whitespace minus the final codepoint.
    return end < text.size() && last_start > offset ? last_start : end;
  }

  return offset + first.length;
}

}  // namespace

MimoTokenizer::MimoTokenizer(tokenization::QwenTokenizer inner)
    : inner_(std::move(inner)) {}

std::unique_ptr<MimoTokenizer> MimoTokenizer::CreateFromGguf(
    const core::GgufReader& reader, std::string* error_msg) {
  // The artifact carries the Qwen2-family pre-tokenizer ('qwen2'); the
  // inner Qwen tokenizer implements exactly that split.
  const auto pre = reader.GetMetadataString("tokenizer.ggml.pre");
  if (pre.has_value() && *pre != "mimo" && *pre != "qwen2") {
    if (error_msg != nullptr) {
      *error_msg = "GGUF tokenizer pre-tokenizer '" + std::string(*pre) +
                   "' is not the MiMo one this runtime implements";
    }
    return nullptr;
  }
  auto inner = tokenization::QwenTokenizer::CreateFromGguf(reader, error_msg);
  if (inner == nullptr) {
    return nullptr;
  }
  // The shared tokenizer only splits for its own pre value; the pieces
  // here drive it directly, so its whole-span fallback must not run.
  return std::unique_ptr<MimoTokenizer>(new MimoTokenizer(std::move(*inner)));
}

std::vector<tokenization::TokenId> MimoTokenizer::Encode(
    std::string_view text) const {
  std::vector<tokenization::TokenId> tokens;
  if (text.empty()) {
    return tokens;
  }
  std::size_t pos = 0;
  while (pos < text.size()) {
    // Specials are the `<|...|>`-shaped vocabulary entries. Only a match
    // in the vocabulary splits; anything else is ordinary text.
    if (text.compare(pos, 2, "<|") == 0) {
      const std::size_t close = text.find("|>", pos + 2);
      if (close != std::string_view::npos && close - pos <= 64) {
        const auto special =
            inner_.FindSpecialToken(text.substr(pos, close + 2 - pos));
        if (special.has_value()) {
          tokens.push_back(*special);
          pos = close + 2;
          continue;
        }
      }
    }
    const std::size_t end = MiMoPieceEnd(text, pos);
    const auto piece = text.substr(pos, end - pos);
    // The inner tokenizer's pre value is not the split one it knows, so an
    // Encode call merges exactly the given span (its whole-text fallback).
    const auto piece_tokens =
        inner_.Encode(piece, {.parse_special_tokens = false});
    tokens.insert(tokens.end(), piece_tokens.begin(), piece_tokens.end());
    pos = end;
  }
  return tokens;
}

std::string MimoTokenizer::Decode(
    std::span<const tokenization::TokenId> tokens) const {
  return inner_.Decode(tokens);
}

std::string MimoTokenizer::DecodeTokenCopy(
    tokenization::TokenId token_id) const {
  return inner_.DecodeTokenCopy(token_id);
}

}  // namespace gufo::models::mimo2
