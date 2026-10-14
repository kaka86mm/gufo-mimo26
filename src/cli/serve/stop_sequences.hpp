#ifndef GUFO_SERVER_STOP_SEQUENCES_HPP_
#define GUFO_SERVER_STOP_SEQUENCES_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "src/core/json.hpp"

namespace gufo::server {

/// On unless GUFO_REPEAT_GUARD says otherwise.
inline bool RepeatGuardEnabled() {
  const char* value = std::getenv("GUFO_REPEAT_GUARD");
  if (value == nullptr || value[0] == '\0') {
    return true;
  }
  return value[0] != '0' && value[0] != 'f' && value[0] != 'F' &&
         value[0] != 'n' && value[0] != 'N';
}

/// On unless GUFO_FOREIGN_STOP says otherwise.
inline bool ForeignTokenStopEnabled() {
  const char* value = std::getenv("GUFO_FOREIGN_STOP");
  if (value == nullptr || value[0] == '\0') {
    return true;
  }
  return value[0] != '0' && value[0] != 'f' && value[0] != 'F' &&
         value[0] != 'n' && value[0] != 'N';
}

inline constexpr std::size_t kMaxStopSequences = 64;
inline constexpr std::size_t kMaxStopSequenceBytes = 4096;
inline constexpr std::size_t kMaxStopTotalBytes = 16384;

inline void ValidateStopSequences(const std::vector<std::string>& sequences,
                                  std::size_t max_count = kMaxStopSequences) {
  if (sequences.size() > max_count)
    throw std::invalid_argument("too many stop sequences (maximum " +
                                std::to_string(max_count) + ")");
  std::size_t bytes = 0;
  for (const auto& sequence : sequences) {
    if (sequence.empty() || sequence.size() > kMaxStopSequenceBytes)
      throw std::invalid_argument("stop sequences must contain 1 to " +
                                  std::to_string(kMaxStopSequenceBytes) +
                                  " UTF-8 bytes");
    bytes += sequence.size();
    if (bytes > kMaxStopTotalBytes)
      throw std::invalid_argument("stop sequences exceed the total byte limit");
  }
}

enum class StopSequenceFormat { kOpenAi, kAnthropic };

inline std::optional<std::string> ParseStopSequences(
    const json::Value* value, StopSequenceFormat format,
    std::vector<std::string>* output) {
  output->clear();
  if (value == nullptr ||
      (format == StopSequenceFormat::kOpenAi && value->is_null()))
    return std::nullopt;
  if (format == StopSequenceFormat::kOpenAi && value->is_string()) {
    output->push_back(value->str());
  } else if (value->is_array()) {
    for (const auto& item : value->items()) {
      if (!item.is_string())
        return "stop sequences must be strings";
      output->push_back(item.str());
    }
  } else {
    return format == StopSequenceFormat::kOpenAi
               ? "'stop' must be null, a string, or an array of strings"
               : "'stop_sequences' must be an array of strings";
  }
  try {
    ValidateStopSequences(
        *output, format == StopSequenceFormat::kOpenAi ? 4 : kMaxStopSequences);
  } catch (const std::invalid_argument& error) {
    return error.what();
  }
  return std::nullopt;
}

/// Matches raw accepted output, retaining only a possible stop prefix.
/// The first completed match wins; ties at the same end prefer the longer
/// sequence. This is independent of how token pieces are split into chunks.
class StopSequenceFilter {
public:
  explicit StopSequenceFilter(std::vector<std::string> sequences = {},
                              bool repeat_guard = RepeatGuardEnabled(),
                              bool foreign_stop = ForeignTokenStopEnabled())
      : sequences_(std::move(sequences)),
        repeat_guard_(repeat_guard),
        foreign_stop_(foreign_stop) {
    ValidateStopSequences(sequences_);
  }

  [[nodiscard]] bool enabled() const noexcept {
    return !sequences_.empty() || repeat_guard_ || foreign_stop_;
  }
  [[nodiscard]] bool stopped() const noexcept {
    return match_.has_value() || repeat_found_ || foreign_found_;
  }
  /// The request ended because the output degenerated into an unbroken repeat
  /// instead of a stop sequence matching.
  [[nodiscard]] bool repetition_detected() const noexcept {
    return repeat_found_;
  }
  /// The request ended because the output drifted into foreign template
  /// tokens rather than matching a stop sequence the caller asked for.
  [[nodiscard]] bool foreign_stop_detected() const noexcept {
    return foreign_found_;
  }
  [[nodiscard]] std::size_t repeat_period() const noexcept {
    return repeat_period_;
  }
  [[nodiscard]] std::size_t repeat_run() const noexcept { return repeat_run_; }
  [[nodiscard]] const std::string& matched_sequence() const {
    static const std::string kEmpty;
    return match_.has_value() ? sequences_.at(match_.value()) : kEmpty;
  }

  std::string Push(std::string_view piece) {
    if (stopped())
      return {};
    if (repeat_guard_) {
      repeat_tail_.append(piece);
      if (repeat_tail_.size() > kRepeatTailBytes) {
        repeat_tail_.erase(0, repeat_tail_.size() - kRepeatTailBytes);
      }
      repeat_pending_ += piece.size();
      if (repeat_pending_ >= kRepeatScanStep) {
        repeat_pending_ = 0;
        CheckRepeat();
      }
    }
    pending_.append(piece);
    std::size_t end = std::string::npos;
    std::size_t start = std::string::npos;
    for (std::size_t i = 0; i < sequences_.size(); ++i) {
      const auto position = pending_.find(sequences_[i]);
      if (position == std::string::npos)
        continue;
      const auto candidate_end = position + sequences_[i].size();
      if (candidate_end < end || (candidate_end == end && position < start)) {
        end = candidate_end;
        start = position;
        match_ = i;
      }
    }
    if (foreign_stop_ && !foreign_found_) {
      for (const auto foreign : kForeignStops) {
        const auto position = pending_.find(foreign);
        if (position == std::string_view::npos)
          continue;
        const auto candidate_end = position + foreign.size();
        if (candidate_end < end ||
            (candidate_end == end && position < start)) {
          end = candidate_end;
          start = position;
          foreign_found_ = true;
          match_.reset();
        }
      }
    }
    if (stopped()) {
      auto safe = pending_.substr(0, start);
      pending_.clear();
      return safe;
    }
    std::size_t keep = 0;
    for (const auto& sequence : sequences_) {
      for (auto length = std::min(pending_.size(), sequence.size() - 1);
           length > keep; --length) {
        if (std::string_view(pending_).ends_with(
                std::string_view(sequence).substr(0, length))) {
          keep = length;
          break;
        }
      }
    }
    if (foreign_stop_ && !foreign_found_) {
      for (const auto foreign : kForeignStops) {
        for (auto length = std::min(pending_.size(), foreign.size() - 1);
             length > keep; --length) {
          if (std::string_view(pending_).ends_with(foreign.substr(0, length))) {
            keep = length;
            break;
          }
        }
      }
    }
    auto safe = pending_.substr(0, pending_.size() - keep);
    pending_.erase(0, safe.size());
    return safe;
  }

  std::string Finish() { return std::exchange(pending_, {}); }

private:
  // A generation that repeats one piece without end: a period of a few bytes
  // to a few hundred, held for well over a kilobyte.  Thresholds are set well
  // above anything an answer does on purpose (the run is reported, so a false
  // positive can be diagnosed from the log).
  static constexpr std::size_t kRepeatTailBytes = 8192;
  static constexpr std::size_t kRepeatScanStep = 256;
  static constexpr std::size_t kRepeatMinPeriod = 2;
  static constexpr std::size_t kRepeatMaxPeriod = 768;
  static constexpr std::size_t kRepeatMinRun = 1536;

  void CheckRepeat() {
    if (repeat_found_ || repeat_tail_.size() < kRepeatMinRun + kRepeatMaxPeriod) {
      return;
    }
    const std::string_view tail(repeat_tail_);
    const std::size_t max_period =
        std::min(kRepeatMaxPeriod, tail.size() / 3);
    for (std::size_t period = kRepeatMinPeriod; period <= max_period;
         ++period) {
      const auto block = tail.substr(tail.size() - period);
      if (tail.substr(tail.size() - 2 * period, period) != block ||
          tail.substr(tail.size() - 3 * period, period) != block) {
        continue;
      }
      std::size_t run = 3 * period;
      while (run + period <= tail.size() &&
             tail.substr(tail.size() - run - period, period) ==
                 tail.substr(tail.size() - run, period)) {
        run += period;
      }
      if (run >= kRepeatMinRun) {
        repeat_found_ = true;
        repeat_period_ = period;
        repeat_run_ = run;
        return;
      }
    }
  }

  // Engine-added stops for drift artifacts: template tokens the model
  // only emits after sliding into playing the transcript instead of
  // answering (2026-10-10: a research chain degenerated through exactly
  // these spellings).
  static constexpr std::array<std::string_view, 5> kForeignStops{
      "<|end|><|start|>",
      "<|function_call_begin|>",
      "<|function_results_begin|>",
      // The template's own markers, spelling themselves as text on short
      // answers instead of arriving as their tokens (seen live on
      // 2026-10-11 on this and the previous build). The turn opener ends
      // the answer exactly like the closer does.
      "<|im_end|>",
      "<|im_start|>",
  };

  std::vector<std::string> sequences_;
  std::string pending_;
  bool repeat_guard_{false};
  bool foreign_stop_{false};
  bool foreign_found_{false};
  bool repeat_found_{false};
  std::size_t repeat_period_{0};
  std::size_t repeat_run_{0};
  std::string repeat_tail_;
  std::size_t repeat_pending_{0};
  std::optional<std::size_t> match_;
};

}  // namespace gufo::server

#endif  // GUFO_SERVER_STOP_SEQUENCES_HPP_
