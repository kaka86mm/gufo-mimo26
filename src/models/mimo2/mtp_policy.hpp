#ifndef GUFO_MODELS_MIMO2_MTP_POLICY_HPP_
#define GUFO_MODELS_MIMO2_MTP_POLICY_HPP_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>

#include "src/models/mimo2/mtp_costs.hpp"

namespace gufo::models::mimo2 {

inline constexpr std::uint32_t kMaxMtpDraftTokens = 7;

/// Weight of the running hit-rate history. The historical 0.75 (effective
/// window ~7 samples) let the per-depth estimate jitter across the
/// draft/no-draft threshold, and every spurious "no width repays" verdict
/// then bought a long retry window: with the calibrated C1 costs the lone
/// path drafted only ~58% of cycles at a true 0.6 depth-1 hit rate.
/// GUFO_MTP_ACCEPT_DECAY=0.9 (window ~19) is the measured follow-up.
[[nodiscard]] inline float MtpAcceptanceDecay() noexcept {
  static const float decay = [] {
    const char* env = getenv("GUFO_MTP_ACCEPT_DECAY");
    if (env == nullptr) {
      return 0.75F;
    }
    const float value = std::strtof(env, nullptr);
    return value >= 0.5F && value <= 0.99F ? value : 0.75F;
  }();
  return decay;
}

/// Retry window (plain-decode cycles) after a near-miss verdict: the
/// estimate sits within kMtpNearMissMargin of the break-even, so a rebound
/// is cheap to catch and re-testing often is worth it.
/// GUFO_MTP_RETRY_TOKENS=4 is the measured follow-up; the historical
/// window is 16.
[[nodiscard]] inline std::uint32_t MtpRetryTokens() noexcept {
  static const std::uint32_t retry = [] {
    const char* env = getenv("GUFO_MTP_RETRY_TOKENS");
    if (env == nullptr) {
      return 16U;
    }
    const long value = std::strtol(env, nullptr, 10);
    return value >= 0 && value <= 64 ? static_cast<std::uint32_t>(value) : 16U;
  }();
  return retry;
}

/// Confidence band of the near-miss rule, in hit-rate units (~1.4 sigma of
/// the smoothed estimate at a 0.9 decay).
inline constexpr float kMtpNearMissMargin = 0.15F;

/// Samples of a width's cycle cost before the batch policy treats it as
/// measured rather than bootstrapped (a caller's verdict may not read the
/// measured cell: see the latch in Session::PlanBatch).
inline constexpr unsigned kArWinsSamples = 2;

struct MtpLengthState {
  std::array<float, kMaxMtpDraftTokens> successes;
  std::array<float, kMaxMtpDraftTokens> failures;
  std::uint32_t retry_tokens{0};
  std::uint32_t probe_depth{0};
  std::uint32_t explored_depth{0};
  std::uint32_t probe_delay{0};
  std::uint32_t failed_depths{0};
};

// Length depends on committed acceptance history, never request timings.
// Only the first rejected draft is an observed failure; its unverified tail
// and a fully accepted, budget-limited chain are censored observations.
class MtpLengthController {
public:
  explicit MtpLengthController(std::uint32_t limit,
                               std::uint32_t concurrency = 1)
      : limit_(std::clamp(limit, 1U, kMaxMtpDraftTokens)),
        concurrency_(std::clamp(concurrency, 1U, 8U)) {
    Reset();
  }

  [[nodiscard]] std::uint32_t Choose(std::uint32_t budget,
                                     std::uint32_t context = 0) const noexcept {
    const auto cap = std::min(budget, limit_);
    if (cap == 0 || state_.retry_tokens != 0) {
      return 0;
    }
    // After a bounded AR interval, try one proposal to detect recovery.
    return std::min(
        cap, std::max({1U, BestLength(cap, context), state_.probe_depth}));
  }

  [[nodiscard]] float ExpectedTokens(std::uint32_t drafts) const noexcept {
    float expected = 1.0F, survival = 1.0F;
    for (unsigned depth = 0; depth < std::min(drafts, limit_); ++depth) {
      survival *= Probability(depth);
      expected += survival;
    }
    return expected;
  }

  /// True when plain decoding without the predictor catch-up beats every
  /// speculative width at this context. That is the depth latch's
  /// precondition: while it holds, each cycle can drop the catch-up a
  /// drafting cycle would need and lose nothing. The comparison uses the
  /// latched AR cost (width-1 cycle minus catch-up), not the width-1 cycle,
  /// so a width that only repays the catch-up itself does not veto the
  /// latch.
  [[nodiscard]] bool LatchAr(std::uint32_t cap,
                             std::uint32_t context) const noexcept {
    const auto costs = MtpCycleCosts(context, concurrency_);
    const float latched_ar =
        std::max(costs[0] - MtpCatchUpCost(context), 0.1F);
    float expected = 1.0F;
    float survival = 1.0F;
    for (std::uint32_t length = 1; length <= cap; ++length) {
      survival *= Probability(length - 1);
      expected += survival;
      if (costs[length] / expected < latched_ar) {
        return false;
      }
    }
    return true;
  }

  /// The depth-latch verdict against a caller-supplied cost row: no width
  /// repays the plain cycle `plain` if every one of them costs at least that
  /// much per expected token. The batch policy prices a whole cohort, which
  /// is a different row than this controller's own occupancy (a session is
  /// constructed for one lane), so the row comes from the caller. `costs` is
  /// indexed by draft count.
  [[nodiscard]] bool ArWinsAgainst(std::span<const float> costs,
                                   std::uint32_t cap, float plain) const
      noexcept {
    float expected = 1.0F;
    float survival = 1.0F;
    for (std::uint32_t length = 1; length <= cap && length < costs.size();
         ++length) {
      survival *= Probability(length - 1);
      expected += survival;
      if (costs[length] / expected < plain) {
        return false;
      }
    }
    return true;
  }

  void Observe(std::size_t accepted, std::size_t drafted,
               std::uint32_t context = 0) noexcept {
    if (drafted == 0)
      return;
    drafted = std::min<std::size_t>(drafted, limit_);
    accepted = std::min(accepted, drafted);
    const auto observed = accepted + (accepted < drafted ? 1 : 0);
    for (std::size_t depth = 0; depth < observed; ++depth) {
      const float decay = MtpAcceptanceDecay();
      state_.successes[depth] =
          decay * state_.successes[depth] + (depth < accepted ? 1.0F : 0.0F);
      state_.failures[depth] =
          decay * state_.failures[depth] + (depth == accepted ? 1.0F : 0.0F);
    }
    // A fully accepted chain supplies no observations for its tail. Probe
    // farther rather than permanently treating those depths as the prior.
    state_.probe_depth = 0;
    if (accepted == drafted && drafted < limit_) {
      if (drafted >= state_.explored_depth || state_.probe_delay == 0) {
        state_.probe_depth = std::min<std::uint32_t>(2 * drafted, limit_);
        state_.probe_delay = 16;
      } else {
        --state_.probe_delay;
      }
    }
    state_.explored_depth =
        std::max<std::uint32_t>(state_.explored_depth, observed);
    if (accepted < drafted)
      state_.failed_depths |= 1U << accepted;
    if (BestLength(limit_, context) == 0) {
      // A near-miss verdict (the estimate is within its own noise of the
      // break-even) retries quickly so a rebound is cheap to catch; a
      // confident verdict keeps the long window.
      const auto retry_costs = MtpCycleCosts(context, concurrency_);
      const float break_even = retry_costs[1] / retry_costs[0] - 1.0F;
      state_.retry_tokens =
          Probability(0) >= break_even - kMtpNearMissMargin
              ? MtpRetryTokens()
              : 16U;
    }
  }

  void ObserveArToken() noexcept {
    if (state_.retry_tokens != 0)
      --state_.retry_tokens;
  }

  void Reset() noexcept {
    state_.successes.fill(1.5F);
    state_.failures.fill(0.5F);
    state_.retry_tokens = 0;
    state_.probe_depth = 0;
    state_.explored_depth = 0;
    state_.probe_delay = 0;
    state_.failed_depths = 0;
  }

  [[nodiscard]] MtpLengthState State() const noexcept { return state_; }

  [[nodiscard]] bool Restore(const MtpLengthState& state) noexcept {
    if (state.retry_tokens > 16 || state.probe_depth > limit_ ||
        state.explored_depth > limit_ || state.probe_delay > 16 ||
        state.failed_depths >= (1U << limit_))
      return false;
    for (std::size_t depth = 0; depth < kMaxMtpDraftTokens; ++depth) {
      const float success = state.successes[depth];
      const float failure = state.failures[depth];
      if (!std::isfinite(success) || !std::isfinite(failure) ||
          success < 0.0F || failure < 0.0F ||
          !std::isfinite(success + failure) || success + failure <= 0.0F)
        return false;
    }
    state_ = state;
    return true;
  }

private:
  [[nodiscard]] float Probability(unsigned depth) const noexcept {
    return depth < state_.explored_depth &&
                   (state_.failed_depths & (1U << depth)) == 0
               ? 1.0F
               : state_.successes[depth] /
                     (state_.successes[depth] + state_.failures[depth]);
  }
  [[nodiscard]] std::uint32_t BestLength(std::uint32_t cap,
                                         std::uint32_t context) const noexcept {
    const auto costs = MtpCycleCosts(context, concurrency_);
    float survival = 1.0F;
    float expected_tokens = 1.0F;  // The sampled target anchor.
    float best_score = 1.0F;       // Ordinary target decoding.
    std::uint32_t best = 0;
    for (std::uint32_t length = 1; length <= cap; ++length) {
      const auto depth = length - 1;
      // An explored depth with no actual rejection remains optimistic.
      // Otherwise the prior alone shortens perfect chains and prevents
      // their less frequently sampled tails from collecting evidence.
      const float probability = Probability(depth);
      survival *= probability;
      expected_tokens += survival;
      const float cost = costs[length] / costs[0];
      const float score = expected_tokens / cost;
      if (score > best_score) {
        best_score = score;
        best = length;
      }
    }
    return best;
  }

  std::uint32_t limit_;
  std::uint32_t concurrency_;
  MtpLengthState state_{};
};

// Runtime costs belong to a physical batch, not an individual sequence.
// Separate occupancy/context bins avoid applying shallow C1 timings to C8.
// Greedy output is independent of the chosen width; sampled requests keep
// the deterministic per-session policy to preserve seeded replay.
class MtpBatchController {
public:
  struct Row {
    const MtpLengthController* acceptance;
    std::uint32_t budget;
  };

  [[nodiscard]] std::uint32_t Choose(std::span<const Row> rows,
                                     std::uint32_t context) const noexcept {
    if (rows.empty() || rows.size() > 8)
      return 0;
    const auto& cell = CellFor(rows.size(), context);
    std::uint32_t cap = kMaxMtpDraftTokens;
    for (const auto& row : rows)
      cap = std::min(cap, row.budget);
    if (cap == 0)
      return 0;
    // Measurement/build knob: pin the cohort width (a comma list rotates one
    // width per cycle) so per-width cycle costs and acceptance compare at
    // matched conditions inside a single run. Width 0 pins plain decoding,
    // which is the alternative the controller measures every shape against.
    // Both ranks walk the same list in the same order, so the decision stays
    // rank-symmetric.
    static const std::vector<std::uint32_t> forced_widths = [] {
      std::vector<std::uint32_t> list;
      const char* env = getenv("GUFO_MTP_BATCH_WIDTH");
      for (const char* p = env; p != nullptr && *p != (char)0;) {
        char* end = nullptr;
        const long value = std::strtol(p, &end, 10);
        if (end == p) {
          ++p;
          continue;
        }
        if (value >= 0) {
          list.push_back(static_cast<std::uint32_t>(value));
        }
        p = end;
      }
      return list;
    }();
    if (!forced_widths.empty()) {
      static std::size_t forced_cycle = 0;
      return std::min<std::uint32_t>(
          forced_widths[forced_cycle++ % forced_widths.size()], cap);
    }
    // Periodic headless cycles measure the actual AR alternative, including
    // predictor catch-up. The next cycle can then resume speculation.
    const bool stable = last_concurrency_ == rows.size() &&
                        last_context_bin_ == ContextBin(context) &&
                        stable_cycles_ >= 2;
    if (stable && (cell.samples[0] == 0 || cell.cycles % 32 < 2))
      return 0;
    auto costs = MtpCycleCosts(context, rows.size());
    const float scale = cell.samples[0] ? cell.ms[0] / costs[0] : 1.0F;
    for (unsigned width = 0; width <= cap; ++width)
      costs[width] =
          cell.samples[width] >= 2 ? cell.ms[width] : costs[width] * scale;
    float best_score = rows.size() / costs[0];
    unsigned best = 0;
    for (unsigned width = 1; width <= cap; ++width) {
      float expected = 0;
      for (const auto& row : rows)
        expected += row.acceptance->ExpectedTokens(width);
      const float score = expected / costs[width];
      if (score > best_score * 1.01F) {
        best_score = score;
        best = width;
      }
    }
    // A cohort whose own table row already prices every width above its
    // plain cycle has nothing to learn from the probe and exploration cycles
    // below: they buy measurements for a decision the row settles, at the
    // cost of a verify round — at six-way that round measures 2.2x a plain
    // cycle for 1.7 tokens, which is where the exploration's few percent
    // went. GUFO_MTP_AR_LATCH=0 keeps them for A/B. The verdict reads the
    // table and the acceptance models only, and this whole choice is rank
    // zero's (the plan is computed once and handed to the peer), so a
    // per-rank number can never split the two onto different widths.
    static const bool plain_latch = [] {
      const char* env = getenv("GUFO_MTP_AR_LATCH");
      return env == nullptr || env[0] != '0' || env[1] != '\0';
    }();
    if (plain_latch && best == 0 && ArWins(rows, context)) {
      return 0;
    }
    // Fully accepted cohorts can qualify deeper shapes promptly. Stop
    // forcing exploration once that physical width has warmed measurements.
    unsigned probe = cap;
    for (const auto& row : rows)
      probe = std::min(probe, row.acceptance->State().probe_depth);
    if (probe > best && cell.samples[probe] < 2)
      return probe;
    // Bounded exploration learns wider costs and conditional acceptance;
    // it also lets an AR-only batch detect a change in the workload.
    if (cell.cycles % 16 == 2)
      return std::min(cap, std::max(1U, best + 1));
    return best;
  }

  /// Whether no speculative width repays its cycle for this cohort, on the
  /// table's own cohort row — the batch form of the depth latch. It reads
  /// only inputs the two ranks share (the table and the per-session
  /// acceptance models): a verdict that differed between ranks would have
  /// one of them post the other's exchange size, and the transport poisons
  /// on a collective mismatch. Measured cycle times are per-rank and must
  /// not be read here. A cohort whose table row still prices a width below
  /// the plain cycle keeps speculating.
  [[nodiscard]] bool ArWins(std::span<const Row> rows,
                            std::uint32_t context) const noexcept {
    if (rows.empty() || rows.size() > 8)
      return false;
    const auto costs = MtpCycleCosts(context, rows.size());
    const float plain = std::max(costs[0] - MtpCatchUpCost(context), 0.1F);
    for (const auto& row : rows) {
      if (row.budget == 0) {
        continue;
      }
      if (!row.acceptance->ArWinsAgainst(costs, row.budget, plain)) {
        return false;
      }
    }
    return true;
  }

  void Observe(std::size_t concurrency, std::uint32_t context,
               std::uint32_t drafts, float ms) noexcept {
    if (concurrency == 0 || concurrency > 8 || drafts > kMaxMtpDraftTokens ||
        !std::isfinite(ms) || ms <= 0)
      return;
    const auto bin = ContextBin(context);
    const bool same_cohort =
        last_concurrency_ == concurrency && last_context_bin_ == bin;
    stable_cycles_ = same_cohort ? std::min(stable_cycles_ + 1, 1000000U) : 1;
    last_concurrency_ = concurrency;
    last_context_bin_ = bin;
    auto& cell = CellFor(concurrency, context);
    // Catch-up belongs to the preceding chain. A transition from seven
    // proposals to three still replays the longer chain, so it cannot
    // measure the steady cost of three proposals (or of plain decoding).
    if (!same_cohort || drafts != cell.last_drafts) {
      cell.last_drafts = drafts;
      ++cell.cycles;
      return;
    }
    cell.last_drafts = drafts;
    auto& count = cell.samples[drafts];
    auto& estimate = cell.ms[drafts];
    // Drop first-use graph/allocation cost once a warmed sample arrives.
    if (count < 2)
      estimate = ms;
    else
      estimate += 0.25F * (ms - estimate);
    count = std::min(count + 1, 1000000U);
    ++cell.cycles;
  }

private:
  struct Cell {
    std::array<float, 8> ms{};
    std::array<unsigned, 8> samples{};
    std::uint64_t cycles{0};
    std::uint32_t last_drafts{0};
  };
  static unsigned ContextBin(std::uint32_t context) noexcept {
    unsigned bin = 0;
    for (unsigned bound = 4096; context >= bound && bin < 7; bound *= 2)
      ++bin;
    return bin;
  }
  Cell& CellFor(std::size_t concurrency, std::uint32_t context) noexcept {
    return cells_[concurrency - 1][ContextBin(context)];
  }
  const Cell& CellFor(std::size_t concurrency,
                      std::uint32_t context) const noexcept {
    return cells_[concurrency - 1][ContextBin(context)];
  }
  std::array<std::array<Cell, 8>, 8> cells_{};
  std::size_t last_concurrency_{0};
  unsigned last_context_bin_{0};
  unsigned stable_cycles_{0};
};

}  // namespace gufo::models::mimo2

#endif  // GUFO_MODELS_MIMO2_MTP_POLICY_HPP_
