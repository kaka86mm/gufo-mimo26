#ifndef GUFO_MODELS_MIMO2_MTP_COSTS_HPP_
#define GUFO_MODELS_MIMO2_MTP_COSTS_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <utility>

namespace gufo::models::mimo2 {

// gfx1151, TP2, MiMo-V2.6-Flash-RL MXFP4 trunk + Q8_0 MTP, 2026-10-04.
// Milliseconds per cohort, including catch-up, recursive proposals and target
// verification. Rows are widths 1..8 (width 1 is headless MTP catch-up plus
// ordinary decode).
//
// The C1 row's first four entries are a paired measurement from one run
// (GUFO_MTP_COST aggregation over the seeded prompt set, sessions 2, context
// 64K): 45.8 / 67.6 / 103.8 ms for widths 1..3, and 152.0 ms for width 4 from
// the 2026-10-03 sweep (ed83c34d). They set the operative break-even: width 2
// pays only above a depth-1 hit rate of 67.6/45.8 - 1 = 0.476, and width 3
// needs ~0.75 on top of that, which the recursion over the first NextN block
// never reaches. Width 5..8 keep the 2026-10-03 sweep (a verified row adds
// roughly 37 ms).
//
// A verified position costs about one more MoE expert read plus attention and
// projections. Verification rows route to nearly disjoint experts (measured
// 1.20 distinct-expert rows at width 2, 1.28 at width 3; prefill chunks reach
// 2.5), so wider batches amortize only the non-expert reads and the expert
// bytes per row stay irreducible. Measured depth-1 hits on this box: Chinese
// prose 0.0-0.52, English prose 0.48-0.57, lists and counting 0.65-0.85,
// Python code 0.36-0.58.
//
// C2..C8 and all context growth factors are inherited from the pinned
// Flash-Next table and scaled so C1 keeps the measured mimo2 width curve.
// The greedy batch path measures its own per-cohort costs online
// (MtpBatchController::Observe); these rows only bootstrap it. Sampled
// sessions stay on the C1 row by construction.
//
// The 2026-10-05 depth sweep (C1, eager path, 65,345-token prompt under
// GUFO_MTP_COST) measured the 65536 row directly: width 1 totalled
// 70.4-84.4 ms (median 74.8), width 2 149.3-177.7 (median 175.0), width 3
// 177.5 ms. The second draft row is nearly free once the first exists (one
// verify pass carries both) while the first costs more than a whole plain
// cycle: the verify batch switches the step to the wide-tile route, and at
// this depth that route costs more than the row it verifies. Width 4..8
// continue the 37 ms-per-row marginal the 2026-10-03 sweep measured.
// C2..C8 keep the 32768 row's cohort ratios; their deep entries are derived,
// and the batch path replaces them with its own observations.
// Cohort 6's first two context rows were re-measured on 2026-10-07 (six-way
// counting, 8x256-token runs, engine BATCHCYCLE aggregation): 77.4/167.3 ms
// and 75.4/172.9 ms over two runs, 254 ms at two drafts (29 + 31 cycles), and
// 410 ms at four (six cycles). The inherited rows had priced a width-1 cycle
// at 1.48x plain decoding; the box measures 2.2-2.3x, which is the difference
// between "draft at a 48% hit rate" and "the verify round costs more than the
// token it verifies". Only the width curve moved -- the same factors applied
// to the 4096 row, whose own 2.4K-context measurement (219.5/105.3 mixed)
// shows the same ratio. Deeper rows keep the shape the 65536 C1 measurement
// gave them. Widths 3 and 5..7 are interpolated and extrapolated on the old
// marginal.
inline constexpr float kMtpCycleMilliseconds[4][5][8] = {
    {
        // Context 0
        {45.8F, 67.6F, 103.8F, 152.0F, 167.0F, 204.0F, 241.0F, 278.0F}, // C1
        {55.0F, 81.1F, 124.7F, 182.5F, 200.5F, 244.9F, 289.3F, 333.7F}, // C2
        {68.7F, 101.4F, 155.7F, 228.0F, 250.5F, 306.0F, 361.5F, 417.0F}, // C4
        {75.4F, 170.0F, 254.0F, 327.0F, 410.0F, 447.0F, 484.0F, 521.0F}, // C6
        {93.8F, 138.5F, 212.7F, 311.5F, 342.3F, 418.1F, 493.9F, 569.8F}, // C8
    },
    {
        // Context 4096
        {47.4F, 69.9F, 107.3F, 157.1F, 172.7F, 210.9F, 249.2F, 287.4F}, // C1
        {58.6F, 86.4F, 132.7F, 194.3F, 213.5F, 260.8F, 308.2F, 355.5F}, // C2
        {74.3F, 109.7F, 168.4F, 246.6F, 270.9F, 331.0F, 391.0F, 451.0F}, // C4
        {82.8F, 186.5F, 278.9F, 359.0F, 450.1F, 490.8F, 531.6F, 572.0F}, // C6
        {104.8F, 154.7F, 237.6F, 348.0F, 382.3F, 467.0F, 551.7F, 636.4F}, // C8
    },
    {
        // Context 32768
        {50.7F, 74.8F, 114.9F, 168.3F, 184.8F, 225.8F, 266.7F, 307.7F}, // C1
        {65.4F, 96.6F, 148.4F, 217.4F, 238.8F, 291.7F, 344.6F, 397.5F}, // C2
        {87.8F, 129.5F, 198.9F, 291.4F, 320.1F, 391.0F, 461.9F, 532.8F}, // C4
        {109.8F, 161.9F, 248.7F, 364.2F, 400.1F, 488.8F, 577.4F, 666.0F}, // C6
        {131.3F, 193.8F, 297.6F, 435.8F, 478.8F, 584.9F, 691.0F, 797.1F}, // C8
    },
    {
        // Context 65536, C1 measured 2026-10-05 (see above)
        {74.8F, 175.0F, 177.5F, 214.5F, 251.5F, 288.5F, 325.5F, 362.5F}, // C1
        {96.5F, 225.8F, 229.0F, 276.7F, 324.4F, 372.2F, 419.9F, 467.6F}, // C2
        {129.6F, 303.1F, 307.4F, 371.5F, 435.6F, 499.7F, 563.8F, 627.9F}, // C4
        {162.0F, 379.1F, 384.5F, 464.6F, 544.7F, 624.9F, 705.0F, 785.2F}, // C6
        {193.8F, 453.3F, 459.7F, 555.6F, 651.4F, 747.2F, 843.0F, 938.9F}, // C8
    },
};

/// Predictor catch-up per cycle at a given depth: one NextN-block forward
/// over the accepted token plus its bookkeeping, paid every cycle while
/// speculation is armed. Width-1 cycle cost minus this is what a latched
/// (speculation-free) cycle costs, and it is the bar a speculative width has
/// to beat. Short-context entries come from the spec-on/spec-off A/B (28.0 vs
/// 23.8 t/s at short context), the 32768 entry interpolates, and the deep
/// entry is the sweep's own prep field (7.2-7.9 ms per AR cycle).
inline constexpr float kMtpCatchUpMilliseconds[4] = {3.8F, 4.2F, 5.5F, 7.3F};

// Sampled policy uses fixed configured capacity for reproducibility; greedy
// batch policy uses these curves to bootstrap each physical occupancy.
// Intermediate capacities use the next measured cohort. Context costs
// interpolate, then extrapolate the measured QSA slope to the native limit.
[[nodiscard]] inline std::array<float, 8> MtpCycleCosts(
    std::uint32_t context, std::uint32_t concurrency) noexcept {
  const auto cohort = concurrency <= 1   ? 0
                      : concurrency <= 2 ? 1
                      : concurrency <= 4 ? 2
                      : concurrency <= 6 ? 3
                                         : 4;
  const auto context_fraction = [](std::uint32_t context) {
    if (context <= 4096) {
      return std::make_pair(0, static_cast<float>(context) / 4096.0F);
    }
    if (context <= 32768) {
      return std::make_pair(
          1, static_cast<float>(std::min(context, 32768U) - 4096) / 28672.0F);
    }
    return std::make_pair(
        2, static_cast<float>(std::min(context, 262144U) - 32768) / 32768.0F);
  };
  const auto [interval, fraction] = context_fraction(context);
  std::array<float, 8> costs{};
  for (std::size_t i = 0; i < costs.size(); ++i) {
    const auto low = kMtpCycleMilliseconds[interval][cohort][i];
    const auto high = kMtpCycleMilliseconds[interval + 1][cohort][i];
    costs[i] = low + fraction * std::max(high - low, 0.0F);
    if (fraction > 1.0F && i != 0) {
      // Independently extrapolated noisy slopes can cross at long contexts.
      // Preserve at least the last measured marginal cost of another draft.
      const auto marginal =
          high - kMtpCycleMilliseconds[interval + 1][cohort][i - 1];
      costs[i] = std::max(costs[i], costs[i - 1] + marginal);
    }
  }
  // GUFO_MTP_COST_SCALE compresses the "another verified row" marginals.
  // The 2026-10-04 rows priced a second-draft cycle at +48% of a plain C1
  // cycle; the 2026-10-13 mtpmeas sweep on the current build measures +37-40%
  // on the MXFP4 trunk (+26% on EXL3), so the lone path demanded p1>0.476 to
  // draft while the true break-even sits at 0.37-0.40. 0.80 makes the C1
  // ratio 1+0.476*0.80 = 1.38, matching the measurement; 1.0 keeps the
  // historical rows.
  static const float verify_scale = [] {
    const char* env = getenv("GUFO_MTP_COST_SCALE");
    if (env == nullptr) {
      return 1.0F;
    }
    const float value = std::strtof(env, nullptr);
    return value > 0.0F && value <= 4.0F ? value : 1.0F;
  }();
  // Only the C1 cohort row carries the lone-request path: the batch rows keep
  // the historical marginals (the 2026-10-13 sweep priced a batched verify
  // row no cheaper than the table does - deeper cohorts must keep latching
  // plain until their rows are re-measured).
  if (verify_scale < 1.0F && cohort == 0) {
    for (std::size_t i = 1; i < costs.size(); ++i) {
      costs[i] = costs[i - 1] + (costs[i] - costs[i - 1]) * verify_scale;
    }
  }
  return costs;
}

/// Catch-up cost at a context, on the same interpolation as the cycle table.
[[nodiscard]] inline float MtpCatchUpCost(std::uint32_t context) noexcept {
  const auto interval = context <= 4096 ? 0 : context <= 32768 ? 1 : 2;
  const float fraction =
      interval == 0
          ? static_cast<float>(context) / 4096.0F
          : interval == 1
                ? static_cast<float>(std::min(context, 32768U) - 4096) / 28672.0F
                : static_cast<float>(std::min(context, 262144U) - 32768) /
                      32768.0F;
  const float low = kMtpCatchUpMilliseconds[interval];
  const float high = kMtpCatchUpMilliseconds[interval + 1];
  return low + fraction * (high - low);
}

}  // namespace gufo::models::mimo2

#endif  // GUFO_MODELS_MIMO2_MTP_COSTS_HPP_