#ifndef GUFO_MODELS_MIMO2_KERNELS_ROCM_KERNELS_HPP_
#define GUFO_MODELS_MIMO2_KERNELS_ROCM_KERNELS_HPP_

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "src/models/mimo2/kernels/rocm/communicator.hpp"
#include "src/models/qwen/vision/rope.hpp"

/// Model-private HIP launchers for everything outside the quantized GEMM
/// tier. Activations are row-major float32 [tokens][dim] unless noted; every
/// launch is asynchronous on `stream`. Weights referenced here are device
/// pointers in their GGUF encoding.
namespace gufo::models::mimo2::rocm {

/// GGUF type ids the runtime accepts for the small-matrix and lookup paths.
enum class WeightType : std::uint32_t {
  kF32 = 0,
  kF16 = 1,
  kQ2_K = 10,
  kQ4_K = 12,
  kQ5_K = 13,
  kQ6_K = 14,
  kQ5_1 = 7,
  kQ8_0 = 8,
  kMXFP4 = 39,
  kBF16 = 30,
};

/// res[t][s][hidden] = table[tokens[t]] for every stream s.
void EmbedTokens(const void* table, WeightType type, const std::int32_t* tokens,
                 float* res, std::uint32_t n_tokens, std::uint32_t hidden,
                 std::uint32_t streams, hipStream_t stream);

/// out[t][d] = rmsnorm(x[t][d]) * gamma[d]; `groups` independent norms of
/// `dim/groups` elements each share the gamma row. gamma may be null.
void RmsNormRows(const float* x, const float* gamma, float* out,
                 std::uint32_t n_rows, std::uint32_t dim, std::uint32_t groups,
                 float eps, hipStream_t stream);

/// x[i] = silu(x[i] * scale), in place over `count` floats.
void SiluScale(float* x, float scale, std::size_t count, hipStream_t stream);
/// gate[i] = silu(gate[i]) * up[i], in place in `gate`.
void Swiglu(float* gate, const float* up, std::size_t count,
            hipStream_t stream);
/// out[i] = silu(gate[i]) * up[i] as F16 (`count` elements).
void SwigluHalf(const float* gate, const float* up, __half* out,
                std::size_t count, hipStream_t stream);
/// silu(gate) * up over n_rows rows of k elements, written only into the
/// W8A8 tiled Q8 layout `out_q8`; false (nothing launched) unless k % 32 == 0.
bool SwigluQ8Tiled(const float* gate, const float* up, void* out_q8,
                   std::size_t n_rows, std::size_t k, hipStream_t stream);
/// x[i] *= sigmoid(g[i]).
void SigmoidMul(float* x, const float* g, std::size_t count,
                hipStream_t stream);

/// x[i] *= scale, in place over `count` floats (the post-attention
/// value_scale multiply; keeps rocBLAS out of the trunk path).
void ScaleInPlace(float* x, float scale, std::size_t count,
                  hipStream_t stream);

/// out[t][m] = sum_k W[m][k] * x[t][k] for F32/BF16/F16 weights; meant for
/// the narrow projections (routers, eh_proj) that the quantized tier does
/// not cover. Converts `count` floats to BF16 (or F16) for a 16-bit hipBLAS
/// GEMM.
void NarrowActivations(const float* x, void* out, bool bf16, std::size_t count,
                       hipStream_t stream);

/// W8A8 route for wide batches over Q8_0 weights: activations quantized per
/// 32-wide block into the tiled fragment layout (`Q8TiledBytes(batch, k)`
/// bytes, padded to the GEMM's 128-token macro tile), then an int8 WMMA
/// GEMM. out is [batch][m]. W8A8Gemm returns false, launching nothing, when
/// k is not a multiple of 32.
std::size_t Q8TiledBytes(std::size_t batch, std::size_t k);
void QuantizeQ8Tiled(const float* x, void* out, std::size_t batch,
                     std::size_t k, hipStream_t stream);
bool W8A8Gemm(const void* w, const void* x_tiled, float* out, std::size_t batch,
              std::size_t m, std::size_t k, hipStream_t stream);
/// Native wave64 launch, selected by W8A8Gemm for wide output projections.
void W8A8GemmWave64(const void* w, const void* x_tiled, float* out,
                    std::size_t batch, std::size_t m, std::size_t k,
                    hipStream_t stream);

/// F16 activation rows [batch][k], quantized weights dequantized to F16 in
/// LDS, F32 accumulation. out is [batch][m]. Unsupported shapes launch
/// nothing.
bool UnquantizedF16Gemm(const void* w, const __half* x, float* out,
                        std::size_t batch, std::size_t m, std::size_t k,
                        hipStream_t stream);
bool DenseF16Gemm(const void* w, const __half* x, float* out, std::size_t batch,
                  std::size_t m, std::size_t k, hipStream_t stream);

/// F32/BF16/F16/Q8_0 projection in groups of up to eight token rows. Each
/// token keeps the same accumulation order at every batch width.
void RouterGemmWmma(const __half* w_half, const __half* x_half, float* out,
                   std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
                   hipStream_t stream);
/// The narrow-batch form of `RouterGemmWmma` (decode rows): one warp per
/// expert row reads the weights once at full width. Same F16 inputs:
/// requires k divisible by 8. The caller keeps the WMMA tiling for wide
/// batches, where its token-by-expert tiles amortize better.
void RouterGemmNarrow(const __half* w_half, const __half* x_half, float* out,
                      std::uint32_t n_tokens, std::uint32_t m,
                      std::uint32_t k, hipStream_t stream);
/// Split-K `RouterGemmWmma` for decode batches, whose two blocks cannot cover
/// the weight sweep's memory latency. `partials` holds `splits * n_tokens * m`
/// floats; the same order of additions runs on every rank, so selection stays
/// symmetric. Requires `k % (16 * splits) == 0`.
void RouterGemmWmmaSplit(const __half* w_half, const __half* x_half, float* out,
                         float* partials, std::uint32_t n_tokens,
                         std::uint32_t m, std::uint32_t k,
                         std::uint32_t splits, hipStream_t stream);
/// Token rows the split-K router accepts (its scratch is sized for that).
inline constexpr std::uint32_t kRouterSplitMaxTokens = 16;
/// k slices the split-K router accepts.
inline constexpr std::uint32_t kRouterSplitMaxSlices = 16;
void SmallGemm(const void* w, WeightType type, const float* x, float* out,
               std::uint32_t n_tokens, std::uint32_t m, std::uint32_t k,
               hipStream_t stream);

/// Slot indirection for the executor-wide decode graphs: a batched body's
/// kernels read each batch row's per-session destinations from these tables
/// instead of baked kernel parameters, so one graph replays across session
/// mixes (the host refreshes a pinned mirror once per launch; the copy node
/// inside the graph re-reads it on every replay). A null pointer inside a
/// slot marks a cancelled row: the kernels skip its private writes and the
/// attention clears its output, matching the eager path's cancellation.
struct SlotKv {
  __half* k_cache;  ///< the store path writes through these (not const)
  __half* v_cache;
  std::uint32_t ring_rows;
  std::uint32_t pad;
  /// int8 twins with an fp16 scale per 32 values (GUFO_MIMO_KV_INT8); the
  /// store writes through them and the decode attention reads them, so a deep
  /// cohort row's KV traffic halves. Null keeps the fp16-only path.
  std::int8_t* k_cache_q;
  std::int8_t* v_cache_q;
  __half* k_scales;
  __half* v_scales;
};
/// Values behind one int8 K/V scale: a 32-value block stores as 32 bytes plus
/// one fp16 scale (GUFO_MIMO_KV_INT8).
constexpr std::uint32_t kKvQuantBlock = 32;

struct SlotPtrs {
  float* target_hidden;
  float* mtp_h;
};
struct SlotControl {
  std::uint32_t position;
  std::uint32_t mtp_position;
  std::int32_t hidden_row;
  std::uint32_t pad;
};

/// NEOX partial rotary over x [t][heads][d] at positions start_pos + t.
/// Positions are read from device memory (`start_pos` points at the
/// session's control block) so a captured decode graph replays at any
/// position.
void Rope(float* x, std::uint32_t n_tokens, std::uint32_t heads,
          std::uint32_t d, std::uint32_t rotary_dim,
          const std::uint32_t* start_pos, float theta, hipStream_t stream,
          const std::uint32_t* row_pos = nullptr);

/// Ropes the query rows in place, ropes the keys and rounds keys and values
/// once into their F16 caches. Q rows are [t][heads*d_qk] f32; the K cache
/// holds kv_heads*d_qk halves per position, the V cache kv_heads*d_v.
/// Requires cache capacity from *start_pos through *start_pos + n_tokens.
/// `ring_rows` 0 stores at absolute positions; a nonzero power-of-two value
/// is the ring size sliding layers store through (row = position &
/// (ring_rows - 1)). The ring must exceed window + n_tokens so a prefill
/// batch cannot overwrite rows earlier queries in the same batch still
/// need (causal masking would break).
void PrepareQkv(const float* q, const float* k, const float* v, float* q_roped,
                __half* k_cache, __half* v_cache, std::uint32_t n_tokens,
                std::uint32_t heads, std::uint32_t kv_heads,
                std::uint32_t d_qk, std::uint32_t d_v,
                std::uint32_t rotary_dim, std::uint32_t ring_rows,
                const std::uint32_t* start_pos, float theta,
                hipStream_t stream, std::int8_t* k_cache_q = nullptr,
                std::int8_t* v_cache_q = nullptr, __half* k_scales = nullptr,
                __half* v_scales = nullptr);
/// Slot-table form: the caches and ring come from kv[slot]; the rope and the
/// store halves share the plain kernels above.
void PrepareQkvSlot(const float* q, const float* k, const float* v,
                    const SlotKv* kv, std::uint32_t slot,
                    std::uint32_t n_tokens, std::uint32_t heads,
                    std::uint32_t kv_heads, std::uint32_t d_qk,
                    std::uint32_t d_v, std::uint32_t rotary_dim,
                    const std::uint32_t* start_pos, float theta,
                    hipStream_t stream, const std::uint32_t* row_slot = nullptr,
                    const std::uint32_t* row_pos = nullptr);

/// Stores f32 rows into the f16 cache at positions start_pos + t:
/// cache[(start_pos + t)][row_dim], ring-mapped when `ring_rows` is nonzero.
void StoreKv(const float* src, __half* cache, std::uint32_t n_tokens,
             std::uint32_t row_dim, std::uint32_t ring_rows,
             const std::uint32_t* start_pos, hipStream_t stream,
             std::int8_t* cache_q = nullptr, __half* scales = nullptr);
/// Slot-table form of the store above: cache/ring come from kv[slot]
/// (`v_side` picks v_cache over k_cache), and a null cache skips the row
/// entirely (cancelled slot).
void StoreKvSlot(const float* src, const SlotKv* kv, std::uint32_t slot,
                 bool v_side, std::uint32_t n_tokens, std::uint32_t row_dim,
                 const std::uint32_t* start_pos, hipStream_t stream,
                 const std::uint32_t* row_slot = nullptr,
                 const std::uint32_t* row_pos = nullptr);
/// Fused cohort attention: one launch for rows that belong to different
/// sessions (row_slot picks the session's caches, row_pos its position).
/// `splits` partitions each row's own visible key range — a deep cohort would
/// otherwise walk thousands of key tiles per row inside one block — and the
/// caller merges through `partials` (a null partials keeps one split).
/// `visible_keys` is the widest row's position count (the caller's split gate);
/// the shared-KV form sizes its own split count from it.
void AttentionRows(const float* q, const SlotKv* kv,
                   const std::uint32_t* row_slot,
                   const std::uint32_t* row_pos, const float* sinks,
                   float* out, float* partials, std::uint32_t splits,
                   std::uint32_t rows, std::uint32_t heads,
                   std::uint32_t kv_heads, std::uint32_t d_qk,
                   std::uint32_t d_v, std::uint32_t window,
                   std::uint32_t visible_keys, hipStream_t stream);

/// Per-query attention over the layer's caches, one block per (head, token,
/// split). Query heads are `d_qk` wide, value heads `d_v` wide (the K and V
/// caches carry their own strides). `window` 0 attends every position below
/// the query; otherwise only the trailing `window` positions (sliding-window
/// layers, window 128). `ring_rows` is the physical row count behind the
/// caches: 0 addresses absolute positions, a power-of-two value maps rows
/// as position & (ring_rows - 1) — it must exceed window + n_tokens so a
/// prefill batch cannot clobber history its own earlier queries read.
/// `sinks`, when non-null, adds one per-head constant
/// to the softmax denominator (attention sinks). With `partials`
/// (n_tokens * heads * splits * (d_v + 2) floats) the key tiles are split
/// across `splits` blocks per row and merged in a second launch, and
/// `visible_keys` lets the shared-KV form size its own split count.
void Attention(const float* q, const __half* k_cache, const __half* v_cache,
               const float* sinks, float* out, float* partials,
               std::uint32_t splits, std::uint32_t wide_splits,
               std::uint32_t n_tokens,
               const std::uint32_t* start_pos, std::uint32_t heads,
               std::uint32_t kv_heads, std::uint32_t d_qk, std::uint32_t d_v,
               std::uint32_t window, std::uint32_t ring_rows,
               std::uint32_t visible_keys, hipStream_t stream,
               const std::int8_t* k_cache_q = nullptr,
               const std::int8_t* v_cache_q = nullptr,
               const __half* k_scales = nullptr,
               const __half* v_scales = nullptr);
/// Slot-table form: the caches and ring come from kv[slot]. A null k_cache
/// (cancelled slot) zeroes `out` and stores nothing, matching the eager
/// path's skip-then-memset. `partials` splits the key range like the eager
/// form (a null partials keeps one split), which is what gives a single-token
/// decode at depth its parallelism: one block per head walks the whole range
/// otherwise.
void AttentionSlot(const float* q, const SlotKv* kv, std::uint32_t slot,
                   const float* sinks, float* out, float* partials,
                   std::uint32_t splits, std::uint32_t n_tokens,
                   const std::uint32_t* start_pos, std::uint32_t heads,
                   std::uint32_t kv_heads, std::uint32_t d_qk,
                   std::uint32_t d_v, std::uint32_t window,
                   std::uint32_t visible_keys, hipStream_t stream);

/// Re-derives an int8 K/V twin from f16 cache rows (rows are the live rows
/// from the cache's start). Snapshots carry only the authoritative f16 rows,
/// so a restored session's int8 twin is rebuilt through this instead of being
/// serialized.
void KvQuantizeFromHalf(const __half* cache, std::int8_t* cache_q,
                        __half* scales, std::uint32_t rows,
                        std::uint32_t row_dim, hipStream_t stream);

/// counts[e] = number of (token, slot) pairs routed to expert e.
void ExpertCounts(const std::int32_t* ids, std::uint32_t* counts,
                  std::uint32_t n_tokens, std::uint32_t n_experts,
                  std::uint32_t k, hipStream_t stream);

/// Scales packed F16 values by 256: restores the MXFP4 up-pass rows that
/// were parked two bins down to survive the F16 intermediate.
void MxUpscaleF16(__half* x, std::size_t count, hipStream_t stream);

/// Sigmoid gating over `n_experts` logits per token (rows `stride` apart):
/// p = sigmoid(logits); selection adds `bias` when non-null; top-k by the
/// biased score; weights are the unbiased probabilities of the selected
/// experts renormalized to sum one. ids [t][k] int32, weights [t][k] f32.
void RouterTopKSigmoid(const float* logits, std::uint32_t stride,
                       const float* bias, std::int32_t* ids, float* weights,
                       std::uint32_t n_tokens, std::uint32_t n_experts,
                       std::uint32_t k, hipStream_t stream);

/// Routed expert GEMMs. RoutedCompact sorts the (token, slot) assignments
/// by expert into `rows_token`/`rows_slot` (RoutedCompactRows(slots,
/// experts) entries, -1 for padding) with every bucket padded to 16 rows
/// (`pad_bounds`, experts + 1 entries) from the per-expert `counts`.
std::size_t RoutedCompactRows(std::size_t slots, std::size_t n_experts);
void RoutedCompact(const std::int32_t* ids, const std::uint32_t* counts,
                   std::int32_t* pad_bounds, std::int32_t* cursors,
                   std::int32_t* rows_token, std::int32_t* rows_slot,
                   std::uint32_t n_tokens, std::uint32_t k,
                   std::uint32_t n_experts, hipStream_t stream);

/// Routed expert GEMM on the F16 WMMA cores over F16 activation rows
/// `x` ([rows][k]): the weights are dequantized to F16 after the LDS read
/// and the whole K extent accumulates in F32. `tiles` (n_tiles entries)
/// packs each launched (expert, token-tile) as expert | tile << 16, so
/// empty tiles cost nothing. Exactly one of `out` (F32) and `out_half`
/// (F16) receives the result; a non-null `swiglu_gate` (the gate
/// projection's F32 output over the same rows) turns it into
/// silu(gate) * result. `tile_rows` is the token rows per tile the map was
/// built with: 16 or 48, or 64 for Q5_1/Q8_0/MXFP4.
/// Q2_K, Q4_K, Q5_K and Q6_K need k % 256 == 0; MXFP4, Q5_1 and Q8_0
/// k % 64 == 0; other types and tile widths return false.
bool RoutedF16Gemm(const void* w, WeightType type, const __half* x,
                   const std::int32_t* tiles, std::uint32_t n_tiles,
                   std::uint32_t tile_rows, const std::int32_t* pad_bounds,
                   const std::int32_t* rows_in, const std::int32_t* rows_out,
                   const float* swiglu_gate, float* out, __half* out_half,
                   std::size_t m, std::size_t k, hipStream_t stream);

/// Paired gate/up expert GEMM with SwiGLU and F16 output. Uses the same
/// routing layout as RoutedF16Gemm, with 64 or 128 token rows per tile. The
/// two projections share one launch and never materialize the gate output.
bool RoutedGatedF16Gemm(const void* gate, const void* up, WeightType type,
                        const __half* x, const std::int32_t* tiles,
                        std::uint32_t n_tiles, std::uint32_t tile_rows,
                        const std::int32_t* pad_bounds,
                        const std::int32_t* rows_in,
                        const std::int32_t* rows_out, __half* out,
                        std::size_t m, std::size_t k, hipStream_t stream);

/// out[t][i] = sum_s weights[t][s] * expert_out[t*k + s][i].
void MoeEpilogue(const float* expert_out, const float* weights, float* out,
                 std::uint32_t n_tokens, std::uint32_t k,
                 std::uint32_t dim, hipStream_t stream);
/// Same contract, four adjacent lanes per thread (`dim % 4 == 0`, else the
/// scalar kernel runs).
void MoeEpilogueVec4(const float* expert_out, const float* weights,
                     float* out, std::uint32_t n_tokens, std::uint32_t k,
                     std::uint32_t dim, hipStream_t stream);
/// MoeEpilogueVec4 over F16 expert rows.
void MoeEpilogueVec4F16(const __half* expert_out, const float* weights,
                        float* out, std::uint32_t n_tokens, std::uint32_t k,
                        std::uint32_t dim, hipStream_t stream);

/// dst[t] = *row < 0 ? alt[t] : base[*row + t] (rows of `width` floats).
void MtpHidden(const float* base, const float* alt, const std::int32_t* row,
               float* dst, std::uint32_t n_tokens, std::uint32_t width,
               hipStream_t stream);
/// Slot-table form: base/alt come from ptrs[slot] and *row from
/// ctrl[slot].hidden_row. A null base (cancelled slot) launches nothing.
void MtpHiddenSlot(const SlotPtrs* ptrs, const SlotControl* ctrl,
                   std::uint32_t slot, float* dst, std::uint32_t n_tokens,
                   std::uint32_t width, hipStream_t stream);

/// dst = *dstp ? src[0..rows*width) : nothing: the slot-indirected
/// device-to-device row copy a replayed graph uses for each session's
/// private hidden rows (a memcpy node cannot read its destination from a
/// table).
void CopyRowsSlot(const float* src, float* const* dstp, std::uint32_t rows,
                  std::uint32_t width, hipStream_t stream);

/// residual[t] += addend[t] for each of `n_tokens` rows, broadcasting the
/// `hidden`-float addend row over the row's `streams` consecutive slices.
/// The MTP block adds its projected embedding this way; the TP all-reduce
/// adds the peer's partial with `streams` = 1.
void StagePartialHalf(const float* partial, __half* send, std::size_t count,
                      std::uint32_t* arrivals, std::uint64_t* ready,
                      std::uint64_t value, hipStream_t stream);
/// The half-payload all-reduce add. The local operand narrows to fp16 before
/// the add so both ranks compute fp16(r_local) + fp16(r_peer) bit-identically;
/// rounding only the wire side makes the two ranks disagree at f16 resolution,
/// which diverges greedy choices and (via speculative chain lengths)
/// desynchronizes the collectives.
void AddRowsBroadcastHalf(const __half* addend, float* residual,
                          std::uint32_t n_tokens, std::uint32_t hidden,
                          std::uint32_t streams, hipStream_t stream);
void AddRowsBroadcast(const float* addend, float* residual,
                      std::uint32_t n_tokens, std::uint32_t hidden,
                      std::uint32_t streams, hipStream_t stream);

/// Copies `count` floats of a partial into a host-visible send window, then
/// publishes `value` in `ready` once every block's copy is visible to the
/// host. `arrivals` counts finished blocks and must start at zero; the last
/// block resets it.
void StagePartial(const float* partial, float* send, std::size_t count,
                  std::uint32_t* arrivals, std::uint64_t* ready,
                  std::uint64_t value, hipStream_t stream);
/// The captured-exchange form: `slot` holds `id + 1` for the replaying
/// exchange, and `send` is the send buffer's payload base — the kernel stages
/// into window `ExchangeWindowOffset(*slot - 1)`, the same window the replayed
/// host function forwards.
void StagePartialSlotted(const float* partial, float* send, std::size_t count,
                         std::uint32_t* arrivals, std::uint64_t* ready,
                         const std::uint64_t* slot, hipStream_t stream);
void WaitValueSlotted(const std::uint64_t* flag, const std::uint64_t* slot,
                      std::uint64_t ticks, std::uint32_t* timed_out,
                      hipStream_t stream);
/// The captured-exchange form of `AddRowsBroadcast`: `peer` is the receive
/// buffer's payload base, and the kernel reads window
/// `ExchangeWindowOffset(*slot - 1)` — where the replayed host function landed
/// the peer's partial.
void AddRowsBroadcastSlotted(const float* peer, float* residual,
                             std::uint32_t n_tokens, std::uint32_t hidden,
                             std::uint32_t streams,
                             const std::uint64_t* slot, hipStream_t stream);


/// Holds the stream until host-written `flag` reaches `value`. After `ticks`
/// of the device wall clock it sets `timed_out` and lets the stream go on.
void WaitValue(const std::uint64_t* flag, std::uint64_t value,
               std::uint64_t ticks, std::uint32_t* timed_out,
               hipStream_t stream);

inline constexpr std::uint32_t kArgmaxParts = 64;
struct ArgmaxCandidate {
  float value;
  std::int32_t index;
};
/// Full-vocabulary argmax into out[t], with the lowest index winning ties.
/// Matches std::max_element, including NaNs. Scratch holds
/// n_tokens * kArgmaxParts candidates.
void Argmax(const float* logits, ArgmaxCandidate* scratch, std::int32_t* out,
            std::uint32_t n_tokens, std::uint32_t vocab, hipStream_t stream);

/// Gather selected IDs and their original logits from independent rows.
/// The caller checks finite values only for the verification prefix it visits.
void GatherArgmaxCandidates(const float* logits, const std::uint32_t* ids,
                            ArgmaxCandidate* out, std::uint32_t rows,
                            std::uint32_t vocab, hipStream_t stream);

/// Number of IDs in each of the two selection buffers.
std::uint32_t MtpCandidateWorkspaceSize(std::uint32_t vocab);

/// Exact MTP top-64 selection, descending score with lowest-token-ID ties.
/// Nonfinite scores become -infinity. Both ID buffers have room for
/// MtpCandidateWorkspaceSize(vocab); scores receives min(vocab, 64) values.
/// Scores may reuse the ID workspace starting at offset 64.
/// All candidate operations are graph-safe.
void MtpTopCandidates(const float* logits, std::uint32_t* ids,
                      std::uint32_t* scratch_ids, float* scores,
                      std::uint32_t vocab, hipStream_t stream);

}  // namespace gufo::models::mimo2::rocm

#endif  // GUFO_MODELS_MIMO2_KERNELS_ROCM_KERNELS_HPP_
