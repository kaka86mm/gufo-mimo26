#ifndef GUFO_MODELS_MIMO2_KERNELS_ROCM_DFLASH_V1_KERNELS_HPP_
#define GUFO_MODELS_MIMO2_KERNELS_ROCM_DFLASH_V1_KERNELS_HPP_

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

/// Launchers private to the DFlash v1 draft sidecar (a five-block 1.5B model
/// with a non-causal block attention over a ring of injected target-feature
/// K/V). Everything else the drafter needs comes from kernels.hpp: RmsNormRows
/// (grouped per-head norms), Rope (NEOX partial rotary), EmbedTokens,
/// SmallGemm/W8A8Gemm and Argmax.
namespace gufo::models::mimo2::rocm {

/// dst[i] += src[i] over `count` floats.
void DFlashV1AddInPlace(float* dst, const float* src, std::size_t count,
                        hipStream_t stream);

/// Replaces the embedding rows of the mask slots with the drafter's own mask
/// embedding: hidden[row] = mask_embedding where tokens[row] == mask_token_id.
/// The trunk's embedding table holds zeros at that token, so every masked slot
/// in a block would otherwise be an all-zero vector.
void DFlashV1MaskSubstitute(float* hidden, const std::int32_t* tokens,
                            std::uint32_t rows, std::uint32_t hidden_size,
                            std::int32_t mask_token_id,
                            const float* mask_embedding, hipStream_t stream);

/// Normalizes every `slice_width`-wide slice of each row of `x` to unit RMS in
/// place: the unit-weight per-tap norm the vLLM drafter applies before its fc
/// (`aux_hidden_norms`, whose weights the checkpoint never carries).
void DFlashV1SliceRmsNorm(float* x, std::uint32_t rows,
                          std::uint32_t slice_width, std::uint32_t slices,
                          float eps, hipStream_t stream);

/// One block of DFlash v1 draft attention: every query row attends a single
/// trailing window of the fp32 history ring plus the block's own K/V rows.
/// GQA groups share a K/V head. Every query sees the history rows
/// [history_length - min(history_length, window), history_length)
/// ring-addressed as `pos % history_capacity`; the whole block shares that one
/// window (the v1 reference's `swa_lo = committed - window`, which a per-row
/// lower bound would silently change). `causal_block` restricts the block part
/// to rows `<=` the query row (sliding layers); otherwise the block is
/// non-causal. `sinks` stays null for v1 drafters (the reference never reads
/// it); a non-null pointer adds one denominator term per head. `position`
/// points at the committed length on the device (the injected cache holds that
/// prefix), so a captured graph replays at any position.
/// q/out: [block_count][q_heads][head_dim] fp32. hist_k/hist_v:
/// [history_capacity][kv_heads*head_dim] fp32. block_k/block_v:
/// [block_count][kv_heads*head_dim]. Requires head_dim % 32 == 0 and
/// head_dim <= 256.
void DFlashV1BlockAttention(const float* q, const float* hist_k,
                            const float* hist_v, const float* block_k,
                            const float* block_v, const float* sinks,
                            float* out, const std::uint32_t* position,
                            std::uint32_t block_count, std::uint32_t window,
                            std::uint32_t q_heads, std::uint32_t kv_heads,
                            std::uint32_t head_dim, float scale,
                            std::uint32_t history_capacity, bool causal_block,
                            hipStream_t stream);

}  // namespace gufo::models::mimo2::rocm

#endif  // GUFO_MODELS_MIMO2_KERNELS_ROCM_DFLASH_V1_KERNELS_HPP_