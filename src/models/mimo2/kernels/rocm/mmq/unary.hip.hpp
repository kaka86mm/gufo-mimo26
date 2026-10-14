#pragma once
#include "common.hip.hpp"

#define CUDA_NEG_BLOCK_SIZE 256
#define CUDA_STEP_BLOCK_SIZE 256
#define CUDA_GELU_BLOCK_SIZE 256
#define CUDA_SILU_BLOCK_SIZE 256
#define CUDA_SILU_BACK_BLOCK_SIZE 256
#define CUDA_TANH_BLOCK_SIZE 256
#define CUDA_RELU_BLOCK_SIZE 256
#define CUDA_SIGMOID_BLOCK_SIZE 256
#define CUDA_HARDSIGMOID_BLOCK_SIZE 256
#define CUDA_EXP_BLOCK_SIZE 256
#define CUDA_HARDSWISH_BLOCK_SIZE 256
#define CUDA_SQR_BLOCK_SIZE 256
#define CUDA_SQRT_BLOCK_SIZE 256
#define CUDA_SIN_BLOCK_SIZE 256
#define CUDA_COS_BLOCK_SIZE 256
#define CUDA_GLU_BLOCK_SIZE 256
#define CUDA_XIELU_BLOCK_SIZE 256

void mimo2_ggml_hip_op_abs(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_sgn(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_neg(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_step(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_gelu(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_silu(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_silu_back(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_gelu_erf(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_gelu_quick(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_tanh(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_relu(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_sigmoid(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_hardsigmoid(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_exp(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_hardswish(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_leaky_relu(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_sqr(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_sqrt(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_sin(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_cos(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_log(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_expm1(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_softplus(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_elu(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_floor(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_ceil(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_round(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_trunc(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_reglu(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_geglu(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_swiglu(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_swiglu_oai(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_geglu_erf(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_geglu_quick(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_xielu(mimo2_ggml_hip_context & ctx, ggml_tensor * dst);

void mimo2_ggml_hip_op_unary_mul(mimo2_ggml_hip_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node);

void mimo2_ggml_hip_op_relu_sqr(mimo2_ggml_hip_context & ctx, ggml_tensor * relu_node, ggml_tensor * sqr_node);

__device__ __forceinline__ float mimo2_ggml_hip_op_silu_single(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float mimo2_ggml_hip_op_gelu_single(float x) {
    const float GELU_COEF_A    = 0.044715f;
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;

    return 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
}

__device__ __forceinline__ float mimo2_ggml_hip_op_swiglu_oai_single(float x, float g, float alpha = 1.702f, float limit = 7.0f) {
    x = fminf(x, limit);
    g = fmaxf(fminf(g, limit), -limit);

    float out_glu = x / (1.0f + expf(-x * alpha));
    out_glu = out_glu * (1.0f + g);
    return out_glu;
}
