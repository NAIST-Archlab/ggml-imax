#pragma once
#include <stddef.h>
#include "ggml-imax.h"

#define IMAX_KERNEL_DEBUG

#ifdef IMAX_KERNEL_DEBUG
#define GGML_IMAX_KERNEL_LOG_DEBUG(...) \
    fprintf(stderr, "ggml-imax-kernel: "); \
    fprintf(stderr, __VA_ARGS__); \
    fprintf(stderr, "\n");
#else
#define GGML_IMAX_KERNEL_LOG_DEBUG(...)
#endif

struct imax_kernel_args {
    void* src0;
    void* src1;
    void* src2;
    void*  dst;
    void* wdata;
    
    int src0_ne[GGML_MAX_DIMS];
    int src0_nb[GGML_MAX_DIMS];
    int src1_ne[GGML_MAX_DIMS];
    int src1_nb[GGML_MAX_DIMS];
    int src2_ne[GGML_MAX_DIMS];
    int src2_nb[GGML_MAX_DIMS];
    int dst_ne [GGML_MAX_DIMS];
    int dst_nb [GGML_MAX_DIMS];

    enum ggml_type src0_type;
    enum ggml_type src1_type;
    enum ggml_type src2_type;
    enum ggml_type dst_type;

    int32_t src0_op_params[16];
    int32_t src1_op_params[16];
    int32_t src2_op_params[16];
    int32_t dst_op_params [16];

    int nb;
    int lane;
};

// On IMAX + CPU
void* kernel_add                 (struct imax_kernel_args* args);
void* kernel_mul                 (struct imax_kernel_args* args);
void* kernel_mul_mm_f32_f32      (struct imax_kernel_args* args);
void* kernel_mul_mm_f16_f32      (struct imax_kernel_args* args);
void* kernel_mul_mm_q4_0_f32     (struct imax_kernel_args* args);
void* kernel_mul_mm_q4_1_f32     (struct imax_kernel_args* args);
void* kernel_mul_mm_q5_0_f32     (struct imax_kernel_args* args);
void* kernel_mul_mm_q5_1_f32     (struct imax_kernel_args* args);
void* kernel_mul_mm_q8_0_f32     (struct imax_kernel_args* args);
void* kernel_mul_mm_q2_K_f32     (struct imax_kernel_args* args);
void* kernel_mul_mm_q3_K_f32     (struct imax_kernel_args* args);
void* kernel_mul_mm_q4_K_f32     (struct imax_kernel_args* args);
void* kernel_mul_mm_q5_K_f32     (struct imax_kernel_args* args);
void* kernel_mul_mm_q6_K_f32     (struct imax_kernel_args* args);

// On Only CPU
void* kernel_scale               (struct imax_kernel_args* args);
void* kernel_upscale_f32         (struct imax_kernel_args* args);
void* kernel_pad_f32             (struct imax_kernel_args* args);
void* kernel_argsort_f32_i32_asc (struct imax_kernel_args* args);
void* kernel_argsort_f32_i32_desc(struct imax_kernel_args* args);
void* kernel_sum_rows            (struct imax_kernel_args* args);
void* kernel_div                 (struct imax_kernel_args* args);
void* kernel_sqr                 (struct imax_kernel_args* args);
void* kernel_softmax             (struct imax_kernel_args* args);
void* kernel_rms_norm            (struct imax_kernel_args* args);
void* kernel_norm                (struct imax_kernel_args* args);
void* kernel_group_norm          (struct imax_kernel_args* args);
void* kernel_get_rows            (struct imax_kernel_args* args);
void* kernel_alibi               (struct imax_kernel_args* args);
void* kernel_rope                (struct imax_kernel_args* args);
void* kernel_im2col              (struct imax_kernel_args* args);
void* kernel_pool_1d             (struct imax_kernel_args* args);
void* kernel_pool_2d             (struct imax_kernel_args* args);
void* kernel_leaky_relu          (struct imax_kernel_args* args);
void* kernel_dup                 (struct imax_kernel_args* args);
void* kernel_cpy                 (struct imax_kernel_args* args);
void* kernel_contiguous          (struct imax_kernel_args* args);
void* kernel_transpose           (struct imax_kernel_args* args);
void* kernel_diag_mask_inf       (struct imax_kernel_args* args);
void* kernel_unary               (struct imax_kernel_args* args);