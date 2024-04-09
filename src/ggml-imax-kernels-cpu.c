#include "ggml-imax-kernels.h"
#include "ggml.h"
#include "ggml-quants.h"

#include <stdio.h>

#ifndef DMA_MMAP_SIZE
#define DMA_MMAP_SIZE 0x0000000000010000LL
#endif

#ifndef DDR_MMAP_SIZE
#define DDR_MMAP_SIZE	 0x0000000100000000LL
#endif

static const ggml_type_traits_t type_traits[GGML_TYPE_COUNT] = {
    [GGML_TYPE_I8] = {
        .type_name                = "i8",
        .blck_size                = 1,
        .type_size                = sizeof(int8_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I16] = {
        .type_name                = "i16",
        .blck_size                = 1,
        .type_size                = sizeof(int16_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_I32] = {
        .type_name                = "i32",
        .blck_size                = 1,
        .type_size                = sizeof(int32_t),
        .is_quantized             = false,
    },
    [GGML_TYPE_F32] = {
        .type_name                = "f32",
        .blck_size                = 1,
        .type_size                = sizeof(float),
        .is_quantized             = false,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f32,
        .vec_dot_type             = GGML_TYPE_F32,
        .nrows                    = 1,
    },
    [GGML_TYPE_F16] = {
        .type_name                = "f16",
        .blck_size                = 1,
        .type_size                = sizeof(ggml_fp16_t),
        .is_quantized             = false,
        .to_float                 = (ggml_to_float_t) ggml_fp16_to_fp32_row,
        .from_float               = (ggml_from_float_t) ggml_fp32_to_fp16_row,
        .from_float_reference     = (ggml_from_float_t) ggml_fp32_to_fp16_row,
        .vec_dot                  = (ggml_vec_dot_t) ggml_vec_dot_f16,
        .vec_dot_type             = GGML_TYPE_F16,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_0] = {
        .type_name                = "q4_0",
        .blck_size                = QK4_0,
        .type_size                = sizeof(block_q4_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_0,
        .from_float               = quantize_row_q4_0,
        .from_float_reference     = (ggml_from_float_t) quantize_row_q4_0_reference,
        .vec_dot                  = ggml_vec_dot_q4_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q4_1] = {
        .type_name                = "q4_1",
        .blck_size                = QK4_1,
        .type_size                = sizeof(block_q4_1),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_1,
        .from_float               = quantize_row_q4_1,
        .from_float_reference     = (ggml_from_float_t) quantize_row_q4_1_reference,
        .vec_dot                  = ggml_vec_dot_q4_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [4] = { // GGML_TYPE_Q4_2
        .type_name                = "DEPRECATED",
        .blck_size                = 0,
        .type_size                = 0,
        .is_quantized             = false,
        .to_float                 = NULL,
        .from_float               = NULL,
        .from_float_reference     = NULL,
        .vec_dot                  = NULL,
        .vec_dot_type             = GGML_TYPE_COUNT,
        .nrows                    = 1,
    },
    [5] = { // GGML_TYPE_Q4_3
        .type_name                = "DEPRECATED",
        .blck_size                = 0,
        .type_size                = 0,
        .is_quantized             = false,
        .to_float                 = NULL,
        .from_float               = NULL,
        .from_float_reference     = NULL,
        .vec_dot                  = NULL,
        .vec_dot_type             = GGML_TYPE_COUNT,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q5_0] = {
        .type_name                = "q5_0",
        .blck_size                = QK5_0,
        .type_size                = sizeof(block_q5_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_0,
        .from_float               = quantize_row_q5_0,
        .from_float_reference     = (ggml_from_float_t) quantize_row_q5_0_reference,
        .vec_dot                  = ggml_vec_dot_q5_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q5_1] = {
        .type_name                = "q5_1",
        .blck_size                = QK5_1,
        .type_size                = sizeof(block_q5_1),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_1,
        .from_float               = quantize_row_q5_1,
        .from_float_reference     = (ggml_from_float_t) quantize_row_q5_1_reference,
        .vec_dot                  = ggml_vec_dot_q5_1_q8_1,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_0] = {
        .type_name                = "q8_0",
        .blck_size                = QK8_0,
        .type_size                = sizeof(block_q8_0),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q8_0,
        .from_float               = quantize_row_q8_0,
        .from_float_reference     = (ggml_from_float_t) quantize_row_q8_0_reference,
        .vec_dot                  = ggml_vec_dot_q8_0_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
#if defined (__ARM_FEATURE_MATMUL_INT8)
        .nrows                    = 2,
#else
        .nrows                    = 1,
#endif
    },
    [GGML_TYPE_Q8_1] = {
        .type_name                = "q8_1",
        .blck_size                = QK8_1,
        .type_size                = sizeof(block_q8_1),
        .is_quantized             = true,
        .from_float               = quantize_row_q8_1,
        .from_float_reference     = (ggml_from_float_t) quantize_row_q8_1_reference,
        .vec_dot_type             = GGML_TYPE_Q8_1,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q2_K] = {
        .type_name                = "q2_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q2_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q2_K,
        .from_float               = quantize_row_q2_K,
        .from_float_reference     = (ggml_from_float_t) quantize_row_q2_K_reference,
        .vec_dot                  = ggml_vec_dot_q2_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q3_K] = {
        .type_name                = "q3_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q3_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q3_K,
        .from_float               = quantize_row_q3_K,
        .from_float_reference     = (ggml_from_float_t) quantize_row_q3_K_reference,
        .vec_dot                  = ggml_vec_dot_q3_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q4_K] = {
        .type_name                = "q4_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q4_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q4_K,
        .from_float               = quantize_row_q4_K,
        .from_float_reference     = (ggml_from_float_t) quantize_row_q4_K_reference,
        .vec_dot                  = ggml_vec_dot_q4_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q5_K] = {
        .type_name                = "q5_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q5_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q5_K,
        .from_float               = quantize_row_q5_K,
        .from_float_reference     = (ggml_from_float_t) quantize_row_q5_K_reference,
        .vec_dot                  = ggml_vec_dot_q5_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_Q6_K] = {
        .type_name                = "q6_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q6_K),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_q6_K,
        .from_float               = quantize_row_q6_K,
        .from_float_reference     = (ggml_from_float_t) quantize_row_q6_K_reference,
        .vec_dot                  = ggml_vec_dot_q6_K_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XXS] = {
        .type_name                = "iq2_xxs",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq2_xxs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_xxs,
        .from_float               = NULL,
        .from_float_reference     = NULL,
        .vec_dot                  = ggml_vec_dot_iq2_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_XS] = {
        .type_name                = "iq2_xs",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq2_xs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_xs,
        .from_float               = NULL,
        .from_float_reference     = NULL,
        .vec_dot                  = ggml_vec_dot_iq2_xs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_XXS] = {
        .type_name                = "iq3_xxs",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq3_xxs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq3_xxs,
        .from_float               = quantize_row_iq3_xxs,
        .from_float_reference     = (ggml_from_float_t)quantize_row_iq3_xxs_reference,
        .vec_dot                  = ggml_vec_dot_iq3_xxs_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ3_S] = {
        .type_name                = "iq3_s",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq3_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq3_s,
        .from_float               = quantize_row_iq3_s,
        .from_float_reference     = (ggml_from_float_t)quantize_row_iq3_s_reference,
        .vec_dot                  = ggml_vec_dot_iq3_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ2_S] = {
        .type_name                = "iq2_s",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq2_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq2_s,
        .from_float               = quantize_row_iq2_s,
        .from_float_reference     = (ggml_from_float_t)quantize_row_iq2_s_reference,
        .vec_dot                  = ggml_vec_dot_iq2_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ1_S] = {
        .type_name                = "iq1_s",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_iq1_s),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq1_s,
        .from_float               = NULL,
        .from_float_reference     = NULL,
        .vec_dot                  = ggml_vec_dot_iq1_s_q8_K,
        .vec_dot_type             = GGML_TYPE_Q8_K,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_NL] = {
        .type_name                = "iq4_nl",
        .blck_size                = QK4_NL,
        .type_size                = sizeof(block_iq4_nl),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq4_nl,
        .from_float               = quantize_row_iq4_nl,
        .from_float_reference     = (ggml_from_float_t)quantize_row_iq4_nl_reference,
        .vec_dot                  = ggml_vec_dot_iq4_nl_q8_0,
        .vec_dot_type             = GGML_TYPE_Q8_0,
        .nrows                    = 1,
    },
    [GGML_TYPE_IQ4_XS] = {
        .type_name                = "iq4_xs",
#if QK_K == 64
        .blck_size                = QK4_NL,
#else
        .blck_size                = QK_K,
#endif
        .type_size                = sizeof(block_iq4_xs),
        .is_quantized             = true,
        .to_float                 = (ggml_to_float_t) dequantize_row_iq4_xs,
        .from_float               = quantize_row_iq4_xs,
        .from_float_reference     = (ggml_from_float_t)quantize_row_iq4_xs_reference,
        .vec_dot                  = ggml_vec_dot_iq4_xs_q8_K,
#if QK_K == 64
        .vec_dot_type             = GGML_TYPE_Q8_0,
#else
        .vec_dot_type             = GGML_TYPE_Q8_K,
#endif
        .nrows                    = 1,
    },
    [GGML_TYPE_Q8_K] = {
        .type_name                = "q8_K",
        .blck_size                = QK_K,
        .type_size                = sizeof(block_q8_K),
        .is_quantized             = true,
        .from_float               = quantize_row_q8_K,
    }
};

void* kernel_upscale_f32(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    return NULL;
}

void* kernel_pad_f32(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    return NULL;

}

void* kernel_argsort_f32_i32_asc(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    return NULL;

}

void* kernel_argsort_f32_i32_desc(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    return NULL;
}

void* kernel_sum_rows(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    return NULL;
}

void* kernel_scale(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    return NULL;

}

void* kernel_scale_4(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    return NULL;
}

void* kernel_div(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    return NULL;

}

void* kernel_sqr(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    return NULL;

}

void* kernel_softmax(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    return NULL;

}

void* kernel_rms_norm(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    return NULL;

}

void* kernel_norm(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    return NULL;

}

void* kernel_group_norm(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    return NULL;
}

static void kernel_get_rows_q(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);

    int64_t nc = args->src0_ne[0];
    int64_t nr = args->src1_ne[0]*args->src1_ne[1]*args->src1_ne[2]*args->src1_ne[3];

    GGML_ASSERT(args->dst_ne[0] == args->src0_ne[0]);
    GGML_ASSERT(args->src0_ne[2] == args->src1_ne[1]);
    GGML_ASSERT(args->src0_nb[0] == sizeof(ggml_fp16_t));
    GGML_ASSERT(args->dst_ne[1]*args->dst_ne[2]*args->dst_ne[3] == nr);

    ggml_to_float_t dequantize_row_q = type_traits[args->src0_type].to_float;

    void** src0_p = args->src0;
    void** src1_p = args->src1;
    void** dst_p  = args->dst;

    for (int64_t i12 = 0; i12 < args->src1_ne[2]; i12++) {
        for (int64_t i11 = 0; i11 < args->src1_ne[1]; i11++) {
            for (int64_t i10 = 0; i10 < args->src1_ne[0]; i10++) {
                uint64_t src1_index = (i12*args->src1_nb[2] + i11*args->src1_nb[1] + i10*src1_nb[0]);
                uint64_t src1_blk = src1_index/DMA_MMAP_SIZE;
                const int64_t i01 = *(int32_t *)((char *)&src1_p[src1_blk][src1_index%DMA_MMAP_SIZE]);

                uint64_t src0_index = (i12*args->src0_nb[3] + i11*args->src0_nb[2] + i01*src0_nb[1]);
                uint64_t src0_blk = src0_index/DMA_MMAP_SIZE;
                uint64_t dst_index  = (i12*args->dst_nb[3] + i11*args->dst_nb[2] + i10*dst_nb[1]);
                uint64_t dst_blk  = dst_index/DMA_MMAP_SIZE;
                dequantize_row_q(
                    (const void *) (&src0_p[src0_blk][src0_index%DMA_MMAP_SIZE]),
                    (void *) (&dst_p[dst_blk][dst_index%DMA_MMAP_SIZE]), nc
                )                
            }
        }
    }
}

static void kernel_get_rows_f16(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);

    int64_t nc = args->src0_ne[0];
    int64_t nr = args->src1_ne[0]*args->src1_ne[1]*args->src1_ne[2]*args->src1_ne[3];

    GGML_ASSERT(args->dst_ne[0] == args->src0_ne[0]);
    GGML_ASSERT(args->src0_ne[2] == args->src1_ne[1]);
    GGML_ASSERT(args->src0_nb[0] == sizeof(ggml_fp16_t));
    GGML_ASSERT(args->dst_ne[1]*args->dst_ne[2]*args->dst_ne[3] == nr);

    void** src0_p = args->src0;
    void** src1_p = args->src1;
    void** dst_p  = args->dst;

    for (int64_t i12 = 0; i12 < args->src1_ne[2]; i12++) {
        for (int64_t i11 = 0; i11 < args->src1_ne[1]; i11++) {
            for (int64_t i10 = 0; i10 < args->src1_ne[0]; i10++) {
                uint64_t src1_index = (i12*args->src1_nb[2] + i11*args->src1_nb[1] + i10*src1_nb[0]);
                uint64_t src1_blk = src1_index/DMA_MMAP_SIZE;
                const int64_t i01 = *(int32_t *)((char *)&src1_p[src1_blk][src1_index%DMA_MMAP_SIZE]);

                // TODO: SIMD
                for (int64_t i0 = 0; i0 < nc; i0++) {
                    uint64_t src0_index = (i12*args->src0_nb[3] + i11*args->src0_nb[2] + i01*src0_nb[1] + i0*src0_nb[0]);
                    uint64_t src0_blk = src0_index/DMA_MMAP_SIZE;
                    uint64_t dst_index  = (i12*args->dst_nb[3] + i11*args->dst_nb[2] + i10*dst_nb[1] + i0*dst_nb[0]);
                    uint64_t dst_blk  = dst_index/DMA_MMAP_SIZE;
                    *(uint32_t*)&dst_p[dst_blk][dst_index%DMA_MMAP_SIZE] = ggml_fp16_to_fp32(*(ggml_fp16_t*)&src0_p[src0_blk][src0_index%DMA_MMAP_SIZE]);
                }
            }
        }
    }
}

static void kernel_get_rows_f32(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);

    int64_t nc = args->src0_ne[0];
    int64_t nr = args->src1_ne[0]*args->src1_ne[1]*args->src1_ne[2]*args->src1_ne[3];

    GGML_ASSERT(args->dst_ne[0] == args->src0_ne[0]);
    GGML_ASSERT(args->src0_ne[2] == args->src1_ne[1]);
    GGML_ASSERT(args->src0_nb[0] == sizeof(float));
    GGML_ASSERT(args->dst_ne[1]*args->dst_ne[2]*args->dst_ne[3] == nr);

    void** src0_p = args->src0;
    void** src1_p = args->src1;
    void** dst_p  = args->dst;

    for (int64_t i12 = 0; i12 < args->src1_ne[2]; i12++) {
        for (int64_t i11 = 0; i11 < args->src1_ne[1]; i11++) {
            for (int64_t i10 = 0; i10 < args->src1_ne[0]; i10++) {
                uint64_t src1_index = (i12*args->src1_nb[2] + i11*args->src1_nb[1] + i10*src1_nb[0]);
                uint64_t src1_blk = src1_index/DMA_MMAP_SIZE;
                const int64_t i01 = *(int32_t *)((char *)&src1_p[src1_blk][src1_index%DMA_MMAP_SIZE]);

                // TODO: SIMD
                for (int64_t i0 = 0; i0 < nc; i0++) {
                    uint64_t src0_index = (i12*args->src0_nb[3] + i11*args->src0_nb[2] + i01*src0_nb[1] + i0*src0_nb[0]);
                    uint64_t src0_blk = src0_index/DMA_MMAP_SIZE;
                    uint64_t dst_index  = (i12*args->dst_nb[3] + i11*args->dst_nb[2] + i10*dst_nb[1] + i0*dst_nb[0]);
                    uint64_t dst_blk  = dst_index/DMA_MMAP_SIZE;
                    *(uint32_t*)&dst_p[dst_blk][dst_index%DMA_MMAP_SIZE] = *(uint32_t*)&src0_p[src0_blk][src0_index%DMA_MMAP_SIZE];
                }
            }
        }
    }
}

void* kernel_get_rows(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);

    switch(args->src0_type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
            {
                kernel_get_rows_q(args);
            } break;
        case GGML_TYPE_F16:
            {
                kernel_get_rows_f16(args);
            } break;
        case GGML_TYPE_F32:
        case GGML_TYPE_I32:
            {
                kernel_get_rows_f32(args);
            } break;
        default:
            {
                GGML_ASSERT(0);
            } break;
    }

    return NULL;
}

void* kernel_alibi(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    return NULL;
}

void* kernel_rope(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    return NULL;
}

void* kernel_im2col(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    return NULL;
}

void* kernel_pool_1d(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    return NULL;
}

void* kernel_pool_2d(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    return NULL;
}

void* kernel_leaky_relu(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    return NULL;
}

void* kernel_dup(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    return NULL;
}

void* kernel_cpy(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    kernel_dup(args);
    return NULL;
}

void* kernel_contiguous(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    kernel_dup(args);
    return NULL;
}

void* kernel_transpose(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    return NULL;
}

void* kernel_diag_mask_inf(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    return NULL;
}

void* kernel_unary(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    return NULL;
}