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

#define load_src01_dst(args_name) \
    uint8_t** src0_p =  (uint8_t**)args_name->src0;      \
    uint8_t** src1_p =  (uint8_t**)args_name->src1;      \
    uint8_t** dst_p  =  (uint8_t**)args_name->dst;       \
    uint64_t    ne00 =  *(uint64_t*)&args_name->src0_ne[0];\
    uint64_t    ne01 =  *(uint64_t*)&args_name->src0_ne[1];\
    uint64_t    ne02 =  *(uint64_t*)&args_name->src0_ne[2];\
    uint64_t    ne03 =  *(uint64_t*)&args_name->src0_ne[3];\
    uint64_t    nb00 =  *(uint64_t*)&args_name->src0_nb[0];\
    uint64_t    nb01 =  *(uint64_t*)&args_name->src0_nb[1];\
    uint64_t    nb02 =  *(uint64_t*)&args_name->src0_nb[2];\
    uint64_t    nb03 =  *(uint64_t*)&args_name->src0_nb[3];\
    uint64_t    ne10 =  *(uint64_t*)&args_name->src1_ne[0];\
    uint64_t    ne11 =  *(uint64_t*)&args_name->src1_ne[1];\
    uint64_t    ne12 =  *(uint64_t*)&args_name->src1_ne[2];\
    uint64_t    ne13 =  *(uint64_t*)&args_name->src1_ne[3];\
    uint64_t    nb10 =  *(uint64_t*)&args_name->src1_nb[0];\
    uint64_t    nb11 =  *(uint64_t*)&args_name->src1_nb[1];\
    uint64_t    nb12 =  *(uint64_t*)&args_name->src1_nb[2];\
    uint64_t    nb13 =  *(uint64_t*)&args_name->src1_nb[3];\
    uint64_t    ne0  =  *(uint64_t*)&args_name->dst_ne[0]; \
    uint64_t    ne1  =  *(uint64_t*)&args_name->dst_ne[1]; \
    uint64_t    ne2  =  *(uint64_t*)&args_name->dst_ne[2]; \
    uint64_t    ne3  =  *(uint64_t*)&args_name->dst_ne[3]; \
    uint64_t    nb0  =  *(uint64_t*)&args_name->dst_nb[0]; \
    uint64_t    nb1  =  *(uint64_t*)&args_name->dst_nb[1]; \
    uint64_t    nb2  =  *(uint64_t*)&args_name->dst_nb[2]; \
    uint64_t    nb3  =  *(uint64_t*)&args_name->dst_nb[3]; \
    int32_t* src0_op_params = args_name->src0_op_params;  \
    int32_t* src1_op_params = args_name->src1_op_params;  \
    int32_t* src2_op_params = args_name->src2_op_params;  \
    int32_t* dst_op_params  = args_name->dst_op_params;


void* kernel_upscale_f32(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    load_src01_dst(args);

    GGML_ASSERT(nb00 == sizeof(float));
    int scale_factor = dst_op_params[0];

    for (int i3 = 0; i3 < ne3; i3++) {
        int64_t i03 = i3 / scale_factor;
        for (int i2 = 0; i2 < ne2; i2++) {
            int64_t i02 = i2 / scale_factor;
            for (int i1 = 0; i1 < ne1; i1++) {
                int64_t i01 = i1 / scale_factor;
                for (int i0 = 0; i0 < ne0; i0++) {
                    int64_t i00 = i0 / scale_factor;
                    uint64_t dst_index  = (i3*nb3  + i2*nb2  + i1*nb1);
                    uint32_t dst_blk  = dst_index/DMA_MMAP_SIZE;
                    uint64_t src0_index = (i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
                    uint32_t src0_blk = src0_index/DMA_MMAP_SIZE;
                    float* src0 = &src0_p[src0_blk][src0_index%DMA_MMAP_SIZE];
                    float* dst  = &dst_p [dst_blk ][dst_index%DMA_MMAP_SIZE ];

                    *dst = *src0;
                }
            }
        }
    }

    return NULL;
}

void* kernel_pad_f32(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    load_src01_dst(args);

    GGML_ASSERT(nb00 == sizeof(float));
    GGML_ASSERT(nb0  == sizeof(float));

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            for (int i1 = 0; i1 < ne1; i1++) {
                for (int i0 = 0; i0 < ne0; i0++) {
                    uint64_t dst_index  = (i3*nb3  + i2*nb2  + i1*nb1);
                    uint32_t dst_blk  = dst_index/DMA_MMAP_SIZE;
                    uint64_t src0_index = (i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                    uint32_t src0_blk = src0_index/DMA_MMAP_SIZE;
                    float* src0 = &src0_p[src0_blk][src0_index%DMA_MMAP_SIZE];
                    float* dst  = &dst_p [dst_blk ][dst_index%DMA_MMAP_SIZE ];

                    if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        *dst = *src0;
                    } else {
                        *dst = 0;
                    }
                }
            }
        }
    }

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

void kernel_sum_rows_f32(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    load_src01_dst(args);

    for (int i3 = 0; i3 < ne03; i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i1 = 0; i1 < ne01; i1++) {
                float row_sum = 0;
                uint64_t dst_index  = (i3*nb3  + i2*nb2  + i1*nb1);
                uint32_t dst_blk  = dst_index/DMA_MMAP_SIZE;
                float* dst  = &dst_p [dst_blk ][dst_index%DMA_MMAP_SIZE ];
                for (int i0 = 0; i0 < ne00; i0++) {
                    uint64_t src0_index = (i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                    uint32_t src0_blk = src0_index/DMA_MMAP_SIZE;
                    float* src0 = &src0_p[src0_blk][src0_index%DMA_MMAP_SIZE];
                    row_sum += *src0;
                }
                *dst = row_sum;
            }
        }
    }
}

void* kernel_sum_rows(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);

    switch (args->src0_type)  {
        case GGML_TYPE_F32: 
            {
                kernel_sum_rows_f32(args);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }

    return NULL;
}

void kernel_scale_f32(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    load_src01_dst(args);

    float v = *(float*)dst_op_params;

    for (int i3 = 0; i3 < ne03; i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i1 = 0; i1 < ne01; i1++) {
                for (int i0 = 0; i0 < ne00; i0++) {
                    uint64_t src0_index = (i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                    uint64_t dst_index  = (i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0 );
                    uint32_t src0_blk = src0_index/DMA_MMAP_SIZE;
                    uint32_t dst_blk  = dst_index/DMA_MMAP_SIZE;
                    float* src0 = &src0_p[src0_blk][src0_index%DMA_MMAP_SIZE];
                    float* dst  = &dst_p [dst_blk ][dst_index%DMA_MMAP_SIZE ];
                    *dst = *src0 * v;
                }
            }
        }
    }
}

void* kernel_scale(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);

    switch (args->src0_type) {
        case GGML_TYPE_F32:
            {
                kernel_scale_f32(args);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }

    return NULL;
}

void kernel_div_f32(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    load_src01_dst(args);

    for (int i3 = 0; i3 < ne03; i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i1 = 0; i1 < ne01; i1++) {
                for (int i0 = 0; i0 < ne00; i0++) {
                    uint64_t src0_index = (i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                    uint64_t src1_index = (i3*nb13 + i2*nb12 + i1*nb11 + i0*nb10);
                    uint64_t dst_index  = (i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0 );
                    uint32_t src0_blk = src0_index/DMA_MMAP_SIZE;
                    uint32_t src1_blk = src1_index/DMA_MMAP_SIZE;
                    uint32_t dst_blk  = dst_index/DMA_MMAP_SIZE;
                    float* src0 = &src0_p[src0_blk][src0_index%DMA_MMAP_SIZE];
                    float* src1 = &src1_p[src1_blk][src1_index%DMA_MMAP_SIZE];
                    float* dst  = &dst_p [dst_blk ][dst_index%DMA_MMAP_SIZE ];
                    *dst = *src0 / *src1;
                }
            }
        }
    }
}

void* kernel_div(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);

    switch (args->src0_type) {
        case GGML_TYPE_F32:
            {
                kernel_div_f32(args);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }

    return NULL;
}

void kernel_sqr_f32(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    load_src01_dst(args);

    for (int i3 = 0; i3 < ne03; i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i1 = 0; i1 < ne01; i1++) {
                for (int i0 = 0; i0 < ne00; i0++) {
                    uint64_t src0_index = (i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                    uint64_t dst_index  = (i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0 );
                    uint32_t src0_blk = src0_index/DMA_MMAP_SIZE;
                    uint32_t dst_blk  = dst_index/DMA_MMAP_SIZE;
                    float* src0 = &src0_p[src0_blk][src0_index%DMA_MMAP_SIZE];
                    float* dst  = &dst_p [dst_blk ][dst_index%DMA_MMAP_SIZE ];
                    *dst = *src0 * *src0;
                }
            }
        }
    }
}

void* kernel_sqr(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);

    switch (args->src0_type) {
        case GGML_TYPE_F32:
            {
                kernel_sqr_f32(args);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }

    return NULL;
}

// TODO: Fit IMAX
void* kernel_softmax(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);

    //void** src0_p = args->src0;
    //void** src1_p = args->src1;
    //void** dst_p  = args->dst;

    //float scale    = 1.0f;
    //float max_bias = 0.0f;

    //memcpy(&scale,    (float *) dst->op_params + 0, sizeof(float));
    //memcpy(&max_bias, (float *) dst->op_params + 1, sizeof(float));

    //const uint32_t n_head_kv   = ne02;
    //const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head_kv));

    //const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    //const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    //const int nc = args->src0_ne[0];
    //const int nr = args->src1_ne[0]*args->src1_ne[1]*args->src1_ne[2]*args->src1_ne[3];

    //// rows per thread
    //const int dr = (nr + nth - 1)/nth;

    //// row range for this thread
    //const int ir0 = dr*ith;
    //const int ir1 = MIN(ir0 + dr, nr);


    //// when max_bias <= 0.0f, src2 is not used and we default it to src0 to avoid branching
    //float * pos = src2 ? (float *) src2->data : src0->data;

    //for (int i1 = 0; i1 < nr; i1++) {
        //for (int i0 = 0; i0 < nc; i0++) {
            //float * sp = (float *)((char *) src0->data + i1*src0->nb[1]);
            //float * wp = (float *) params->wdata + (nc + CACHE_LINE_SIZE_F32) * ith; 
            //float * mp = src1 ? (float *)((char *) src1->data + (i1%ne11)*src1->nb[1]) : NULL;

            //*wp *= scale *sp;
            //if (mp)  {
                //*wp += *mp;
            //}
        //}        

        //for (max_bias > 0.0f) {
            //const uint32_t h = (i1/ne01)%ne02;

            //const float slope = h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1);

            //for (int i = 0; i < nc; i++) {
                //wp[i] = wp[i] + slope*pos[i];
            //}
        //}

        //float max = -INFINITY;
        //ggml_vec_max_f32(nc, &max, wp);

        //ggml_float sum = 0.0;

        //uint16_t scvt;
        //for (int i = 0; i < nc; i++) {
            //if (wp[i] == -INFINITY) {
                //dp[i] = 0.0f;
            //} else {
                //// const float val = (wp[i] == -INFINITY) ? 0.0 : exp(wp[i] - max);
                //ggml_fp16_t s = GGML_FP32_TO_FP16(wp[i] - max);
                //memcpy(&scvt, &s, sizeof(scvt));
                //const float val = GGML_FP16_TO_FP32(ggml_table_exp_f16[scvt]);
                //sum += (ggml_float)val;
                //dp[i] = val;
            //}
        //}

        //sum = 1.0/sum;
        //ggml_vec_scale_f32(nc, dp, sum);
    //}

    return NULL;

}

void* kernel_rms_norm(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    return NULL;

}

void kernel_norm_f32(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);

    GGML_ASSERT(args->src0_type == GGML_TYPE_F32);

    const float eps = 1e-6f;

    void** src0_p = args->src0;
    void** src1_p = args->src1;
    void** dst_p  = args->dst;

    for (int64_t i03 = 0; i03 < args->src0_ne[3]; i03++) {
        for (int64_t i02 = 0; i02 < args->src0_ne[2]; i02++) {
            for (int64_t i01 = 0; i01 < args->src0_ne[1]; i01++) {

                double sum = 0.0;
                for (int64_t i00 = 0; i00 < args->src0_ne[0]; i00++) {
                    uint64_t src0_index = (i03*args->src0_nb[3] + i02*args->src0_nb[2] + i01*args->src0_nb[1] + i00*args->src0_nb[0]);
                    uint64_t src0_blk = src0_index/DMA_MMAP_SIZE;
                    sum += (double)(*(float *) (&src0_p[src0_blk][src0_index%DMA_MMAP_SIZE]));
                }

                float mean = sum/args->src0_ne[0];

                double sum2 = 0.0;
                for (int64_t i00 = 0; i00 < args->src0_ne[0]; i00++) {
                    uint64_t src0_index = (i03*args->src0_nb[3] + i02*args->src0_nb[2] + i01*args->src0_nb[1] + i00*args->src0_nb[0]);
                    uint64_t src0_blk = src0_index/DMA_MMAP_SIZE;
                    uint64_t dst_index  = (i03*args->dst_nb[3] + i02*args->dst_nb[2] + i01*args->dst_nb[1] + i00*args->dst_nb[0]);
                    uint64_t dst_blk  = dst_index/DMA_MMAP_SIZE;
                    float x = (float)(*(float *) (&src0_p[src0_blk][src0_index%DMA_MMAP_SIZE]));
                    float v = x - mean;
                    *(float *) (&dst_p[dst_blk][dst_index%DMA_MMAP_SIZE]) = v;
                    sum2 += (double) (v*v);
                }

                float variance = sum2/args->src0_ne[0];
                const float scale = 1.0f/sqrtf(variance + eps);

                for (int64_t i00 = 0; i00 < args->src0_ne[0]; i00++) {
                    uint64_t dst_index  = (i03*args->dst_nb[3] + i02*args->dst_nb[2] + i01*args->dst_nb[1] + i00*args->dst_nb[0]);
                    uint64_t dst_blk  = dst_index/DMA_MMAP_SIZE;
                    *(float *) (&dst_p[dst_blk][dst_index%DMA_MMAP_SIZE]) *= scale;
                }
            }
        }
    }
}

void* kernel_norm(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);

    switch (args->src0_type) {
        case GGML_TYPE_F32: 
            {
                kernel_norm_f32(args);
            } break;
        default : 
            {
                GGML_ASSERT(false);
            } break;
    }

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
                uint64_t src1_index = (i12*args->src1_nb[2] + i11*args->src1_nb[1] + i10*args->src1_nb[0]);
                uint64_t src1_blk = src1_index/DMA_MMAP_SIZE;
                const int64_t i01 = *(int32_t *)((char *)&src1_p[src1_blk][src1_index%DMA_MMAP_SIZE]);

                uint64_t src0_index = (i12*args->src0_nb[3] + i11*args->src0_nb[2] + i01*args->src0_nb[1]);
                uint64_t src0_blk = src0_index/DMA_MMAP_SIZE;
                uint64_t dst_index  = (i12*args->dst_nb[3] + i11*args->dst_nb[2] + i10*args->dst_nb[1]);
                uint64_t dst_blk  = dst_index/DMA_MMAP_SIZE;
                dequantize_row_q(
                    (const void *) (&src0_p[src0_blk][src0_index%DMA_MMAP_SIZE]),
                    (void *) (&dst_p[dst_blk][dst_index%DMA_MMAP_SIZE]), nc
                );
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
                uint64_t src1_index = (i12*args->src1_nb[2] + i11*args->src1_nb[1] + i10*args->src1_nb[0]);
                uint64_t src1_blk = src1_index/DMA_MMAP_SIZE;
                const int64_t i01 = *(int32_t *)((char *)&src1_p[src1_blk][src1_index%DMA_MMAP_SIZE]);

                // TODO: SIMD
                for (int64_t i0 = 0; i0 < nc; i0++) {
                    uint64_t src0_index = (i12*args->src0_nb[3] + i11*args->src0_nb[2] + i01*args->src0_nb[1] + i0*args->src0_nb[0]);
                    uint64_t src0_blk = src0_index/DMA_MMAP_SIZE;
                    uint64_t dst_index  = (i12*args->dst_nb[3] + i11*args->dst_nb[2] + i10*args->dst_nb[1] + i0*args->dst_nb[0]);
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
                uint64_t src1_index = (i12*args->src1_nb[2] + i11*args->src1_nb[1] + i10*args->src1_nb[0]);
                uint64_t src1_blk = src1_index/DMA_MMAP_SIZE;
                const int64_t i01 = *(int32_t *)((char *)&src1_p[src1_blk][src1_index%DMA_MMAP_SIZE]);

                // TODO: SIMD
                for (int64_t i0 = 0; i0 < nc; i0++) {
                    uint64_t src0_index = (i12*args->src0_nb[3] + i11*args->src0_nb[2] + i01*args->src0_nb[1] + i0*args->src0_nb[0]);
                    uint64_t src0_blk = src0_index/DMA_MMAP_SIZE;
                    uint64_t dst_index  = (i12*args->dst_nb[3] + i11*args->dst_nb[2] + i10*args->dst_nb[1] + i0*args->dst_nb[0]);
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

void kernel_alibi_f32(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    load_src01_dst(args);

    int n_head = ((int32_t *)dst_op_params)[1];
    float max_bias = *(float*)(&((int32_t *)dst_op_params)[2]);

    int64_t n = ne01*ne02*ne03;
    int64_t ne2_ne3 = n/ne01;

    GGML_ASSERT(nb00 == sizeof(float));
    GGML_ASSERT(n_head == ne2);

    int n_heads_log2_floor = 1 << (int) floor(log2(n_head));

    float m0 = powf(2.0f, -(max_bias) / n_heads_log2_floor);
    float m1 = powf(2.0f, -(max_bias / 2.0f) /n_heads_log2_floor);

    for (int64_t k = 0; k < ne2_ne3; k++) {
        float m_k;

        if (k < n_heads_log2_floor) {
            m_k = powf(m0, k + 1);
        } else {
            m_k = powf(m1, 2 * (k = n_heads_log2_floor) + 1);
        }

        for (int64_t i = 0; i < ne00; i++) {
            for (int64_t j = 0; j < ne01; j++) {
                    uint64_t src0_index = (k*nb02 + j*nb01 + i*nb00);
                    uint64_t dst_index  = (k*nb03 + j*nb02 + i*nb01);
                    uint32_t src0_blk = src0_index/DMA_MMAP_SIZE;
                    uint32_t dst_blk  = dst_index/DMA_MMAP_SIZE;
                    float* src0 = &src0_p[src0_blk][src0_index%DMA_MMAP_SIZE];
                    float* dst  = &dst_p [dst_blk ][dst_index%DMA_MMAP_SIZE ];
                    *dst = i * m_k + *src0;
            }
        }
    }
}

void* kernel_alibi(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);

    switch(args->src0_type) {
        case GGML_TYPE_F32:
            {
                kernel_alibi_f32(args);
            } break;
        case GGML_TYPE_F16:
            {
                // TODO: Impl kernel_alibi_f16
                GGML_ASSERT(false);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }
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

void kernel_dup_bytes(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    load_src01_dst(args);

    // at same types
    if (args->src0_type == args->dst_type && 
        ne00 == ne0 &&
        nb00 == ggml_type_size(args->src0_type) && nb0 == ggml_type_size(args->dst_type)) {
        for (int i3 = 0; i3 < ne03; i3++) {
            for (int i2 = 0; i2 < ne02; i2++) {
                for (int i1 = 0; i1 < ne01; i1++) {
                    for (int i0 = 0; i0 < ne00; i0++) {
                        uint64_t src0_index = (i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                        uint64_t dst_index  = (i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0 );
                        uint32_t src0_blk = src0_index/DMA_MMAP_SIZE;
                        uint32_t dst_blk  = dst_index/DMA_MMAP_SIZE;
                        uint8_t* src0 = &src0_p[src0_blk][src0_index%DMA_MMAP_SIZE];
                        uint8_t* dst  = &dst_p [dst_blk ][dst_index%DMA_MMAP_SIZE ];
                        *dst = *src0;
                    }
                }
            }
        }
        return;
    }

    // TODO: impl
    GGML_ASSERT(false);

}

void kernel_dup_f32(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    load_src01_dst(args);

    // at same types
    if (args->src0_type == args->dst_type && 
        ne00 == ne0 &&
        nb00 == ggml_type_size(args->src0_type) && nb0 == ggml_type_size(args->dst_type)) {
        for (int i3 = 0; i3 < ne03; i3++) {
            for (int i2 = 0; i2 < ne02; i2++) {
                for (int i1 = 0; i1 < ne01; i1++) {
                    for (int i0 = 0; i0 < ne00; i0++) {
                        uint64_t src0_index = (i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                        uint64_t dst_index  = (i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0 );
                        uint32_t src0_blk = src0_index/DMA_MMAP_SIZE;
                        uint32_t dst_blk  = dst_index/DMA_MMAP_SIZE;
                        float* src0 = &src0_p[src0_blk][src0_index%DMA_MMAP_SIZE];
                        float* dst  = &dst_p [dst_blk ][dst_index%DMA_MMAP_SIZE ];
                        *dst = *src0;
                    }
                }
            }
        }
        return;
    }

    // TODO: impl
    GGML_ASSERT(false);
}

void* kernel_dup(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);

    if (args->src0_type == args->dst_type) {
        kernel_dup_bytes(args);
        return;
    }

    switch (args->src0_type) {
        case GGML_TYPE_F32:
            {
                kernel_dup_f32(args);
            } break;
        case GGML_TYPE_F16:
            {
                GGML_ASSERT(false);
            } break;
        default:
            {
                GGML_ASSERT(false);
            } break;
    }

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