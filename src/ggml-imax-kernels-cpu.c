#include "ggml-imax-kernels.h"
#include "ggml.h"
#include "ggml-quants.h"

#include <stdio.h>
#include <signal.h>

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
    uint8_t* src0_p =  (uint8_t*)args_name->src0;      \
    uint8_t* src1_p =  (uint8_t*)args_name->src1;      \
    uint8_t* src2_p =  (uint8_t*)args_name->src2;      \
    uint8_t* dst_p  =  (uint8_t*)args_name->dst;       \
    uint8_t* wdata = (uint8_t*)args_name->wdata;       \
    uint64_t    ne00 =  args_name->src0_ne[0];\
    uint64_t    ne01 =  args_name->src0_ne[1];\
    uint64_t    ne02 =  args_name->src0_ne[2];\
    uint64_t    ne03 =  args_name->src0_ne[3];\
    uint64_t    *SRC0NE  =  args_name->src0_ne;\
    uint64_t    nb00 =  args_name->src0_nb[0];\
    uint64_t    nb01 =  args_name->src0_nb[1];\
    uint64_t    nb02 =  args_name->src0_nb[2];\
    uint64_t    nb03 =  args_name->src0_nb[3];\
    uint64_t    *SRC0NB  =  args_name->src0_nb;\
    uint64_t    ne10 =  args_name->src1_ne[0];\
    uint64_t    ne11 =  args_name->src1_ne[1];\
    uint64_t    ne12 =  args_name->src1_ne[2];\
    uint64_t    ne13 =  args_name->src1_ne[3];\
    uint64_t    *SRC1NE  =  args_name->src1_ne;\
    uint64_t    nb10 =  args_name->src1_nb[0];\
    uint64_t    nb11 =  args_name->src1_nb[1];\
    uint64_t    nb12 =  args_name->src1_nb[2];\
    uint64_t    nb13 =  args_name->src1_nb[3];\
    uint64_t    *SRC1NB  =  args_name->src1_nb;\
    uint64_t    ne0  =  args_name->dst_ne[0]; \
    uint64_t    ne1  =  args_name->dst_ne[1]; \
    uint64_t    ne2  =  args_name->dst_ne[2]; \
    uint64_t    ne3  =  args_name->dst_ne[3]; \
    uint64_t    *DSTNE =  args_name->dst_ne; \
    uint64_t    nb0  =  args_name->dst_nb[0]; \
    uint64_t    nb1  =  args_name->dst_nb[1]; \
    uint64_t    nb2  =  args_name->dst_nb[2]; \
    uint64_t    nb3  =  args_name->dst_nb[3]; \
    uint64_t    *DSTNB =  args_name->dst_nb; \
    int32_t* src0_op_params = args_name->src0_op_params;  \
    int32_t* src1_op_params = args_name->src1_op_params;  \
    int32_t* src2_op_params = args_name->src2_op_params;  \
    int32_t* dst_op_params  = args_name->dst_op_params;   \
    int32_t  nlane = args_name->nlane; \
    int32_t   lane = args_name->lane

void* kernel_upscale_f32(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);
    load_src01_dst(args);

    GGML_ASSERT(nb00 == sizeof(float));
    int scale_factor = dst_op_params[0];

    for (int i3 = lane*(ne3/nlane); i3 < (lane+1)*(ne3/nlane); i3++) {
        int64_t i03 = i3 / scale_factor;
        for (int i2 = 0; i2 < ne2; i2++) {
            int64_t i02 = i2 / scale_factor;
            for (int i1 = 0; i1 < ne1; i1++) {
                int64_t i01 = i1 / scale_factor;
                float* src0 = (float*)&src0_p[i03*nb03 + i02*nb02 + i01*nb01];
                float* dst  = (float*)&dst_p [i3*nb3  + i2*nb2  + i1*nb1];
                for (int i0 = 0; i0 < ne0; i0++) {
                    int64_t i00 = i0 / scale_factor;
                    dst[i0] = src0[i00];
                }
            }
        }
    }

    return NULL;
}

void* kernel_pad_f32(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);
    load_src01_dst(args);

    GGML_ASSERT(nb00 == sizeof(float));
    GGML_ASSERT(nb0  == sizeof(float));

    for (int i3 = lane*(ne3/nlane); i3 < (lane+1)*(ne3/nlane); i3++) {
        for (int i2 = 0; i2 < ne2; i2++) {
            for (int i1 = 0; i1 < ne1; i1++) {
                float* src0 = (float*)&src0_p[i3*nb03 + i2*nb02 + i1*nb01];
                float* dst  = (float*)&dst_p [i3*nb3 + i2*nb2 + i1*nb1];
                for (int i0 = 0; i0 < ne0; i0++) {
                    if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
                        dst[i0] = src0[i0];
                    } else {
                        dst[i0] = 0;
                    }
                }
            }
        }
    }

    return NULL;
}

void* kernel_argsort_f32_i32_asc(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);
    load_src01_dst(args);

    const int64_t nr = ne01*ne02*ne03;

    GGML_ASSERT(nb0 == sizeof(float));

    for (int64_t i = lane*(nr/nlane); i < (lane+1)*(nr/nlane); i++) {
        int32_t *dst_data = (int32_t*)((char*)dst_p + i*nb1);
        const float *src_data = (float*)((char*)src0_p + i*nb01);

        for (int64_t j = 0; j < ne0; j++) {
            dst_data[j] = j;
        }

        // TODO: change the sort argorithm
        for (int64_t j = 0; j < ne0; j++) {
            for (int64_t k = j + 1; k < ne0; k++) {
                if (src_data[dst_data[j]] > src_data[dst_data[k]]) {
                    int32_t tmp = dst_data[j];
                    dst_data[j] = dst_data[k];
                    dst_data[k] = tmp;
                }
            }
        }
    }
    
    return NULL;
}

void* kernel_argsort_f32_i32_desc(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);
    load_src01_dst(args);

    const int64_t nr = ne01*ne02*ne03;

    GGML_ASSERT(nb0 == sizeof(float));

    for (int64_t i = lane*(nr/nlane); i < (lane+1)*(nr/nlane); i++) {
        int32_t *dst_data = (int32_t*)((char*)dst_p + i*nb1);
        const float *src_data = (float*)((char*)src0_p + i*nb01);

        for (int64_t j = 0; j < ne0; j++) {
            dst_data[j] = j;
        }

        // TODO: change the sort argorithm
        for (int64_t j = 0; j < ne0; j++) {
            for (int64_t k = j + 1; k < ne0; k++) {
                if (src_data[dst_data[j]] < src_data[dst_data[k]]) {
                    int32_t tmp = dst_data[j];
                    dst_data[j] = dst_data[k];
                    dst_data[k] = tmp;
                }
            }
        }
    }

    return NULL;
}

void kernel_sum_rows_f32(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);
    load_src01_dst(args);

    for (int i3 = lane*(ne03/nlane); i3 < (lane+1)*(ne03/nlane); i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i1 = 0; i1 < ne01; i1++) {
                float row_sum = 0;
                float* dst  = (float*)&dst_p [i3*nb3  + i2*nb2  + i1*nb1];
                float* src0 = (float*)&src0_p[i3*nb03 + i2*nb02 + i1*nb01];
                for (int i0 = 0; i0 < ne00; i0++) {
                    row_sum += src0[i0];
                }
                dst[0] = row_sum;
            }
        }
    }
}

void* kernel_sum_rows(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);

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
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);
    load_src01_dst(args);

    float v = *(float*)dst_op_params;

    for (int i3 = lane*(ne03/nlane); i3 < (lane+1)*(ne03/nlane); i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i1 = 0; i1 < ne01; i1++) {
                float* src0 = (float*)&src0_p[i3*nb03 + i2*nb02 + i1*nb01];
                float* dst  = (float*)&dst_p [i3*nb3  + i2*nb2  + i1*nb1 ];
                for (int i0 = 0; i0 < ne00; i0++) {
                    dst[i0] = src0[i0] * v;
                }
            }
        }
    }
}

void* kernel_scale(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);

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
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);
    load_src01_dst(args);

    for (int i3 = lane*(ne03/nlane); i3 < (lane+1)*(ne03/nlane); i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i1 = 0; i1 < ne01; i1++) {
                float* src0 = (float*)&src0_p[i3*nb03 + i2*nb02 + i1*nb01];
                float* src1 = (float*)&src1_p[i3*nb13 + i2*nb12 + i1*nb11];
                float* dst  = (float*)&dst_p [i3*nb3  + i2*nb2  + i1*nb1];
                for (int i0 = 0; i0 < ne00; i0++) {
                    dst[i0] = src0[i0] / src1[i0];
                }
            }
        }
    }
}

void* kernel_div(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);

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
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);
    load_src01_dst(args);

    for (int i3 = lane*(ne03/nlane); i3 < (lane+1)*(ne03/nlane); i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i1 = 0; i1 < ne01; i1++) {
                float* src0 = (float*)&src0_p[i3*nb03 + i2*nb02 + i1*nb01];
                float* dst  = (float*)&dst_p [i3*nb3  + i2*nb2  + i1*nb1 ];
                for (int i0 = 0; i0 < ne00; i0++) {
                    dst[i0] = src0[i0] * src0[i0];
                }
            }
        }
    }
}

void* kernel_sqr(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);

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
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);
    load_src01_dst(args);

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale,    (float *) dst_op_params + 0, sizeof(float));
    memcpy(&max_bias, (float *) dst_op_params + 1, sizeof(float));

    const uint32_t n_head_kv   = ne02;
    const uint32_t n_head_log2 = 1u << (uint32_t) floor(log2(n_head_kv));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const int nc = ne00;
    const int nr = ne10*ne11*ne12*ne13;

    // when max_bias <= 0.0f, src2 is not used and we default it to src0 to avoid branching
    float * pos = src2_p ? (float *) src2_p : src0_p;

    for (int i1 = lane*(nr/nlane); i1 < (lane+1)*(nr/nlane); i1++) {
        float * sp = (float *)((char *) src0_p + i1*nb01);
        float * wp = (float *) wdata; 
        float * mp = src1_p ? (float *)((char *) src1_p + (i1%ne11)*nb11) : NULL;
        float * dp = (float *)((char *) dst_p + i1*nb1);
        for (int i0 = 0; i0 < nc; i0++) {
            wp[i0] *= scale * sp[i0];
            if (mp)  {
                wp[i0] += mp[i0];
            }
        }

        if (max_bias > 0.0f) {
            const uint32_t h = (i1/ne01)%ne02;

            const float slope = h < n_head_log2 ? powf(m0, h + 1) : powf(m1, 2*(h - n_head_log2) + 1);

            for (int i = 0; i < nc; i++) {
                wp[i] = wp[i] + slope*pos[i];
            }
        }

        float max = -INFINITY;
        for (int i = 0; i < nc; i++) {
            if (wp[i] > max) {
                max = wp[i];
            }
        }

        double sum = 0.0;

        for (int i = 0; i < nc; i++) {
            if (wp[i] == -INFINITY) {
                dp[i] = 0.0f;
            } else {
                const float val = (wp[i] == -INFINITY) ? 0.0 : exp(wp[i] - max);
                sum += (double)val;
                dp[i] = val;
            }
        }

        sum = 1.0/sum;
        for (int i = 0; i < nc; i++) {
            dp[i] *= sum;
        }
    }

    return NULL;
}

void kernel_rms_norm_f32(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);
    load_src01_dst(args);

    GGML_ASSERT(nb00 == sizeof(float));

    float eps = ((float*)dst_op_params)[0];

    // TODO: optimize
    for (int64_t i03 = lane*(ne03/nlane); i03 < (lane+1)*(ne03/nlane); i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = 0; i01 < ne01; i01++) {
                const float * x = (float *) ((char *) src0_p + i01*nb01 + i02*nb02 + i03*nb03);

                double sum = 0.0;
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum += (double)(x[i00] * x[i00]);
                }

                const float mean = sum/ne00;

                float * y = (float *) ((char *) dst_p + i01*nb1 + i02*nb2 + i03*nb3);

                for (int i00 = 0; i00 < ne00; i00++) {
                    y[i00] = x[i00];
                }

                const float scale = 1.0f/sqrtf(mean + eps);

                for (int i00 = 0; i00 < ne00; i00++) {
                    y[i00] *= scale;
                }
            }
        }
    }
}

void* kernel_rms_norm(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);

    switch (args->src0_type) {
        case GGML_TYPE_F32: {
            kernel_rms_norm_f32(args);
        } break;
        default: {
            GGML_ASSERT(false);
        } break;
    }

    return NULL;
}

void kernel_norm_f32(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);
    load_src01_dst(args);

    GGML_ASSERT(args->src0_type == GGML_TYPE_F32);
    GGML_ASSERT(args->dst_type == GGML_TYPE_F32);

    const float eps = 1e-6f;

    for (int64_t i03 = lane*(ne03/nlane); i03 < (lane+1)*(ne03/nlane); i03++) {
        for (int64_t i02 = 0; i02 < ne02; i02++) {
            for (int64_t i01 = 0; i01 < ne01; i01++) {
                double sum = 0.0;
                float* src0 = (float*)&src0_p[i03*nb03 + i02*nb02 + i01*nb01];
                float* dst = (float*)&dst_p[i03*nb3 + i02*nb2 + i01*nb1];
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    sum += (double)src0[i00];
                }

                float mean = sum/ne00;

                double sum2 = 0.0;
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    float x = src0[i00];
                    float v = x - mean;
                    dst[i00] = v;
                    sum2 += (double) (v*v);
                }

                float variance = sum2/ne00;
                const float scale = 1.0f/sqrtf(variance + eps);
                for (int64_t i00 = 0; i00 < ne00; i00++) {
                    dst[i00] *= scale;
                }
            }
        }
    }
}

void* kernel_norm(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);

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
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);
    return NULL;
}

static void kernel_get_rows_q(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);
    load_src01_dst(args);

    int64_t nc = ne00;
    int64_t nr = ne10 * ne11 * ne12 * ne13;

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne02 == ne11);
    GGML_ASSERT(nb00 == sizeof(ggml_fp16_t)); // TODO: row_q
    GGML_ASSERT(ne1*ne2*ne3 == nr);

    ggml_to_float_t dequantize_row_q = type_traits[args->src0_type].to_float;

    for (int64_t i12 = lane*(ne12/nlane); i12 < (lane+1)*(ne12/nlane); i12++) {
        for (int64_t i11 = 0; i11 < ne11; i11++) {
            for (int64_t i10 = 0; i10 < ne10; i10++) {
                const int64_t i01 = *(int32_t *)(&src1_p[i12*nb12 + i11*nb11 + i10*nb10]);
                dequantize_row_q(
                    (const void *) (&src0_p[i12*nb03 + i11*nb02 + i01*nb01]),
                    (void *) (&dst_p[i12*nb3 + i11*nb2 + i10*nb1]), nc
                );
            }
        }
    }
}

static void kernel_get_rows_f16(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);
    load_src01_dst(args);

    int64_t nc = ne00;
    int64_t nr = ne10 * ne11 * ne12 * ne13;

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne02 == ne11);
    GGML_ASSERT(nb00 == sizeof(ggml_fp16_t));
    GGML_ASSERT(ne1*ne2*ne3 == nr);

    for (int64_t i12 = lane*(ne12/nlane); i12 < (lane+1)*(ne12/nlane); i12++) {
        for (int64_t i11 = 0; i11 < ne11; i11++) {
            float* dst = (float*)&dst_p[i12*nb3 + i11*nb2];
            ggml_fp16_t* src0 = (ggml_fp16_t*)&src0_p[i12*nb03 + i11*nb02];
            for (int64_t i10 = 0; i10 < ne10; i10++) {
                const int64_t i01 = *(int32_t *)(&src1_p[i12*nb12 + i11*nb11 + i10*nb10]);

                // TODO: SIMD
                for (int64_t i0 = 0; i0 < nc; i0++) {
                    dst[i0] = ggml_fp16_to_fp32(src0[i0]);
                }
            }
        }
    }
}

static void kernel_get_rows_f32(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);
    load_src01_dst(args);

    int64_t nc = ne00;
    int64_t nr = ne10 * ne11 * ne12 * ne13;

    GGML_ASSERT(ne0 == ne00);
    GGML_ASSERT(ne02 == ne11);
    GGML_ASSERT(nb00 == sizeof(float));
    GGML_ASSERT(ne1*ne2*ne3 == nr);

    for (int64_t i12 = lane*(ne12/nlane); i12 < (lane+1)*(ne12/nlane); i12++) {
        for (int64_t i11 = 0; i11 < ne11; i11++) {
            float* dst = (float*)&dst_p[i12*nb3 + i11*nb2];
            float* src0 = (float*)&src0_p[i12*nb03 + i11*nb02];
            for (int64_t i10 = 0; i10 < ne10; i10++) {
                const int64_t i01 = *(int32_t *)(&src1_p[i12*nb12 + i11*nb11 + i10*nb10]);

                // TODO: SIMD
                for (int64_t i0 = 0; i0 < nc; i0++) {
                    dst[i0] = src0[i0];
                }
            }
        }
    }
}

void* kernel_get_rows(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);

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
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);
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

    for (int64_t k = lane*(ne2_ne3/nlane); k < (lane+1)*(ne2_ne3/nlane); k++) {
        float m_k;

        if (k < n_heads_log2_floor) {
            m_k = powf(m0, k + 1);
        } else {
            m_k = powf(m1, 2 * (k = n_heads_log2_floor) + 1);
        }

        for (int64_t i = 0; i < ne00; i++) {
            for (int64_t j = 0; j < ne01; j++) {
                    float* src0 = (float*)&src0_p[k*nb02 + j*nb01 + i*nb00];
                    float* dst  = (float*)&dst_p [k*nb03 + j*nb02 + i*nb01];
                    *dst = i * m_k + *src0;
            }
        }
    }
}

void* kernel_alibi(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);

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

// TODO
void* kernel_rope(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);
    return NULL;
}

// TODO
void* kernel_im2col(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);
    load_src01_dst(args);

    int s0 = dst_op_params[0];
    int s1 = dst_op_params[1];
    int p0 = dst_op_params[2];
    int p1 = dst_op_params[3];
    int d0 = dst_op_params[4];
    int d1 = dst_op_params[5];
    bool is_2D = dst_op_params[6] == 1;

    int N = is_2D ? ne13 : ne12;
    int IC = is_2D ? ne12 : ne11;
    int IH = is_2D ? ne11 : 1;
    int IW = ne10;

    int KH = is_2D ? ne01 : 1;
    int KW = ne00;

    int OH = is_2D ? ne2 : 1;
    int OW = ne1;

    int ofs0 = is_2D ? nb13 : nb12;
    int ofs1 = is_2D ? nb12 : nb11;

    // im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
    float * wdata = (float *) &dst_p;

    // lanes on N
    for (int in = lane*(N/nlane); in < (lane+1)*(N/nlane); in++) {
        for (int ioh = 0; ioh<OH; ioh++) {
            for (int64_t iow = 0; iow < OW; iow++) {
                for (int64_t iic = 0; iic < IC; iic++) {
                    
                    // micro kernel
                    float * dst_data = (float *) &wdata[(in*OH*OW + ioh*OW + iow)*(IC*KH*KW)];
                    float * src_data = (float *) &src1_p[(in*ofs0 + iic*ofs1)];

                    for (int64_t ikh = 0; ikh < KH; ikh++) {  // 1
                        for (int64_t ikw = 0; ikw < KW; ikw++) {
                            const int64_t iiw = iow*s0 + ikw*d0 - p0;
                            const int64_t iih = ioh*s1 + ikh*d1 - p1;

                            if (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
                                dst_data[iic*(KH*KW) + ikh*KW + ikw] = 0;
                            } else {
                                dst_data[iic*(KH*KW) + ikh*KW + ikw] = (src_data[iih*IW + iiw]);
                            }
                        }
                    }

                }
            }
        }
    }


    return NULL;
}

// TODO
size_t cal_nbytes(struct imax_kernel_args* args) {
    load_src01_dst(args);
    size_t nbytes;
    size_t blck_size = ggml_blck_size(args->src0_type);
    if (blck_size == 1) {
        nbytes = ggml_type_size(args->src0_type);
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            nbytes += (SRC0NE[i] - 1)*SRC0NB[i];
        }
    }
    else {
        nbytes = ne00*nb00/blck_size;
        for (int i = 1; i < GGML_MAX_DIMS; ++i) {
            nbytes += (SRC0NE[i] - 1)*SRC0NB[i];
        }
    }

    return nbytes;
}

// TODO
void* kernel_pool_1d(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);
    load_src01_dst(args);

    enum ggml_op_pool op = dst_op_params[0];
    const int k0 = dst_op_params[1];
    const int s0 = dst_op_params[2];
    const int p0 = dst_op_params[3];
    
    // ggml_compute_forward_pool_1d_sk_p0(params, op, k0, dst);
    const char * cdata = (const char *) src0_p;
    const char * data_end = cdata + cal_nbytes(args);
    float * drow = (float *) dst_p;
    int64_t total_elements = (data_end - cdata) / nb01;

    const int64_t rs = ne00;

    while (cdata < data_end) {
        const int lane_size = rs / nlane;

        const float *srow = (const float *) (cdata + lane * lane_size * sizeof(float) );
        float *drow_lane = drow + lane * lane_size;

        int j = 0; // j for loop drow

        for (int64_t i = 0; i < lane_size; ++i) {
            switch (op) {
                case GGML_OP_POOL_AVG:   drow_lane[i] = 0;        break;
                case GGML_OP_POOL_MAX:   drow_lane[i] = -3e33; break;
                case GGML_OP_POOL_COUNT: GGML_ASSERT(false); break;
            }
            for (int ki = 0; ki < k0; ++ki) {
                switch (op) {
                    case GGML_OP_POOL_AVG:                          drow_lane[i] += srow[j]; break;
                    case GGML_OP_POOL_MAX:   if (srow[j] > drow_lane[i]) drow_lane[i]  = srow[j]; break;
                    case GGML_OP_POOL_COUNT:                        GGML_ASSERT(false); break;
                }
                ++j;
            }
            switch (op) {
                case GGML_OP_POOL_AVG:         drow_lane[i] /= k0; break;
                case GGML_OP_POOL_MAX:                       break;
                case GGML_OP_POOL_COUNT: GGML_ASSERT(false); break;
            }
        }

        cdata += nb01;
        drow  += rs;
    }


    return NULL;
}

// TODO
void* kernel_pool_2d(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);
    load_src01_dst(args);

    enum ggml_op_pool op = dst_op_params[0];
    const int k0 = dst_op_params[1];
    const int k1 = dst_op_params[2];
    const int s0 = dst_op_params[3];
    const int s1 = dst_op_params[4];
    const int p0 = dst_op_params[5];
    const int p1 = dst_op_params[6];

    const char * cdata = (const char *) src0_p;
    const char * const data_end = cdata + cal_nbytes(args);

    const int64_t px = ne0;
    const int64_t py = ne1;
    const int64_t pa = px * py;

    float * dplane = dst_p;

    const int ka = k0 * k1;
    const int offset0 = -p0;
    const int offset1 = -p1;

    while (cdata < data_end) {
        for (int oy = 0; oy < py; ++oy) {
            float * const drow = dplane + oy * px;
            for (int ox = 0; ox < px; ++ox) {
                float * const out =  drow + ox;
                switch (op) {
                    case GGML_OP_POOL_AVG:     *out = 0;        break;
                    case GGML_OP_POOL_MAX:     *out = -3e33; break;
                    case GGML_OP_POOL_COUNT: GGML_ASSERT(false); break;
                }

                const int ix = offset0 + ox * s0;
                const int iy = offset1 + oy * s1;

                for (int ky = 0; ky < k1; ++ky) {
                    if (iy + ky < 0 || iy + ky >= ne01) continue;
                    const float * const srow = (const float *)(cdata + nb01 * (iy + ky));
                    for (int kx = 0; kx < k0; ++kx) {
                        int j = ix + kx;
                        if (j < 0 || j >= ne00) continue;
                        switch (op) {
                            case GGML_OP_POOL_AVG:                     *out += srow[j]; break;
                            case GGML_OP_POOL_MAX: if (srow[j] > *out) *out  = srow[j]; break;
                            case GGML_OP_POOL_COUNT:                GGML_ASSERT(false); break;
                        }
                    }
                }
                switch (op) {
                    case GGML_OP_POOL_AVG:           *out /= ka; break;
                    case GGML_OP_POOL_MAX:                       break;
                    case GGML_OP_POOL_COUNT: GGML_ASSERT(false); break;
                }
            }
        }

        cdata  += nb02;
        dplane += pa;
    }


    return NULL;
}

// TODO
void* kernel_leaky_relu(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);
    load_src01_dst(args);

    const int n = ne01*ne02*ne03; //ggml_nrows
    const int nc = ne00; // ncols

    float negative_slope;
    memcpy(&negative_slope, dst_op_params, sizeof(float));

    for (int i = 0; i < n; i++) {
        ggml_vec_leaky_relu_f32(nc,
                (float *) ((char *) dst_p  + i*(nb1)),
                (float *) ((char *) src0_p + i*(nb01)), negative_slope);
    }

    return NULL;
}

#define IS_CONTIGUOUS(nb0, nb1, nb2, nb3, ne0, ne1, ne2, ne3, type) \
    nb0 == ggml_type_size(type) && \
    nb1 == nb0*ne0/ggml_blck_size(type) && \
    nb2 == nb1*ne1 && \
    nb3 == nb2*ne2

void kernel_dup_bytes(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);
    load_src01_dst(args);

    // at same size
    if (ne00 == ne0 && args->src0_type == args->dst_type) {
        for (int64_t i3 = lane*(ne03/nlane); i3 < (lane+1)*(ne03/nlane); i3++) {
            for (int i2 = 0; i2 < ne02; i2++) {
                for (int i1 = 0; i1 < ne01; i1++) {
                    uint8_t* src0 = &src0_p[i3*nb03 + i2*nb02 + i1*nb01];
                    uint8_t* dst  = &dst_p [i3*nb3  + i2*nb2  + i1*nb1 ];
                    for (int i0 = 0; i0 < ne00; i0++) {
                        dst[i0] = src0[i0];
                    }
                }
            }
        }
        return;
    }

    int64_t i10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    for (int64_t i3 = lane*(ne03/nlane); i3 < (lane+1)*(ne03/nlane); i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            while (i10 >= ne0) {
                i10 -= ne0;
                i11++;
                while (i11 >= ne1) {
                    i11 -= ne1;
                    i12++;
                    while (i12 >= ne2) {
                        i12 -= ne2;
                        i13++;
                    }
                }
            }
            for (int i1 = 0; i1 < ne01; i1++) {
                uint8_t* src0 = &src0_p[i3*nb03 + i2*nb02 + i1*nb01];
                uint8_t* dst  = &dst_p [i13*nb3  + i12*nb2  + i11*nb1 ];
                for (int i0 = 0; i0 < ne00; i0++) {
                    memcpy(&dst[i10], &src0[i0], nb00);
                    if (++i10 >= ne0) {
                        i10 = 0;
                        i11++;
                        if (i11 >= ne1) {
                            i11 = 0;
                            i12++;
                            if (i12 >= ne2) {
                                i12 = 0;
                                i13++;
                            }
                        }
                    }
                }
            }
            i10 += ne00 * ne01;
            while (i10 >= ne0) {
                i10 -= ne0;
                i11++;
                while (i11 >= ne1) {
                    i11 -= ne1;
                    i12++;
                    while (i12 >= ne2) {
                        i12 -= ne2;
                        i13++;
                    }
                }
            }
        }
    }
}

void kernel_dup_f32(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);
    load_src01_dst(args);

    // at same types
    if (args->src0_type == args->dst_type && ne00 == ne0) {
        for (int64_t i3 = lane*(ne03/nlane); i3 < (lane+1)*(ne03/nlane); i3++) {
            for (int i2 = 0; i2 < ne02; i2++) {
                for (int i1 = 0; i1 < ne01; i1++) {
                    float* src0 = &src0_p[i3*nb03 + i2*nb02 + i1*nb01];
                    float* dst  = &dst_p [i3*nb3  + i2*nb2  + i1*nb1 ];
                    for (int i0 = 0; i0 < ne00; i0++) {
                        dst[i0] = src0[i0];
                    }
                }
            }
        }
        return;
    }


    if (type_traits[args->dst_type].from_float) {
        ggml_from_float_t const quantize_row_q = type_traits[args->dst_type].from_float;

        for (int64_t i3 = lane*(ne03/nlane); i3 < (lane+1)*(ne03/nlane); i3++) {
            for (int i2 = 0; i2 < ne02; i2++) {
                for (int i1 = 0; i1 < ne01; i1++) {
                    float* src0 = (float*)&src0_p[i3*nb03 + i2*nb02 + i1*nb01];
                    float* dst  = (float*)&dst_p [i3*nb3  + i2*nb2  + i1*nb1 ];
                    quantize_row_q(src0, dst, ne00);
                }
            }
        }

        return;
    }

    int64_t i10 = 0;
    int64_t i11 = 0;
    int64_t i12 = 0;
    int64_t i13 = 0;

    for (int64_t i3 = lane*(ne03/nlane); i3 < (lane+1)*(ne03/nlane); i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            while (i10 >= ne0) {
                i10 -= ne0;
                i11++;
                while (i11 >= ne1) {
                    i11 -= ne1;
                    i12++;
                    while (i12 >= ne2) {
                        i12 -= ne2;
                        i13++;
                    }
                }
            }
            for (int i1 = 0; i1 < ne01; i1++) {
                uint8_t* src0 = &src0_p[i3*nb03 + i2*nb02 + i1*nb01];
                uint8_t* dst  = &dst_p [i13*nb3  + i12*nb2  + i11*nb1 ];
                for (int i0 = 0; i0 < ne00; i0++) {
                    memcpy(&dst[i10], &src0[i0], nb00);
                    if (++i10 >= ne0) {
                        i10 = 0;
                        i11++;
                        if (i11 >= ne1) {
                            i11 = 0;
                            i12++;
                            if (i12 >= ne2) {
                                i12 = 0;
                                i13++;
                            }
                        }
                    }
                }
            }
            i10 += ne00 * ne01;
            while (i10 >= ne0) {
                i10 -= ne0;
                i11++;
                while (i11 >= ne1) {
                    i11 -= ne1;
                    i12++;
                    while (i12 >= ne2) {
                        i12 -= ne2;
                        i13++;
                    }
                }
            }
        }
    }
}

void* kernel_dup(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);

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
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);
    kernel_dup(args);
    return NULL;
}

void* kernel_contiguous(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);
    kernel_dup(args);
    return NULL;
}

// TODO TRANSPOSE_2D
void* kernel_transpose(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);
    return NULL;
}

// TODO
void* kernel_diag_mask_inf(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);
    return NULL;
}

// TODO
void* kernel_unary(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("name: %s, lane: %d", __func__, args->lane);
    return NULL;
}