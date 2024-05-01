#include "ggml-imax-kernels.h"
#include "ggml.h"
#include "ggml-quants.h"
#include "emax7.h"
#include "emax7lib.h"

#include <stdio.h>

#define load_src01_dst(args_name) \
    Uchar* src0_p = (Uchar*)args_name->src0;   \
    Uchar* src1_p = (Uchar*)args_name->src1;   \
    Uchar* dst_p  = (Uchar*)args_name->dst;    \
    Uchar* wdata = (Uchar*)args_name->wdata;       \
    Ull    ne00 =  args_name->src0_ne[0];\
    Ull    ne01 =  args_name->src0_ne[1];\
    Ull    ne02 =  args_name->src0_ne[2];\
    Ull    ne03 =  args_name->src0_ne[3];\
    Ull    nb00 =  args_name->src0_nb[0];\
    Ull    nb01 =  args_name->src0_nb[1];\
    Ull    nb02 =  args_name->src0_nb[2];\
    Ull    nb03 =  args_name->src0_nb[3];\
    Ull    ne10 =  args_name->src1_ne[0];\
    Ull    ne11 =  args_name->src1_ne[1];\
    Ull    ne12 =  args_name->src1_ne[2];\
    Ull    ne13 =  args_name->src1_ne[3];\
    Ull    nb10 =  args_name->src1_nb[0];\
    Ull    nb11 =  args_name->src1_nb[1];\
    Ull    nb12 =  args_name->src1_nb[2];\
    Ull    nb13 =  args_name->src1_nb[3];\
    Ull    ne0  =  args_name->dst_ne[0]; \
    Ull    ne1  =  args_name->dst_ne[1]; \
    Ull    ne2  =  args_name->dst_ne[2]; \
    Ull    ne3  =  args_name->dst_ne[3]; \
    Ull    nb0  =  args_name->dst_nb[0]; \
    Ull    nb1  =  args_name->dst_nb[1]; \
    Ull    nb2  =  args_name->dst_nb[2]; \
    Ull    nb3  =  args_name->dst_nb[3]; \
    Uint* src0_op_params = args_name->src0_op_params;\
    Uint* src1_op_params = args_name->src1_op_params;\
    Uint* src2_op_params = args_name->src2_op_params;\
    Uint* dst_op_params  = args_name->dst_op_params;

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define QK 32

void* kernel_add(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    load_src01_dst(args);

    for (int i3 = 0; i3 < ne03; i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i1 = 0; i1 < ne01; i1++) {
                for (int i0 = 0; i0 < ne00; i0++) {
                    Uchar* src0 = &src0_p[i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00];
                    Uchar* src1 = &src1_p[i3*nb13 + i2*nb12 + i1*nb11 + i0*nb10];
                    Uchar* dst  = &dst_p [i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0 ];
                    *dst = *src0 + *src1;
                }
            }
        }
    }

    return NULL;
}

void* kernel_mul(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    load_src01_dst(args);

    for (int i3 = 0; i3 < ne03; i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i1 = 0; i1 < ne01; i1++) {
                for (int i0 = 0; i0 < ne00; i0++) {
                    Uchar* src0 = &src0_p[i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00];
                    Uchar* src1 = &src1_p[i3*nb13 + i2*nb12 + i1*nb11 + i0*nb10];
                    Uchar* dst  = &dst_p [i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0 ];
                    *dst = *src0 * *src1;
                }
            }
        }
    }

    return NULL;
}

void* kernel_mul_mm_f32_f32(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    load_src01_dst(args);

    for (int i3 = 0; i3 < ne03; i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i00 = 0; i00 < ne00; i00++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    for (int i11 = 0; i11 < ne11; i11++) {
                        float* src0 = &src0_p[i3*nb03 + i2*nb02 + i01*nb01 + i00*nb00];
                        float* src1 = &src1_p[i3*nb13 + i2*nb12 + i01*nb11 + i11*nb10];
                        float* dst  = &dst_p [i3*nb3  + i2*nb2  + i00*nb1  + i11*nb0 ];
                        *dst += *src0 * *src1;
                    }
                }
            }
        }
    }

    return NULL;
}

// Under construction (not working)
void* kernel_mul_mm_f16_f32(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    load_src01_dst(args);

    for (int i3 = 0; i3 < ne03; i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i00 = 0; i00 < ne00; i00++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    for (int i11 = 0; i11 < ne11; i11++) {
                        Uchar* src0 = &src0_p[i3*nb03 + i2*nb02 + i01*nb01 + i00*nb00];
                        Uchar* src1 = &src1_p[i3*nb13 + i2*nb12 + i01*nb11 + i11*nb10];
                        Uchar* dst  = &dst_p [i3*nb3  + i2*nb2  + i00*nb1  + i11*nb0 ];
                        *dst += *src0 * *src1;
                    }
                }
            }
        }
    }

    return NULL;
}

void *kernel_mul_mm_q4_0_f32(struct imax_kernel_args *args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    load_src01_dst(args);

    const int ne = ne0 * ne1 * ne2 * ne3;
    const int nr = ne01 * ne02 * ne03;
    //const int ith = params->ith;
    //const int nth = params->nth;
    const int ith = 0;
    const int nth = 0;
    const int nb = ne00 / QK;
    const size_t bs = sizeof(float) + QK / 2; /* 20B */

//#if !defined(EMAX7)
#if 1
    // nb01 >= nb00 - src0 is not transposed. compute by src0 rows
    // rows per thread
    const int dr = (nr + nth - 1) / nth;
    // row range for this thread
    const int ir0 = dr * ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) { /* 5120, 20480, 50288 */
        const int i03 = ir / (ne02 * ne01);
        const int i02 = (ir - i03 * ne02 * ne01) / ne01;
        const int i01 = (ir - i03 * ne02 * ne01 - i02 * ne01);
        const int i13 = i03;
        const int i12 = i02;
        const int i0 = i01;
        const int i2 = i02;
        const int i3 = i03;

        void *src0_row = (void *)((char *)src0_p + (i01 * nb01 + i02 * nb02 + i03 * nb03));
        char *src1_col = ((char *)wdata + ((0 + i12 * ne11 + i13 * ne12 * ne11) * ne00 * ggml_type_size(GGML_TYPE_Q4_0)) / ggml_blck_size(GGML_TYPE_Q4_0));
        float *dst_col = (float *)((char *)dst_p + (i0 * nb0 + 0 * nb1 + i2 * nb2 + i3 * nb3));

        for (int ic = 0; ic < ne11; ++ic) { /* 1, 5 */
            const uint8_t *restrict pd0 = ((const uint8_t *)src0_row + 0 * bs);
            const uint8_t *restrict pd1 = ((const uint8_t *)(src1_col + (ic * ne00 * ggml_type_size(GGML_TYPE_Q4_0)) / ggml_blck_size(GGML_TYPE_Q4_0)) + 0 * bs);
            const uint8_t *restrict pb0 = ((const uint8_t *)src0_row + 0 * bs + sizeof(float));
            const uint8_t *restrict pb1 = ((const uint8_t *)(src1_col + (ic * ne00 * ggml_type_size(GGML_TYPE_Q4_0)) / ggml_blck_size(GGML_TYPE_Q4_0)) + 0 * bs + sizeof(float));
            float sumf = 0.0;

            for (int i = 0; i < nb; i++) { /* 320, 1280 */
                const float d0 = *(const float *)(pd0 + i * bs);
                const float d1 = *(const float *)(pd1 + i * bs);
                const uint8_t *restrict p0 = pb0 + i * bs;
                const uint8_t *restrict p1 = pb1 + i * bs;

                for (int j = 0; j < QK / 2; j++) { /* 16 */
                    const uint8_t v0 = p0[j];
                    const uint8_t v1 = p1[j];
                    const float f0 = d0 * ((int8_t)(v0 & 0xf) - 8);
                    const float f1 = d0 * ((int8_t)(v0 >> 4) - 8);
                    const float f2 = d1 * ((int8_t)(v1 & 0xf) - 8);
                    const float f3 = d1 * ((int8_t)(v1 >> 4) - 8);
                    sumf += f0 * f2 + f1 * f3;
                }
            }
            dst_col[ic * ne0] = sumf;
        }
    }
#else
    if (ith != 0 || nth != 1 || src0->n_dims > 2 || src1->n_dims > 2) {
        printf("imax_ggml_compute_forward_mul_mat_q4_0_f32: ith=%d(!=0), nth=%d(!=1), src0->n_dims=%d(>2), src1->n_dims=%d(>2)\n", ith, nth, src0->n_dims, src1->n_dims);
        exit(1);
    }
    if (ne02 != 1 || ne03 != 1 || ne12 != 1 || ne13 != 1 || ne2 != 1 || ne3 != 1) {
        printf("imax_ggml_compute_forward_mul_mat_q4_0_f32: ne02=%d(!=1), ne03=%d(!=1), ne12=%d(!=1), ne13=%d(!=1), ne2=%d(!=1), ne3=%d(!=1)\n", ne02, ne03, ne12, ne13, ne2, ne3);
        exit(1);
    }
    /* output: Hi there, how are you doing? I am Open Assistant and here to help... */
    /*                    <|BEGIN>  50278  12092  2  0  50281  12764  627  13  849  403  368  2509  32 ... <END|> */
    /* output: Hi there!  <|BEGIN>  50278  12092  2  0  50281  12764  627   2                              <END|> */
    /* output: Hey there! <|BEGIN>  50278  12092  2  0  50281   8262  627   2                              <END|> */
    Ull CHIP, rofs, cofs, iofs, oofs;
    Ull LOOP1, LOOP0;
    Ull INIT1, INIT0;
    Ull AR[64][4];    /* output of EX     in each unit */
    Ull BR[64][4][4]; /* output registers in each unit */
    Ull r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
    Ull r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30, r31;
    Ull cc0, cc1, cc2, cc3, ex0, ex1;
    Ull NRNB01d4 = nr * nb01 / sizeof(Uint);         /* 50288*3200B/4:40230400 (160MB) max total words of sd */
    Ull NBNB00 = (Ull)(nb * nb00);                   /* 160 * 20B                                            */
    Ull NBNB00d4 = NBNB00 / sizeof(int);             /* 160 * 20B /4  :     800  (3KB) max LMM words of sd   */
    Ull NBNB00xNE11d4 = NBNB00 * ne11 / sizeof(int); /* 160 * 20B /4*5:    4000 (16KB) max LMM words of wd   */
    Ull MNBNB00_MNE0 = (0LL - NBNB00) << 32 | ((0LL - (Ull)ne0 * sizeof(int)) & 0xffffffffLL);
    Ull NBNB00_NE0 = (NBNB00) << 32 | (((Ull)ne0 * sizeof(int)) & 0xffffffffLL);
    Ull MBS = (0LL - (Ull)bs) << 32 | ((0LL - (Ull)0LL) & 0xffffffffLL);
    Ull BS = ((Ull)bs) << 32 | (((Ull)0LL) & 0xffffffffLL);
    Ull NE01NE11 = ne01 * ne11; /* 50288 * 5    :  251440 (1MB)   max LMM words of dst_col */
    Ull Force = 1;              /* force wdat load to LMM */

    static int nrnb01d4;
    static int nbnb00d4;
    static int nbnb00xne11d4;
    static int ne01ne11;
    static int updated;
    static int check_lmm;
    static int check_lmm_ovf;
    static int check_lmm_fit;

    /* check LMM_SIZE */
    if (NBNB00d4 > LMM_SIZE / sizeof(Uint) || NBNB00xNE11d4 > LMM_SIZE / sizeof(Uint) || NE01NE11 > LMM_SIZE / sizeof(Uint)) {
        check_lmm_ovf++;
        for (int ir = 0; ir < nr; ir++) {                                                               /* 5120, 20480, 50288■ */
            const uint8_t *restrict sd = (const uint8_t *)((char *)src0_p + (ir * nb01));           /* nb01:     3200B/ir★, 12800B/ir● */
            float *dst_col = (float *)((char *)dst_p + (ir * nb0));                                 /* nb0:      4B                      */
            for (int ic = 0; ic < ne11; ic++) {                                                         /* 1,5,8,9 */
                const uint8_t *restrict wd = (const uint8_t *)((char *)wdata + ic * nb * nb00); /* nb*nb00:3200B(x9=28800B))★, 12800B(x9=115200B)● */
                float sumf = 0.0;
                for (int i = 0; i < nb; i++) {                                                   /* 160, 640 */
                    const float *sdf32 = (const float *)(sd + i * bs);                           /* min:160*20=3200B★, max:640*20=12800B● */
                    const float *wdf32 = (const float *)(wd + i * bs);                           /* min:160*20=3200B★, max:640*20=12800B● */
                    const uint8_t *restrict s0 = (const uint8_t *)(sd + i * bs + sizeof(float)); /* min:160*20=3200B★, max:640*20=12800B● */
                    const uint8_t *restrict w0 = (const uint8_t *)(wd + i * bs + sizeof(float)); /* min:160*20=3200B★, max:640*20=12800B● */
                    for (int j = 0; j < QK / 2; j++) {                                           /* 16 */
                        const float slo = *sdf32 * ((int8_t)(s0[j] & 0xf) - 8);
                        const float shi = *sdf32 * ((int8_t)(s0[j] >> 4) - 8);
                        const float wlo = *wdf32 * ((int8_t)(w0[j] & 0xf) - 8);
                        const float whi = *wdf32 * ((int8_t)(w0[j] >> 4) - 8);
                        sumf += slo * wlo + shi * whi;
                    }
                }
                dst_col[ic * ne0] = sumf;
            }
        }
    } else { /* IMAX */
#undef NCHIP
#define NCHIP 1
        check_lmm_fit++;
        int tmp;
        //monitor_time_start(THREAD, IMAX_CPYIN);
        //xmax_cpyin(3, i_m0A[LANE], &tmp, src0->data, 1, 1, 1, NRNB01d4, 1);
        //xmax_cpyin(3, i_m0B[LANE], &tmp, params->wdata, 1, 1, 1, NBNB00xNE11d4, 1);
        //xmax_bzero(i_m0C[LANE], NE01NE11);
        //monitor_time_end(THREAD, IMAX_CPYIN);

        for (int ir = 0; ir < nr; ir++) { /* 5120, 20480, 50288■ */
            //const uint8_t *restrict sd = (const uint8_t *)((char *)i_m0A[LANE] + (ir * nb01));
            //const uint8_t *restrict wd = (const uint8_t *)((char *)i_m0B[LANE]);
            // Already Copied to Buffers
            const uint8_t *restrict sd = (const uint8_t *)((char *)src0_p + (ir * nb01));
            const uint8_t *restrict wd = (const uint8_t *)((char *)wdata);
            const uint8_t *restrict sdp[4];
            const uint8_t *restrict wdp[4];
            sdp[0] = sd + sizeof(float) * 1;
            wdp[0] = wd + sizeof(float) * 1;
            sdp[1] = sd + sizeof(float) * 2;
            wdp[1] = wd + sizeof(float) * 2;
            sdp[2] = sd + sizeof(float) * 3;
            wdp[2] = wd + sizeof(float) * 3;
            sdp[3] = sd + sizeof(float) * 4;
            wdp[3] = wd + sizeof(float) * 4;
            float *dst_col = (float *)((char *)dst_p + (ir * nb0)); /* nb0: 4B */

#define mul_mat_cores(r, c, d0, d1, d2, d3)                                                                                       \
    mop(OP_LDWR, 1, &BR[r][0][1], sd, cofs, MSK_W1, (Ull)sd, NBNB00d4, 0, 0, (Ull)NULL, NBNB00d4);                                \
    mop(OP_LDWR, 1, &BR[r][2][1], sdp[c], cofs, MSK_W1, (Ull)sd, NBNB00d4, 0, 0, (Ull)NULL, NBNB00d4);                            \
    exe(OP_FML3, &d0, BR[r][0][1], EXP_H1010, BR[r][2][1], EXP_H1010, 0x0003000200010000LL, EXP_B5410, OP_NOP, 0LL, OP_NOP, 0LL); \
    exe(OP_FML3, &d1, BR[r][0][1], EXP_H1010, BR[r][2][1], EXP_H1010, 0x0003000200010000LL, EXP_B7632, OP_NOP, 0LL, OP_NOP, 0LL); \
    exe(OP_FML3, &d2, BR[r][0][1], EXP_H1010, BR[r][2][1], EXP_H1010, 0x0007000600050004LL, EXP_B5410, OP_NOP, 0LL, OP_NOP, 0LL); \
    exe(OP_FML3, &d3, BR[r][0][1], EXP_H1010, BR[r][2][1], EXP_H1010, 0x0007000600050004LL, EXP_B7632, OP_NOP, 0LL, OP_NOP, 0LL)

#define mul_mat_corew(r, c, d0, d1, d2, d3, Force)                                                                                \
    mop(OP_LDWR, 1, &BR[r][0][1], wd, iofs, MSK_W1, (Ull)wd, NBNB00xNE11d4, 0, Force, (Ull)NULL, NBNB00xNE11d4);                  \
    mop(OP_LDWR, 1, &BR[r][2][1], wdp[c], iofs, MSK_W1, (Ull)wd, NBNB00xNE11d4, 0, Force, (Ull)NULL, NBNB00xNE11d4);              \
    exe(OP_FML3, &d0, BR[r][0][1], EXP_H1010, BR[r][2][1], EXP_H1010, 0x0003000200010000LL, EXP_B5410, OP_NOP, 0LL, OP_NOP, 0LL); \
    exe(OP_FML3, &d1, BR[r][0][1], EXP_H1010, BR[r][2][1], EXP_H1010, 0x0003000200010000LL, EXP_B7632, OP_NOP, 0LL, OP_NOP, 0LL); \
    exe(OP_FML3, &d2, BR[r][0][1], EXP_H1010, BR[r][2][1], EXP_H1010, 0x0007000600050004LL, EXP_B5410, OP_NOP, 0LL, OP_NOP, 0LL); \
    exe(OP_FML3, &d3, BR[r][0][1], EXP_H1010, BR[r][2][1], EXP_H1010, 0x0007000600050004LL, EXP_B7632, OP_NOP, 0LL, OP_NOP, 0LL)

// EMAX5A begin mul_mat_q4_0_f32 mapdist=0
            /*3*/ for (CHIP = 0; CHIP < NCHIP; CHIP++) {                                                                                  /* will be parallelized by multi-chip (M/#chip) */
                /*2*/ for (INIT1 = 1, LOOP1 = ne11, rofs = MNBNB00_MNE0; LOOP1--; INIT1 = 0) { /* stage#0 */                              /* mapped to FOR() on BR[63][1][0] */
                    /*1*/ for (INIT0 = 1, LOOP0 = nb, cofs = MBS; LOOP0--; INIT0 = 0) { /* stage#0 */                                     /* mapped to FOR() on BR[63][0][0] */
                        exe(OP_ADD, &cofs, INIT0 ? cofs : cofs, EXP_H3210, BS, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);      /* stage#0 */
                        exe(OP_ADD, &rofs, rofs, EXP_H3210, INIT0 ? NBNB00_NE0 : 0, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#0 */
                        exe(OP_ADD, &iofs, rofs, EXP_H3210, cofs, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xffffffff00000000LL, OP_NOP, 0LL);  /* stage#1 */
                        exe(OP_ADD, &oofs, rofs, EXP_H3210, cofs, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0x00000000ffffffffLL, OP_NOP, 0LL);  /* stage#1 */

                        mul_mat_cores(2, 0, r16, r17, r18, r19);                                                     /* stage #2-#3  */
                        mul_mat_corew(4, 0, r20, r21, r22, r23, Force);                                              /* stage #4-#5 */
                        exe(OP_FML, &r24, r16, EXP_H3210, r20, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#6 */
                        exe(OP_FML, &r25, r17, EXP_H3210, r21, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#6 */
                        exe(OP_FML, &r26, r18, EXP_H3210, r22, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#6 */
                        exe(OP_FML, &r27, r19, EXP_H3210, r23, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#6 */

                        mul_mat_cores(7, 1, r16, r17, r18, r19);                                                     /* stage #7-#8 */
                        mul_mat_corew(9, 1, r20, r21, r22, r23, Force);                                              /* stage #9-#10 */
                        exe(OP_FMA, &r28, r24, EXP_H3210, r16, EXP_H3210, r20, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
                        exe(OP_FMA, &r29, r25, EXP_H3210, r17, EXP_H3210, r21, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
                        exe(OP_FMA, &r30, r26, EXP_H3210, r18, EXP_H3210, r22, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */
                        exe(OP_FMA, &r31, r27, EXP_H3210, r19, EXP_H3210, r23, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#11 */

                        mul_mat_cores(12, 2, r16, r17, r18, r19);                                                    /* stage #12-#13 */
                        mul_mat_corew(14, 2, r20, r21, r22, r23, Force);                                             /* stage #14-#15 */
                        exe(OP_FMA, &r24, r28, EXP_H3210, r16, EXP_H3210, r20, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#16 */
                        exe(OP_FMA, &r25, r29, EXP_H3210, r17, EXP_H3210, r21, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#16 */
                        exe(OP_FMA, &r26, r30, EXP_H3210, r18, EXP_H3210, r22, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#16 */
                        exe(OP_FMA, &r27, r31, EXP_H3210, r19, EXP_H3210, r23, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#16 */

                        mul_mat_cores(17, 3, r16, r17, r18, r19);                                                    /* stage #17-#18 */
                        mul_mat_corew(19, 3, r20, r21, r22, r23, Force);                                             /* stage #19-#20 */
                        exe(OP_FMA, &r28, r24, EXP_H3210, r16, EXP_H3210, r20, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#21 */
                        exe(OP_FMA, &r29, r25, EXP_H3210, r17, EXP_H3210, r21, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#21 */
                        exe(OP_FMA, &r30, r26, EXP_H3210, r18, EXP_H3210, r22, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#21 */
                        exe(OP_FMA, &r31, r27, EXP_H3210, r19, EXP_H3210, r23, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#21 */

                        /* FAD tree */
                        exe(OP_FAD, &r3, r28, EXP_H3210, r29, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#22 */
                        exe(OP_FAD, &r4, r30, EXP_H3210, r31, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#22 */

                        exe(OP_FAD, &r2, r3, EXP_H3210, r4, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#23 */

                        exe(OP_FAD, &r1, r2, EXP_H3232, r2, EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#24 */

                        exe(OP_NOP, &AR[25][0], 0LL, EXP_H3210, 0LL, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL); /* stage#25 (dummy to set target location) */
                        //mop(OP_LDWR, 1, &r0, dst_col, oofs, MSK_W0, i_m0C[LANE], NE01NE11, 0, Force, (Ull)NULL, NE01NE11); /* stage#25 */
                        mop(OP_LDWR, 1, &r0, dst_col, oofs, MSK_W0, dst_p, NE01NE11, 0, Force, (Ull)NULL, NE01NE11); /* stage#25 */
                        exe(OP_FAD, &r0, INIT0 ? r0 : r0, EXP_H3210, r1, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                        //mop(OP_STWR, 1, &r0, oofs, dst_col, MSK_D0, i_m0C[LANE], NE01NE11, 0, Force, (Ull)NULL, NE01NE11);
                        mop(OP_STWR, 1, &r0, oofs, dst_col, MSK_D0, dst_p, NE01NE11, 0, Force, (Ull)NULL, NE01NE11);
                    }
                }
            }
// EMAX5A end
            if (Force)
                Force = 0; /* reset wdat load to LMM */
        }
// EMAX5A drain_dirty_lmm
        //monitor_time_start(THREAD, IMAX_CPYOUT);
        //xmax_cpyout(2, dst->data, 1, 1, i_m0C[LANE], NE01NE11, 1, 1);
        //monitor_time_end(THREAD, IMAX_CPYOUT);
    }
#endif

    return NULL;
}

void* kernel_mul_mm_q4_1_f32(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    load_src01_dst(args);

    for (int i3 = 0; i3 < ne03; i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i00 = 0; i00 < ne00; i00++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    for (int i11 = 0; i11 < ne11; i11++) {
                        Uchar* src0 = &src0_p[i3*nb03 + i2*nb02 + i01*nb01 + i00*nb00];
                        Uchar* src1 = &src1_p[i3*nb13 + i2*nb12 + i01*nb11 + i11*nb10];
                        Uchar* dst  = &dst_p [i3*nb3  + i2*nb2  + i00*nb1  + i11*nb0 ];
                        *dst += *src0 * *src1;
                    }
                }
            }
        }
    }

    return NULL;
}

void* kernel_mul_mm_q5_0_f32(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    load_src01_dst(args);

    for (int i3 = 0; i3 < ne03; i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i00 = 0; i00 < ne00; i00++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    for (int i11 = 0; i11 < ne11; i11++) {
                        Uchar* src0 = &src0_p[i3*nb03 + i2*nb02 + i01*nb01 + i00*nb00];
                        Uchar* src1 = &src1_p[i3*nb13 + i2*nb12 + i01*nb11 + i11*nb10];
                        Uchar* dst  = &dst_p [i3*nb3  + i2*nb2  + i00*nb1  + i11*nb0 ];
                        *dst += *src0 * *src1;
                    }
                }
            }
        }
    }

    return NULL;
}

void* kernel_mul_mm_q5_1_f32(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    load_src01_dst(args);

    for (int i3 = 0; i3 < ne03; i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i00 = 0; i00 < ne00; i00++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    for (int i11 = 0; i11 < ne11; i11++) {
                        Uchar* src0 = &src0_p[i3*nb03 + i2*nb02 + i01*nb01 + i00*nb00];
                        Uchar* src1 = &src1_p[i3*nb13 + i2*nb12 + i01*nb11 + i11*nb10];
                        Uchar* dst  = &dst_p [i3*nb3  + i2*nb2  + i00*nb1  + i11*nb0 ];
                        *dst += *src0 * *src1;
                    }
                }
            }
        }
    }

    return NULL;
}

void* kernel_mul_mm_q8_0_f32(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    load_src01_dst(args);

    for (int i3 = 0; i3 < ne03; i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i00 = 0; i00 < ne00; i00++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    for (int i11 = 0; i11 < ne11; i11++) {
                        Uchar* src0 = &src0_p[i3*nb03 + i2*nb02 + i01*nb01 + i00*nb00];
                        Uchar* src1 = &src1_p[i3*nb13 + i2*nb12 + i01*nb11 + i11*nb10];
                        Uchar* dst  = &dst_p [i3*nb3  + i2*nb2  + i00*nb1  + i11*nb0 ];
                        *dst += *src0 * *src1;
                    }
                }
            }
        }
    }

    return NULL;
}

void* kernel_mul_mm_q2_K_f32(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    load_src01_dst(args);

    for (int i3 = 0; i3 < ne03; i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i00 = 0; i00 < ne00; i00++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    for (int i11 = 0; i11 < ne11; i11++) {
                        Uchar* src0 = &src0_p[i3*nb03 + i2*nb02 + i01*nb01 + i00*nb00];
                        Uchar* src1 = &src1_p[i3*nb13 + i2*nb12 + i01*nb11 + i11*nb10];
                        Uchar* dst  = &dst_p [i3*nb3  + i2*nb2  + i00*nb1  + i11*nb0 ];
                        *dst += *src0 * *src1;
                    }
                }
            }
        }
    }

    return NULL;
}

void* kernel_mul_mm_q3_K_f32(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    load_src01_dst(args);

    for (int i3 = 0; i3 < ne03; i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i00 = 0; i00 < ne00; i00++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    for (int i11 = 0; i11 < ne11; i11++) {
                        Uchar* src0 = &src0_p[i3*nb03 + i2*nb02 + i01*nb01 + i00*nb00];
                        Uchar* src1 = &src1_p[i3*nb13 + i2*nb12 + i01*nb11 + i11*nb10];
                        Uchar* dst  = &dst_p [i3*nb3  + i2*nb2  + i00*nb1  + i11*nb0 ];
                        *dst += *src0 * *src1;
                    }
                }
            }
        }
    }

    return NULL;
}

void* kernel_mul_mm_q4_K_f32(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    load_src01_dst(args);

    for (int i3 = 0; i3 < ne03; i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i00 = 0; i00 < ne00; i00++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    for (int i11 = 0; i11 < ne11; i11++) {
                        Uchar* src0 = &src0_p[i3*nb03 + i2*nb02 + i01*nb01 + i00*nb00];
                        Uchar* src1 = &src1_p[i3*nb13 + i2*nb12 + i01*nb11 + i11*nb10];
                        Uchar* dst  = &dst_p [i3*nb3  + i2*nb2  + i00*nb1  + i11*nb0 ];
                        *dst += *src0 * *src1;
                    }
                }
            }
        }
    }

    return NULL;
}

void* kernel_mul_mm_q5_K_f32(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    load_src01_dst(args);

    for (int i3 = 0; i3 < ne03; i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i00 = 0; i00 < ne00; i00++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    for (int i11 = 0; i11 < ne11; i11++) {
                        Uchar* src0 = &src0_p[i3*nb03 + i2*nb02 + i01*nb01 + i00*nb00];
                        Uchar* src1 = &src1_p[i3*nb13 + i2*nb12 + i01*nb11 + i11*nb10];
                        Uchar* dst  = &dst_p [i3*nb3  + i2*nb2  + i00*nb1  + i11*nb0 ];
                        *dst += *src0 * *src1;
                    }
                }
            }
        }
    }

    return NULL;
}

void* kernel_mul_mm_q6_K_f32(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    load_src01_dst(args);

    for (int i3 = 0; i3 < ne03; i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i00 = 0; i00 < ne00; i00++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    for (int i11 = 0; i11 < ne11; i11++) {
                        Uchar* src0 = &src0_p[i3*nb03 + i2*nb02 + i01*nb01 + i00*nb00];
                        Uchar* src1 = &src1_p[i3*nb13 + i2*nb12 + i01*nb11 + i11*nb10];
                        Uchar* dst  = &dst_p [i3*nb3  + i2*nb2  + i00*nb1  + i11*nb0 ];
                        *dst += *src0 * *src1;
                    }
                }
            }
        }
    }

    return NULL;
}