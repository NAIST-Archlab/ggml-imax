#include "ggml-imax-kernels.h"
#include "ggml.h"
#include "ggml-imax.h"
#include "emax7.h"
#include "emax7lib.h"

#include <stdio.h>

#define NCHIP 1 //Temp

void* kernel_add(struct imax_kernel_args* args) {
    Uchar** src0 = (Uchar**)args->args[0];
    Uchar** src1 = (Uchar**)args->args[1];
    Uchar** dst  = (Uchar**)args->args[2];
    Ull    ne00 =  *(Ull*)args->args[3];
    Ull    ne01 =  *(Ull*)args->args[4];
    Ull    ne02 =  *(Ull*)args->args[5];
    Ull    ne03 =  *(Ull*)args->args[6];
    Ull    nb00 =  *(Ull*)args->args[7];
    Ull    nb01 =  *(Ull*)args->args[8];
    Ull    nb02 =  *(Ull*)args->args[9];
    Ull    nb03 =  *(Ull*)args->args[10];
    Ull    ne10 =  *(Ull*)args->args[11];
    Ull    ne11 =  *(Ull*)args->args[12];
    Ull    ne12 =  *(Ull*)args->args[13];
    Ull    ne13 =  *(Ull*)args->args[14];
    Ull    nb10 =  *(Ull*)args->args[15];
    Ull    nb11 =  *(Ull*)args->args[16];
    Ull    nb12 =  *(Ull*)args->args[17];
    Ull    nb13 =  *(Ull*)args->args[18];
    Ull    ne0  =  *(Ull*)args->args[19];
    Ull    ne1  =  *(Ull*)args->args[20];
    Ull    ne2  =  *(Ull*)args->args[21];
    Ull    ne3  =  *(Ull*)args->args[22];
    Ull    nb0  =  *(Ull*)args->args[23];
    Ull    nb1  =  *(Ull*)args->args[24];
    Ull    nb2  =  *(Ull*)args->args[25];
    Ull    nb3  =  *(Ull*)args->args[26];
    Ull    nb   =  *(Ull*)args->args[27];
    Ull CHIP;
    Ull LOOP1, LOOP0, L;
    Ull INIT1, INIT0;
    Ull AR[64][4]; /* output of EX in each unit */
    Ull BR[64][4][4]; /* output registers in each unit */
    Ull r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
    Ull r16, r17, r18, r19, r20, r21, r22, r23, r24, r25;
    Ull cofs, rofs, oofs;
    Ull cofs_init = (0LL-nb00)<<32|(0LL-nb10);
    Ull rofs_init = (0LL-nb01)<<32|(0LL-nb11);
    Ull col_stride = (nb00<<32)|nb10;
    Ull row_stride = (nb01<<32)|nb11;


#define ADD_BLK 0x100 //Temp
    for (int i3 = 0; i3 < ne03; i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            // Should be Contigious
            for (int i1blk = 0; i1blk < ne01; i1blk += ADD_BLK*4) { 
                Uchar* src0i1blk = src0 + nb03*i3 + nb02*i2 + nb01*i1blk*ADD_BLK;
                Uchar* src1i1blk = src1 + nb13*i3 + nb12*i2 + nb11*i1blk*ADD_BLK;
                Uchar* dsti1blk  = dst + nb1*i3 + nb2*i2 + nb1*i1blk*ADD_BLK;

//EMAX5A begin add mapdist=0
                for (CHIP=0; CHIP<NCHIP; CHIP++) {
                    for (INIT1=1,LOOP1=cofs_init; LOOP1--; INIT1=0) {
                        for (INIT0=1,LOOP0=rofs_init; LOOP0--; INIT0=0) {
                            exe(OP_ADD, &cofs, INIT0?cofs:cofs, EXP_H3210, col_stride, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xffffffffffffffffLL, OP_NOP, 0LL);
                            exe(OP_ADD, &rofs, rofs, EXP_H3210, INIT0?row_stride:0, EXP_H3210, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                            exe(OP_ADD, &oofs, rofs, EXP_H3210, cofs, EXP_H3210, 0LL, EXP_H3210, OP_AND, 0xffffffffLL, OP_NOP, 0LL);

                            mop(OP_LDR,  3, &BR[1][2][1], (Ull)src0[i1blk+0], (Ull)cofs, MSK_W0, (Ull)src0[i1blk+0], ADD_BLK*4, 0, 0, (Ull)NULL, ADD_BLK*4);
                            mop(OP_LDR,  3, &BR[1][2][0], (Ull)src0[i1blk+1], (Ull)cofs, MSK_W0, (Ull)src0[i1blk+1], ADD_BLK*4, 0, 0, (Ull)NULL, ADD_BLK*4);
                            mop(OP_LDR,  3, &BR[1][1][1], (Ull)src0[i1blk+2], (Ull)cofs, MSK_W0, (Ull)src0[i1blk+2], ADD_BLK*4, 0, 0, (Ull)NULL, ADD_BLK*4);
                            mop(OP_LDR,  3, &BR[1][1][0], (Ull)src0[i1blk+3], (Ull)cofs, MSK_W0, (Ull)src0[i1blk+3], ADD_BLK*4, 0, 0, (Ull)NULL, ADD_BLK*4);
                            mop(OP_LDR,  3, &BR[1][0][1], (Ull)src1[i1blk+0], (Ull)cofs, MSK_W1, (Ull)src1[i1blk+0], ADD_BLK*4, 0, 0, (Ull)NULL, ADD_BLK*4);
                            mop(OP_LDR,  3, &BR[1][0][0], (Ull)src1[i1blk+1], (Ull)cofs, MSK_W1, (Ull)src1[i1blk+1], ADD_BLK*4, 0, 0, (Ull)NULL, ADD_BLK*4);
                            mop(OP_LDR,  3, &BR[1][0][1], (Ull)src1[i1blk+2], (Ull)cofs, MSK_W1, (Ull)src1[i1blk+2], ADD_BLK*4, 0, 0, (Ull)NULL, ADD_BLK*4);
                            mop(OP_LDR,  3, &BR[1][0][0], (Ull)src1[i1blk+3], (Ull)cofs, MSK_W1, (Ull)src1[i1blk+3], ADD_BLK*4, 0, 0, (Ull)NULL, ADD_BLK*4);
                            mop(OP_LDR,  3, &BR[2][1][1], (Ull)dst[i1blk+0],  (Ull)cofs, MSK_W0, (Ull)dst[i1blk+0],  ADD_BLK*4, 0, 0, (Ull)NULL, ADD_BLK*4);
                            mop(OP_LDR,  3, &BR[2][1][0], (Ull)dst[i1blk+1],  (Ull)cofs, MSK_W0, (Ull)dst[i1blk+1],  ADD_BLK*4, 0, 0, (Ull)NULL, ADD_BLK*4);
                            mop(OP_LDR,  3, &BR[2][0][1], (Ull)dst[i1blk+2],  (Ull)cofs, MSK_W0, (Ull)dst[i1blk+2],  ADD_BLK*4, 0, 0, (Ull)NULL, ADD_BLK*4);
                            mop(OP_LDR,  3, &BR[2][0][0], (Ull)dst[i1blk+3],  (Ull)cofs, MSK_W0, (Ull)dst[i1blk+3],  ADD_BLK*4, 0, 0, (Ull)NULL, ADD_BLK*4);
                            exe(OP_FAD, &AR[2][0], BR[1][0][1], EXP_H3210, BR[1][2][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                            exe(OP_FAD, &AR[2][1], BR[1][0][0], EXP_H3210, BR[1][2][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                            exe(OP_FAD, &AR[2][2], BR[1][1][1], EXP_H3210, BR[1][2][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                            exe(OP_FAD, &AR[2][3], BR[1][1][0], EXP_H3210, BR[1][2][1], EXP_H1010, 0LL, EXP_H3210, OP_NOP, 0LL, OP_NOP, 0LL);
                            mop(OP_STR,  3, &AR[2][0], (Ull)dst[i1blk+0], (Ull)cofs, MSK_W1, (Ull)dst[i1blk+0], ADD_BLK*4, 0, 0, (Ull)NULL, ADD_BLK*4);
                            mop(OP_STR,  3, &AR[2][1], (Ull)dst[i1blk+1], (Ull)cofs, MSK_W1, (Ull)dst[i1blk+1], ADD_BLK*4, 0, 0, (Ull)NULL, ADD_BLK*4);
                            mop(OP_STR,  3, &AR[2][2], (Ull)dst[i1blk+2], (Ull)cofs, MSK_W1, (Ull)dst[i1blk+2], ADD_BLK*4, 0, 0, (Ull)NULL, ADD_BLK*4);
                            mop(OP_STR,  3, &AR[2][3], (Ull)dst[i1blk+3], (Ull)cofs, MSK_W1, (Ull)dst[i1blk+3], ADD_BLK*4, 0, 0, (Ull)NULL, ADD_BLK*4);
                        }
                    }
                }
//EMAX5A end
//EMAX5A drain_dirty_lmm
            }
        }
    }

    return NULL;
}

void* kernel_add_row             (struct imax_kernel_args* args){}
void* kernel_mul                 (struct imax_kernel_args* args){}
void* kernel_mul_row             (struct imax_kernel_args* args){}
void* kernel_scale               (struct imax_kernel_args* args){}
void* kernel_scale_4             (struct imax_kernel_args* args){}
void* kernel_mul_mm_f32_f32      (struct imax_kernel_args* args){}
void* kernel_mul_mm_f16_f32      (struct imax_kernel_args* args){}
void* kernel_mul_mm_q4_0_f32     (struct imax_kernel_args* args){}
void* kernel_mul_mm_q4_1_f32     (struct imax_kernel_args* args){}
void* kernel_mul_mm_q5_0_f32     (struct imax_kernel_args* args){}
void* kernel_mul_mm_q5_1_f32     (struct imax_kernel_args* args){}
void* kernel_mul_mm_q8_0_f32     (struct imax_kernel_args* args){}
void* kernel_mul_mm_q2_K_f32     (struct imax_kernel_args* args){}
void* kernel_mul_mm_q3_K_f32     (struct imax_kernel_args* args){}
void* kernel_mul_mm_q4_K_f32     (struct imax_kernel_args* args){}
void* kernel_mul_mm_q5_K_f32     (struct imax_kernel_args* args){}
void* kernel_mul_mm_q6_K_f32     (struct imax_kernel_args* args){}
void* kernel_mul_mm_iq2_xxs_f32  (struct imax_kernel_args* args){}
void* kernel_mul_mm_iq2_xs_f32   (struct imax_kernel_args* args){}
void* kernel_mul_mm_iq3_xxs_f32  (struct imax_kernel_args* args){}
void* kernel_mul_mm_iq3_s_f32    (struct imax_kernel_args* args){}
void* kernel_mul_mm_iq2_s_f32    (struct imax_kernel_args* args){}
void* kernel_mul_mm_iq1_s_f32    (struct imax_kernel_args* args){}
void* kernel_mul_mm_iq4_nl_f32   (struct imax_kernel_args* args){}
void* kernel_mul_mm_iq4_xs_f32   (struct imax_kernel_args* args){}
void* kernel_upscale_f32         (struct imax_kernel_args* args){}
void* kernel_pad_f32             (struct imax_kernel_args* args){}
void* kernel_argsort_f32_i32_asc (struct imax_kernel_args* args){}
void* kernel_argsort_f32_i32_desc(struct imax_kernel_args* args){}
void* kernel_sum_rows            (struct imax_kernel_args* args){}