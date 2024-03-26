#include "ggml-imax-kernels.h"
#include "ggml.h"
#include "ggml-imax.h"
#include "emax7.h"
#include "emax7lib.h"

#include <stdio.h>

#define NCHIP 1 //Temp

#define load_src01_dst(args_name) \
    Uchar** src0_p = (Uchar**)args_name->args[0];\
    Uchar** src1_p = (Uchar**)args_name->args[1];\
    Uchar** dst_p  = (Uchar**)args_name->args[2];\
    Ull    ne00 =  *(Ull*)args_name->args[3];    \
    Ull    ne01 =  *(Ull*)args_name->args[4];    \
    Ull    ne02 =  *(Ull*)args_name->args[5];    \
    Ull    ne03 =  *(Ull*)args_name->args[6];    \
    Ull    nb00 =  *(Ull*)args_name->args[7];    \
    Ull    nb01 =  *(Ull*)args_name->args[8];    \
    Ull    nb02 =  *(Ull*)args_name->args[9];    \
    Ull    nb03 =  *(Ull*)args_name->args[10];   \
    Ull    ne10 =  *(Ull*)args_name->args[11];   \
    Ull    ne11 =  *(Ull*)args_name->args[12];   \
    Ull    ne12 =  *(Ull*)args_name->args[13];   \
    Ull    ne13 =  *(Ull*)args_name->args[14];   \
    Ull    nb10 =  *(Ull*)args_name->args[15];   \
    Ull    nb11 =  *(Ull*)args_name->args[16];   \
    Ull    nb12 =  *(Ull*)args_name->args[17];   \
    Ull    nb13 =  *(Ull*)args_name->args[18];   \
    Ull    ne0  =  *(Ull*)args_name->args[19];   \
    Ull    ne1  =  *(Ull*)args_name->args[20];   \
    Ull    ne2  =  *(Ull*)args_name->args[21];   \
    Ull    ne3  =  *(Ull*)args_name->args[22];   \
    Ull    nb0  =  *(Ull*)args_name->args[23];   \
    Ull    nb1  =  *(Ull*)args_name->args[24];   \
    Ull    nb2  =  *(Ull*)args_name->args[25];   \
    Ull    nb3  =  *(Ull*)args_name->args[26]

void* kernel_add(struct imax_kernel_args* args) {
    load_src01_dst(args);
    Ull    nb   =  *(Ull*)args->args[27];

    for (int i3 = 0; i3 < ne03; i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i1 = 0; i1 < ne01; i1++) {
                for (int i0 = 0; i0 < ne00; i0++) {
                    Ull src0_index = (i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                    Ull src1_index = (i3*nb13 + i2*nb12 + i1*nb11 + i0*nb10);
                    Ull dst_index  = (i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0 );
                    Uint src0_blk = src0_index/DMA_REG_SIZE;
                    Uint src1_blk = src1_index/DMA_REG_SIZE;
                    Uint dst_blk  = dst_index/DMA_REG_SIZE;
                    Uchar* src0 = &src0_p[src0_blk][src0_index%DMA_REG_SIZE];
                    Uchar* src1 = &src1_p[src1_blk][src1_index%DMA_REG_SIZE];
                    Uchar* dst  = &dst_p [dst_blk ][dst_index%DMA_REG_SIZE ];
                    *dst = *src0 + *src1;
                }
            }
        }
    }

    return NULL;
}

void* kernel_mul(struct imax_kernel_args* args) {
    load_src01_dst(args);
    Ull    nb   =  *(Ull*)args->args[27];

    for (int i3 = 0; i3 < ne03; i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i1 = 0; i1 < ne01; i1++) {
                for (int i0 = 0; i0 < ne00; i0++) {
                    Ull src0_index = (i3*nb03 + i2*nb02 + i1*nb01 + i0*nb00);
                    Ull src1_index = (i3*nb13 + i2*nb12 + i1*nb11 + i0*nb10);
                    Ull dst_index  = (i3*nb3  + i2*nb2  + i1*nb1  + i0*nb0 );
                    Uint src0_blk = src0_index/DMA_REG_SIZE;
                    Uint src1_blk = src1_index/DMA_REG_SIZE;
                    Uint dst_blk  = dst_index/DMA_REG_SIZE;
                    Uchar* src0 = &src0_p[src0_blk][src0_index%DMA_REG_SIZE];
                    Uchar* src1 = &src1_p[src1_blk][src1_index%DMA_REG_SIZE];
                    Uchar* dst  = &dst_p [dst_blk ][dst_index%DMA_REG_SIZE ];
                    *dst = *src0 * *src1;
                }
            }
        }
    }

    return NULL;
}

void* kernel_mul_mm_f32_f32(struct imax_kernel_args* args) {
    load_src01_dst(args);

    for (int i3 = 0; i3 < ne03; i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i00 = 0; i00 < ne00; i00++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    for (int i11 = 0; i11 < ne11; i11++) {
                        Ull src0_index = (i3*nb03 + i2*nb02 + i01*nb01 + i00*nb00);
                        Ull src1_index = (i3*nb13 + i2*nb12 + i01*nb11 + i11*nb10);
                        Ull dst_index  = (i3*nb3  + i2*nb2  + i00*nb1  + i11*nb0 );
                        Uint src0_blk = src0_index/DMA_REG_SIZE;
                        Uint src1_blk = src1_index/DMA_REG_SIZE;
                        Uint dst_blk  = dst_index/DMA_REG_SIZE;
                        float  src0 = src0_p[src0_blk][src0_index%DMA_REG_SIZE];
                        float  src1 = src1_p[src1_blk][src1_index%DMA_REG_SIZE];
                        float* dst  = &dst_p[dst_blk ][dst_index%DMA_REG_SIZE ];
                        *dst += src0 * src1;
                    }
                }
            }
        }
    }

    return NULL;
}

// Under construction (not working)
void* kernel_mul_mm_f16_f32(struct imax_kernel_args* args) {
    load_src01_dst(args);

    for (int i3 = 0; i3 < ne03; i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i00 = 0; i00 < ne00; i00++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    for (int i11 = 0; i11 < ne11; i11++) {
                        Ull src0_index = (i3*nb03 + i2*nb02 + i01*nb01 + i00*nb00);
                        Ull src1_index = (i3*nb13 + i2*nb12 + i01*nb11 + i11*nb10);
                        Ull dst_index  = (i3*nb3  + i2*nb2  + i00*nb1  + i11*nb0 );
                        Uint src0_blk = src0_index/DMA_REG_SIZE;
                        Uint src1_blk = src1_index/DMA_REG_SIZE;
                        Uint dst_blk  = dst_index/DMA_REG_SIZE;
                        float  src0 = src0_p[src0_blk][src0_index%DMA_REG_SIZE];
                        float  src1 = src1_p[src1_blk][src1_index%DMA_REG_SIZE];
                        float* dst  = &dst_p[dst_blk ][dst_index%DMA_REG_SIZE ];
                        *dst += src0 * src1;
                    }
                }
            }
        }
    }

    return NULL;
}

void* kernel_mul_mm_q4_0_f32(struct imax_kernel_args* args) {
    load_src01_dst(args);

    for (int i3 = 0; i3 < ne03; i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i00 = 0; i00 < ne00; i00++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    for (int i11 = 0; i11 < ne11; i11++) {
                        Ull src0_index = (i3*nb03 + i2*nb02 + i01*nb01 + i00*nb00);
                        Ull src1_index = (i3*nb13 + i2*nb12 + i01*nb11 + i11*nb10);
                        Ull dst_index  = (i3*nb3  + i2*nb2  + i00*nb1  + i11*nb0 );
                        Uint src0_blk = src0_index/DMA_REG_SIZE;
                        Uint src1_blk = src1_index/DMA_REG_SIZE;
                        Uint dst_blk  = dst_index/DMA_REG_SIZE;
                        float  src0 = src0_p[src0_blk][src0_index%DMA_REG_SIZE];
                        float  src1 = src1_p[src1_blk][src1_index%DMA_REG_SIZE];
                        float* dst  = &dst_p[dst_blk ][dst_index%DMA_REG_SIZE ];
                        *dst += src0 * src1;
                    }
                }
            }
        }
    }

    return NULL;
}

void* kernel_mul_mm_q4_1_f32(struct imax_kernel_args* args) {
    load_src01_dst(args);

    for (int i3 = 0; i3 < ne03; i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i00 = 0; i00 < ne00; i00++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    for (int i11 = 0; i11 < ne11; i11++) {
                        Ull src0_index = (i3*nb03 + i2*nb02 + i01*nb01 + i00*nb00);
                        Ull src1_index = (i3*nb13 + i2*nb12 + i01*nb11 + i11*nb10);
                        Ull dst_index  = (i3*nb3  + i2*nb2  + i00*nb1  + i11*nb0 );
                        Uint src0_blk = src0_index/DMA_REG_SIZE;
                        Uint src1_blk = src1_index/DMA_REG_SIZE;
                        Uint dst_blk  = dst_index/DMA_REG_SIZE;
                        float  src0 = src0_p[src0_blk][src0_index%DMA_REG_SIZE];
                        float  src1 = src1_p[src1_blk][src1_index%DMA_REG_SIZE];
                        float* dst  = &dst_p[dst_blk ][dst_index%DMA_REG_SIZE ];
                        *dst += src0 * src1;
                    }
                }
            }
        }
    }

    return NULL;
}

void* kernel_mul_mm_q5_0_f32(struct imax_kernel_args* args) {
    load_src01_dst(args);

    for (int i3 = 0; i3 < ne03; i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i00 = 0; i00 < ne00; i00++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    for (int i11 = 0; i11 < ne11; i11++) {
                        Ull src0_index = (i3*nb03 + i2*nb02 + i01*nb01 + i00*nb00);
                        Ull src1_index = (i3*nb13 + i2*nb12 + i01*nb11 + i11*nb10);
                        Ull dst_index  = (i3*nb3  + i2*nb2  + i00*nb1  + i11*nb0 );
                        Uint src0_blk = src0_index/DMA_REG_SIZE;
                        Uint src1_blk = src1_index/DMA_REG_SIZE;
                        Uint dst_blk  = dst_index/DMA_REG_SIZE;
                        float  src0 = src0_p[src0_blk][src0_index%DMA_REG_SIZE];
                        float  src1 = src1_p[src1_blk][src1_index%DMA_REG_SIZE];
                        float* dst  = &dst_p[dst_blk ][dst_index%DMA_REG_SIZE ];
                        *dst += src0 * src1;
                    }
                }
            }
        }
    }

    return NULL;
}

void* kernel_mul_mm_q5_1_f32(struct imax_kernel_args* args) {
    load_src01_dst(args);

    for (int i3 = 0; i3 < ne03; i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i00 = 0; i00 < ne00; i00++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    for (int i11 = 0; i11 < ne11; i11++) {
                        Ull src0_index = (i3*nb03 + i2*nb02 + i01*nb01 + i00*nb00);
                        Ull src1_index = (i3*nb13 + i2*nb12 + i01*nb11 + i11*nb10);
                        Ull dst_index  = (i3*nb3  + i2*nb2  + i00*nb1  + i11*nb0 );
                        Uint src0_blk = src0_index/DMA_REG_SIZE;
                        Uint src1_blk = src1_index/DMA_REG_SIZE;
                        Uint dst_blk  = dst_index/DMA_REG_SIZE;
                        float  src0 = src0_p[src0_blk][src0_index%DMA_REG_SIZE];
                        float  src1 = src1_p[src1_blk][src1_index%DMA_REG_SIZE];
                        float* dst  = &dst_p[dst_blk ][dst_index%DMA_REG_SIZE ];
                        *dst += src0 * src1;
                    }
                }
            }
        }
    }

    return NULL;
}

void* kernel_mul_mm_q8_0_f32(struct imax_kernel_args* args) {
    load_src01_dst(args);

    for (int i3 = 0; i3 < ne03; i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i00 = 0; i00 < ne00; i00++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    for (int i11 = 0; i11 < ne11; i11++) {
                        Ull src0_index = (i3*nb03 + i2*nb02 + i01*nb01 + i00*nb00);
                        Ull src1_index = (i3*nb13 + i2*nb12 + i01*nb11 + i11*nb10);
                        Ull dst_index  = (i3*nb3  + i2*nb2  + i00*nb1  + i11*nb0 );
                        Uint src0_blk = src0_index/DMA_REG_SIZE;
                        Uint src1_blk = src1_index/DMA_REG_SIZE;
                        Uint dst_blk  = dst_index/DMA_REG_SIZE;
                        float  src0 = src0_p[src0_blk][src0_index%DMA_REG_SIZE];
                        float  src1 = src1_p[src1_blk][src1_index%DMA_REG_SIZE];
                        float* dst  = &dst_p[dst_blk ][dst_index%DMA_REG_SIZE ];
                        *dst += src0 * src1;
                    }
                }
            }
        }
    }

    return NULL;
}

void* kernel_mul_mm_q2_K_f32(struct imax_kernel_args* args) {
    load_src01_dst(args);

    for (int i3 = 0; i3 < ne03; i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i00 = 0; i00 < ne00; i00++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    for (int i11 = 0; i11 < ne11; i11++) {
                        Ull src0_index = (i3*nb03 + i2*nb02 + i01*nb01 + i00*nb00);
                        Ull src1_index = (i3*nb13 + i2*nb12 + i01*nb11 + i11*nb10);
                        Ull dst_index  = (i3*nb3  + i2*nb2  + i00*nb1  + i11*nb0 );
                        Uint src0_blk = src0_index/DMA_REG_SIZE;
                        Uint src1_blk = src1_index/DMA_REG_SIZE;
                        Uint dst_blk  = dst_index/DMA_REG_SIZE;
                        float  src0 = src0_p[src0_blk][src0_index%DMA_REG_SIZE];
                        float  src1 = src1_p[src1_blk][src1_index%DMA_REG_SIZE];
                        float* dst  = &dst_p[dst_blk ][dst_index%DMA_REG_SIZE ];
                        *dst += src0 * src1;
                    }
                }
            }
        }
    }

    return NULL;
}

void* kernel_mul_mm_q3_K_f32(struct imax_kernel_args* args) {
    load_src01_dst(args);

    for (int i3 = 0; i3 < ne03; i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i00 = 0; i00 < ne00; i00++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    for (int i11 = 0; i11 < ne11; i11++) {
                        Ull src0_index = (i3*nb03 + i2*nb02 + i01*nb01 + i00*nb00);
                        Ull src1_index = (i3*nb13 + i2*nb12 + i01*nb11 + i11*nb10);
                        Ull dst_index  = (i3*nb3  + i2*nb2  + i00*nb1  + i11*nb0 );
                        Uint src0_blk = src0_index/DMA_REG_SIZE;
                        Uint src1_blk = src1_index/DMA_REG_SIZE;
                        Uint dst_blk  = dst_index/DMA_REG_SIZE;
                        float  src0 = src0_p[src0_blk][src0_index%DMA_REG_SIZE];
                        float  src1 = src1_p[src1_blk][src1_index%DMA_REG_SIZE];
                        float* dst  = &dst_p[dst_blk ][dst_index%DMA_REG_SIZE ];
                        *dst += src0 * src1;
                    }
                }
            }
        }
    }

    return NULL;
}

void* kernel_mul_mm_q4_K_f32(struct imax_kernel_args* args) {
    load_src01_dst(args);

    for (int i3 = 0; i3 < ne03; i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i00 = 0; i00 < ne00; i00++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    for (int i11 = 0; i11 < ne11; i11++) {
                        Ull src0_index = (i3*nb03 + i2*nb02 + i01*nb01 + i00*nb00);
                        Ull src1_index = (i3*nb13 + i2*nb12 + i01*nb11 + i11*nb10);
                        Ull dst_index  = (i3*nb3  + i2*nb2  + i00*nb1  + i11*nb0 );
                        Uint src0_blk = src0_index/DMA_REG_SIZE;
                        Uint src1_blk = src1_index/DMA_REG_SIZE;
                        Uint dst_blk  = dst_index/DMA_REG_SIZE;
                        float  src0 = src0_p[src0_blk][src0_index%DMA_REG_SIZE];
                        float  src1 = src1_p[src1_blk][src1_index%DMA_REG_SIZE];
                        float* dst  = &dst_p[dst_blk ][dst_index%DMA_REG_SIZE ];
                        *dst += src0 * src1;
                    }
                }
            }
        }
    }

    return NULL;
}

void* kernel_mul_mm_q5_K_f32(struct imax_kernel_args* args) {
    load_src01_dst(args);

    for (int i3 = 0; i3 < ne03; i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i00 = 0; i00 < ne00; i00++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    for (int i11 = 0; i11 < ne11; i11++) {
                        Ull src0_index = (i3*nb03 + i2*nb02 + i01*nb01 + i00*nb00);
                        Ull src1_index = (i3*nb13 + i2*nb12 + i01*nb11 + i11*nb10);
                        Ull dst_index  = (i3*nb3  + i2*nb2  + i00*nb1  + i11*nb0 );
                        Uint src0_blk = src0_index/DMA_REG_SIZE;
                        Uint src1_blk = src1_index/DMA_REG_SIZE;
                        Uint dst_blk  = dst_index/DMA_REG_SIZE;
                        float  src0 = src0_p[src0_blk][src0_index%DMA_REG_SIZE];
                        float  src1 = src1_p[src1_blk][src1_index%DMA_REG_SIZE];
                        float* dst  = &dst_p[dst_blk ][dst_index%DMA_REG_SIZE ];
                        *dst += src0 * src1;
                    }
                }
            }
        }
    }

    return NULL;
}

void* kernel_mul_mm_q6_K_f32(struct imax_kernel_args* args) {
    load_src01_dst(args);

    for (int i3 = 0; i3 < ne03; i3++) {
        for (int i2 = 0; i2 < ne02; i2++) {
            for (int i00 = 0; i00 < ne00; i00++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    for (int i11 = 0; i11 < ne11; i11++) {
                        Ull src0_index = (i3*nb03 + i2*nb02 + i01*nb01 + i00*nb00);
                        Ull src1_index = (i3*nb13 + i2*nb12 + i01*nb11 + i11*nb10);
                        Ull dst_index  = (i3*nb3  + i2*nb2  + i00*nb1  + i11*nb0 );
                        Uint src0_blk = src0_index/DMA_REG_SIZE;
                        Uint src1_blk = src1_index/DMA_REG_SIZE;
                        Uint dst_blk  = dst_index/DMA_REG_SIZE;
                        float  src0 = src0_p[src0_blk][src0_index%DMA_REG_SIZE];
                        float  src1 = src1_p[src1_blk][src1_index%DMA_REG_SIZE];
                        float* dst  = &dst_p[dst_blk ][dst_index%DMA_REG_SIZE ];
                        *dst += src0 * src1;
                    }
                }
            }
        }
    }

    return NULL;
}

void* kernel_upscale_f32         (struct imax_kernel_args* args){}
void* kernel_pad_f32             (struct imax_kernel_args* args){}
void* kernel_argsort_f32_i32_asc (struct imax_kernel_args* args){}
void* kernel_argsort_f32_i32_desc(struct imax_kernel_args* args){}
void* kernel_sum_rows            (struct imax_kernel_args* args){}
void* kernel_scale               (struct imax_kernel_args* args){}
void* kernel_scale_4             (struct imax_kernel_args* args){}
