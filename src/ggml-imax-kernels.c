#include "ggml-imax-kernels.h"
#include "ggml.h"
#include "emax7.h"
#include "emax7lib.h"

#include <stdio.h>

#define load_src01_dst(args_name) \
    Uchar* src0_p = (Uchar*)args_name->src0;   \
    Uchar* src1_p = (Uchar*)args_name->src1;   \
    Uchar* dst_p  = (Uchar*)args_name->dst;    \
    Ull    ne00 =  *(Ull*)&args_name->src0_ne[0];\
    Ull    ne01 =  *(Ull*)&args_name->src0_ne[1];\
    Ull    ne02 =  *(Ull*)&args_name->src0_ne[2];\
    Ull    ne03 =  *(Ull*)&args_name->src0_ne[3];\
    Ull    nb00 =  *(Ull*)&args_name->src0_nb[0];\
    Ull    nb01 =  *(Ull*)&args_name->src0_nb[1];\
    Ull    nb02 =  *(Ull*)&args_name->src0_nb[2];\
    Ull    nb03 =  *(Ull*)&args_name->src0_nb[3];\
    Ull    ne10 =  *(Ull*)&args_name->src1_ne[0];\
    Ull    ne11 =  *(Ull*)&args_name->src1_ne[1];\
    Ull    ne12 =  *(Ull*)&args_name->src1_ne[2];\
    Ull    ne13 =  *(Ull*)&args_name->src1_ne[3];\
    Ull    nb10 =  *(Ull*)&args_name->src1_nb[0];\
    Ull    nb11 =  *(Ull*)&args_name->src1_nb[1];\
    Ull    nb12 =  *(Ull*)&args_name->src1_nb[2];\
    Ull    nb13 =  *(Ull*)&args_name->src1_nb[3];\
    Ull    ne0  =  *(Ull*)&args_name->dst_ne[0]; \
    Ull    ne1  =  *(Ull*)&args_name->dst_ne[1]; \
    Ull    ne2  =  *(Ull*)&args_name->dst_ne[2]; \
    Ull    ne3  =  *(Ull*)&args_name->dst_ne[3]; \
    Ull    nb0  =  *(Ull*)&args_name->dst_nb[0]; \
    Ull    nb1  =  *(Ull*)&args_name->dst_nb[1]; \
    Ull    nb2  =  *(Ull*)&args_name->dst_nb[2]; \
    Ull    nb3  =  *(Ull*)&args_name->dst_nb[3]; \
    Uint* src0_op_params = args_name->src0_op_params;\
    Uint* src1_op_params = args_name->src1_op_params;\
    Uint* src2_op_params = args_name->src2_op_params;\
    Uint* dst_op_params  = args_name->dst_op_params;

void* kernel_add(struct imax_kernel_args* args) {
    GGML_IMAX_KERNEL_LOG_DEBUG("%s", __func__);
    load_src01_dst(args);
    Ull    nb   =  *(Ull*)args->nb;

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
    Ull    nb   =  *(Ull*)args->nb;

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

void* kernel_mul_mm_q4_0_f32(struct imax_kernel_args* args) {
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