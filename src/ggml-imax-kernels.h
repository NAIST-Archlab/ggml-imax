#pragma once
#include <stddef.h>

struct imax_kernel_args {
    void** args;
};

void* kernel_add                 (struct imax_kernel_args* args);
void* kernel_mul                 (struct imax_kernel_args* args);
void* kernel_scale               (struct imax_kernel_args* args);
void* kernel_scale_4             (struct imax_kernel_args* args);
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
void* kernel_upscale_f32         (struct imax_kernel_args* args);
void* kernel_pad_f32             (struct imax_kernel_args* args);
void* kernel_argsort_f32_i32_asc (struct imax_kernel_args* args);
void* kernel_argsort_f32_i32_desc(struct imax_kernel_args* args);
void* kernel_sum_rows            (struct imax_kernel_args* args);