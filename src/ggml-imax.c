#include "ggml-imax.h"
#include "ggml-imax-kernels.h"

#include "ggml-backend-impl.h"
#include "ggml.h"
#include "emax7.h"
#include "emax7lib.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <stdarg.h>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#ifdef GGML_IMAX_NDEBUG
#define GGML_IMAX_LOG_INFO(...)
#define GGML_IMAX_LOG_WARN(...)
#define GGML_IMAX_LOG_ERROR(...)
#else
#define GGML_IMAX_LOG_INFO(...)  ggml_imax_log(GGML_LOG_LEVEL_INFO, __VA_ARGS__)
#define GGML_IMAX_LOG_WARN(...)  ggml_imax_log(GGML_LOG_LEVEL_WARN, __VA_ARGS__)
#define GGML_IMAX_LOG_ERROR(...) ggml_imax_log(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#endif

#ifndef DMA_MMAP_SIZE
#define DMA_MMAP_SIZE 0x0000000000010000LL
#endif

#ifndef DDR_MMAP_SIZE
#define DDR_MMAP_SIZE	 0x0000000100000000LL
#endif

#define UNUSED(x) (void)(x)

struct imax_kernel_pipeline {
    pthread_t tid;
    int stat;
    sigset_t sigset;
    void* (*kernel)(struct imax_kernel_args*);
    struct imax_kernel_args args;
    struct imax_kernel_pipeline* prev;
    struct imax_kernel_pipeline* next;
};

struct ggml_imax_kernel_queue {
    struct imax_kernel_pipeline* head;
    struct imax_kernel_pipeline* tail;
};

static void ggml_imax_default_log_callback(enum ggml_log_level level, const char * msg, void * user_data) {
    fprintf(stderr, "%s", msg);

    UNUSED(level);
    UNUSED(user_data);
}

ggml_log_callback ggml_imax_log_callback = ggml_imax_default_log_callback;
void * ggml_imax_log_user_data = NULL;

GGML_ATTRIBUTE_FORMAT(2, 3)
static void ggml_imax_log(enum ggml_log_level level, const char * format, ...){
    if (ggml_imax_log_callback != NULL) {
        va_list args;
        va_start(args, format);
        char buffer[128];
        int len = vsnprintf(buffer, 128, format, args);
        if (len < 128) {
            ggml_imax_log_callback(level, buffer, ggml_imax_log_user_data);
        } else {
            char* buffer2 = malloc(len+1);
            va_end(args);
            va_start(args, format);
            vsnprintf(buffer2, len+1, format, args);
            buffer2[len] = 0;
            ggml_imax_log_callback(level, buffer2, ggml_imax_log_user_data);
            free(buffer2);
        }
        va_end(args);
    }
}

void ggml_imax_kernel_queue_push(struct ggml_imax_kernel_queue* queue, struct imax_kernel_pipeline* pipeline) {
    if (queue->head == NULL) {
        queue->head = pipeline;
        queue->tail = pipeline;
    } else {
        queue->tail->next = pipeline;
        pipeline->prev = queue->tail;
        queue->tail = pipeline;
    }
}

struct imax_kernel_pipeline* ggml_imax_kernel_queue_pop(struct ggml_imax_kernel_queue* queue) {
    if (queue->head != NULL) {
        struct imax_kernel_pipeline* pipeline = queue->head;
        queue->head = pipeline->next;
        if (queue->head != NULL) {
            queue->head->prev = NULL;
        }
        pipeline->next = NULL;
        return pipeline;
    }
    return NULL;
}

void* ggml_imax_pipeline_runtime(struct ggml_imax_kernel_queue *queue) {
    struct imax_kernel_pipeline* pipeline = ggml_imax_kernel_queue_pop(queue);

    while(pipeline != NULL) {
        pthread_create(&pipeline->tid, NULL, pipeline->kernel, &pipeline->args);
        pthread_join(pipeline->tid, NULL);
        pipeline = ggml_imax_kernel_queue_pop(queue);
    }
}

pthread_t ggml_imax_kernel_queue_run_async(struct ggml_imax_kernel_queue *queue) {
    pthread_t tid;

    pthread_create(&tid, NULL, ggml_imax_pipeline_runtime, queue);
    return tid;
}

void ggml_imax_kernel_queue_wait(pthread_t tid) {
    pthread_join(tid, NULL);
}

void ggml_imax_kernel_queue_run_sync(struct ggml_imax_kernel_queue *queue) {
    ggml_imax_kernel_queue_wait(ggml_imax_kernel_queue_run_async(queue));
}

enum ggml_imax_kernel_type {
    GGML_IMAX_KERNEL_TYPE_ADD,
    GGML_IMAX_KERNEL_TYPE_MUL,
    GGML_IMAX_KERNEL_TYPE_SCALE,
    GGML_IMAX_KERNEL_TYPE_MUL_MM_F32_F32,
    GGML_IMAX_KERNEL_TYPE_MUL_MM_F16_F32,
    GGML_IMAX_KERNEL_TYPE_MUL_MM_Q4_0_F32,
    GGML_IMAX_KERNEL_TYPE_MUL_MM_Q4_1_F32,
    GGML_IMAX_KERNEL_TYPE_MUL_MM_Q5_0_F32,
    GGML_IMAX_KERNEL_TYPE_MUL_MM_Q5_1_F32,
    GGML_IMAX_KERNEL_TYPE_MUL_MM_Q8_0_F32,
    GGML_IMAX_KERNEL_TYPE_MUL_MM_Q2_K_F32,
    GGML_IMAX_KERNEL_TYPE_MUL_MM_Q3_K_F32,
    GGML_IMAX_KERNEL_TYPE_MUL_MM_Q4_K_F32,
    GGML_IMAX_KERNEL_TYPE_MUL_MM_Q5_K_F32,
    GGML_IMAX_KERNEL_TYPE_MUL_MM_Q6_K_F32,
    GGML_IMAX_KERNEL_TYPE_UPSCALE_F32,
    GGML_IMAX_KERNEL_TYPE_PAD_F32,
    GGML_IMAX_KERNEL_TYPE_ARGSORT_F32_I32_ASC,
    GGML_IMAX_KERNEL_TYPE_ARGSORT_F32_I32_DESC,
    GGML_IMAX_KERNEL_TYPE_SUM_ROWS,
    GGML_IMAX_KERNEL_TYPE_DIV,
    GGML_IMAX_KERNEL_TYPE_SQR,
    GGML_IMAX_KERNEL_TYPE_SOFTMAX,
    GGML_IMAX_KERNEL_TYPE_RMS_NORM,
    GGML_IMAX_KERNEL_TYPE_NORM,
    GGML_IMAX_KERNEL_TYPE_GROUP_NORM,
    GGML_IMAX_KERNEL_TYPE_GET_ROWS,
    GGML_IMAX_KERNEL_TYPE_ALIBI,
    GGML_IMAX_KERNEL_TYPE_ROPE,
    GGML_IMAX_KERNEL_TYPE_IM2COL,
    GGML_IMAX_KERNEL_TYPE_POOL_1D,
    GGML_IMAX_KERNEL_TYPE_POOL_2D,
    GGML_IMAX_KERNEL_TYPE_LEAKY_RELU,
    GGML_IMAX_KERNEL_TYPE_CPY,
    GGML_IMAX_KERNEL_TYPE_DUP,
    GGML_IMAX_KERNEL_TYPE_CONT,
    GGML_IMAX_KERNEL_TYPE_DIAG_MASK_INF,
    GGML_IMAX_KERNEL_TYPE_UNARY,
    GGML_IMAX_KERNEL_TYPE_COUNT
};

struct ggml_imax_context {
    int n_cb;

    struct emax7* device;
    struct ggml_imax_kernel_queue queue;

    struct imax_kernel_pipeline* kernels[GGML_IMAX_KERNEL_TYPE_COUNT];
};

// TODO: memorize provisional buffers allocation because it is not possible to free it
char block_flags[DDR_MMAP_SIZE / DMA_MMAP_SIZE] = {0};
uint32_t block_ptr = 0;

// TODO: ggml_imax_host_malloc is not implemented
static void* ggml_imax_host_malloc(size_t n) {
    void* data = emax_info[0].ddr_mmap + (block_ptr * DMA_MMAP_SIZE);
    for (int i = 0; i < (n + DMA_MMAP_SIZE - 1) / DMA_MMAP_SIZE; i++) {
        if (block_flags[block_ptr + i] == 1) {
            GGML_IMAX_LOG_ERROR("%s: error: block %d is already allocated\n", __func__, block_ptr + i);
            return NULL;
        }
        block_flags[block_ptr+i] = 1;
    }
    block_ptr += (n + DMA_MMAP_SIZE - 1) / DMA_MMAP_SIZE;

    return data;
}

static struct ggml_imax_context* ggml_imax_init(int n_cb) {
    GGML_IMAX_LOG_INFO("%s: Initialize IMAX3\n", __func__);

#define EMAX7 // Temp
#define ARMZYNQ // Temp
#if defined(EMAX7)
if (emax7_open(n_cb) == NULL) exit(1);
for (int i = 0; i < n_cb; i++) {
    char* membase = emax_info[i].ddr_mmap;
#if __AARCH64EL__ == 1
    Dll zero = 0;
#else
    Dll zero = {0, 0};
#endif
    {int j;for (j = 0; j < (DDR_MMAP_SIZE + sizeof(Dll) - 1) / sizeof(Dll); j++)*((Dll *)membase + j) = zero;}

#if !defined(ARMZYNQ)
    emax_info[i].dma_phys = DMA_BASE2_PHYS; /* defined in emax7lib.h */
    emax_info[i].dma_mmap = emax_info[i].dma_phys;
    emax_info[i].reg_phys = REG_BASE2_PHYS; /* defined in emax7lib.h */
    emax_info[i].reg_mmap = emax_info[i].reg_phys;
    emax_info[i].lmm_phys = LMM_BASE2_PHYS;
    emax_info[i].lmm_mmap = emax_info[i].lmm_phys;
    emax_info[i].ddr_phys = membase;
    emax_info[i].ddr_mmap = emax_info[i].ddr_phys;
#elif (defined(ARMSIML) || defined(ARMZYNQ))
    emax7[i].dma_ctrl = emax_info[i].dma_mmap;
    emax7[i].reg_ctrl = emax_info[i].reg_mmap;
    ((struct reg_ctrl*)emax7[i].reg_ctrl)->i[i].cmd = CMD_RESET;
#endif
#if defined(ARMZYNQ)
    usleep(1);
#endif
    switch (((struct reg_ctrl*)emax7[i].reg_ctrl)->i[i].stat >> 8 & 0xf) {
    case 3:
        EMAX_DEPTH = 64;
        break;
    case 2:
        EMAX_DEPTH = 32;
        break;
    case 1:
        EMAX_DEPTH = 16;
        break;
    default:
        EMAX_DEPTH = 8;
        break;
    }
    ((struct reg_ctrl*)emax7[i].reg_ctrl)->i[i].adtr = emax_info[i].ddr_mmap - emax_info[i].lmm_phys;
    ((struct reg_ctrl*)emax7[i].reg_ctrl)->i[i].dmrp = 0LL;
}
#endif

    // Configure context
    struct ggml_imax_context* ctx = (struct ggml_imax_context*) malloc(sizeof(struct ggml_imax_context));
    ctx->device = emax7;
    ctx->n_cb   = MIN(n_cb, GGML_IMAX_MAX_BUFFERS);

    // load kernels
    {
        for (int i = 0; i < GGML_IMAX_KERNEL_TYPE_COUNT; ++i) {
            ctx->kernels[i] = NULL;
        }

#define GGML_IMAX_ADD_KERNEL(e, name, supported) \
        if (supported) { \
            struct imax_kernel_pipeline* kernel = malloc(sizeof(struct imax_kernel_pipeline)); \
            kernel->kernel = kernel_##name; \
            ctx->kernels[e] = kernel; \
            GGML_IMAX_LOG_INFO("loading kernel_%-32s\n", #name); \
        } else { \
            GGML_IMAX_LOG_WARN("skipping kernle_%-32s (not supported)\n", #name); \
        }

        // On IMAX + CPU
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_ADD,                       add,                    true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_MUL,                       mul,                    true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_MUL_MM_F32_F32,            mul_mm_f32_f32,         true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_MUL_MM_F16_F32,            mul_mm_f16_f32,         true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_MUL_MM_Q4_0_F32,           mul_mm_q4_0_f32,        true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_MUL_MM_Q4_1_F32,           mul_mm_q4_1_f32,        true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_MUL_MM_Q5_0_F32,           mul_mm_q5_0_f32,        true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_MUL_MM_Q5_1_F32,           mul_mm_q5_1_f32,        true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_MUL_MM_Q8_0_F32,           mul_mm_q8_0_f32,        true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_MUL_MM_Q2_K_F32,           mul_mm_q2_K_f32,        true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_MUL_MM_Q3_K_F32,           mul_mm_q3_K_f32,        true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_MUL_MM_Q4_K_F32,           mul_mm_q4_K_f32,        true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_MUL_MM_Q5_K_F32,           mul_mm_q5_K_f32,        true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_MUL_MM_Q6_K_F32,           mul_mm_q6_K_f32,        true);

        // On CPU Only
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_SCALE,                     scale,                  true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_UPSCALE_F32,               upscale_f32,            true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_PAD_F32,                   pad_f32,                true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_ARGSORT_F32_I32_ASC,       argsort_f32_i32_asc,    true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_ARGSORT_F32_I32_DESC,      argsort_f32_i32_desc,   true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_SUM_ROWS,                  sum_rows,               true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_DIV,                       div,                    true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_SQR,                       sqr,                    true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_SOFTMAX,                   softmax,                true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_RMS_NORM,                  rms_norm,               true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_NORM,                      norm,                   true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_GROUP_NORM,                group_norm,             true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_GET_ROWS,                  get_rows,               true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_ALIBI,                     alibi,                  true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_ROPE,                      rope,                   true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_IM2COL,                    im2col,                 true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_POOL_1D,                   pool_1d,                true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_POOL_2D,                   pool_2d,                true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_LEAKY_RELU,                leaky_relu,             true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_CPY,                       cpy,                    true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_DUP,                       dup,                    true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_CONT,                      contiguous,             true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_DIAG_MASK_INF,             diag_mask_inf,          true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_UNARY,                     unary,                  true);
    }

    return ctx;
}

static void ggml_imax_free(struct ggml_imax_context * ctx) {
    GGML_IMAX_LOG_INFO("%s: deallocating\n", __func__);

    for (int i = 0; i < GGML_IMAX_KERNEL_TYPE_COUNT; ++i) {
        free(ctx->kernels[i]);
    }

    free(ctx->device);
    free(ctx);
}

// finds the IMAX buffer
static void* ggml_imax_get_buffer(struct ggml_tensor * t, size_t * offs) {
    const int64_t tsize = ggml_nbytes(t);

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;
    struct ggml_backend_imax_buffer_context * buf_ctx = (struct ggml_backend_imax_buffer_context *) buffer->context;

    // find the view that contains the tensor fully
    for (int i = 0; i < buf_ctx->n_buffers; ++i) {
        const int64_t ioffs = (int64_t) buf_ctx->buffers[i].data - (int64_t) buf_ctx->buffers[i].data;
        if (ioffs >= 0) {
            *offs = (size_t) ioffs;
            return buf_ctx->buffers[i].data;
        }
    }

    GGML_IMAX_LOG_ERROR("%s: error: tensor '%s' buffer is NULL\n", __func__, t->name);

    return NULL;
}

static bool ggml_imax_supports_op(const struct ggml_imax_context * ctx, const struct ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_ADD:
        case GGML_OP_MUL:
            return true;
        case GGML_OP_MUL_MAT:
            return true;
            //return (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F32);
        case GGML_OP_ACC:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_SCALE:
        case GGML_OP_PAD:
        case GGML_OP_ARGSORT:
        case GGML_OP_UPSCALE:
            return true;
        case GGML_OP_DIV:
        case GGML_OP_SQR:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
        case GGML_OP_CONCAT:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_RMS_NORM:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_NORM:
        case GGML_OP_GET_ROWS:
            return true;
        case GGML_OP_ALIBI:
        case GGML_OP_ROPE:
        case GGML_OP_IM2COL:
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
        case GGML_OP_DIAG_MASK_INF:
        case GGML_OP_UNARY:
        case GGML_OP_MUL_MAT_ID:
            return true;
        default:
            return false;
    }
}

static bool ggml_imax_graph_compute(
        struct ggml_imax_context * ctx,
               struct ggml_cgraph * gf) {
    const int n_nodes  = gf->n_nodes;
    const int n_cb = ctx->n_cb;
    const int n_nodes_per_cb = (n_nodes + n_cb - 1) / n_cb;

    // TODO: cb adapt to IMAX LANE
    for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {

        size_t offs_src0 = 0;
        size_t offs_src1 = 0;
        size_t offs_src2 = 0;
        size_t offs_dst  = 0;

        const int node_start =                                      (cb_idx + 0) * n_nodes_per_cb;
        const int node_end   = MIN((cb_idx == n_cb - 1) ? n_nodes : (cb_idx + 1) * n_nodes_per_cb, n_nodes);

        for (int i = node_start; i < node_end; ++i) {
            if (i == -1) {
                continue;
            }

            struct ggml_tensor * src0 = gf->nodes[i]->src[0];
            struct ggml_tensor * src1 = gf->nodes[i]->src[1];
            struct ggml_tensor * src2 = gf->nodes[i]->src[2];
            struct ggml_tensor * dst  = gf->nodes[i];

            switch (dst->op) {
                case GGML_OP_NONE:
                case GGML_OP_RESHAPE:
                case GGML_OP_VIEW:
                case GGML_OP_TRANSPOSE:
                case GGML_OP_PERMUTE:
                    {
                        // noop -> next node
                    } continue;
                default:
                    {
                    } break;
            }

            if (!ggml_imax_supports_op(ctx, dst)) {
                GGML_IMAX_LOG_ERROR("%s: error: unsupported op '%s'\n", __func__, ggml_op_desc(dst));
                GGML_ASSERT(!"unsupported op");
            }

            const int64_t  ne00 = src0 ? src0->ne[0] : 0;
            const int64_t  ne01 = src0 ? src0->ne[1] : 0;
            const int64_t  ne02 = src0 ? src0->ne[2] : 0;
            const int64_t  ne03 = src0 ? src0->ne[3] : 0;

            const uint64_t nb00 = src0 ? src0->nb[0] : 0;
            const uint64_t nb01 = src0 ? src0->nb[1] : 0;
            const uint64_t nb02 = src0 ? src0->nb[2] : 0;
            const uint64_t nb03 = src0 ? src0->nb[3] : 0;

            const int64_t  ne10 = src1 ? src1->ne[0] : 0;
            const int64_t  ne11 = src1 ? src1->ne[1] : 0;
            const int64_t  ne12 = src1 ? src1->ne[2] : 0;
            const int64_t  ne13 = src1 ? src1->ne[3] : 0; UNUSED(ne13);

            const uint64_t nb10 = src1 ? src1->nb[0] : 0;
            const uint64_t nb11 = src1 ? src1->nb[1] : 0;
            const uint64_t nb12 = src1 ? src1->nb[2] : 0;
            const uint64_t nb13 = src1 ? src1->nb[3] : 0; UNUSED(nb13);

            const int64_t  ne0  = dst ? dst->ne[0] : 0;
            const int64_t  ne1  = dst ? dst->ne[1] : 0;
            const int64_t  ne2  = dst ? dst->ne[2] : 0;
            const int64_t  ne3  = dst ? dst->ne[3] : 0;

            const uint64_t nb0  = dst ? dst->nb[0] : 0;
            const uint64_t nb1  = dst ? dst->nb[1] : 0;
            const uint64_t nb2  = dst ? dst->nb[2] : 0;
            const uint64_t nb3  = dst ? dst->nb[3] : 0;

            const enum ggml_type src0t = src0 ? src0->type : GGML_TYPE_COUNT;
            const enum ggml_type src1t = src1 ? src1->type : GGML_TYPE_COUNT;
            const enum ggml_type src2t = src2 ? src2->type : GGML_TYPE_COUNT;
            const enum ggml_type dstt  = dst  ? dst->type  : GGML_TYPE_COUNT;

            void* id_src0 = src0 ? ggml_imax_get_buffer(src0, &offs_src0) : NULL;
            void* id_src1 = src1 ? ggml_imax_get_buffer(src1, &offs_src1) : NULL;
            void* id_src2 = src2 ? ggml_imax_get_buffer(src2, &offs_src2) : NULL;
            void* id_dst  = dst  ? ggml_imax_get_buffer(dst,  &offs_dst)  : NULL;

#define set_src01_dst(args_name) \
    args_name.src0 = id_src0;args_name.src1 = id_src1;args_name.src2 = id_src2;args_name.dst  = id_dst;             \
    args_name.src0_type = src0t;args_name.src1_type = src1t;args_name.src2_type = src2t;args_name.dst_type  = dstt; \
    if(src0 && src0->op_params!=NULL) {memcpy(args_name.src0_op_params, src0->op_params, sizeof(int32_t)*16);}      \
    if(src1 && src1->op_params!=NULL) {memcpy(args_name.src1_op_params, src1->op_params, sizeof(int32_t)*16);}      \
    if(src2 && src2->op_params!=NULL) {memcpy(args_name.src2_op_params, src2->op_params, sizeof(int32_t)*16);}      \
    if(dst  &&  dst->op_params!=NULL) {memcpy(args_name.dst_op_params,   dst->op_params, sizeof(int32_t)*16);}      \
    args_name.src0_ne[0] = ne00;args_name.src0_ne[1] = ne01;args_name.src0_ne[2] = ne02;args_name.src0_ne[3] = ne03;\
    args_name.src0_nb[0] = nb00;args_name.src0_nb[1] = nb01;args_name.src0_nb[2] = nb02;args_name.src0_nb[3] = nb03;\
    args_name.src1_ne[0] = ne10;args_name.src1_ne[1] = ne11;args_name.src1_ne[2] = ne12;args_name.src1_ne[3] = ne13;\
    args_name.src1_nb[0] = nb10;args_name.src1_nb[1] = nb11;args_name.src1_nb[2] = nb12;args_name.src1_nb[3] = nb13;\
    args_name.dst_ne[0]  = ne0; args_name.dst_ne[1]  = ne1;args_name.dst_ne[2]   = ne2; args_name.dst_ne[3]  =  ne3;\
    args_name.dst_nb[0]  = nb0; args_name.dst_nb[1]  = nb1;args_name.dst_nb[2]   = nb2; args_name.dst_nb[3]  =  nb3

            switch (dst->op) {
                case GGML_OP_ADD:
                case GGML_OP_MUL:
                    {
                        const size_t offs = 0;

                        bool bcast_row = false;

                        int64_t nb = ne00;

                        struct imax_kernel_pipeline* pipeline = NULL;

                        switch (dst->op) {
                            case GGML_OP_ADD: pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_ADD]; break;
                            case GGML_OP_MUL: pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL]; break;
                            default: GGML_ASSERT(false);
                        }

                        set_src01_dst(pipeline->args);
                        ggml_imax_kernel_queue_push(&(ctx->queue), pipeline);
                    } break;
                case GGML_OP_MUL_MAT:
                    {
                        GGML_ASSERT(ne00 == ne10);

                        // TODO: assert that dim2 and dim3 are contiguous
                        GGML_ASSERT(ne12 % ne02 == 0);
                        GGML_ASSERT(ne13 % ne03 == 0);

                        const unsigned int r2 = ne12/ne02;
                        const unsigned int r3 = ne13/ne03;

                        // MM kernels on IMAX
                        // TODO: extend the matrix fit to the kernel size
                        if (!ggml_is_transposed(src0) &&
                            !ggml_is_transposed(src1) &&
                            src1t == GGML_TYPE_F32) {

                            struct imax_kernel_pipeline* pipeline = NULL;

                            switch (src0->type) {
                                case GGML_TYPE_F32:     pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_MM_F32_F32    ]; break;
                                case GGML_TYPE_F16:     pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_MM_F16_F32    ]; break;
                                case GGML_TYPE_Q4_0:    pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_MM_Q4_0_F32   ]; break;
                                case GGML_TYPE_Q4_1:    pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_MM_Q4_1_F32   ]; break;
                                case GGML_TYPE_Q5_0:    pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_MM_Q5_0_F32   ]; break;
                                case GGML_TYPE_Q5_1:    pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_MM_Q5_1_F32   ]; break;
                                case GGML_TYPE_Q8_0:    pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_MM_Q8_0_F32   ]; break;
                                // TODO: Implement the following operations
                                //case GGML_TYPE_Q2_K:    pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_MM_Q2_K_F32   ]; break;
                                //case GGML_TYPE_Q3_K:    pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_MM_Q3_K_F32   ]; break;
                                //case GGML_TYPE_Q4_K:    pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_MM_Q4_K_F32   ]; break;
                                //case GGML_TYPE_Q5_K:    pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_MM_Q5_K_F32   ]; break;
                                //case GGML_TYPE_Q6_K:    pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_MM_Q6_K_F32   ]; break;
                                default: GGML_ASSERT(false && "MUL MAT-MAT not implemented: unsupported type");
                            }
                            set_src01_dst(pipeline->args);
                            if (pipeline != NULL) ggml_imax_kernel_queue_push(&(ctx->queue), pipeline);
                        } else GGML_ASSERT(false && "MUL MAT-MAT not implemented: unsupported dimensions");
                    } break;
// TODO: Impelment the following operations
                case GGML_OP_ACC:
                    {
                        GGML_ASSERT(src0t == GGML_TYPE_F32);
                        GGML_ASSERT(src1t == GGML_TYPE_F32);
                        GGML_ASSERT(dstt  == GGML_TYPE_F32);

                        GGML_ASSERT(ggml_is_contiguous(src0));
                        GGML_ASSERT(ggml_is_contiguous(src1));

                        const size_t pnb1 = ((int32_t*) dst->op_params)[0];
                        const size_t pnb2 = ((int32_t*) dst->op_params)[1];
                        const size_t pnb3 = ((int32_t*) dst->op_params)[2];
                        const size_t offs = ((int32_t*) dst->op_params)[3];

                        const bool inplace = (bool) ((int32_t*) dst->op_params)[4];

                        struct imax_kernel_pipeline* pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_ADD];
                        set_src01_dst(pipeline->args);
                        ggml_imax_kernel_queue_push(&(ctx->queue), pipeline);
                    } break;
                case GGML_OP_SCALE:
                    {
                        GGML_ASSERT(ggml_is_contiguous(src0));

                        const float scale = *(const float*) dst->op_params;

                        int64_t n = ggml_nelements(dst);

                        struct imax_kernel_pipeline* pipeline = NULL;

                        pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_SCALE];
                        set_src01_dst(pipeline->args);
                        ggml_imax_kernel_queue_push(&(ctx->queue), pipeline);
                    } break;
                case GGML_OP_SUM_ROWS:
                    {
                        GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));

                        struct imax_kernel_pipeline* pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_SUM_ROWS];
                        set_src01_dst(pipeline->args);
                        ggml_imax_kernel_queue_push(&(ctx->queue), pipeline);
                    } break;
                case GGML_OP_UPSCALE:
                    {
                        GGML_ASSERT(src0->type == GGML_TYPE_F32);

                        const int sf = dst->op_params[0];

                        struct imax_kernel_pipeline* pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_UPSCALE_F32];
                        set_src01_dst(pipeline->args);
                        ggml_imax_kernel_queue_push(&(ctx->queue), pipeline);
                    } break;
                case GGML_OP_PAD:
                    {
                        GGML_ASSERT(src0->type == GGML_TYPE_F32);

                        struct imax_kernel_pipeline* pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_PAD_F32];
                        set_src01_dst(pipeline->args);
                        ggml_imax_kernel_queue_push(&(ctx->queue), pipeline);
                    } break;
                case GGML_OP_ARGSORT:
                    {
                        GGML_ASSERT(src0->type == GGML_TYPE_F32);
                        GGML_ASSERT( dst->type == GGML_TYPE_I32);

                        const int nrows = ggml_nrows(src0);

                        enum ggml_sort_order order = (enum ggml_sort_order) dst->op_params[0];

                        struct imax_kernel_pipeline* pipeline = NULL;

                        switch (order) {
                            case GGML_SORT_ORDER_ASC:  pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_ARGSORT_F32_I32_ASC];  break;
                            case GGML_SORT_ORDER_DESC: pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_ARGSORT_F32_I32_DESC]; break;
                            default: GGML_ASSERT(false);
                        };
                        set_src01_dst(pipeline->args);
                        ggml_imax_kernel_queue_push(&(ctx->queue), pipeline);
                    } break;
                case GGML_OP_DIV:
                    {
                        GGML_ASSERT(src0->type == GGML_TYPE_F32);
                        GGML_ASSERT(src1->type == GGML_TYPE_F32);
                        GGML_ASSERT(dst->type  == GGML_TYPE_F32);

                        const size_t offs = 0;

                        struct imax_kernel_pipeline* pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_DIV];
                        set_src01_dst(pipeline->args);
                        ggml_imax_kernel_queue_push(&(ctx->queue), pipeline);
                    } break;
                case GGML_OP_SQR:
                    {
                        GGML_ASSERT(src0->type == GGML_TYPE_F32);
                        GGML_ASSERT(dst->type  == GGML_TYPE_F32);

                        const size_t offs = 0;

                        struct imax_kernel_pipeline* pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_SQR];
                        set_src01_dst(pipeline->args);
                        ggml_imax_kernel_queue_push(&(ctx->queue), pipeline);
                    } break;
                case GGML_OP_SOFT_MAX:
                    {
                        GGML_ASSERT(src0->type == GGML_TYPE_F32);
                        GGML_ASSERT(dst->type  == GGML_TYPE_F32);

                        const size_t offs = 0;

                        struct imax_kernel_pipeline* pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_SOFTMAX];
                        set_src01_dst(pipeline->args);
                        ggml_imax_kernel_queue_push(&(ctx->queue), pipeline);
                    } break;
                case GGML_OP_RMS_NORM:
                    {
                        GGML_ASSERT(src0->type == GGML_TYPE_F32);
                        GGML_ASSERT(dst->type  == GGML_TYPE_F32);

                        const size_t offs = 0;

                        struct imax_kernel_pipeline* pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_RMS_NORM];
                        set_src01_dst(pipeline->args);
                        ggml_imax_kernel_queue_push(&(ctx->queue), pipeline);
                    } break;
                case GGML_OP_NORM:
                    {
                        GGML_ASSERT(src0->type == GGML_TYPE_F32);
                        GGML_ASSERT(dst->type  == GGML_TYPE_F32);

                        const size_t offs = 0;

                        struct imax_kernel_pipeline* pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_NORM];
                        set_src01_dst(pipeline->args);
                        ggml_imax_kernel_queue_push(&(ctx->queue), pipeline);
                    } break;
                case GGML_OP_GROUP_NORM:
                    {
                        GGML_ASSERT(src0->type == GGML_TYPE_F32);
                        GGML_ASSERT(dst->type  == GGML_TYPE_F32);

                        const size_t offs = 0;

                        struct imax_kernel_pipeline* pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_GROUP_NORM];
                        set_src01_dst(pipeline->args);
                        ggml_imax_kernel_queue_push(&(ctx->queue), pipeline);
                    } break;
                case GGML_OP_GET_ROWS:
                    {
                        const size_t offs = 0;

                        struct imax_kernel_pipeline* pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_GET_ROWS];
                        set_src01_dst(pipeline->args);
                        ggml_imax_kernel_queue_push(&(ctx->queue), pipeline);
                    } break;
                case GGML_OP_ALIBI:
                    {
                        const size_t offs = 0;

                        struct imax_kernel_pipeline* pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_ALIBI];
                        set_src01_dst(pipeline->args);
                        ggml_imax_kernel_queue_push(&(ctx->queue), pipeline);
                    } break;
                case GGML_OP_ROPE:
                    {
                        const size_t offs = 0;

                        struct imax_kernel_pipeline* pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_ROPE];
                        set_src01_dst(pipeline->args);
                        ggml_imax_kernel_queue_push(&(ctx->queue), pipeline);
                    } break;
                case GGML_OP_IM2COL:
                    {
                        const size_t offs = 0;

                        struct imax_kernel_pipeline* pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_IM2COL];
                        set_src01_dst(pipeline->args);
                        ggml_imax_kernel_queue_push(&(ctx->queue), pipeline);
                    } break;
                case GGML_OP_POOL_1D:
                    {
                        const size_t offs = 0;

                        struct imax_kernel_pipeline* pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_POOL_1D];
                        set_src01_dst(pipeline->args);
                        ggml_imax_kernel_queue_push(&(ctx->queue), pipeline);
                    } break;
                case GGML_OP_POOL_2D:
                    {
                        const size_t offs = 0;

                        struct imax_kernel_pipeline* pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_POOL_2D];
                        set_src01_dst(pipeline->args);
                        ggml_imax_kernel_queue_push(&(ctx->queue), pipeline);
                    } break;
                case GGML_OP_LEAKY_RELU:
                    {
                        const size_t offs = 0;

                        struct imax_kernel_pipeline* pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_LEAKY_RELU];
                        set_src01_dst(pipeline->args);
                        ggml_imax_kernel_queue_push(&(ctx->queue), pipeline);
                    } break;
                case GGML_OP_CPY:
                    {
                        const size_t offs = 0;

                        struct imax_kernel_pipeline* pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_CPY];
                        set_src01_dst(pipeline->args);
                        ggml_imax_kernel_queue_push(&(ctx->queue), pipeline);
                    } break;
                case GGML_OP_DUP:
                    {
                        const size_t offs = 0;

                        struct imax_kernel_pipeline* pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_DUP];
                        set_src01_dst(pipeline->args);
                        ggml_imax_kernel_queue_push(&(ctx->queue), pipeline);
                    } break;
                case GGML_OP_CONT:
                    {
                        const size_t offs = 0;

                        struct imax_kernel_pipeline* pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_CONT];
                        set_src01_dst(pipeline->args);
                        ggml_imax_kernel_queue_push(&(ctx->queue), pipeline);
                    } break;
                case GGML_OP_DIAG_MASK_INF:
                    {
                        const size_t offs = 0;

                        struct imax_kernel_pipeline* pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_DIAG_MASK_INF];
                        set_src01_dst(pipeline->args);
                        ggml_imax_kernel_queue_push(&(ctx->queue), pipeline);
                    } break;
                case GGML_OP_UNARY:
                    {
                        const size_t offs = 0;

                        struct imax_kernel_pipeline* pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_UNARY];
                        set_src01_dst(pipeline->args);
                        ggml_imax_kernel_queue_push(&(ctx->queue), pipeline);
                    } break;
                default:
                    {
                        GGML_IMAX_LOG_ERROR("%s: error: node %3d, op = %8s not implemented\n", __func__, i, ggml_op_name(dst->op));
                        GGML_ASSERT(false);
                    }
            }
        }
    }

    ggml_imax_kernel_queue_run_sync(&(ctx->queue));

    return true;
}

////////////////////////////////////////////////////////////////////////////////

// backend interface

// default buffer
static struct emax7* g_backend_device = NULL;
static int g_backend_device_ref_count = 0;

static struct emax7* ggml_backend_imax_get_device(void) {
    return emax7;
}

GGML_CALL static const char * ggml_backend_imax_buffer_get_name(ggml_backend_buffer_t buffer) {
    return "IMAX3";

    UNUSED(buffer);
}

GGML_CALL static void ggml_backend_imax_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    struct ggml_backend_imax_buffer_context* ctx = (struct ggml_backend_imax_buffer_context*)buffer->context;

    for (int i = 0; i < ctx->n_buffers; i++) {
        free(ctx->buffers[i].data);
    }

    if (ctx->owned) {
        free(ctx->all_data);
    }

    free(ctx);
}

GGML_CALL static void * ggml_backend_imax_buffer_get_base(ggml_backend_buffer_t buffer) {
    struct ggml_backend_imax_buffer_context* ctx = (struct ggml_backend_imax_buffer_context*)buffer->context;

    return ctx->all_data;
}

GGML_CALL static void ggml_backend_imax_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor* tensor, const void* data, size_t offset, size_t size) {
    //GGML_IMAX_LOG_INFO("%s: tensor '%s', offset = %zu, size = %zu\n", __func__, tensor->name, offset, size);
    struct ggml_backend_imax_buffer_context* ctx = (struct ggml_backend_imax_buffer_context*)buffer->context;

    memcpy(&((char*)(ctx->buffers[0].data))[offset], data, size);
    // TODO: Below is not working (Bus Error)
    //if (size % DMA_MMAP_SIZE != 0) {
        //memset(&((char*)(ctx->buffers[0].data))[offset + size], 0, DMA_MMAP_SIZE - (size % DMA_MMAP_SIZE));
    //}

    UNUSED(buffer);
}

GGML_CALL static void ggml_backend_imax_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor* tensor, void* data, size_t offset, size_t size) {
    GGML_IMAX_LOG_INFO("%s: tensor '%s', offset = %zu, size = %zu\n", __func__, tensor->name, offset, size);
    struct ggml_backend_imax_buffer_context* ctx = (struct ggml_backend_imax_buffer_context*)buffer->context;

    memcpy(data, &((char*)(&ctx->buffers[0].data))[offset], size);

    UNUSED(buffer);
}

GGML_CALL static bool ggml_backend_imax_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor* src, struct ggml_tensor* dst) {
    GGML_IMAX_LOG_INFO("%s: src '%s', dst '%s'\n", __func__, src->name, dst->name);

    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    UNUSED(buffer);
}

GGML_CALL static void ggml_backend_imax_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    struct ggml_backend_imax_buffer_context* ctx = (struct ggml_backend_imax_buffer_contexts *)buffer->context;

    memset(ctx->all_data, value, ctx->all_size);
}

static struct ggml_backend_buffer_i ggml_backend_imax_buffer_i = {
    /* .get_name        = */ ggml_backend_imax_buffer_get_name,
    /* .free_buffer     = */ ggml_backend_imax_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_imax_buffer_get_base,
    /* .init_tensor     = */ NULL,
    /* .set_tensor      = */ ggml_backend_imax_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_imax_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_imax_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_imax_buffer_clear,
    /* .reset           = */ NULL,
};

// default buffer type

GGML_CALL static const char* ggml_backend_imax_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "IMAX3";

    UNUSED(buft);
}

GGML_CALL static size_t ggml_backend_imax_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 32;
    UNUSED(buft);
}

GGML_CALL static ggml_backend_buffer_t ggml_backend_imax_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    GGML_IMAX_LOG_INFO("%s: size = %8.2f MiB %d\n", __func__, size / 1024.0 / 1024.0, block_ptr);

    struct ggml_backend_imax_buffer_context* ctx = malloc(sizeof(struct ggml_backend_imax_buffer_context));

    const size_t size_page = DMA_MMAP_SIZE;

    size_t size_aligned = size;
    if ((size_aligned % (size_page*ggml_backend_imax_buffer_type_get_alignment(buft))) != 0) {
        size_aligned += (size_page - (size_aligned % (size_page*ggml_backend_imax_buffer_type_get_alignment(buft))));
    }

    struct emax7* device = ggml_backend_imax_get_device();

    // TODO: ggml_imax_host_malloc() 直す
    ctx->all_data = ggml_imax_host_malloc(size_aligned);
    ctx->all_size = size_aligned;
    ctx->owned = true;
    ctx->n_buffers = 1;

    ctx->buffers[0].data = ctx->all_data;
    ctx->buffers[0].size = ctx->all_size;

    GGML_IMAX_LOG_INFO("%s: allocated buffer, size = %8.2f MiB\n", __func__, size_aligned / 1024.0 / 1024.0);

    return ggml_backend_buffer_init(buft, ggml_backend_imax_buffer_i, ctx, size);
}

GGML_CALL static size_t ggml_backend_imax_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    struct emax7* device = ggml_backend_imax_get_device();
    size_t max_size = DDR_MMAP_SIZE;

    return max_size;

    UNUSED(buft);
}

GGML_CALL static bool ggml_backend_imax_buffer_type_supports_backend(ggml_backend_buffer_type_t buft, ggml_backend_t backend) {
    return ggml_backend_is_imax(backend);

    UNUSED(buft);
}

GGML_CALL static bool ggml_backend_imax_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    UNUSED(buft);
}

GGML_CALL ggml_backend_buffer_type_t ggml_backend_imax_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_buffer_type_imax = {
        /* .iface = */ {
            /* .get_name         = */ ggml_backend_imax_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_imax_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_imax_buffer_type_get_alignment,
            /* .get_max_size     = */ ggml_backend_imax_buffer_type_get_max_size,
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .supports_backend = */ ggml_backend_imax_buffer_type_supports_backend,
            /* .is_host          = */ ggml_backend_imax_buffer_type_is_host,
        },
        /* .context = */ NULL,
    };

    return &ggml_backend_buffer_type_imax;
}

// backend

GGML_CALL static const char* ggml_backend_imax_name(ggml_backend_t backend) {
    return "IMAX3";

    UNUSED(backend);
}

GGML_CALL static void ggml_backend_imax_free(ggml_backend_t backend) {
    struct ggml_imax_context* ctx = (struct ggml_imax_context *)backend->context;
    ggml_imax_free(ctx);
    free(backend);
}

GGML_CALL static ggml_backend_buffer_type_t ggml_backend_imax_get_default_buffer_type(ggml_backend_t backend) {
    return ggml_backend_imax_buffer_type();

    UNUSED(backend);
}

GGML_CALL static bool ggml_backend_imax_graph_compute(ggml_backend_t backend, struct ggml_cgraph* cgraph) {
    struct ggml_imax_context* imax_ctx = (struct ggml_imax_context *)backend->context;

    return ggml_imax_graph_compute(imax_ctx, cgraph);
}

GGML_CALL static bool ggml_backend_imax_supports_op(ggml_backend_t backend, const struct ggml_tensor* op) {
    struct ggml_imax_context* imax_ctx = (struct ggml_imax_context*)backend->context;

    return ggml_imax_supports_op(imax_ctx, op);
}

static struct ggml_backend_i ggml_backend_imax_i = {
    /* .get_name                = */ ggml_backend_imax_name,
    /* .free                    = */ ggml_backend_imax_free,
    /* .get_default_buffer_type = */ ggml_backend_imax_get_default_buffer_type,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_imax_graph_compute,
    /* .supports_op             = */ ggml_backend_imax_supports_op,
};

void ggml_backend_imax_log_set_callback(ggml_log_callback log_callback, void*  user_data) {
    ggml_imax_log_callback  = log_callback;
    ggml_imax_log_user_data = user_data;
}

static ggml_guid_t ggml_backend_imax_guid(void) {
     static ggml_guid guid = { 0x99, 0xd5, 0x91, 0x68, 0xb0, 0x49, 0x42, 0x3d, 0xa4, 0x26, 0xfe, 0x26, 0x9d, 0xca, 0x98, 0xea };
    return &guid;
}

ggml_backend_t ggml_backend_imax_init(void) {
    //struct ggml_imax_context* ctx = ggml_imax_init(GGML_DEFAULT_N_THREADS);
    struct ggml_imax_context* ctx = ggml_imax_init(1);

    if (ctx == NULL) {
        return NULL;
    }

    ggml_backend_t imax_backend = malloc(sizeof(struct ggml_backend));

    *imax_backend = (struct ggml_backend) {
        /* .guid      = */ ggml_backend_imax_guid(),
        /* .interface = */ ggml_backend_imax_i,
        /* .context   = */ ctx,
    };

    return imax_backend;
}

bool ggml_backend_is_imax(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_imax_guid());
}

void ggml_backend_imax_set_n_cb(ggml_backend_t backend, int n_cb) {
    GGML_ASSERT(ggml_backend_is_imax(backend));

    struct ggml_imax_context* ctx = (struct ggml_imax_context*)backend->context;

    ctx->n_cb = MIN(n_cb, GGML_IMAX_MAX_BUFFERS);
}

GGML_CALL ggml_backend_t ggml_backend_reg_imax_init(const char* params, void* user_data); // silence warning

GGML_CALL ggml_backend_t ggml_backend_reg_imax_init(const char* params, void* user_data) {
    return ggml_backend_imax_init();

    GGML_UNUSED(params);
    GGML_UNUSED(user_data);
}
