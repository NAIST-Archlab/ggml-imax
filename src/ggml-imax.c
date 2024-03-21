#include "ggml-imax.h"
#include "ggml-imax-kernels.h"

#include "ggml-backend-impl.h"
#include "ggml.h"
#include "emax7lib.h"

#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <asm/signal.h>
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

#define UNUSED(x) (void)(x)

#ifndef DMA_MMAP_SIZE
#define DMA_MMAP_SIZE 0x10000000
#endif

#ifndef DMA_REG_SIZE
#define DMA_REG_SIZE 0x1000
#endif

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
    struct imax_kernel_pipeline* pipeline = queue->head;

    while(pipeline = ggml_imax_kernel_queue_pop(queue)) {
        pthread_create(&pipeline->tid, NULL, pipeline->kernel, &pipeline->args);
        pthread_join(&pipeline->tid, NULL);
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
    GGML_IMAX_KERNEL_TYPE_ADD_ROW,
    GGML_IMAX_KERNEL_TYPE_MUL,
    GGML_IMAX_KERNEL_TYPE_MUL_ROW,
    GGML_IMAX_KERNEL_TYPE_SCALE,
    GGML_IMAX_KERNEL_TYPE_SCALE_4,
    GGML_IMAX_KERNEL_TYPE_GET_ROWS_F32,
    GGML_IMAX_KERNEL_TYPE_GET_ROWS_F16,
    GGML_IMAX_KERNEL_TYPE_GET_ROWS_Q4_0,
    GGML_IMAX_KERNEL_TYPE_GET_ROWS_Q4_1,
    GGML_IMAX_KERNEL_TYPE_GET_ROWS_Q5_0,
    GGML_IMAX_KERNEL_TYPE_GET_ROWS_Q5_1,
    GGML_IMAX_KERNEL_TYPE_GET_ROWS_Q8_0,
    GGML_IMAX_KERNEL_TYPE_GET_ROWS_Q2_K,
    GGML_IMAX_KERNEL_TYPE_GET_ROWS_Q3_K,
    GGML_IMAX_KERNEL_TYPE_GET_ROWS_Q4_K,
    GGML_IMAX_KERNEL_TYPE_GET_ROWS_Q5_K,
    GGML_IMAX_KERNEL_TYPE_GET_ROWS_Q6_K,
    GGML_IMAX_KERNEL_TYPE_GET_ROWS_IQ2_XXS,
    GGML_IMAX_KERNEL_TYPE_GET_ROWS_IQ2_XS,
    GGML_IMAX_KERNEL_TYPE_GET_ROWS_IQ3_XXS,
    GGML_IMAX_KERNEL_TYPE_GET_ROWS_IQ3_S,
    GGML_IMAX_KERNEL_TYPE_GET_ROWS_IQ2_S,
    GGML_IMAX_KERNEL_TYPE_GET_ROWS_IQ1_S,
    GGML_IMAX_KERNEL_TYPE_GET_ROWS_IQ4_NL,
    GGML_IMAX_KERNEL_TYPE_GET_ROWS_IQ4_XS,
    GGML_IMAX_KERNEL_TYPE_GET_ROWS_I32,
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
    GGML_IMAX_KERNEL_TYPE_MUL_MM_IQ2_XXS_F32,
    GGML_IMAX_KERNEL_TYPE_MUL_MM_IQ2_XS_F32,
    GGML_IMAX_KERNEL_TYPE_MUL_MM_IQ3_XXS_F32,
    GGML_IMAX_KERNEL_TYPE_MUL_MM_IQ3_S_F32,
    GGML_IMAX_KERNEL_TYPE_MUL_MM_IQ2_S_F32,
    GGML_IMAX_KERNEL_TYPE_MUL_MM_IQ1_S_F32,
    GGML_IMAX_KERNEL_TYPE_MUL_MM_IQ4_NL_F32,
    GGML_IMAX_KERNEL_TYPE_MUL_MM_IQ4_XS_F32,
    GGML_IMAX_KERNEL_TYPE_UPSCALE_F32,
    GGML_IMAX_KERNEL_TYPE_PAD_F32,
    GGML_IMAX_KERNEL_TYPE_ARGSORT_F32_I32_ASC,
    GGML_IMAX_KERNEL_TYPE_ARGSORT_F32_I32_DESC,
    GGML_IMAX_KERNEL_TYPE_SUM_ROWS,
    GGML_IMAX_KERNEL_TYPE_COUNT
};

struct ggml_imax_context {
    int n_cb;

    struct emax7* device;
    struct ggml_imax_kernel_queue queue;

    struct imax_kernel_pipeline* kernels[GGML_IMAX_KERNEL_TYPE_COUNT];
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

// TODO: memorize provisional buffers allocation because it is not possible to free it
static void* ggml_imax_host_malloc(size_t n) {
    void** data = malloc(sizeof(void*)*(n/DMA_REG_SIZE+(n%DMA_REG_SIZE?1:0)));
    for (int i = 0; i < (n/DMA_REG_SIZE)+(n%DMA_REG_SIZE?1:0); ++i) {
        // TODO: find a free block
        if (block_ptr >= DMA_MMAP_SIZE / DMA_REG_SIZE) {
            GGML_IMAX_LOG_ERROR("%s: error: out of memory\n", __func__);
            return NULL;
        } else {
            data[i] = DMA_BASE2_PHYS + block_ptr * DMA_REG_SIZE;
            block_flags[block_ptr] = 1;
            block_ptr++;
        }
    }

    return data;
}

static struct ggml_imax_context* ggml_imax_init(int n_cb) {
    GGML_IMAX_LOG_INFO("%s: allocating\n", __func__);

#define EMAX7 // Temp
#define ARMZYNQ // Temp
#if defined(EMAX7)
for (int i = 0; i < n_cb; i++) {
    if (emax7_open(n_cb) == NULL) exit(1);
    char* membase = emax_info[i].ddr_mmap;
#if __AARCH64EL__ == 1
    Dll zero = 0;
#else
    Dll zero = {0, 0};
#endif
    {int j;for (j = 0; j < (DMA_MMAP_SIZE + sizeof(Dll) - 1) / sizeof(Dll); j++)*((Dll *)membase + j) = zero;}

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
    struct ggml_imax_context* ctx = malloc(sizeof(struct ggml_imax_context));
    ctx->device = emax7;
    ctx->n_cb   = MIN(n_cb, GGML_IMAX_MAX_BUFFERS);

    // load kernels
    {
        for (int i = 0; i < GGML_IMAX_KERNEL_TYPE_COUNT; ++i) {
            ctx->kernels[i] = NULL;
        }

#define GGML_IMAX_ADD_KERNEL(e, name, supported) \
        if (supported) { \
            struct imax_kernel_pipeline* kernel = &ctx->kernels[e]; \
            kernel->kernel = kernel_##name; \
        } else { \
            GGML_IMAX_LOG_WARN("skipping %-32s (not supported)\n", #name); \
        }

        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_ADD,                       add,                    true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_ADD_ROW,                   add_row,                true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_MUL,                       mul,                    true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_MUL_ROW,                   mul_row,                true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_SCALE,                     scale,                  true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_SCALE_4,                   scale_4,                true);
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
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_MUL_MM_IQ2_XXS_F32,        mul_mm_iq2_xxs_f32,     true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_MUL_MM_IQ2_XS_F32,         mul_mm_iq2_xs_f32,      true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_MUL_MM_IQ3_XXS_F32,        mul_mm_iq3_xxs_f32,     true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_MUL_MM_IQ3_S_F32,          mul_mm_iq3_s_f32,       true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_MUL_MM_IQ2_S_F32,          mul_mm_iq2_s_f32,       true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_MUL_MM_IQ1_S_F32,          mul_mm_iq1_s_f32,       true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_MUL_MM_IQ4_NL_F32,         mul_mm_iq4_nl_f32,      true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_MUL_MM_IQ4_XS_F32,         mul_mm_iq4_xs_f32,      true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_UPSCALE_F32,               upscale_f32,            true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_PAD_F32,                   pad_f32,                true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_ARGSORT_F32_I32_ASC,       argsort_f32_i32_asc,    true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_ARGSORT_F32_I32_DESC,      argsort_f32_i32_desc,   true);
        GGML_IMAX_ADD_KERNEL(GGML_IMAX_KERNEL_TYPE_SUM_ROWS,                  sum_rows,               true);
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


char block_flags[DMA_MMAP_SIZE / DMA_REG_SIZE] = {0};
uint32_t block_ptr = 0;

struct ggml_backend_imax_buffer {
    void   **data; //data[]: index of block, data[b][n]: data
    size_t   size;
};

struct ggml_backend_imax_buffer_context {
    void * all_data;
    size_t all_size;

    bool owned;

    int n_buffers;
    struct ggml_backend_imax_buffer buffers[GGML_IMAX_MAX_BUFFERS];
};

// finds the Metal buffer that contains the tensor data on the GPU device
// the assumption is that there is 1-to-1 mapping between the host and device memory buffers, so we can find the
// Metal buffer based on the host memory pointer
static void* ggml_imax_get_buffer(struct ggml_tensor * t, size_t * offs) {
    const int64_t tsize = ggml_nbytes(t);

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;
    struct ggml_backend_imax_buffer_context * buf_ctx = (struct ggml_backend_imax_buffer_context *) buffer->context;

    // find the view that contains the tensor fully
    for (int i = 0; i < buf_ctx->n_buffers; ++i) {
        const int64_t ioffs = (int64_t) t->data - (int64_t) buf_ctx->buffers[i].data;
        if (ioffs >= 0 && ioffs + tsize <= (int64_t) buf_ctx->buffers[i].size) {
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
        case GGML_OP_ACC:
        case GGML_OP_MUL:
        case GGML_OP_SUM_ROWS:
        case GGML_OP_SCALE:
        case GGML_OP_PAD:
        case GGML_OP_ARGSORT:
        case GGML_OP_UPSCALE:
            return true;
        case GGML_OP_MUL_MAT:
            return (op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F32);
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
        case GGML_OP_GET_ROWS:
        case GGML_OP_UNARY:
        case GGML_OP_MUL_MAT_ID:
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
            const enum ggml_type dstt  = dst  ? dst->type  : GGML_TYPE_COUNT;

            void* id_src0 = src0 ? ggml_imax_get_buffer(src0, &offs_src0) : NULL;
            void* id_src1 = src1 ? ggml_imax_get_buffer(src1, &offs_src1) : NULL;
            void* id_src2 = src2 ? ggml_imax_get_buffer(src2, &offs_src2) : NULL;
            void* id_dst  = dst  ? ggml_imax_get_buffer(dst,  &offs_dst)  : NULL;

            switch (dst->op) {
                case GGML_OP_ADD:
                case GGML_OP_MUL:
                case GGML_OP_DIV:
                    {
                        const size_t offs = 0;

                        bool bcast_row = false;

                        int64_t nb = ne00;

                        struct imax_kernel_pipeline* pipeline = NULL;

                        if (ggml_nelements(src1) == ne10 && ggml_is_contiguous(src1) && ne00 % 4 == 0 && ne10 % 4 == 0) {
                            GGML_ASSERT(ggml_is_contiguous(src0));

                            // src1 is a row
                            GGML_ASSERT(ne11 == 1);


                            nb = ne00 / 4;
                            switch (dst->op) {
                                case GGML_OP_ADD: pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_ADD_ROW]; break;
                                case GGML_OP_MUL: pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_ROW]; break;
                                default: GGML_ASSERT(false);
                            }

                            bcast_row = true;
                        } else {
                            switch (dst->op) {
                                case GGML_OP_ADD: pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_ADD]; break;
                                case GGML_OP_MUL: pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL]; break;
                                default: GGML_ASSERT(false);
                            }
                        }

                        pipeline->args.args = malloc(sizeof(void*)*28);
                        pipeline->args.args[ 0] = &id_src0;
                        pipeline->args.args[ 1] = &id_src1;
                        pipeline->args.args[ 2] = &id_dst;
                        pipeline->args.args[ 3] = &ne00;pipeline->args.args[ 4] = &ne01;pipeline->args.args[ 5] = &ne02;pipeline->args.args[ 6] = &ne03;
                        pipeline->args.args[ 7] = &nb00;pipeline->args.args[ 8] = &nb01;pipeline->args.args[ 9] = &nb02;pipeline->args.args[10] = &nb03;
                        pipeline->args.args[11] = &ne10;pipeline->args.args[12] = &ne11;pipeline->args.args[13] = &ne12;pipeline->args.args[14] = &ne13;
                        pipeline->args.args[15] = &nb10;pipeline->args.args[16] = &nb11;pipeline->args.args[17] = &nb12;pipeline->args.args[18] = &nb13;
                        pipeline->args.args[19] = & ne0;pipeline->args.args[20] = & ne1;pipeline->args.args[21] = & ne2;pipeline->args.args[22] = & ne3;
                        pipeline->args.args[23] = & ne0;pipeline->args.args[24] = & ne1;pipeline->args.args[25] = & nb2;pipeline->args.args[26] = & nb3;
                        pipeline->args.args[27] = &  nb;
                        ggml_imax_kernel_queue_push(&(ctx->queue), pipeline);
                    } break;
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

                        const struct imax_kernel_pipeline* pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_ADD];
                        ggml_imax_kernel_queue_push(&(ctx->queue), pipeline);
                    } break;
                case GGML_OP_SCALE:
                    {
                        GGML_ASSERT(ggml_is_contiguous(src0));

                        const float scale = *(const float*) dst->op_params;

                        int64_t n = ggml_nelements(dst);

                        struct imax_kernel_pipeline* pipeline = NULL;

                        if (n % 4 == 0) {
                            n /= 4;
                            pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_SCALE_4];
                        } else {
                            pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_SCALE];
                        }
                        ggml_imax_kernel_queue_push(&(ctx->queue), pipeline);
                    } break;
                case GGML_OP_SUM_ROWS:
                    {
                        GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));

                        struct imax_kernel_pipeline* pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_SUM_ROWS];
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

                        // find the break-even point where the matrix-matrix kernel becomes more efficient compared
                        // to the matrix-vector kernel
                        int ne11_mm_min = 1;

                        // MM kernels on IMAX
                        // TODO: extend the matrix fit to the kernel size
                        if (!ggml_is_transposed(src0) &&
                            !ggml_is_transposed(src1) &&
                            src1t == GGML_TYPE_F32 &&
                            ne00 % 32 == 0 && ne00 >= 64 &&
                            (ne11 > ne11_mm_min || (ggml_is_quantized(src0t) && ne12 > 1))) {

                            struct imax_kernel_pipeline* pipeline = NULL;

                            switch (src0->type) {
                                case GGML_TYPE_F32:     pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_MM_F32_F32    ]; break;
                                case GGML_TYPE_F16:     pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_MM_F16_F32    ]; break;
                                case GGML_TYPE_Q4_0:    pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_MM_Q4_0_F32   ]; break;
                                case GGML_TYPE_Q4_1:    pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_MM_Q4_1_F32   ]; break;
                                case GGML_TYPE_Q5_0:    pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_MM_Q5_0_F32   ]; break;
                                case GGML_TYPE_Q5_1:    pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_MM_Q5_1_F32   ]; break;
                                case GGML_TYPE_Q8_0:    pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_MM_Q8_0_F32   ]; break;
                                case GGML_TYPE_Q2_K:    pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_MM_Q2_K_F32   ]; break;
                                case GGML_TYPE_Q3_K:    pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_MM_Q3_K_F32   ]; break;
                                case GGML_TYPE_Q4_K:    pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_MM_Q4_K_F32   ]; break;
                                case GGML_TYPE_Q5_K:    pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_MM_Q5_K_F32   ]; break;
                                case GGML_TYPE_Q6_K:    pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_MM_Q6_K_F32   ]; break;
                                case GGML_TYPE_IQ2_XXS: pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_MM_IQ2_XXS_F32]; break;
                                case GGML_TYPE_IQ2_XS:  pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_MM_IQ2_XS_F32 ]; break;
                                case GGML_TYPE_IQ3_XXS: pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_MM_IQ3_XXS_F32]; break;
                                case GGML_TYPE_IQ3_S:   pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_MM_IQ3_S_F32  ]; break;
                                case GGML_TYPE_IQ2_S:   pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_MM_IQ2_S_F32  ]; break;
                                case GGML_TYPE_IQ1_S:   pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_MM_IQ1_S_F32  ]; break;
                                case GGML_TYPE_IQ4_NL:  pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_MM_IQ4_NL_F32 ]; break;
                                case GGML_TYPE_IQ4_XS:  pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_MUL_MM_IQ4_XS_F32 ]; break;
                                default: GGML_ASSERT(false && "MUL MAT-MAT not implemented");
                            }
                            if (pipeline != NULL) ggml_imax_kernel_queue_push(&(ctx->queue), pipeline);
                        } else GGML_ASSERT(false && "MUL MAT-MAT not implemented");
                    } break;
                case GGML_OP_GET_ROWS:
                    {
                        struct imax_kernel_pipeline* pipeline = NULL;

                        switch (src0->type) {
                            case GGML_TYPE_F32:     pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_GET_ROWS_F32    ]; break;
                            case GGML_TYPE_F16:     pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_GET_ROWS_F16    ]; break;
                            case GGML_TYPE_Q4_0:    pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_GET_ROWS_Q4_0   ]; break;
                            case GGML_TYPE_Q4_1:    pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_GET_ROWS_Q4_1   ]; break;
                            case GGML_TYPE_Q5_0:    pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_GET_ROWS_Q5_0   ]; break;
                            case GGML_TYPE_Q5_1:    pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_GET_ROWS_Q5_1   ]; break;
                            case GGML_TYPE_Q8_0:    pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_GET_ROWS_Q8_0   ]; break;
                            case GGML_TYPE_Q2_K:    pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_GET_ROWS_Q2_K   ]; break;
                            case GGML_TYPE_Q3_K:    pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_GET_ROWS_Q3_K   ]; break;
                            case GGML_TYPE_Q4_K:    pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_GET_ROWS_Q4_K   ]; break;
                            case GGML_TYPE_Q5_K:    pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_GET_ROWS_Q5_K   ]; break;
                            case GGML_TYPE_Q6_K:    pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_GET_ROWS_Q6_K   ]; break;
                            case GGML_TYPE_IQ2_XXS: pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_GET_ROWS_IQ2_XXS]; break;
                            case GGML_TYPE_IQ2_XS:  pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_GET_ROWS_IQ2_XS ]; break;
                            case GGML_TYPE_IQ3_XXS: pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_GET_ROWS_IQ3_XXS]; break;
                            case GGML_TYPE_IQ3_S:   pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_GET_ROWS_IQ3_S  ]; break;
                            case GGML_TYPE_IQ2_S:   pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_GET_ROWS_IQ2_S  ]; break;
                            case GGML_TYPE_IQ1_S:   pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_GET_ROWS_IQ1_S  ]; break;
                            case GGML_TYPE_IQ4_NL:  pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_GET_ROWS_IQ4_NL ]; break;
                            case GGML_TYPE_IQ4_XS:  pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_GET_ROWS_IQ4_XS ]; break;
                            case GGML_TYPE_I32:     pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_GET_ROWS_I32    ]; break;
                            default: GGML_ASSERT(false && "not implemented");
                        }
                        if (pipeline != NULL) ggml_imax_kernel_queue_push(&(ctx->queue), pipeline);
                    } break;
                case GGML_OP_UPSCALE:
                    {
                        GGML_ASSERT(src0->type == GGML_TYPE_F32);

                        const int sf = dst->op_params[0];

                        const struct imax_kernel_pipeline* pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_UPSCALE_F32];
                        ggml_imax_kernel_queue_push(&(ctx->queue), pipeline);
                    } break;
                case GGML_OP_PAD:
                    {
                        GGML_ASSERT(src0->type == GGML_TYPE_F32);

                        struct imax_kernel_pipeline* pipeline = ctx->kernels[GGML_IMAX_KERNEL_TYPE_PAD_F32];
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

    // TODO: Wait for completion and check status of each command buffer
    for (int i = 0; i < n_cb; ++i) {
    }

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
    for (int i = 0; i < size / DMA_REG_SIZE; i++) {
        if (DMA_REG_SIZE * (i + 1) > size) {
            memcpy((char*)tensor->data + offset + DMA_REG_SIZE * i, data, size - DMA_REG_SIZE * i);
        } else {
            memcpy((char*)tensor->data + offset + DMA_REG_SIZE * i, data, DMA_REG_SIZE);
        }
    }

    UNUSED(buffer);
}

GGML_CALL static void ggml_backend_imax_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor* tensor, void* data, size_t offset, size_t size) {
    for (int i = 0; i < size / DMA_REG_SIZE; i++) {
        if (DMA_REG_SIZE * (i + 1) > size) {
            memcpy(data, (char*)tensor->data + offset + DMA_REG_SIZE * i, size - DMA_REG_SIZE * i);
        } else {
            memcpy(data, (char*)tensor->data + offset + DMA_REG_SIZE * i, DMA_REG_SIZE);
        }
    }

    UNUSED(buffer);
}

GGML_CALL static bool ggml_backend_imax_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor* src, struct ggml_tensor* dst) {
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

GGML_CALL static ggml_backend_buffer_t ggml_backend_imax_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    struct ggml_backend_imax_buffer_context* ctx = malloc(sizeof(struct ggml_backend_imax_buffer_context));

    const size_t size_page = DMA_REG_SIZE;

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    struct emax7* device = ggml_backend_imax_get_device();

    ctx->all_data = ggml_imax_host_malloc(size_aligned);
    ctx->all_size = size_aligned;
    ctx->owned = true;
    ctx->n_buffers = 1;

    ctx->buffers[0].data = ctx->all_data;
    ctx->buffers[0].size = size;

    GGML_IMAX_LOG_INFO("%s: allocated buffer, size = %8.2f MiB", __func__, size_aligned / 1024.0 / 1024.0);

    return ggml_backend_buffer_init(buft, ggml_backend_imax_buffer_i, ctx, size);
}

GGML_CALL static size_t ggml_backend_imax_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 32;
    UNUSED(buft);
}

GGML_CALL static size_t ggml_backend_imax_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    struct emax7* device = ggml_backend_imax_get_device();
    size_t max_size = DMA_MMAP_SIZE;

    return max_size;

    UNUSED(buft);
}

GGML_CALL static bool ggml_backend_imax_buffer_type_supports_backend(ggml_backend_buffer_type_t buft, ggml_backend_t backend) {
    return ggml_backend_is_imax(backend) || ggml_backend_is_cpu(backend);

    UNUSED(buft);
}

GGML_CALL static bool ggml_backend_imax_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

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

// TODO: buffer from ptr
GGML_CALL ggml_backend_buffer_t ggml_backend_imax_buffer_from_ptr(void* data, size_t size, size_t max_size) {
    struct ggml_backend_imax_buffer_context * ctx = malloc(sizeof(struct ggml_backend_imax_buffer_context));

    ctx->all_data = data;
    ctx->all_size = size;
    ctx->owned = false;
    ctx->n_buffers = 0;

    const size_t size_page = 1000; //Temp

    // page-align the data ptr
    {
        const uintptr_t offs = (uintptr_t) data % size_page;
        data  = (void*)((char*) data - offs);
        size += offs;
    }

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    struct emax7* device = ggml_backend_imax_get_device();

    // the buffer fits into the max buffer size allowed by the device
    if (size_aligned <= DMA_MMAP_SIZE) {
        ctx->buffers[ctx->n_buffers].data = data;
        ctx->buffers[ctx->n_buffers].size = size;
        GGML_IMAX_LOG_INFO("%s: allocated buffer, size = %8.2f MiB", __func__, size_aligned / 1024.0 / 1024.0);

        ++ctx->n_buffers;
    } else {
        // this overlap between the views will guarantee that the tensor with the maximum size will fully fit into
        // one of the views
        const size_t size_ovlp = ((max_size + size_page - 1) / size_page + 1) * size_page; // round-up 2 pages just in case
        const size_t size_step = DMA_MMAP_SIZE - size_ovlp;
        const size_t size_view = DMA_MMAP_SIZE;

        for (size_t i = 0; i < size; i += size_step) {
            const size_t size_step_aligned = (i + size_view <= size) ? size_view : (size_aligned - i);

            ctx->buffers[ctx->n_buffers].data = (void *) ((uint8_t *) data + i);
            ctx->buffers[ctx->n_buffers].size = size_step_aligned;

            GGML_IMAX_LOG_INFO("%s: allocated buffer, size = %8.2f MiB, offs = %12ld", __func__, size_step_aligned / 1024.0 / 1024.0, i);
            if (i + size_step < size) {
                GGML_IMAX_LOG_INFO("\n");
            }

            ++ctx->n_buffers;
        }
    }

    return ggml_backend_buffer_init(ggml_backend_imax_buffer_type(), ggml_backend_imax_buffer_i, ctx, size);
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
    struct ggml_imax_context* ctx = ggml_imax_init(GGML_DEFAULT_N_THREADS);

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
