#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "session.h"
#include "buffer.h"
#include "dispatch.h"
#include "data.h"
#include "state.h"
#include "validation.h"
#include "generated_vocab.h"

static void validate_softmax_v(const float* x, const float* v, float* o, int n) {
    float max_val = -FLT_MAX;
    float sum_val = 0.0f;
    float acc[256] = {0};

    for (int k = 0; k < n; k += 256) {
        int tile_size = (n - k < 256) ? (n - k) : 256;

        float tile_max = -FLT_MAX;
        for (int i = 0; i < tile_size; i++) {
            if (x[k + i] > tile_max) tile_max = x[k + i];
        }

        float exp_vals[256];
        float tile_sum = 0.0f;
        for (int i = 0; i < tile_size; i++) {
            exp_vals[i] = expf(x[k + i] - tile_max);
            tile_sum += exp_vals[i];
        }

        if (tile_sum > 0.0f) {
            float new_max = fmaxf(max_val, tile_max);
            float scale_prev = (sum_val == 0.0f) ? 0.0f : expf(max_val - new_max);
            float scale_tile = expf(tile_max - new_max);

            sum_val = sum_val * scale_prev + tile_sum * scale_tile;

            for (int col = 0; col < 256; col++) {
                float tile_acc = 0.0f;
                for (int i = 0; i < tile_size; i++) {
                    tile_acc += exp_vals[i] * v[(k + i) * 256 + col];
                }
                acc[col] = acc[col] * scale_prev + tile_acc * scale_tile;
            }
            max_val = new_max;
        }
    }

    for (int col = 0; col < 256; col++) {
        o[col] = (sum_val > 0.0f) ? acc[col] / sum_val : 0.0f;
    }
}

static void validate_attention(const float* q, const float* k, const float* v, float* o, int seq, int heads, int kv_heads, int dim) {
    for (int h = 0; h < heads; h++) {
        int kv = h / (heads / kv_heads);
        const float* qh = q + h * dim;
        const float* kh = k + kv * dim * seq;
        const float* vh = v + kv * dim * seq;

        float max_val = -FLT_MAX;
        float sum_val = 0.0f;
        float acc[256] = {0};

        for (int t = 0; t < seq; t += 256) {
            int tile = (seq - t < 256) ? (seq - t) : 256;

            float scores[256];
            float tile_max = -FLT_MAX;
            for (int i = 0; i < tile; i++) {
                float sc = 0.0f;
                for (int d = 0; d < dim; d++) sc += qh[d] * kh[d * seq + (t + i)];
                scores[i] = sc;
                if (sc > tile_max) tile_max = sc;
            }

            float expv[256];
            float tile_sum = 0.0f;
            for (int i = 0; i < tile; i++) {
                expv[i] = expf(scores[i] - tile_max);
                tile_sum += expv[i];
            }

            float new_max = fmaxf(max_val, tile_max);
            float scale_prev = (sum_val == 0.0f) ? 0.0f : expf(max_val - new_max);
            float scale_tile = expf(tile_max - new_max);
            sum_val = sum_val * scale_prev + tile_sum * scale_tile;

            for (int d = 0; d < dim; d++) {
                float tile_acc = 0.0f;
                for (int i = 0; i < tile; i++) tile_acc += expv[i] * vh[d * seq + (t + i)];
                acc[d] = acc[d] * scale_prev + tile_acc * scale_tile;
            }
            max_val = new_max;
        }

        for (int d = 0; d < dim; d++) o[h * dim + d] = (sum_val > 0.0f) ? acc[d] / sum_val : 0.0f;
    }
}

static void validate_attention_multi(const float* q, const float* k, const float* v, float* o, int seq, int heads, int kv_heads, int dim) {
    for (int m = 0; m < seq; m++)
        for (int h = 0; h < heads; h++) {
            int kv = h / (heads / kv_heads);
            const float* qh = q + (m * heads + h) * dim;
            const float* kh = k + kv * dim * seq;
            const float* vh = v + kv * dim * seq;
            float max_val = -FLT_MAX, sum_val = 0.0f;
            float acc[256] = {0};
            for (int t = 0; t <= m; t += 256) {
                int tile = ((m - t + 1) < 256) ? (m - t + 1) : 256;
                float scores[256]; float tile_max = -FLT_MAX;
                for (int i = 0; i < tile; i++) {
                    float sc = 0.0f;
                    for (int d = 0; d < dim; d++) sc += qh[d] * kh[d * seq + (t + i)];
                    scores[i] = sc; if (sc > tile_max) tile_max = sc;
                }
                float expv[256]; float tile_sum = 0.0f;
                for (int i = 0; i < tile; i++) { expv[i] = expf(scores[i] - tile_max); tile_sum += expv[i]; }
                float new_max = fmaxf(max_val, tile_max);
                float sp = (sum_val == 0.0f) ? 0.0f : expf(max_val - new_max);
                float st = expf(tile_max - new_max);
                sum_val = sum_val * sp + tile_sum * st;
                for (int d = 0; d < dim; d++) {
                    float ta = 0.0f;
                    for (int i = 0; i < tile; i++) ta += expv[i] * vh[d * seq + (t + i)];
                    acc[d] = acc[d] * sp + ta * st;
                }
                max_val = new_max;
            }
            for (int d = 0; d < dim; d++) o[(m * heads + h) * dim + d] = (sum_val > 0.0f) ? acc[d] / sum_val : 0.0f;
        }
}

static double run_ops(session s, operation ops[], int n) {
    execute(s, ops, n);
    return getExecutionTime(s);
}

static void destroy_buffers(session s, buffer bufs[], int n) {
    for (int i = 0; i < n; i++) destroyBuffer(s.dev.device, bufs[i]);
}

static void report(const char* name, int idx, const float* out, const float* ref, int count, double ms) {
    float err = 0.0f;
    for (int i = 0; i < count; i++) {
        float e = fabsf(out[i] - ref[i]);
        if (e > err) err = e;
    }
    printf("%s: shader[%d]= %f ref[%d]= %f max_err= %f\n", name, idx, out[idx], idx, ref[idx], err);
    printf("%s time: %.3f ms\n", name, ms);
}

static void rms_norm_apply(const float* x, const float* g, float* xn, int k) {
    float sum = 0.0f;
    for (int i = 0; i < k; i++) sum += x[i] * x[i];
    float rms = sqrtf(sum / (float)k + 1e-6f);
    float inv = 1.0f / rms;
    for (int i = 0; i < k; i++) xn[i] = x[i] * g[i] * inv;
}

static void gemv_ref_f32(const float* x, const float* w, float* o, int n, int k) {
    for (int j = 0; j < n; j++) {
        float acc = 0.0f;
        for (int i = 0; i < k; i++) acc += x[i] * w[i * n + j];
        o[j] = acc;
    }
}

static void gemv_ref_fp16(const float* x, const uint16_t* w, float* o, int n, int k) {
    for (int j = 0; j < n; j++) {
        float acc = 0.0f;
        for (int i = 0; i < k; i++) acc += x[i] * fp16_to_float(w[i * n + j]);
        o[j] = acc;
    }
}

static void gemv_ref_int8(const float* x, const QuantizedData* q, float* o, int n, int k) {
    for (int j = 0; j < n; j++) {
        float acc = 0.0f;
        int bj = j / q->group_size;
        for (int i = 0; i < k; i++) {
            float dq = (float)q->data[i * n + j] * q->scale[bj * k + i] - q->z[bj * k + i];
            acc += x[i] * dq;
        }
        o[j] = acc;
    }
}

static void gemv_ref_int4(const float* x, const QuantizedData* q, float* o, int n, int k) {
    for (int j = 0; j < n; j++) {
        float acc = 0.0f;
        int bj = j / q->group_size;
        for (int i = 0; i < k; i++) {
            uint8_t b = q->data[i * (n / 2) + j / 2];
            int nib = (j & 1) ? (b & 0x0F) : (b >> 4);
            float dq = (float)nib * q->scale[bj * k + i] - q->z[bj * k + i];
            acc += x[i] * dq;
        }
        o[j] = acc;
    }
}

static void gemm_ref_fp16(const float* x, const uint16_t* w, float* o, int m, int n, int k) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            for (int t = 0; t < k; t++) acc += x[i * k + t] * fp16_to_float(w[t * n + j]);
            o[i * n + j] = acc;
        }
}

static void gemm_ref_int8(const float* x, const QuantizedData* q, float* o, int m, int n, int k) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            int bj = j / q->group_size;
            for (int t = 0; t < k; t++) {
                float dq = (float)q->data[t * n + j] * q->scale[bj * k + t] - q->z[bj * k + t];
                acc += x[i * k + t] * dq;
            }
            o[i * n + j] = acc;
        }
}

static void gemm_ref_int4(const float* x, const QuantizedData* q, float* o, int m, int n, int k) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float acc = 0.0f;
            int bj = j / q->group_size;
            for (int t = 0; t < k; t++) {
                uint8_t b = q->data[t * (n / 2) + j / 2];
                int nib = (j & 1) ? (b & 0x0F) : (b >> 4);
                acc += x[i * k + t] * ((float)nib * q->scale[bj * k + t] - q->z[bj * k + t]);
            }
            o[i * n + j] = acc;
        }
}

static void swiglu_ref_f32(const float* x, const float* g, const float* wg, const float* wu, float* o, int n, int k) {
    float* xn = (float*)malloc(sizeof(float) * k);
    rms_norm_apply(x, g, xn, k);
    for (int j = 0; j < n; j++) {
        float gate = 0.0f;
        float up = 0.0f;
        for (int i = 0; i < k; i++) {
            gate += xn[i] * wg[i * n + j];
            up += xn[i] * wu[i * n + j];
        }
        o[j] = (gate / (1.0f + exp2f(-gate * 1.44269504f))) * up;
    }
    free(xn);
}

static void swiglu_ref_fp16(const float* x, const float* g, const uint16_t* wg, const uint16_t* wu, float* o, int n, int k) {
    float* xn = (float*)malloc(sizeof(float) * k);
    rms_norm_apply(x, g, xn, k);
    for (int j = 0; j < n; j++) {
        float gate = 0.0f;
        float up = 0.0f;
        for (int i = 0; i < k; i++) {
            gate += xn[i] * fp16_to_float(wg[i * n + j]);
            up += xn[i] * fp16_to_float(wu[i * n + j]);
        }
        o[j] = (gate / (1.0f + exp2f(-gate * 1.44269504f))) * up;
    }
    free(xn);
}

static void swiglu_ref_int8(const float* x, const float* g, const QuantizedData* qg, const QuantizedData* qu, float* o, int n, int k) {
    float* xn = (float*)malloc(sizeof(float) * k);
    rms_norm_apply(x, g, xn, k);
    for (int j = 0; j < n; j++) {
        float gate = 0.0f;
        float up = 0.0f;
        int bj = j / qg->group_size;
        for (int i = 0; i < k; i++) {
            float dg = (float)qg->data[i * n + j] * qg->scale[bj * k + i] - qg->z[bj * k + i];
            float du = (float)qu->data[i * n + j] * qu->scale[bj * k + i] - qu->z[bj * k + i];
            gate += xn[i] * dg;
            up += xn[i] * du;
        }
        o[j] = (gate / (1.0f + exp2f(-gate * 1.44269504f))) * up;
    }
    free(xn);
}

static void swiglu_ref_int4(const float* x, const float* g, const QuantizedData* qg, const QuantizedData* qu, float* o, int n, int k) {
    float* xn = (float*)malloc(sizeof(float) * k);
    rms_norm_apply(x, g, xn, k);
    for (int j = 0; j < n; j++) {
        float gate = 0.0f;
        float up = 0.0f;
        int bj = j / qg->group_size;
        for (int i = 0; i < k; i++) {
            uint8_t bg = qg->data[i * (n / 2) + j / 2];
            uint8_t bu = qu->data[i * (n / 2) + j / 2];
            int ng = (j & 1) ? (bg & 0x0F) : (bg >> 4);
            int nu = (j & 1) ? (bu & 0x0F) : (bu >> 4);
            float dg = (float)ng * qg->scale[bj * k + i] - qg->z[bj * k + i];
            float du = (float)nu * qu->scale[bj * k + i] - qu->z[bj * k + i];
            gate += xn[i] * dg;
            up += xn[i] * du;
        }
        o[j] = (gate / (1.0f + exp2f(-gate * 1.44269504f))) * up;
    }
    free(xn);
}

void validateGEMV(session s, int M, int N, int K, float* input, float* weight) {
    float* out = (float*)calloc(M * N, sizeof(float));

    float* transposed = (float*)malloc(sizeof(float) * K * N);
    transpose_block16((uint8_t*)weight, (uint8_t*)transposed, K, N, QUANT_FP32);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(float) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, weightBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 3);
    free(transposed);

    operation ops[] = {
        {.shader = "GEMV.spv", .buffers = {inputBuffer, weightBuffer, outBuffer}, .bufferCount = 3,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = (N + 255) / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemv_ref_f32(input, weight, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMV", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 3);
    free(out);
    free(ref);
}

void validateGEMVFP16(session s, int M, int N, int K, float* input, uint16_t* weightFP16) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint16_t* transposed = (uint16_t*)malloc(sizeof(uint16_t) * K * N);
    transpose_block16((uint8_t*)weightFP16, (uint8_t*)transposed, K, N, QUANT_FP16);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint16_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, weightBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 3);
    free(transposed);

    operation ops[] = {
        {.shader = "GEMV-FP16.spv", .buffers = {inputBuffer, weightBuffer, outBuffer}, .bufferCount = 3,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = (N + 255) / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemv_ref_fp16(input, weightFP16, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMV-FP16", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 3);
    free(out);
    free(ref);
}

void validateGEMVINT8(session s, int M, int N, int K, float* input, QuantizedData weightINT8) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N);
    transpose_block16(weightINT8.data, transposed, K, N, QUANT_INT8);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.scale, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.z, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 5);
    free(transposed);

    operation ops[] = {
        {.shader = "GEMV-INT8.spv", .buffers = {inputBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer}, .bufferCount = 5,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = (N + 255) / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemv_ref_int8(input, &weightINT8, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMV-INT8", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 5);
    free(out);
    free(ref);
}

void validateGEMVINT4(session s, int M, int N, int K, float* input, QuantizedData weightINT4) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N / 2);
    transpose_block16(weightINT4.data, transposed, K, N, QUANT_INT4);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N / 2, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.scale, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.z, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 5);
    free(transposed);

    operation ops[] = {
        {.shader = "GEMV-INT4.spv", .buffers = {inputBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer}, .bufferCount = 5,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = (N + 255) / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemv_ref_int4(input, &weightINT4, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMV-INT4", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 5);
    free(out);
    free(ref);
}

void validateGEMMFP16(session s, int M, int N, int K, float* input, uint16_t* weightFP16) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint16_t* transposed = (uint16_t*)malloc(sizeof(uint16_t) * K * N);
    transpose_block16((uint8_t*)weightFP16, (uint8_t*)transposed, K, N, QUANT_FP16);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint16_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, weightBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 3);
    free(transposed);

    operation ops[] = {
        {.shader = "GEMM-FP16.spv", .buffers = {inputBuffer, weightBuffer, outBuffer}, .bufferCount = 3,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = N / 64, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemm_ref_fp16(input, weightFP16, ref, M, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMM-FP16", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 3);
    free(out);
    free(ref);
}

void validateGEMMINT8(session s, int M, int N, int K, float* input, QuantizedData weightINT8) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N);
    transpose_block16(weightINT8.data, transposed, K, N, QUANT_INT8);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.scale, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.z, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 5);
    free(transposed);

    operation ops[] = {
        {.shader = "GEMM-INT8.spv", .buffers = {inputBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer}, .bufferCount = 5,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = N / 64, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemm_ref_int8(input, &weightINT8, ref, M, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMM-INT8", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 5);
    free(out);
    free(ref);
}

void validateGEMMINT4(session s, int M, int N, int K, float* input, QuantizedData weightINT4) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N / 2);
    transpose_block16(weightINT4.data, transposed, K, N, QUANT_INT4);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N / 2, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.scale, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.z, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 5);
    free(transposed);

    operation ops[] = {
        {.shader = "GEMM-INT4.spv", .buffers = {inputBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer}, .bufferCount = 5,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = N / 64, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemm_ref_int4(input, &weightINT4, ref, M, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMM-INT4", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 5);
    free(out);
    free(ref);
}

void validateRmsNormGEMVFP16(session s, int M, int N, int K, float* input, float* gamma, uint16_t* weightFP16) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint16_t* transposed = (uint16_t*)malloc(sizeof(uint16_t) * K * N);
    transpose_block16((uint8_t*)weightFP16, (uint8_t*)transposed, K, N, QUANT_FP16);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint16_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, gammaBuffer, weightBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 4);
    free(transposed);

    operation ops[] = {
        {.shader = "RmsNorm-GEMV-FP16.spv", .buffers = {inputBuffer, gammaBuffer, weightBuffer, outBuffer}, .bufferCount = 4,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = (N + 255) / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* xn = (float*)malloc(sizeof(float) * K);
    rms_norm_apply(input, gamma, xn, K);
    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemv_ref_fp16(xn, weightFP16, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("RmsNorm-GEMV-FP16", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 4);
    free(out);
    free(ref);
    free(xn);
}

void validateRmsNormGEMVINT8(session s, int M, int N, int K, float* input, float* gamma, QuantizedData weightINT8) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N);
    transpose_block16(weightINT8.data, transposed, K, N, QUANT_INT8);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.scale, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.z, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, gammaBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 6);
    free(transposed);

    operation ops[] = {
        {.shader = "RmsNorm-GEMV-INT8.spv", .buffers = {inputBuffer, gammaBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer}, .bufferCount = 6,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = (N + 255) / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* xn = (float*)malloc(sizeof(float) * K);
    rms_norm_apply(input, gamma, xn, K);
    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemv_ref_int8(xn, &weightINT8, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("RmsNorm-GEMV-INT8", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 6);
    free(out);
    free(ref);
    free(xn);
}

void validateRmsNormGEMVINT4(session s, int M, int N, int K, float* input, float* gamma, QuantizedData weightINT4) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N / 2);
    transpose_block16(weightINT4.data, transposed, K, N, QUANT_INT4);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N / 2, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.scale, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.z, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, gammaBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 6);
    free(transposed);

    operation ops[] = {
        {.shader = "RmsNorm-GEMV-INT4.spv", .buffers = {inputBuffer, gammaBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer}, .bufferCount = 6,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = (N + 255) / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* xn = (float*)malloc(sizeof(float) * K);
    rms_norm_apply(input, gamma, xn, K);
    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemv_ref_int4(xn, &weightINT4, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("RmsNorm-GEMV-INT4", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 6);
    free(out);
    free(ref);
    free(xn);
}

void validateGemvAddFP16(session s, int M, int N, int K, float* input, float* residual, uint16_t* weightFP16) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint16_t* transposed = (uint16_t*)malloc(sizeof(uint16_t) * K * N);
    transpose_block16((uint8_t*)weightFP16, (uint8_t*)transposed, K, N, QUANT_FP16);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint16_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer residualBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, residual, sizeof(float) * N, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, weightBuffer, outBuffer, residualBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 4);
    free(transposed);

    operation ops[] = {
        {.shader = "GEMV-ADD-FP16.spv", .buffers = {inputBuffer, weightBuffer, outBuffer, residualBuffer}, .bufferCount = 4,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = (N + 255) / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemv_ref_fp16(input, weightFP16, ref, N, K);
    for (int j = 0; j < N; j++) ref[j] += residual[j];
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMV-ADD FP16", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 4);
    free(out);
    free(ref);
}

void validateGemvAddINT8(session s, int M, int N, int K, float* input, float* residual, QuantizedData weightINT8) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N);
    transpose_block16(weightINT8.data, transposed, K, N, QUANT_INT8);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.scale, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.z, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer residualBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, residual, sizeof(float) * N, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer, residualBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 6);
    free(transposed);

    operation ops[] = {
        {.shader = "GEMV-ADD-INT8.spv", .buffers = {inputBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer, residualBuffer}, .bufferCount = 6,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = (N + 255) / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemv_ref_int8(input, &weightINT8, ref, N, K);
    for (int j = 0; j < N; j++) ref[j] += residual[j];
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMV-ADD INT8", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 6);
    free(out);
    free(ref);
}

void validateGemvAddINT4(session s, int M, int N, int K, float* input, float* residual, QuantizedData weightINT4) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N / 2);
    transpose_block16(weightINT4.data, transposed, K, N, QUANT_INT4);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N / 2, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.scale, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.z, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer residualBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, residual, sizeof(float) * N, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer, residualBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 6);
    free(transposed);

    operation ops[] = {
        {.shader = "GEMV-ADD-INT4.spv", .buffers = {inputBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer, residualBuffer}, .bufferCount = 6,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = (N + 255) / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemv_ref_int4(input, &weightINT4, ref, N, K);
    for (int j = 0; j < N; j++) ref[j] += residual[j];
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMV-ADD INT4", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 6);
    free(out);
    free(ref);
}

void validateGemvSplitKINT4(session s, int M, int N, int K, float* input, float* residual, QuantizedData weightINT4) {
    float* out = (float*)calloc(M * N, sizeof(float));
    float* partials = (float*)calloc(4 * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N / 2);
    transpose_block16(weightINT4.data, transposed, K, N, QUANT_INT4);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N / 2, MEMORY_RAM);
    buffer partialBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, partials, sizeof(float) * 4 * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.scale, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.z, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer residualBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, residual, sizeof(float) * N, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, weightBuffer, partialBuffer, outBuffer, scaleBuffer, zeroBuffer, residualBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 7);
    free(transposed);

    operation ops[] = {
        {.shader = "GEMV-SplitK-INT4.spv", .buffers = {inputBuffer, weightBuffer, partialBuffer, scaleBuffer, zeroBuffer}, .bufferCount = 5,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = (N + 255) / 256, .dispatchY = 4, .dispatchZ = 1},
        {.shader = "Reduce-GEMV-ADD.spv", .buffers = {partialBuffer, residualBuffer, outBuffer}, .bufferCount = 3,
         .pushConstants = {N, 4}, .pushConstantCount = 2,
         .dispatchX = (N + 255) / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 2);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemv_ref_int4(input, &weightINT4, ref, N, K);
    for (int j = 0; j < N; j++) ref[j] += residual[j];
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMV SplitK INT4", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 7);
    free(out);
    free(ref);
    free(partials);
}

void validateGemvSplitKINT8(session s, int M, int N, int K, float* input, float* residual, QuantizedData weightINT8) {
    float* out = (float*)calloc(M * N, sizeof(float));
    float* partials = (float*)calloc(4 * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N);
    transpose_block16(weightINT8.data, transposed, K, N, QUANT_INT8);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N, MEMORY_RAM);
    buffer partialBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, partials, sizeof(float) * 4 * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.scale, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.z, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer residualBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, residual, sizeof(float) * N, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, weightBuffer, partialBuffer, outBuffer, scaleBuffer, zeroBuffer, residualBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 7);
    free(transposed);

    operation ops[] = {
        {.shader = "GEMV-SplitK-INT8.spv", .buffers = {inputBuffer, weightBuffer, partialBuffer, scaleBuffer, zeroBuffer}, .bufferCount = 5,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = (N + 255) / 256, .dispatchY = 4, .dispatchZ = 1},
        {.shader = "Reduce-GEMV-ADD.spv", .buffers = {partialBuffer, residualBuffer, outBuffer}, .bufferCount = 3,
         .pushConstants = {N, 4}, .pushConstantCount = 2,
         .dispatchX = (N + 255) / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 2);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemv_ref_int8(input, &weightINT8, ref, N, K);
    for (int j = 0; j < N; j++) ref[j] += residual[j];
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMV SplitK INT8", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 7);
    free(out);
    free(ref);
    free(partials);
}

void validateGemvSplitKFP16(session s, int M, int N, int K, float* input, float* residual, uint16_t* weightFP16) {
    float* out = (float*)calloc(M * N, sizeof(float));
    float* partials = (float*)calloc(4 * N, sizeof(float));

    uint16_t* transposed = (uint16_t*)malloc(sizeof(uint16_t) * K * N);
    transpose_block16((uint8_t*)weightFP16, (uint8_t*)transposed, K, N, QUANT_FP16);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint16_t) * K * N, MEMORY_RAM);
    buffer partialBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, partials, sizeof(float) * 4 * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer residualBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, residual, sizeof(float) * N, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, weightBuffer, partialBuffer, outBuffer, residualBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 5);
    free(transposed);

    operation ops[] = {
        {.shader = "GEMV-SplitK-FP16.spv", .buffers = {inputBuffer, weightBuffer, partialBuffer}, .bufferCount = 3,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = (N + 255) / 256, .dispatchY = 4, .dispatchZ = 1},
        {.shader = "Reduce-GEMV-ADD.spv", .buffers = {partialBuffer, residualBuffer, outBuffer}, .bufferCount = 3,
         .pushConstants = {N, 4}, .pushConstantCount = 2,
         .dispatchX = (N + 255) / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 2);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemv_ref_fp16(input, weightFP16, ref, N, K);
    for (int j = 0; j < N; j++) ref[j] += residual[j];
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMV SplitK FP16", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 5);
    free(out);
    free(ref);
    free(partials);
}

void validateRmsNormSwigluFfn(session s, int M, int N, int K, float* input, float* gamma, float* weight) {
    float* out = (float*)calloc(M * N, sizeof(float));

    float* transposed = (float*)malloc(sizeof(float) * K * N);
    transpose_block16((uint8_t*)weight, (uint8_t*)transposed, K, N, QUANT_FP32);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(float) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, gammaBuffer, weightBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 4);
    free(transposed);

    operation ops[] = {
        {.shader = "RmsNorm-swiglu-ffn.spv", .buffers = {inputBuffer, gammaBuffer, weightBuffer, weightBuffer, outBuffer}, .bufferCount = 5,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = (N + 255) / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    swiglu_ref_f32(input, gamma, weight, weight, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("RmsNorm-swiglu-ffn", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 4);
    free(out);
    free(ref);
}

void validateRmsNormSwigluFfnFP16(session s, int M, int N, int K, float* input, float* gamma, uint16_t* weightFP16, uint16_t* weight2FP16) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint16_t* transposed = (uint16_t*)malloc(sizeof(uint16_t) * K * N);
    uint16_t* transposed2 = (uint16_t*)malloc(sizeof(uint16_t) * K * N);
    transpose_block16((uint8_t*)weightFP16, (uint8_t*)transposed, K, N, QUANT_FP16);
    transpose_block16((uint8_t*)weight2FP16, (uint8_t*)transposed2, K, N, QUANT_FP16);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * M * K, MEMORY_RAM);
    buffer gateBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint16_t) * K * N, MEMORY_RAM);
    buffer upBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed2, sizeof(uint16_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 5);
    free(transposed);
    free(transposed2);

    operation ops[] = {
        {.shader = "RmsNorm-swiglu-ffn-FP16.spv", .buffers = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, outBuffer}, .bufferCount = 5,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = (N + 255) / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    swiglu_ref_fp16(input, gamma, weightFP16, weight2FP16, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("RmsNorm-swiglu-ffn-FP16", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 5);
    free(out);
    free(ref);
}

void validateRmsNormSwigluFfnINT8(session s, int M, int N, int K, float* input, float* gamma, QuantizedData weightINT8, QuantizedData weight2INT8) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N);
    uint8_t* transposed2 = (uint8_t*)malloc(sizeof(uint8_t) * K * N);
    transpose_block16(weightINT8.data, transposed, K, N, QUANT_INT8);
    transpose_block16(weight2INT8.data, transposed2, K, N, QUANT_INT8);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * M * K, MEMORY_RAM);
    buffer gateBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N, MEMORY_RAM);
    buffer upBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed2, sizeof(uint8_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer gateScale = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.scale, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer gateZero = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.z, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer upScale = createBuffer(s.dev.device, s.dev.physicalDevice, weight2INT8.scale, sizeof(float) * K * N / weight2INT8.group_size, MEMORY_RAM);
    buffer upZero = createBuffer(s.dev.device, s.dev.physicalDevice, weight2INT8.z, sizeof(float) * K * N / weight2INT8.group_size, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, outBuffer, gateScale, gateZero, upScale, upZero};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 9);
    free(transposed);
    free(transposed2);

    operation ops[] = {
        {.shader = "RmsNorm-swiglu-ffn-INT8.spv", .buffers = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, outBuffer, gateScale, gateZero, upScale, upZero}, .bufferCount = 9,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = (N + 255) / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    swiglu_ref_int8(input, gamma, &weightINT8, &weight2INT8, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("RmsNorm-swiglu-ffn-INT8", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 9);
    free(out);
    free(ref);
}

void validateRmsNormSwigluFfnINT4(session s, int M, int N, int K, float* input, float* gamma, QuantizedData weightINT4, QuantizedData weight2INT4) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N / 2);
    uint8_t* transposed2 = (uint8_t*)malloc(sizeof(uint8_t) * K * N / 2);
    transpose_block16(weightINT4.data, transposed, K, N, QUANT_INT4);
    transpose_block16(weight2INT4.data, transposed2, K, N, QUANT_INT4);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * M * K, MEMORY_RAM);
    buffer gateBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N / 2, MEMORY_RAM);
    buffer upBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed2, sizeof(uint8_t) * K * N / 2, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer gateScale = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.scale, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer gateZero = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.z, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer upScale = createBuffer(s.dev.device, s.dev.physicalDevice, weight2INT4.scale, sizeof(float) * K * N / weight2INT4.group_size, MEMORY_RAM);
    buffer upZero = createBuffer(s.dev.device, s.dev.physicalDevice, weight2INT4.z, sizeof(float) * K * N / weight2INT4.group_size, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, outBuffer, gateScale, gateZero, upScale, upZero};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 9);
    free(transposed);
    free(transposed2);

    operation ops[] = {
        {.shader = "RmsNorm-swiglu-ffn-INT4.spv", .buffers = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, outBuffer, gateScale, gateZero, upScale, upZero}, .bufferCount = 9,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = (N + 255) / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    swiglu_ref_int4(input, gamma, &weightINT4, &weight2INT4, ref, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("RmsNorm-swiglu-ffn-INT4", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 9);
    free(out);
    free(ref);
}

void validateOnlineSoftmax(session s, int softmax_n, float* softmax_x, float* softmax_v) {
    int n = softmax_n;
    float* out = (float*)calloc(256, sizeof(float));

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, softmax_x, sizeof(float) * n, MEMORY_VRAM);
    buffer vBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, softmax_v, sizeof(float) * n * 256, MEMORY_VRAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * 256, MEMORY_VRAM);
    buffer bufs[] = {xBuffer, vBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 3);

    operation ops[] = {
        {.shader = "online-softmax.spv", .buffers = {xBuffer, vBuffer, outBuffer}, .bufferCount = 3,
         .pushConstants = {n}, .pushConstantCount = 1,
         .dispatchX = 1, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * 256);
    validate_softmax_v(softmax_x, softmax_v, ref, n);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("online-softmax", 40, out, ref, 256, ms);

    destroy_buffers(s, bufs, 3);
    free(out);
    free(ref);
}

void validateAttentionFP16(session s, int att_seq, int att_heads, int att_kv_heads, int att_dim, float* att_q, uint16_t* att_k, uint16_t* att_v) {
    int seq = att_seq;
    int heads = att_heads;
    int kv_heads = att_kv_heads;
    int dim = att_dim;
    int rows = kv_heads * dim;
    int kvs = rows * seq;
    float* kf = (float*)malloc(sizeof(float) * kvs);
    float* vf = (float*)malloc(sizeof(float) * kvs);
    for (int i = 0; i < kvs; i++) {
        kf[i] = fp16_to_float(att_k[i]);
        vf[i] = fp16_to_float(att_v[i]);
    }
    uint16_t* att_k_t = (uint16_t*)calloc((size_t)rows * 32768, sizeof(uint16_t));
    uint16_t* att_v_t = (uint16_t*)malloc(sizeof(uint16_t) * kvs);
    for (int s2 = 0; s2 < seq; s2++) {
        for (int r = 0; r < rows; r++) {
            att_k_t[r * 32768 + s2] = att_k[r * seq + s2];
            att_v_t[s2 * rows + r] = att_v[r * seq + s2];
        }
    }
    float* out = (float*)calloc(heads * dim, sizeof(float));

    uint32_t posVal = (uint32_t)(seq - 1);
    buffer keyBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_k_t, sizeof(uint16_t) * (size_t)rows * 32768, MEMORY_RAM);
    buffer valueBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_v_t, sizeof(uint16_t) * kvs, MEMORY_RAM);
    buffer queryBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_q, sizeof(float) * heads * dim, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * heads * dim, MEMORY_RAM);
    buffer posBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, &posVal, sizeof(uint32_t), MEMORY_VRAM);
    buffer bufs[] = {keyBuffer, valueBuffer, queryBuffer, outBuffer, posBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 5);
    free(att_k_t);
    free(att_v_t);

    operation ops[] = {
        {.shader = "Att-full-FP16.spv", .buffers = {keyBuffer, valueBuffer, queryBuffer, outBuffer, posBuffer}, .bufferCount = 5,
         .pushConstants = {32768, 4, 1024, 4, 256, 16}, .pushConstantCount = 6,
         .dispatchX = heads, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * heads * dim);
    validate_attention(att_q, kf, vf, ref, seq, heads, kv_heads, dim);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("Attention FP16", 100, out, ref, heads * dim, ms);

    destroy_buffers(s, bufs, 5);
    free(kf);
    free(vf);
    free(out);
    free(ref);
}

void validateAttentionINT8(session s, int att_seq, int att_heads, int att_kv_heads, int att_dim, float* att_q, uint16_t* att_k, uint16_t* att_v) {
    int seq = att_seq;
    int heads = att_heads;
    int kv_heads = att_kv_heads;
    int dim = att_dim;
    int rows = kv_heads * dim;
    int kvs = rows * seq;
    uint8_t* kq = (uint8_t*)malloc(kvs);
    uint8_t* vq = (uint8_t*)malloc(kvs);
    float* kScale = (float*)malloc(sizeof(float) * kv_heads * 32768);
    float* kZero = (float*)malloc(sizeof(float) * kv_heads * 32768);
    float* vScale = (float*)malloc(sizeof(float) * kv_heads * 32768);
    float* vZero = (float*)malloc(sizeof(float) * kv_heads * 32768);
    for (int h = 0; h < kv_heads; h++) {
        for (int s2 = 0; s2 < seq; s2++) {
            float mn = FLT_MAX;
            float mx = -FLT_MAX;
            for (int d = 0; d < dim; d++) {
                float kv = fp16_to_float(att_k[(h * dim + d) * seq + s2]);
                mn = fminf(mn, kv);
                mx = fmaxf(mx, kv);
            }
            float ks = (mx - mn) / 255.0f;
            if (ks == 0.0f) ks = 1.0f;
            kScale[h * 32768 + s2] = ks;
            kZero[h * 32768 + s2] = -mn;
            for (int d = 0; d < dim; d++) {
                float kv = fp16_to_float(att_k[(h * dim + d) * seq + s2]);
                int q = (int)rintf((kv + kZero[h * 32768 + s2]) / ks);
                if (q < 0) q = 0;
                if (q > 255) q = 255;
                kq[s2 * rows + (h * dim + d)] = (uint8_t)q;
            }
        }
    }
    for (int h = 0; h < kv_heads; h++) {
        for (int s2 = 0; s2 < seq; s2++) {
            float mn = FLT_MAX;
            float mx = -FLT_MAX;
            for (int d = 0; d < dim; d++) {
                float vv = fp16_to_float(att_v[(h * dim + d) * seq + s2]);
                mn = fminf(mn, vv);
                mx = fmaxf(mx, vv);
            }
            float vs = (mx - mn) / 255.0f;
            if (vs == 0.0f) vs = 1.0f;
            vScale[h * 32768 + s2] = vs;
            vZero[h * 32768 + s2] = -mn;
            for (int d = 0; d < dim; d++) {
                float vv = fp16_to_float(att_v[(h * dim + d) * seq + s2]);
                int q = (int)rintf((vv + vZero[h * 32768 + s2]) / vs);
                if (q < 0) q = 0;
                if (q > 255) q = 255;
                vq[s2 * rows + (h * dim + d)] = (uint8_t)q;
            }
        }
    }
    float* kf = (float*)malloc(sizeof(float) * kvs);
    float* vf = (float*)malloc(sizeof(float) * kvs);
    for (int r = 0; r < rows; r++) {
        int h = r / dim;
        for (int s2 = 0; s2 < seq; s2++) {
            kf[r * seq + s2] = (float)kq[s2 * rows + r] * kScale[h * 32768 + s2] - kZero[h * 32768 + s2];
            vf[r * seq + s2] = (float)vq[s2 * rows + r] * vScale[h * 32768 + s2] - vZero[h * 32768 + s2];
        }
    }
    float* out = (float*)calloc(heads * dim, sizeof(float));

    uint32_t posVal = (uint32_t)(seq - 1);
    uint8_t* kq_t = (uint8_t*)calloc((size_t)rows * 32768, sizeof(uint8_t));
    for (int s2 = 0; s2 < seq; s2++)
        for (int r = 0; r < rows; r++)
            kq_t[r * 32768 + s2] = kq[s2 * rows + r];
    buffer keyBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kq_t, sizeof(uint8_t) * (size_t)rows * 32768, MEMORY_RAM);
    free(kq_t);
    buffer valueBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vq, sizeof(uint8_t) * kvs, MEMORY_RAM);
    buffer queryBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_q, sizeof(float) * heads * dim, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * heads * dim, MEMORY_RAM);
    buffer kScaleBuf = createBuffer(s.dev.device, s.dev.physicalDevice, kScale, sizeof(float) * kv_heads * 32768, MEMORY_RAM);
    buffer kZeroBuf = createBuffer(s.dev.device, s.dev.physicalDevice, kZero, sizeof(float) * kv_heads * 32768, MEMORY_RAM);
    buffer vScaleBuf = createBuffer(s.dev.device, s.dev.physicalDevice, vScale, sizeof(float) * kv_heads * 32768, MEMORY_RAM);
    buffer vZeroBuf = createBuffer(s.dev.device, s.dev.physicalDevice, vZero, sizeof(float) * kv_heads * 32768, MEMORY_RAM);
    buffer posBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, &posVal, sizeof(uint32_t), MEMORY_VRAM);
    buffer bufs[] = {keyBuffer, valueBuffer, queryBuffer, outBuffer, kScaleBuf, kZeroBuf, vScaleBuf, vZeroBuf, posBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 9);

    operation ops[] = {
        {.shader = "Att-full-INT8.spv", .buffers = {keyBuffer, valueBuffer, queryBuffer, outBuffer, kScaleBuf, kZeroBuf, vScaleBuf, vZeroBuf, posBuffer}, .bufferCount = 9,
         .pushConstants = {32768, 4, 1024, 4, 256, 16}, .pushConstantCount = 6,
         .dispatchX = heads, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * heads * dim);
    validate_attention(att_q, kf, vf, ref, seq, heads, kv_heads, dim);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("Attention INT8", 100, out, ref, heads * dim, ms);

    destroy_buffers(s, bufs, 9);
    free(kq);
    free(vq);
    free(kScale);
    free(kZero);
    free(vScale);
    free(vZero);
    free(kf);
    free(vf);
    free(out);
    free(ref);
}

void validateAttentionINT4(session s, int att_seq, int att_heads, int att_kv_heads, int att_dim, float* att_q, uint16_t* att_k, uint16_t* att_v) {
    int seq = att_seq;
    int heads = att_heads;
    int kv_heads = att_kv_heads;
    int dim = att_dim;
    int rows = kv_heads * dim;
    int kvs = rows * seq;
    uint8_t* kq = (uint8_t*)malloc(kvs);
    uint8_t* vq = (uint8_t*)malloc(kvs);
    float* kScale = (float*)malloc(sizeof(float) * kv_heads * 32768);
    float* kZero = (float*)malloc(sizeof(float) * kv_heads * 32768);
    float* vScale = (float*)malloc(sizeof(float) * kv_heads * 32768);
    float* vZero = (float*)malloc(sizeof(float) * kv_heads * 32768);
    for (int h = 0; h < kv_heads; h++) {
        for (int s2 = 0; s2 < seq; s2++) {
            float mn = FLT_MAX;
            float mx = -FLT_MAX;
            for (int d = 0; d < dim; d++) {
                float kv = fp16_to_float(att_k[(h * dim + d) * seq + s2]);
                mn = fminf(mn, kv);
                mx = fmaxf(mx, kv);
            }
            float ks = (mx - mn) / 255.0f;
            if (ks == 0.0f) ks = 1.0f;
            kScale[h * 32768 + s2] = ks;
            kZero[h * 32768 + s2] = -mn;
            for (int d = 0; d < dim; d++) {
                float kv = fp16_to_float(att_k[(h * dim + d) * seq + s2]);
                int q = (int)rintf((kv + kZero[h * 32768 + s2]) / ks);
                if (q < 0) q = 0;
                if (q > 255) q = 255;
                kq[s2 * rows + (h * dim + d)] = (uint8_t)q;
            }
        }
    }
    for (int h = 0; h < kv_heads; h++) {
        for (int s2 = 0; s2 < seq; s2++) {
            float mn = FLT_MAX;
            float mx = -FLT_MAX;
            for (int d = 0; d < dim; d++) {
                float vv = fp16_to_float(att_v[(h * dim + d) * seq + s2]);
                mn = fminf(mn, vv);
                mx = fmaxf(mx, vv);
            }
            float vs = (mx - mn) / 255.0f;
            if (vs == 0.0f) vs = 1.0f;
            vScale[h * 32768 + s2] = vs;
            vZero[h * 32768 + s2] = -mn;
            for (int d = 0; d < dim; d++) {
                float vv = fp16_to_float(att_v[(h * dim + d) * seq + s2]);
                int q = (int)rintf((vv + vZero[h * 32768 + s2]) / vs);
                if (q < 0) q = 0;
                if (q > 255) q = 255;
                vq[s2 * rows + (h * dim + d)] = (uint8_t)q;
            }
        }
    }
    float* kf = (float*)malloc(sizeof(float) * kvs);
    float* vf = (float*)malloc(sizeof(float) * kvs);
    for (int r = 0; r < rows; r++) {
        int h = r / dim;
        for (int s2 = 0; s2 < seq; s2++) {
            kf[r * seq + s2] = (float)kq[s2 * rows + r] * kScale[h * 32768 + s2] - kZero[h * 32768 + s2];
            vf[r * seq + s2] = (float)vq[s2 * rows + r] * vScale[h * 32768 + s2] - vZero[h * 32768 + s2];
        }
    }
    float* out = (float*)calloc(heads * dim, sizeof(float));

    uint32_t posVal = (uint32_t)(seq - 1);
    uint8_t* kq_t = (uint8_t*)calloc((size_t)rows * 32768, sizeof(uint8_t));
    for (int s2 = 0; s2 < seq; s2++)
        for (int r = 0; r < rows; r++)
            kq_t[r * 32768 + s2] = kq[s2 * rows + r];
    buffer keyBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kq_t, sizeof(uint8_t) * (size_t)rows * 32768, MEMORY_RAM);
    free(kq_t);
    buffer valueBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vq, sizeof(uint8_t) * kvs, MEMORY_RAM);
    buffer queryBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_q, sizeof(float) * heads * dim, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * heads * dim, MEMORY_RAM);
    buffer kScaleBuf = createBuffer(s.dev.device, s.dev.physicalDevice, kScale, sizeof(float) * kv_heads * 32768, MEMORY_RAM);
    buffer kZeroBuf = createBuffer(s.dev.device, s.dev.physicalDevice, kZero, sizeof(float) * kv_heads * 32768, MEMORY_RAM);
    buffer vScaleBuf = createBuffer(s.dev.device, s.dev.physicalDevice, vScale, sizeof(float) * kv_heads * 32768, MEMORY_RAM);
    buffer vZeroBuf = createBuffer(s.dev.device, s.dev.physicalDevice, vZero, sizeof(float) * kv_heads * 32768, MEMORY_RAM);
    buffer posBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, &posVal, sizeof(uint32_t), MEMORY_VRAM);
    buffer bufs[] = {keyBuffer, valueBuffer, queryBuffer, outBuffer, kScaleBuf, kZeroBuf, vScaleBuf, vZeroBuf, posBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 9);

    operation ops[] = {
        {.shader = "Att-full-INT4.spv", .buffers = {keyBuffer, valueBuffer, queryBuffer, outBuffer, kScaleBuf, kZeroBuf, vScaleBuf, vZeroBuf, posBuffer}, .bufferCount = 9,
         .pushConstants = {32768, 4, 1024, 4, 256, 16}, .pushConstantCount = 6,
         .dispatchX = heads, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * heads * dim);
    validate_attention(att_q, kf, vf, ref, seq, heads, kv_heads, dim);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("Attention INT4", 100, out, ref, heads * dim, ms);

    destroy_buffers(s, bufs, 9);
    free(kq);
    free(vq);
    free(kScale);
    free(kZero);
    free(vScale);
    free(vZero);
    free(kf);
    free(vf);
    free(out);
    free(ref);
}

void validateAttentionGEMMFP16(session s, int seq, int heads, int kv_heads, int dim) {
    int rows = kv_heads * dim;
    int kvs = rows * seq;
    float* att_q = getData(7777, seq, heads * dim);
    uint16_t* att_k = getDataFP16(8100, rows, seq);
    uint16_t* att_v = getDataFP16(9100, rows, seq);

    float* kf = (float*)malloc(sizeof(float) * kvs);
    float* vf = (float*)malloc(sizeof(float) * kvs);
    for (int i = 0; i < kvs; i++) { kf[i] = fp16_to_float(att_k[i]); vf[i] = fp16_to_float(att_v[i]); }

    uint16_t* att_k_t = (uint16_t*)malloc(sizeof(uint16_t) * kvs);
    uint16_t* att_v_t = (uint16_t*)malloc(sizeof(uint16_t) * kvs);
    for (int s2 = 0; s2 < seq; s2++)
        for (int r = 0; r < rows; r++) {
            att_k_t[s2 * rows + r] = att_k[r * seq + s2];
            att_v_t[s2 * rows + r] = att_v[r * seq + s2];
        }

    float* out = (float*)calloc(seq * heads * dim, sizeof(float));

    buffer keyBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_k_t, sizeof(uint16_t) * kvs, MEMORY_RAM);
    buffer valueBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_v_t, sizeof(uint16_t) * kvs, MEMORY_RAM);
    buffer queryBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_q, sizeof(float) * seq * heads * dim, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * seq * heads * dim, MEMORY_RAM);
    buffer bufs[] = {keyBuffer, valueBuffer, queryBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 4);

    operation ops[] = {
        {.shader = "Att-full-GEMM-FP16.spv", .buffers = {keyBuffer, valueBuffer, queryBuffer, outBuffer}, .bufferCount = 4,
         .pushConstants = {seq}, .pushConstantCount = 1,
         .dispatchX = heads, .dispatchY = seq / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * seq * heads * dim);
    validate_attention_multi(att_q, kf, vf, ref, seq, heads, kv_heads, dim);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("Attention-GEMM FP16", 100, out, ref, seq * heads * dim, ms);

    destroy_buffers(s, bufs, 4);
    free(att_q); free(att_k); free(att_v);
    free(kf); free(vf); free(att_k_t); free(att_v_t);
    free(out); free(ref);
}

void validateAttentionGEMMINT8(session s, int seq, int heads, int kv_heads, int dim) {
    int rows = kv_heads * dim;
    int kvs = rows * seq;
    float* att_q = getData(7777, seq, heads * dim);
    uint16_t* att_k = getDataFP16(8100, rows, seq);
    uint16_t* att_v = getDataFP16(9100, rows, seq);

    uint8_t* kq = (uint8_t*)malloc(kvs);
    uint8_t* vq = (uint8_t*)malloc(kvs);
    float* kScale = (float*)malloc(sizeof(float) * kv_heads * seq);
    float* kZero = (float*)malloc(sizeof(float) * kv_heads * seq);
    float* vScale = (float*)malloc(sizeof(float) * kv_heads * seq);
    float* vZero = (float*)malloc(sizeof(float) * kv_heads * seq);
    for (int h = 0; h < kv_heads; h++) {
        for (int s2 = 0; s2 < seq; s2++) {
            float mn = FLT_MAX;
            float mx = -FLT_MAX;
            for (int d = 0; d < dim; d++) {
                float kv = fp16_to_float(att_k[(h * dim + d) * seq + s2]);
                mn = fminf(mn, kv);
                mx = fmaxf(mx, kv);
            }
            float ks = (mx - mn) / 255.0f;
            if (ks == 0.0f) ks = 1.0f;
            kScale[h * seq + s2] = ks;
            kZero[h * seq + s2] = -mn;
            for (int d = 0; d < dim; d++) {
                float kv = fp16_to_float(att_k[(h * dim + d) * seq + s2]);
                int q = (int)rintf((kv + kZero[h * seq + s2]) / ks);
                if (q < 0) q = 0;
                if (q > 255) q = 255;
                kq[s2 * rows + (h * dim + d)] = (uint8_t)q;
            }
        }
    }
    for (int h = 0; h < kv_heads; h++) {
        for (int s2 = 0; s2 < seq; s2++) {
            float mn = FLT_MAX;
            float mx = -FLT_MAX;
            for (int d = 0; d < dim; d++) {
                float vv = fp16_to_float(att_v[(h * dim + d) * seq + s2]);
                mn = fminf(mn, vv);
                mx = fmaxf(mx, vv);
            }
            float vs = (mx - mn) / 255.0f;
            if (vs == 0.0f) vs = 1.0f;
            vScale[h * seq + s2] = vs;
            vZero[h * seq + s2] = -mn;
            for (int d = 0; d < dim; d++) {
                float vv = fp16_to_float(att_v[(h * dim + d) * seq + s2]);
                int q = (int)rintf((vv + vZero[h * seq + s2]) / vs);
                if (q < 0) q = 0;
                if (q > 255) q = 255;
                vq[s2 * rows + (h * dim + d)] = (uint8_t)q;
            }
        }
    }
    float* kf = (float*)malloc(sizeof(float) * kvs);
    float* vf = (float*)malloc(sizeof(float) * kvs);
    for (int r = 0; r < rows; r++) {
        int h = r / dim;
        for (int s2 = 0; s2 < seq; s2++) {
            kf[r * seq + s2] = (float)kq[s2 * rows + r] * kScale[h * seq + s2] - kZero[h * seq + s2];
            vf[r * seq + s2] = (float)vq[s2 * rows + r] * vScale[h * seq + s2] - vZero[h * seq + s2];
        }
    }
    float* out = (float*)calloc(seq * heads * dim, sizeof(float));

    buffer keyBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kq, sizeof(uint8_t) * kvs, MEMORY_RAM);
    buffer valueBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vq, sizeof(uint8_t) * kvs, MEMORY_RAM);
    buffer queryBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_q, sizeof(float) * seq * heads * dim, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * seq * heads * dim, MEMORY_RAM);
    buffer kScaleBuf = createBuffer(s.dev.device, s.dev.physicalDevice, kScale, sizeof(float) * kv_heads * seq, MEMORY_RAM);
    buffer kZeroBuf = createBuffer(s.dev.device, s.dev.physicalDevice, kZero, sizeof(float) * kv_heads * seq, MEMORY_RAM);
    buffer vScaleBuf = createBuffer(s.dev.device, s.dev.physicalDevice, vScale, sizeof(float) * kv_heads * seq, MEMORY_RAM);
    buffer vZeroBuf = createBuffer(s.dev.device, s.dev.physicalDevice, vZero, sizeof(float) * kv_heads * seq, MEMORY_RAM);
    buffer bufs[] = {keyBuffer, valueBuffer, queryBuffer, outBuffer, kScaleBuf, kZeroBuf, vScaleBuf, vZeroBuf};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 8);

    operation ops[] = {
        {.shader = "Att-full-GEMM-INT8.spv", .buffers = {keyBuffer, valueBuffer, queryBuffer, outBuffer, kScaleBuf, kZeroBuf, vScaleBuf, vZeroBuf}, .bufferCount = 8,
         .pushConstants = {seq}, .pushConstantCount = 1,
         .dispatchX = heads, .dispatchY = seq / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * seq * heads * dim);
    validate_attention_multi(att_q, kf, vf, ref, seq, heads, kv_heads, dim);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("Attention-GEMM INT8", 100, out, ref, seq * heads * dim, ms);

    destroy_buffers(s, bufs, 8);
    free(att_q); free(att_k); free(att_v);
    free(kq);
    free(vq);
    free(kScale);
    free(kZero);
    free(vScale);
    free(vZero);
    free(kf);
    free(vf);
    free(out);
    free(ref);
}

void validateAttentionGEMMINT4(session s, int seq, int heads, int kv_heads, int dim) {
    int rows = kv_heads * dim;
    int kvs = rows * seq;
    float* att_q = getData(7777, seq, heads * dim);
    uint16_t* att_k = getDataFP16(8100, rows, seq);
    uint16_t* att_v = getDataFP16(9100, rows, seq);

    uint8_t* kq = (uint8_t*)malloc(kvs);
    uint8_t* vq = (uint8_t*)malloc(kvs);
    float* kScale = (float*)malloc(sizeof(float) * kv_heads * seq);
    float* kZero = (float*)malloc(sizeof(float) * kv_heads * seq);
    float* vScale = (float*)malloc(sizeof(float) * kv_heads * seq);
    float* vZero = (float*)malloc(sizeof(float) * kv_heads * seq);
    for (int h = 0; h < kv_heads; h++) {
        for (int s2 = 0; s2 < seq; s2++) {
            float mn = FLT_MAX;
            float mx = -FLT_MAX;
            for (int d = 0; d < dim; d++) {
                float kv = fp16_to_float(att_k[(h * dim + d) * seq + s2]);
                mn = fminf(mn, kv);
                mx = fmaxf(mx, kv);
            }
            float ks = (mx - mn) / 255.0f;
            if (ks == 0.0f) ks = 1.0f;
            kScale[h * seq + s2] = ks;
            kZero[h * seq + s2] = -mn;
            for (int d = 0; d < dim; d++) {
                float kv = fp16_to_float(att_k[(h * dim + d) * seq + s2]);
                int q = (int)rintf((kv + kZero[h * seq + s2]) / ks);
                if (q < 0) q = 0;
                if (q > 255) q = 255;
                kq[s2 * rows + (h * dim + d)] = (uint8_t)q;
            }
        }
    }
    for (int h = 0; h < kv_heads; h++) {
        for (int s2 = 0; s2 < seq; s2++) {
            float mn = FLT_MAX;
            float mx = -FLT_MAX;
            for (int d = 0; d < dim; d++) {
                float vv = fp16_to_float(att_v[(h * dim + d) * seq + s2]);
                mn = fminf(mn, vv);
                mx = fmaxf(mx, vv);
            }
            float vs = (mx - mn) / 255.0f;
            if (vs == 0.0f) vs = 1.0f;
            vScale[h * seq + s2] = vs;
            vZero[h * seq + s2] = -mn;
            for (int d = 0; d < dim; d++) {
                float vv = fp16_to_float(att_v[(h * dim + d) * seq + s2]);
                int q = (int)rintf((vv + vZero[h * seq + s2]) / vs);
                if (q < 0) q = 0;
                if (q > 255) q = 255;
                vq[s2 * rows + (h * dim + d)] = (uint8_t)q;
            }
        }
    }
    float* kf = (float*)malloc(sizeof(float) * kvs);
    float* vf = (float*)malloc(sizeof(float) * kvs);
    for (int r = 0; r < rows; r++) {
        int h = r / dim;
        for (int s2 = 0; s2 < seq; s2++) {
            kf[r * seq + s2] = (float)kq[s2 * rows + r] * kScale[h * seq + s2] - kZero[h * seq + s2];
            vf[r * seq + s2] = (float)vq[s2 * rows + r] * vScale[h * seq + s2] - vZero[h * seq + s2];
        }
    }
    float* out = (float*)calloc(seq * heads * dim, sizeof(float));

    buffer keyBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kq, sizeof(uint8_t) * kvs, MEMORY_RAM);
    buffer valueBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vq, sizeof(uint8_t) * kvs, MEMORY_RAM);
    buffer queryBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_q, sizeof(float) * seq * heads * dim, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * seq * heads * dim, MEMORY_RAM);
    buffer kScaleBuf = createBuffer(s.dev.device, s.dev.physicalDevice, kScale, sizeof(float) * kv_heads * seq, MEMORY_RAM);
    buffer kZeroBuf = createBuffer(s.dev.device, s.dev.physicalDevice, kZero, sizeof(float) * kv_heads * seq, MEMORY_RAM);
    buffer vScaleBuf = createBuffer(s.dev.device, s.dev.physicalDevice, vScale, sizeof(float) * kv_heads * seq, MEMORY_RAM);
    buffer vZeroBuf = createBuffer(s.dev.device, s.dev.physicalDevice, vZero, sizeof(float) * kv_heads * seq, MEMORY_RAM);
    buffer bufs[] = {keyBuffer, valueBuffer, queryBuffer, outBuffer, kScaleBuf, kZeroBuf, vScaleBuf, vZeroBuf};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 8);

    operation ops[] = {
        {.shader = "Att-full-GEMM-INT4.spv", .buffers = {keyBuffer, valueBuffer, queryBuffer, outBuffer, kScaleBuf, kZeroBuf, vScaleBuf, vZeroBuf}, .bufferCount = 8,
         .pushConstants = {seq}, .pushConstantCount = 1,
         .dispatchX = heads, .dispatchY = seq / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * seq * heads * dim);
    validate_attention_multi(att_q, kf, vf, ref, seq, heads, kv_heads, dim);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("Attention-GEMM INT4", 100, out, ref, seq * heads * dim, ms);

    destroy_buffers(s, bufs, 8);
    free(att_q); free(att_k); free(att_v);
    free(kq);
    free(vq);
    free(kScale);
    free(kZero);
    free(vScale);
    free(vZero);
    free(kf);
    free(vf);
    free(out);
    free(ref);
}

#define QKV_ROPE_DIM 64

static void qkv_rope_ref(const float* proj, const float* theta, float* qref, float* kref, float* vref, float* gref,
                         int n_total, int g_offset, int k_offset, int v_offset, int dim, int token,
                         const float* qgamma, const float* kgamma) {
    int num_heads = n_total / dim;
    int q_heads = g_offset / dim;
    int k_head_base = k_offset / dim;
    int v_head_base = v_offset / dim;
    float* hn = (float*)calloc(n_total, sizeof(float));
    for (int h = 0; h < num_heads; h++) {
        float sum = 0.0f;
        for (int i = 0; i < dim; i++) sum += proj[h * dim + i] * proj[h * dim + i];
        float inv = 1.0f / (sqrtf(sum / (float)dim + 1e-6f));
        const float* gm = NULL;
        if (h < q_heads) gm = qgamma;
        else if (h >= k_head_base && h < v_head_base) gm = kgamma;
        if (!gm) continue;
        for (int i = 0; i < dim; i++) hn[h * dim + i] = proj[h * dim + i] * inv * gm[i];
    }
    for (int j = 0; j < n_total; j++) {
        if (j >= v_offset) {
            vref[j - v_offset] = proj[j];
            continue;
        }
        if (j >= g_offset && j < k_offset) {
            gref[j - g_offset] = proj[j];
            continue;
        }
        int col = j % dim;
        int head = (j / dim) * dim;
        float val = hn[j];
        if (col < QKV_ROPE_DIM) {
            float ang = token * theta[col & (QKV_ROPE_DIM / 2 - 1)];
            float cs = cosf(ang);
            float sn = sinf(ang);
            if (col < QKV_ROPE_DIM / 2) {
                val = hn[head + col] * cs - hn[head + col + QKV_ROPE_DIM / 2] * sn;
            } else {
                val = hn[head + col - QKV_ROPE_DIM / 2] * sn + hn[head + col] * cs;
            }
        }
        if (j < g_offset) qref[j] = val * 0.0625f;
        else kref[j - k_offset] = val;
    }
    free(hn);
}



void validateQkvRopeFP16(session s, int K, int qkv_heads, int qkv_kv_heads, int qkv_dim, float* input, float* gamma, uint16_t* qkv_weightFP16, float* qkv_theta) {
    int k = K;
    int heads = qkv_heads;
    int kv_heads = qkv_kv_heads;
    int dim = qkv_dim;
    int gate_heads = heads;
    int n_total = (heads + gate_heads + 2 * kv_heads) * dim;
    int g_offset = heads * dim;
    int k_offset = (heads + gate_heads) * dim;
    int v_offset = (heads + gate_heads + kv_heads) * dim;
    int rows = kv_heads * dim;
    int ctx = 33;
    int seq_len = ctx + 1;
    uint32_t posVal = (uint32_t)ctx;

    float* xn = (float*)malloc(sizeof(float) * k);
    rms_norm_apply(input, gamma, xn, k);
    float* proj = (float*)malloc(sizeof(float) * n_total);
    gemv_ref_fp16(xn, qkv_weightFP16, proj, n_total, k);
    float* qref = (float*)malloc(sizeof(float) * heads * dim);
    float* kref = (float*)malloc(sizeof(float) * rows);
    float* vref = (float*)malloc(sizeof(float) * rows);
    float* gref = (float*)malloc(sizeof(float) * gate_heads * dim);
    float* qgamma = getData(61001, 1, dim);
    float* kgamma = getData(62001, 1, dim);
    qkv_rope_ref(proj, qkv_theta, qref, kref, vref, gref, n_total, g_offset, k_offset, v_offset, dim, ctx, qgamma, kgamma);

    float* qOut = (float*)calloc(heads * dim, sizeof(float));
    uint16_t* kCache = (uint16_t*)calloc(rows * seq_len, sizeof(uint16_t));
    uint16_t* vCache = (uint16_t*)calloc(rows * seq_len, sizeof(uint16_t));

    uint16_t* transposed = (uint16_t*)malloc(sizeof(uint16_t) * k * n_total);
    transpose_block16((uint8_t*)qkv_weightFP16, (uint8_t*)transposed, k, n_total, QUANT_FP16);

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * k, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * k, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint16_t) * k * n_total, MEMORY_RAM);
    buffer thetaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_theta, sizeof(float) * (QKV_ROPE_DIM / 2), MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * heads * dim, MEMORY_VRAM);
    buffer kCacheBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kCache, sizeof(uint16_t) * rows * seq_len, MEMORY_VRAM);
    buffer vCacheBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vCache, sizeof(uint16_t) * rows * seq_len, MEMORY_VRAM);
    buffer posBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, &posVal, sizeof(uint32_t), MEMORY_VRAM);
    float* gOut = (float*)calloc(gate_heads * dim, sizeof(float));
    buffer qGammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qgamma, sizeof(float) * dim, MEMORY_VRAM);
    buffer kGammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kgamma, sizeof(float) * dim, MEMORY_VRAM);
    buffer gOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gOut, sizeof(float) * gate_heads * dim, MEMORY_VRAM);
    buffer bufs[] = {xBuffer, gammaBuffer, weightBuffer, thetaBuffer, qOutBuffer, kCacheBuffer, vCacheBuffer, posBuffer, qGammaBuffer, kGammaBuffer, gOutBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 11);
    free(transposed);

    operation ops[] = {
        {.shader = "RmsNorm-QKV-FP16.spv", .buffers = {xBuffer, gammaBuffer, weightBuffer, thetaBuffer, qOutBuffer, kCacheBuffer, vCacheBuffer, posBuffer, qGammaBuffer, kGammaBuffer, gOutBuffer}, .bufferCount = 11,
         .pushConstants = {1, n_total, k, g_offset, k_offset, v_offset, 32768, 256, 64}, .pushConstantCount = 9,
         .dispatchX = n_total / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, qOutBuffer, qOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kCacheBuffer, kCache);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vCacheBuffer, vCache);
    float* kstored = (float*)malloc(sizeof(float) * rows);
    float* vstored = (float*)malloc(sizeof(float) * rows);
    for (int r = 0; r < rows; r++) kstored[r] = fp16_to_float(kCache[(seq_len - 1) * rows + r]);
    for (int r = 0; r < rows; r++) vstored[r] = fp16_to_float(vCache[(seq_len - 1) * rows + r]);
    float* kvOut = (float*)malloc(sizeof(float) * 2 * rows);
    float* kvRef = (float*)malloc(sizeof(float) * 2 * rows);
    for (int r = 0; r < rows; r++) {
        kvOut[r] = kstored[r];
        kvOut[rows + r] = vstored[r];
        kvRef[r] = kref[r];
        kvRef[rows + r] = vref[r];
    }


    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, gOutBuffer, gOut);
    report("QKV-Rope FP16", 100, qOut, qref, heads * dim, ms);
    report("QKV-Rope-g FP16", 100, gOut, gref, gate_heads * dim, ms);
    report("QKV-Rope-kv FP16", 100, kvOut, kvRef, 2 * rows, ms);
    uint32_t posRead = 0;
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, posBuffer, &posRead);
    printf("QKV-Rope-pos FP16: shader[0]= %u ref= %d\n", posRead, ctx);

    destroy_buffers(s, bufs, 11);
    free(xn);
    free(proj);
    free(qref);
    free(kref);
    free(vref);
    free(gref);
    free(qgamma);
    free(kgamma);
    free(gOut);
    free(qOut);
    free(kCache);
    free(vCache);
    free(kstored);
    free(vstored);
    free(kvOut);
    free(kvRef);
}

void validateQkvRopeINT8(session s, int K, int qkv_heads, int qkv_kv_heads, int qkv_dim, float* input, float* gamma, QuantizedData qkv_weightINT8, float* qkv_theta) {
    int k = K;
    int heads = qkv_heads;
    int kv_heads = qkv_kv_heads;
    int dim = qkv_dim;
    int gate_heads = heads;
    int n_total = (heads + gate_heads + 2 * kv_heads) * dim;
    int g_offset = heads * dim;
    int k_offset = (heads + gate_heads) * dim;
    int v_offset = (heads + gate_heads + kv_heads) * dim;
    int rows = kv_heads * dim;
    int ctx = 33;
    int seq_len = ctx + 1;
    int scaleCount = k * n_total / qkv_weightINT8.group_size;
    uint32_t posVal = (uint32_t)ctx;

    float* xn = (float*)malloc(sizeof(float) * k);
    rms_norm_apply(input, gamma, xn, k);
    float* proj = (float*)malloc(sizeof(float) * n_total);
    gemv_ref_int8(xn, &qkv_weightINT8, proj, n_total, k);
    float* qref = (float*)malloc(sizeof(float) * heads * dim);
    float* kref = (float*)malloc(sizeof(float) * rows);
    float* vref = (float*)malloc(sizeof(float) * rows);
    float* gref = (float*)malloc(sizeof(float) * gate_heads * dim);
    float* qgamma = getData(61001, 1, dim);
    float* kgamma = getData(62001, 1, dim);
    qkv_rope_ref(proj, qkv_theta, qref, kref, vref, gref, n_total, g_offset, k_offset, v_offset, dim, ctx, qgamma, kgamma);

    float* qOut = (float*)calloc(heads * dim, sizeof(float));
    uint8_t* kCache = (uint8_t*)calloc(rows * seq_len, sizeof(uint8_t));
    uint8_t* vCache = (uint8_t*)calloc(rows * seq_len, sizeof(uint8_t));
    float* kScale = (float*)calloc(kv_heads * seq_len, sizeof(float));
    float* kZero = (float*)calloc(kv_heads * seq_len, sizeof(float));
    float* vScale = (float*)calloc(kv_heads * seq_len, sizeof(float));
    float* vZero = (float*)calloc(kv_heads * seq_len, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * k * n_total);
    transpose_block16(qkv_weightINT8.data, transposed, k, n_total, QUANT_INT8);

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * k, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * k, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * k * n_total, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_weightINT8.scale, sizeof(float) * scaleCount, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_weightINT8.z, sizeof(float) * scaleCount, MEMORY_RAM);
    buffer thetaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_theta, sizeof(float) * (QKV_ROPE_DIM / 2), MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * heads * dim, MEMORY_VRAM);
    buffer kCacheBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kCache, sizeof(uint8_t) * rows * seq_len, MEMORY_VRAM);
    buffer vCacheBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vCache, sizeof(uint8_t) * rows * seq_len, MEMORY_VRAM);
    buffer kScaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kScale, sizeof(float) * kv_heads * seq_len, MEMORY_VRAM);
    buffer kZeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kZero, sizeof(float) * kv_heads * seq_len, MEMORY_VRAM);
    buffer vScaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vScale, sizeof(float) * kv_heads * seq_len, MEMORY_VRAM);
    buffer vZeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vZero, sizeof(float) * kv_heads * seq_len, MEMORY_VRAM);
    buffer posBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, &posVal, sizeof(uint32_t), MEMORY_VRAM);
    float* gOut = (float*)calloc(gate_heads * dim, sizeof(float));
    buffer qGammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qgamma, sizeof(float) * dim, MEMORY_VRAM);
    buffer kGammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kgamma, sizeof(float) * dim, MEMORY_VRAM);
    buffer gOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gOut, sizeof(float) * gate_heads * dim, MEMORY_VRAM);
    buffer bufs[] = {xBuffer, gammaBuffer, weightBuffer, scaleBuffer, zeroBuffer, thetaBuffer, qOutBuffer, kCacheBuffer, vCacheBuffer, kScaleBuffer, kZeroBuffer, vScaleBuffer, vZeroBuffer, posBuffer, qGammaBuffer, kGammaBuffer, gOutBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 17);
    free(transposed);

    operation ops[] = {
        {.shader = "RmsNorm-QKV-INT8.spv", .buffers = {xBuffer, gammaBuffer, weightBuffer, scaleBuffer, zeroBuffer, thetaBuffer, qOutBuffer, kCacheBuffer, vCacheBuffer, kScaleBuffer, kZeroBuffer, vScaleBuffer, vZeroBuffer, posBuffer, qGammaBuffer, kGammaBuffer, gOutBuffer}, .bufferCount = 17,
         .pushConstants = {1, n_total, k, g_offset, k_offset, v_offset, 32768, 256, 64}, .pushConstantCount = 9,
         .dispatchX = n_total / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, qOutBuffer, qOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kCacheBuffer, kCache);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vCacheBuffer, vCache);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kScaleBuffer, kScale);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kZeroBuffer, kZero);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vScaleBuffer, vScale);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vZeroBuffer, vZero);
    float* kstored = (float*)malloc(sizeof(float) * rows);
    float* vstored = (float*)malloc(sizeof(float) * rows);
    for (int r = 0; r < rows; r++) {
        int h = r / dim;
        kstored[r] = (float)kCache[(seq_len - 1) * rows + r] * kScale[h * seq_len + seq_len - 1] - kZero[h * seq_len + seq_len - 1];
        vstored[r] = (float)vCache[(seq_len - 1) * rows + r] * vScale[h * seq_len + seq_len - 1] - vZero[h * seq_len + seq_len - 1];
    }
    float* kvOut = (float*)malloc(sizeof(float) * 2 * rows);
    float* kvRef = (float*)malloc(sizeof(float) * 2 * rows);
    for (int r = 0; r < rows; r++) {
        kvOut[r] = kstored[r];
        kvOut[rows + r] = vstored[r];
        kvRef[r] = kref[r];
        kvRef[rows + r] = vref[r];
    }

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, gOutBuffer, gOut);
    report("QKV-Rope INT8", 100, qOut, qref, heads * dim, ms);
    report("QKV-Rope-g INT8", 100, gOut, gref, gate_heads * dim, ms);
    report("QKV-Rope-kv INT8", 100, kvOut, kvRef, 2 * rows, ms);
    uint32_t posRead = 0;
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, posBuffer, &posRead);
    printf("QKV-Rope-pos INT8: shader[0]= %u ref= %d\n", posRead, ctx);

    destroy_buffers(s, bufs, 17);
    free(xn);
    free(proj);
    free(qref);
    free(kref);
    free(vref);
    free(gref);
    free(qgamma);
    free(kgamma);
    free(gOut);
    free(qOut);
    free(kCache);
    free(vCache);
    free(kScale);
    free(kZero);
    free(vScale);
    free(vZero);
    free(kstored);
    free(vstored);
    free(kvOut);
    free(kvRef);
}

void validateQkvRopeINT4(session s, int K, int qkv_heads, int qkv_kv_heads, int qkv_dim, float* input, float* gamma, QuantizedData qkv_weightINT4, float* qkv_theta) {
    int k = K;
    int heads = qkv_heads;
    int kv_heads = qkv_kv_heads;
    int dim = qkv_dim;
    int gate_heads = heads;
    int n_total = (heads + gate_heads + 2 * kv_heads) * dim;
    int g_offset = heads * dim;
    int k_offset = (heads + gate_heads) * dim;
    int v_offset = (heads + gate_heads + kv_heads) * dim;
    int rows = kv_heads * dim;
    int ctx = 33;
    int seq_len = ctx + 1;
    int scaleCount = k * n_total / qkv_weightINT4.group_size;
    uint32_t posVal = (uint32_t)ctx;

    float* xn = (float*)malloc(sizeof(float) * k);
    rms_norm_apply(input, gamma, xn, k);
    float* proj = (float*)malloc(sizeof(float) * n_total);
    gemv_ref_int4(xn, &qkv_weightINT4, proj, n_total, k);
    float* qref = (float*)malloc(sizeof(float) * heads * dim);
    float* kref = (float*)malloc(sizeof(float) * rows);
    float* vref = (float*)malloc(sizeof(float) * rows);
    float* gref = (float*)malloc(sizeof(float) * gate_heads * dim);
    float* qgamma = getData(61001, 1, dim);
    float* kgamma = getData(62001, 1, dim);
    qkv_rope_ref(proj, qkv_theta, qref, kref, vref, gref, n_total, g_offset, k_offset, v_offset, dim, ctx, qgamma, kgamma);

    float* qOut = (float*)calloc(heads * dim, sizeof(float));
    uint8_t* kCache = (uint8_t*)calloc(rows * seq_len, sizeof(uint8_t));
    uint8_t* vCache = (uint8_t*)calloc(rows * seq_len, sizeof(uint8_t));
    float* kScale = (float*)calloc(kv_heads * seq_len, sizeof(float));
    float* kZero = (float*)calloc(kv_heads * seq_len, sizeof(float));
    float* vScale = (float*)calloc(kv_heads * seq_len, sizeof(float));
    float* vZero = (float*)calloc(kv_heads * seq_len, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * k * n_total / 2);
    transpose_block16(qkv_weightINT4.data, transposed, k, n_total, QUANT_INT4);

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * k, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * k, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * k * n_total / 2, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_weightINT4.scale, sizeof(float) * scaleCount, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_weightINT4.z, sizeof(float) * scaleCount, MEMORY_RAM);
    buffer thetaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_theta, sizeof(float) * (QKV_ROPE_DIM / 2), MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * heads * dim, MEMORY_VRAM);
    buffer kCacheBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kCache, sizeof(uint8_t) * rows * seq_len, MEMORY_VRAM);
    buffer vCacheBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vCache, sizeof(uint8_t) * rows * seq_len, MEMORY_VRAM);
    buffer kScaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kScale, sizeof(float) * kv_heads * seq_len, MEMORY_VRAM);
    buffer kZeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kZero, sizeof(float) * kv_heads * seq_len, MEMORY_VRAM);
    buffer vScaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vScale, sizeof(float) * kv_heads * seq_len, MEMORY_VRAM);
    buffer vZeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vZero, sizeof(float) * kv_heads * seq_len, MEMORY_VRAM);
    buffer posBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, &posVal, sizeof(uint32_t), MEMORY_VRAM);
    float* gOut = (float*)calloc(gate_heads * dim, sizeof(float));
    buffer qGammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qgamma, sizeof(float) * dim, MEMORY_VRAM);
    buffer kGammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kgamma, sizeof(float) * dim, MEMORY_VRAM);
    buffer gOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gOut, sizeof(float) * gate_heads * dim, MEMORY_VRAM);
    buffer bufs[] = {xBuffer, gammaBuffer, weightBuffer, scaleBuffer, zeroBuffer, thetaBuffer, qOutBuffer, kCacheBuffer, vCacheBuffer, kScaleBuffer, kZeroBuffer, vScaleBuffer, vZeroBuffer, posBuffer, qGammaBuffer, kGammaBuffer, gOutBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 17);
    free(transposed);

    operation ops[] = {
        {.shader = "RmsNorm-QKV-INT4.spv", .buffers = {xBuffer, gammaBuffer, weightBuffer, scaleBuffer, zeroBuffer, thetaBuffer, qOutBuffer, kCacheBuffer, vCacheBuffer, kScaleBuffer, kZeroBuffer, vScaleBuffer, vZeroBuffer, posBuffer, qGammaBuffer, kGammaBuffer, gOutBuffer}, .bufferCount = 17,
         .pushConstants = {1, n_total, k, g_offset, k_offset, v_offset, 32768, 256, 64}, .pushConstantCount = 9,
         .dispatchX = n_total / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, qOutBuffer, qOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kCacheBuffer, kCache);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vCacheBuffer, vCache);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kScaleBuffer, kScale);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kZeroBuffer, kZero);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vScaleBuffer, vScale);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vZeroBuffer, vZero);
    float* kstored = (float*)malloc(sizeof(float) * rows);
    float* vstored = (float*)malloc(sizeof(float) * rows);
    for (int r = 0; r < rows; r++) {
        int h = r / dim;
        kstored[r] = (float)kCache[(seq_len - 1) * rows + r] * kScale[h * seq_len + seq_len - 1] - kZero[h * seq_len + seq_len - 1];
        vstored[r] = (float)vCache[(seq_len - 1) * rows + r] * vScale[h * seq_len + seq_len - 1] - vZero[h * seq_len + seq_len - 1];
    }
    float* kvOut = (float*)malloc(sizeof(float) * 2 * rows);
    float* kvRef = (float*)malloc(sizeof(float) * 2 * rows);
    for (int r = 0; r < rows; r++) {
        kvOut[r] = kstored[r];
        kvOut[rows + r] = vstored[r];
        kvRef[r] = kref[r];
        kvRef[rows + r] = vref[r];
    }

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, gOutBuffer, gOut);
    report("QKV-Rope INT4", 100, qOut, qref, heads * dim, ms);
    report("QKV-Rope-g INT4", 100, gOut, gref, gate_heads * dim, ms);
    report("QKV-Rope-kv INT4", 100, kvOut, kvRef, 2 * rows, ms);
    uint32_t posRead = 0;
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, posBuffer, &posRead);
    printf("QKV-Rope-pos INT4: shader[0]= %u ref= %d\n", posRead, ctx);

    destroy_buffers(s, bufs, 17);
    free(xn);
    free(proj);
    free(qref);
    free(kref);
    free(vref);
    free(gref);
    free(qgamma);
    free(kgamma);
    free(gOut);
    free(qOut);
    free(kCache);
    free(vCache);
    free(kScale);
    free(kZero);
    free(vScale);
    free(vZero);
    free(kstored);
    free(vstored);
    free(kvOut);
    free(kvRef);
}

static void deltanet_ref(float* proj, const float* aLog, const float* dtBias, const float* wnorm, float* S, float* ygated, int n_qk, int n_v, int dim) {
    float* Q = proj;
    float* K = proj + n_qk * dim;
    const float* V = proj + 2 * n_qk * dim;
    const float* Z = proj + 2 * n_qk * dim + n_v * dim;
    const float* A = Z + n_v * dim;
    const float* B = A + n_v;

    float qscale = 1.0f / sqrtf((float)dim);
    for (int qk = 0; qk < n_qk; qk++) {
        float ksum = 0.0f;
        float qsum = 0.0f;
        for (int d = 0; d < dim; d++) {
            ksum += K[qk * dim + d] * K[qk * dim + d];
            qsum += Q[qk * dim + d] * Q[qk * dim + d];
        }
        float k_inv = 1.0f / sqrtf(ksum + 1e-6f);
        float q_inv = (1.0f / sqrtf(qsum + 1e-6f)) * qscale;
        for (int d = 0; d < dim; d++) {
            K[qk * dim + d] *= k_inv;
            Q[qk * dim + d] *= q_inv;
        }
    }

    for (int h = 0; h < n_v; h++) {
        int qk = h / 2;
        float g = -expf(aLog[h]) * logf(1.0f + expf(A[h] + dtBias[h]));
        float alpha = expf(g);
        float beta = 1.0f / (1.0f + expf(-B[h]));
        float* Sh = S + h * dim * dim;
        const float* Kh = K + qk * dim;
        const float* Qh = Q + qk * dim;
        const float* Vh = V + h * dim;
        const float* Zh = Z + h * dim;
        float* yh = ygated + h * dim;
        float delta[128];
        for (int i = 0; i < dim; i++) {
            float vhat = 0.0f;
            for (int j = 0; j < dim; j++) vhat += Sh[i * dim + j] * Kh[j];
            delta[i] = Vh[i] - alpha * vhat;
        }
        for (int i = 0; i < dim; i++) {
            float y = 0.0f;
            for (int j = 0; j < dim; j++) y += (alpha * Sh[i * dim + j] + beta * delta[i] * Kh[j]) * Qh[j];
            yh[i] = y;
        }
        for (int i = 0; i < dim; i++) {
            for (int j = 0; j < dim; j++) {
                Sh[i * dim + j] = alpha * Sh[i * dim + j] + beta * delta[i] * Kh[j];
            }
        }
        float sum = 0.0f;
        for (int i = 0; i < dim; i++) sum += yh[i] * yh[i];
        float inv = 1.0f / sqrtf(sum / (float)dim + 1e-6f);
        for (int i = 0; i < dim; i++) {
            float zv = Zh[i];
            yh[i] = yh[i] * inv * wnorm[i] * (zv / (1.0f + expf(-zv)));
        }
    }
}

void validateGatedDeltaNetFP16(session s, int K, float* input, float* input2, float* gamma, uint16_t* w_inFP16, uint16_t* woFP16) {
    int proj_n = 12352;
    int out_n = 4096;
    int n_qk = 16;
    int n_v = 32;
    int dim = 128;
    int smat = n_v * dim * dim;

    float* xn = (float*)malloc(sizeof(float) * K);
    float* xn2 = (float*)malloc(sizeof(float) * K);
    rms_norm_apply(input, gamma, xn, K);
    rms_norm_apply(input2, gamma, xn2, K);
    float* p1 = (float*)malloc(sizeof(float) * proj_n);
    float* p2 = (float*)malloc(sizeof(float) * proj_n);
    gemv_ref_fp16(xn, w_inFP16, p1, proj_n, K);
    gemv_ref_fp16(xn2, w_inFP16, p2, proj_n, K);
    float* wn = getData(64051, 1, dim);
    float* aLog = getData(64052, 1, n_v);
    float* dtBias = getData(64053, 1, n_v);
    float* S = (float*)calloc(smat, sizeof(float));
    float* yg1 = (float*)malloc(sizeof(float) * out_n);
    float* yg2 = (float*)malloc(sizeof(float) * out_n);
    deltanet_ref(p1, aLog, dtBias, wn, S, yg1, n_qk, n_v, dim);
    deltanet_ref(p2, aLog, dtBias, wn, S, yg2, n_qk, n_v, dim);
    float* ref = (float*)malloc(sizeof(float) * out_n);
    gemv_ref_fp16(yg2, woFP16, ref, out_n, K);

    float* qOut = (float*)calloc(n_qk * dim, sizeof(float));
    float* kOut = (float*)calloc(n_qk * dim, sizeof(float));
    float* vOut = (float*)calloc(n_v * dim, sizeof(float));
    float* zOut = (float*)calloc(n_v * dim, sizeof(float));
    float* aOut = (float*)calloc(n_v, sizeof(float));
    float* bOut = (float*)calloc(n_v, sizeof(float));
    float* yGated = (float*)calloc(out_n, sizeof(float));
    float* out = (float*)calloc(out_n, sizeof(float));
    float* Sbuf = (float*)calloc(smat, sizeof(float));

    uint16_t* twIn = (uint16_t*)malloc(sizeof(uint16_t) * K * proj_n);
    uint16_t* twOut = (uint16_t*)malloc(sizeof(uint16_t) * K * out_n);
    transpose_block16((uint8_t*)w_inFP16, (uint8_t*)twIn, K, proj_n, QUANT_FP16);
    transpose_block16((uint8_t*)woFP16, (uint8_t*)twOut, K, out_n, QUANT_FP16);

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * K, MEMORY_RAM);
    buffer x2Buffer = createBuffer(s.dev.device, s.dev.physicalDevice, input2, sizeof(float) * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer wInBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, twIn, sizeof(uint16_t) * K * proj_n, MEMORY_RAM);
    buffer wOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, twOut, sizeof(uint16_t) * K * out_n, MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * n_qk * dim, MEMORY_VRAM);
    buffer kOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kOut, sizeof(float) * n_qk * dim, MEMORY_VRAM);
    buffer vOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vOut, sizeof(float) * n_v * dim, MEMORY_VRAM);
    buffer zOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, zOut, sizeof(float) * n_v * dim, MEMORY_VRAM);
    buffer aOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, aOut, sizeof(float) * n_v, MEMORY_VRAM);
    buffer bOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, bOut, sizeof(float) * n_v, MEMORY_VRAM);
    buffer sBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, Sbuf, sizeof(float) * smat, MEMORY_VRAM);
    buffer yGatedBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, yGated, sizeof(float) * out_n, MEMORY_VRAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * out_n, MEMORY_VRAM);
    buffer wnBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, wn, sizeof(float) * dim, MEMORY_RAM);
    buffer aLogBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, aLog, sizeof(float) * n_v, MEMORY_RAM);
    buffer dtBiasBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, dtBias, sizeof(float) * n_v, MEMORY_RAM);
    buffer bufs[] = {xBuffer, x2Buffer, gammaBuffer, wInBuffer, wOutBuffer, qOutBuffer, kOutBuffer, vOutBuffer, zOutBuffer, aOutBuffer, bOutBuffer, sBuffer, yGatedBuffer, outBuffer, wnBuffer, aLogBuffer, dtBiasBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 17);
    free(twIn);
    free(twOut);

    operation ops[] = {
        {.shader = "RmsNorm-LinearProj-FP16.spv", .buffers = {xBuffer, gammaBuffer, wInBuffer, qOutBuffer, kOutBuffer, vOutBuffer, zOutBuffer, aOutBuffer, bOutBuffer}, .bufferCount = 9,
         .pushConstants = {1, proj_n, K, 2048, 4096, 8192, 12288, 12320}, .pushConstantCount = 8,
         .dispatchX = (proj_n + 255) / 256, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "GatedDeltaNet.spv", .buffers = {qOutBuffer, kOutBuffer, vOutBuffer, aOutBuffer, bOutBuffer, sBuffer, yGatedBuffer, zOutBuffer, wnBuffer, aLogBuffer, dtBiasBuffer}, .bufferCount = 11,
         .pushConstants = {n_v, n_qk, dim}, .pushConstantCount = 3,
         .dispatchX = n_v, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "GEMV-FP16.spv", .buffers = {yGatedBuffer, wOutBuffer, outBuffer}, .bufferCount = 3,
         .pushConstants = {1, out_n, K}, .pushConstantCount = 3,
         .dispatchX = out_n / 256, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "RmsNorm-LinearProj-FP16.spv", .buffers = {x2Buffer, gammaBuffer, wInBuffer, qOutBuffer, kOutBuffer, vOutBuffer, zOutBuffer, aOutBuffer, bOutBuffer}, .bufferCount = 9,
         .pushConstants = {1, proj_n, K, 2048, 4096, 8192, 12288, 12320}, .pushConstantCount = 8,
         .dispatchX = (proj_n + 255) / 256, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "GatedDeltaNet.spv", .buffers = {qOutBuffer, kOutBuffer, vOutBuffer, aOutBuffer, bOutBuffer, sBuffer, yGatedBuffer, zOutBuffer, wnBuffer, aLogBuffer, dtBiasBuffer}, .bufferCount = 11,
         .pushConstants = {n_v, n_qk, dim}, .pushConstantCount = 3,
         .dispatchX = n_v, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "GEMV-FP16.spv", .buffers = {yGatedBuffer, wOutBuffer, outBuffer}, .bufferCount = 3,
         .pushConstants = {1, out_n, K}, .pushConstantCount = 3,
         .dispatchX = out_n / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 6);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, sBuffer, Sbuf);
    report("GatedDeltaNet FP16", 100, out, ref, out_n, ms);
    report("GatedDeltaNet-S FP16", 100, Sbuf, S, smat, ms);

    destroy_buffers(s, bufs, 17);
    free(xn);
    free(xn2);
    free(p1);
    free(p2);
    free(wn);
    free(aLog);
    free(dtBias);
    free(S);
    free(yg1);
    free(yg2);
    free(ref);
    free(qOut);
    free(kOut);
    free(vOut);
    free(zOut);
    free(aOut);
    free(bOut);
    free(yGated);
    free(out);
    free(Sbuf);
}

void validateGatedDeltaNetINT8(session s, int K, float* input, float* input2, float* gamma, QuantizedData w_inINT8, QuantizedData woINT8) {
    int proj_n = 12352;
    int out_n = 4096;
    int n_qk = 16;
    int n_v = 32;
    int dim = 128;
    int smat = n_v * dim * dim;
    int inBlocks = (proj_n + 255) / 256;
    int outBlocks = out_n / 256;

    float* xn = (float*)malloc(sizeof(float) * K);
    float* xn2 = (float*)malloc(sizeof(float) * K);
    rms_norm_apply(input, gamma, xn, K);
    rms_norm_apply(input2, gamma, xn2, K);
    float* p1 = (float*)malloc(sizeof(float) * proj_n);
    float* p2 = (float*)malloc(sizeof(float) * proj_n);
    gemv_ref_int8(xn, &w_inINT8, p1, proj_n, K);
    gemv_ref_int8(xn2, &w_inINT8, p2, proj_n, K);
    float* wn = getData(64051, 1, dim);
    float* aLog = getData(64052, 1, n_v);
    float* dtBias = getData(64053, 1, n_v);
    float* S = (float*)calloc(smat, sizeof(float));
    float* yg1 = (float*)malloc(sizeof(float) * out_n);
    float* yg2 = (float*)malloc(sizeof(float) * out_n);
    deltanet_ref(p1, aLog, dtBias, wn, S, yg1, n_qk, n_v, dim);
    deltanet_ref(p2, aLog, dtBias, wn, S, yg2, n_qk, n_v, dim);
    float* ref = (float*)malloc(sizeof(float) * out_n);
    gemv_ref_int8(yg2, &woINT8, ref, out_n, K);

    float* qOut = (float*)calloc(n_qk * dim, sizeof(float));
    float* kOut = (float*)calloc(n_qk * dim, sizeof(float));
    float* vOut = (float*)calloc(n_v * dim, sizeof(float));
    float* zOut = (float*)calloc(n_v * dim, sizeof(float));
    float* aOut = (float*)calloc(n_v, sizeof(float));
    float* bOut = (float*)calloc(n_v, sizeof(float));
    float* yGated = (float*)calloc(out_n, sizeof(float));
    float* out = (float*)calloc(out_n, sizeof(float));
    float* Sbuf = (float*)calloc(smat, sizeof(float));

    uint8_t* twIn = (uint8_t*)malloc(sizeof(uint8_t) * K * proj_n);
    uint8_t* twOut = (uint8_t*)malloc(sizeof(uint8_t) * K * out_n);
    transpose_block16(w_inINT8.data, twIn, K, proj_n, QUANT_INT8);
    transpose_block16(woINT8.data, twOut, K, out_n, QUANT_INT8);

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * K, MEMORY_RAM);
    buffer x2Buffer = createBuffer(s.dev.device, s.dev.physicalDevice, input2, sizeof(float) * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer wInBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, twIn, sizeof(uint8_t) * K * proj_n, MEMORY_RAM);
    buffer wInScale = createBuffer(s.dev.device, s.dev.physicalDevice, w_inINT8.scale, sizeof(float) * K * inBlocks, MEMORY_RAM);
    buffer wInZero = createBuffer(s.dev.device, s.dev.physicalDevice, w_inINT8.z, sizeof(float) * K * inBlocks, MEMORY_RAM);
    buffer wOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, twOut, sizeof(uint8_t) * K * out_n, MEMORY_RAM);
    buffer wOutScale = createBuffer(s.dev.device, s.dev.physicalDevice, woINT8.scale, sizeof(float) * K * outBlocks, MEMORY_RAM);
    buffer wOutZero = createBuffer(s.dev.device, s.dev.physicalDevice, woINT8.z, sizeof(float) * K * outBlocks, MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * n_qk * dim, MEMORY_VRAM);
    buffer kOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kOut, sizeof(float) * n_qk * dim, MEMORY_VRAM);
    buffer vOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vOut, sizeof(float) * n_v * dim, MEMORY_VRAM);
    buffer zOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, zOut, sizeof(float) * n_v * dim, MEMORY_VRAM);
    buffer aOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, aOut, sizeof(float) * n_v, MEMORY_VRAM);
    buffer bOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, bOut, sizeof(float) * n_v, MEMORY_VRAM);
    buffer sBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, Sbuf, sizeof(float) * smat, MEMORY_VRAM);
    buffer yGatedBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, yGated, sizeof(float) * out_n, MEMORY_VRAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * out_n, MEMORY_VRAM);
    buffer wnBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, wn, sizeof(float) * dim, MEMORY_RAM);
    buffer aLogBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, aLog, sizeof(float) * n_v, MEMORY_RAM);
    buffer dtBiasBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, dtBias, sizeof(float) * n_v, MEMORY_RAM);
    buffer bufs[] = {xBuffer, x2Buffer, gammaBuffer, wInBuffer, wInScale, wInZero, wOutBuffer, wOutScale, wOutZero, qOutBuffer, kOutBuffer, vOutBuffer, zOutBuffer, aOutBuffer, bOutBuffer, sBuffer, yGatedBuffer, outBuffer, wnBuffer, aLogBuffer, dtBiasBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 21);
    free(twIn);
    free(twOut);

    operation ops[] = {
        {.shader = "RmsNorm-LinearProj-INT8.spv", .buffers = {xBuffer, gammaBuffer, wInBuffer, wInScale, wInZero, qOutBuffer, kOutBuffer, vOutBuffer, zOutBuffer, aOutBuffer, bOutBuffer}, .bufferCount = 11,
         .pushConstants = {1, proj_n, K, 2048, 4096, 8192, 12288, 12320}, .pushConstantCount = 8,
         .dispatchX = (proj_n + 255) / 256, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "GatedDeltaNet.spv", .buffers = {qOutBuffer, kOutBuffer, vOutBuffer, aOutBuffer, bOutBuffer, sBuffer, yGatedBuffer, zOutBuffer, wnBuffer, aLogBuffer, dtBiasBuffer}, .bufferCount = 11,
         .pushConstants = {n_v, n_qk, dim}, .pushConstantCount = 3,
         .dispatchX = n_v, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "GEMV-INT8.spv", .buffers = {yGatedBuffer, wOutBuffer, outBuffer, wOutScale, wOutZero}, .bufferCount = 5,
         .pushConstants = {1, out_n, K}, .pushConstantCount = 3,
         .dispatchX = out_n / 256, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "RmsNorm-LinearProj-INT8.spv", .buffers = {x2Buffer, gammaBuffer, wInBuffer, wInScale, wInZero, qOutBuffer, kOutBuffer, vOutBuffer, zOutBuffer, aOutBuffer, bOutBuffer}, .bufferCount = 11,
         .pushConstants = {1, proj_n, K, 2048, 4096, 8192, 12288, 12320}, .pushConstantCount = 8,
         .dispatchX = (proj_n + 255) / 256, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "GatedDeltaNet.spv", .buffers = {qOutBuffer, kOutBuffer, vOutBuffer, aOutBuffer, bOutBuffer, sBuffer, yGatedBuffer, zOutBuffer, wnBuffer, aLogBuffer, dtBiasBuffer}, .bufferCount = 11,
         .pushConstants = {n_v, n_qk, dim}, .pushConstantCount = 3,
         .dispatchX = n_v, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "GEMV-INT8.spv", .buffers = {yGatedBuffer, wOutBuffer, outBuffer, wOutScale, wOutZero}, .bufferCount = 5,
         .pushConstants = {1, out_n, K}, .pushConstantCount = 3,
         .dispatchX = out_n / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 6);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, sBuffer, Sbuf);
    report("GatedDeltaNet INT8", 100, out, ref, out_n, ms);
    report("GatedDeltaNet-S INT8", 100, Sbuf, S, smat, ms);

    destroy_buffers(s, bufs, 21);
    free(xn);
    free(xn2);
    free(p1);
    free(p2);
    free(wn);
    free(aLog);
    free(dtBias);
    free(S);
    free(yg1);
    free(yg2);
    free(ref);
    free(qOut);
    free(kOut);
    free(vOut);
    free(zOut);
    free(aOut);
    free(bOut);
    free(yGated);
    free(out);
    free(Sbuf);
}

void validateGatedDeltaNetINT4(session s, int K, float* input, float* input2, float* gamma, QuantizedData w_inINT4, QuantizedData woINT4) {
    int proj_n = 12352;
    int out_n = 4096;
    int n_qk = 16;
    int n_v = 32;
    int dim = 128;
    int smat = n_v * dim * dim;
    int inBlocks = (proj_n + 255) / 256;
    int outBlocks = out_n / 256;

    float* xn = (float*)malloc(sizeof(float) * K);
    float* xn2 = (float*)malloc(sizeof(float) * K);
    rms_norm_apply(input, gamma, xn, K);
    rms_norm_apply(input2, gamma, xn2, K);
    float* p1 = (float*)malloc(sizeof(float) * proj_n);
    float* p2 = (float*)malloc(sizeof(float) * proj_n);
    gemv_ref_int4(xn, &w_inINT4, p1, proj_n, K);
    gemv_ref_int4(xn2, &w_inINT4, p2, proj_n, K);
    float* wn = getData(64051, 1, dim);
    float* aLog = getData(64052, 1, n_v);
    float* dtBias = getData(64053, 1, n_v);
    float* S = (float*)calloc(smat, sizeof(float));
    float* yg1 = (float*)malloc(sizeof(float) * out_n);
    float* yg2 = (float*)malloc(sizeof(float) * out_n);
    deltanet_ref(p1, aLog, dtBias, wn, S, yg1, n_qk, n_v, dim);
    deltanet_ref(p2, aLog, dtBias, wn, S, yg2, n_qk, n_v, dim);
    float* ref = (float*)malloc(sizeof(float) * out_n);
    gemv_ref_int4(yg2, &woINT4, ref, out_n, K);

    float* qOut = (float*)calloc(n_qk * dim, sizeof(float));
    float* kOut = (float*)calloc(n_qk * dim, sizeof(float));
    float* vOut = (float*)calloc(n_v * dim, sizeof(float));
    float* zOut = (float*)calloc(n_v * dim, sizeof(float));
    float* aOut = (float*)calloc(n_v, sizeof(float));
    float* bOut = (float*)calloc(n_v, sizeof(float));
    float* yGated = (float*)calloc(out_n, sizeof(float));
    float* out = (float*)calloc(out_n, sizeof(float));
    float* Sbuf = (float*)calloc(smat, sizeof(float));

    uint8_t* twIn = (uint8_t*)malloc(sizeof(uint8_t) * K * proj_n / 2);
    uint8_t* twOut = (uint8_t*)malloc(sizeof(uint8_t) * K * out_n / 2);
    transpose_block16(w_inINT4.data, twIn, K, proj_n, QUANT_INT4);
    transpose_block16(woINT4.data, twOut, K, out_n, QUANT_INT4);

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * K, MEMORY_RAM);
    buffer x2Buffer = createBuffer(s.dev.device, s.dev.physicalDevice, input2, sizeof(float) * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer wInBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, twIn, sizeof(uint8_t) * K * proj_n / 2, MEMORY_RAM);
    buffer wInScale = createBuffer(s.dev.device, s.dev.physicalDevice, w_inINT4.scale, sizeof(float) * K * inBlocks, MEMORY_RAM);
    buffer wInZero = createBuffer(s.dev.device, s.dev.physicalDevice, w_inINT4.z, sizeof(float) * K * inBlocks, MEMORY_RAM);
    buffer wOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, twOut, sizeof(uint8_t) * K * out_n / 2, MEMORY_RAM);
    buffer wOutScale = createBuffer(s.dev.device, s.dev.physicalDevice, woINT4.scale, sizeof(float) * K * outBlocks, MEMORY_RAM);
    buffer wOutZero = createBuffer(s.dev.device, s.dev.physicalDevice, woINT4.z, sizeof(float) * K * outBlocks, MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * n_qk * dim, MEMORY_VRAM);
    buffer kOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kOut, sizeof(float) * n_qk * dim, MEMORY_VRAM);
    buffer vOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vOut, sizeof(float) * n_v * dim, MEMORY_VRAM);
    buffer zOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, zOut, sizeof(float) * n_v * dim, MEMORY_VRAM);
    buffer aOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, aOut, sizeof(float) * n_v, MEMORY_VRAM);
    buffer bOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, bOut, sizeof(float) * n_v, MEMORY_VRAM);
    buffer sBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, Sbuf, sizeof(float) * smat, MEMORY_VRAM);
    buffer yGatedBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, yGated, sizeof(float) * out_n, MEMORY_VRAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * out_n, MEMORY_VRAM);
    buffer wnBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, wn, sizeof(float) * dim, MEMORY_RAM);
    buffer aLogBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, aLog, sizeof(float) * n_v, MEMORY_RAM);
    buffer dtBiasBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, dtBias, sizeof(float) * n_v, MEMORY_RAM);
    buffer bufs[] = {xBuffer, x2Buffer, gammaBuffer, wInBuffer, wInScale, wInZero, wOutBuffer, wOutScale, wOutZero, qOutBuffer, kOutBuffer, vOutBuffer, zOutBuffer, aOutBuffer, bOutBuffer, sBuffer, yGatedBuffer, outBuffer, wnBuffer, aLogBuffer, dtBiasBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 21);
    free(twIn);
    free(twOut);

    operation ops[] = {
        {.shader = "RmsNorm-LinearProj-INT4.spv", .buffers = {xBuffer, gammaBuffer, wInBuffer, wInScale, wInZero, qOutBuffer, kOutBuffer, vOutBuffer, zOutBuffer, aOutBuffer, bOutBuffer}, .bufferCount = 11,
         .pushConstants = {1, proj_n, K, 2048, 4096, 8192, 12288, 12320}, .pushConstantCount = 8,
         .dispatchX = (proj_n + 255) / 256, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "GatedDeltaNet.spv", .buffers = {qOutBuffer, kOutBuffer, vOutBuffer, aOutBuffer, bOutBuffer, sBuffer, yGatedBuffer, zOutBuffer, wnBuffer, aLogBuffer, dtBiasBuffer}, .bufferCount = 11,
         .pushConstants = {n_v, n_qk, dim}, .pushConstantCount = 3,
         .dispatchX = n_v, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "GEMV-INT4.spv", .buffers = {yGatedBuffer, wOutBuffer, outBuffer, wOutScale, wOutZero}, .bufferCount = 5,
         .pushConstants = {1, out_n, K}, .pushConstantCount = 3,
         .dispatchX = out_n / 256, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "RmsNorm-LinearProj-INT4.spv", .buffers = {x2Buffer, gammaBuffer, wInBuffer, wInScale, wInZero, qOutBuffer, kOutBuffer, vOutBuffer, zOutBuffer, aOutBuffer, bOutBuffer}, .bufferCount = 11,
         .pushConstants = {1, proj_n, K, 2048, 4096, 8192, 12288, 12320}, .pushConstantCount = 8,
         .dispatchX = (proj_n + 255) / 256, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "GatedDeltaNet.spv", .buffers = {qOutBuffer, kOutBuffer, vOutBuffer, aOutBuffer, bOutBuffer, sBuffer, yGatedBuffer, zOutBuffer, wnBuffer, aLogBuffer, dtBiasBuffer}, .bufferCount = 11,
         .pushConstants = {n_v, n_qk, dim}, .pushConstantCount = 3,
         .dispatchX = n_v, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "GEMV-INT4.spv", .buffers = {yGatedBuffer, wOutBuffer, outBuffer, wOutScale, wOutZero}, .bufferCount = 5,
         .pushConstants = {1, out_n, K}, .pushConstantCount = 3,
         .dispatchX = out_n / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 6);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, sBuffer, Sbuf);
    report("GatedDeltaNet INT4", 100, out, ref, out_n, ms);
    report("GatedDeltaNet-S INT4", 100, Sbuf, S, smat, ms);

    destroy_buffers(s, bufs, 21);
    free(xn);
    free(xn2);
    free(p1);
    free(p2);
    free(wn);
    free(aLog);
    free(dtBias);
    free(S);
    free(yg1);
    free(yg2);
    free(ref);
    free(qOut);
    free(kOut);
    free(vOut);
    free(zOut);
    free(aOut);
    free(bOut);
    free(yGated);
    free(out);
    free(Sbuf);
}

void validateGatedDeltaNetGEMMFP16(session s, int M, int K, float* input, float* gamma, uint16_t* w_inFP16, uint16_t* woFP16) {
    int proj_n = 12352;
    int out_n = 4096;
    int n_qk = 16;
    int n_v = 32;
    int dim = 128;
    int smat = n_v * dim * dim;

    const float wscale = 1.0f / 64.0f;
    uint16_t* w_in_scaled = (uint16_t*)malloc(sizeof(uint16_t) * K * proj_n);
    for (int i = 0; i < K * proj_n; i++)
        w_in_scaled[i] = float_to_fp16(fp16_to_float(w_inFP16[i]) * wscale);

    float* S_ref = (float*)calloc(smat, sizeof(float));
    float* yg_all = (float*)malloc(sizeof(float) * M * out_n);
    float* wn = getData(64051, 1, dim);
    float* aLog = getData(64052, 1, n_v);
    float* dtBias = getData(64053, 1, n_v);
    for (int m = 0; m < M; m++) {
        float* xn = (float*)malloc(sizeof(float) * K);
        rms_norm_apply(input + m * K, gamma, xn, K);
        float* proj = (float*)malloc(sizeof(float) * proj_n);
        gemv_ref_fp16(xn, w_in_scaled, proj, proj_n, K);
        deltanet_ref(proj, aLog, dtBias, wn, S_ref, yg_all + m * out_n, n_qk, n_v, dim);
        free(xn);
        free(proj);
    }
    float* ref = (float*)malloc(sizeof(float) * M * out_n);
    gemm_ref_fp16(yg_all, woFP16, ref, M, out_n, K);

    float* qOut = (float*)calloc(M * n_qk * dim, sizeof(float));
    float* kOut = (float*)calloc(M * n_qk * dim, sizeof(float));
    float* vOut = (float*)calloc(M * n_v * dim, sizeof(float));
    float* zOut = (float*)calloc(M * n_v * dim, sizeof(float));
    float* aOut = (float*)calloc(M * n_v, sizeof(float));
    float* bOut = (float*)calloc(M * n_v, sizeof(float));
    float* yGated = (float*)calloc(M * out_n, sizeof(float));
    float* out = (float*)calloc(M * out_n, sizeof(float));
    float* Sbuf = (float*)calloc(smat, sizeof(float));

    uint16_t* twIn = (uint16_t*)malloc(sizeof(uint16_t) * K * proj_n);
    uint16_t* twOut = (uint16_t*)malloc(sizeof(uint16_t) * K * out_n);
    transpose_block16((uint8_t*)w_in_scaled, (uint8_t*)twIn, K, proj_n, QUANT_FP16);
    transpose_block16((uint8_t*)woFP16, (uint8_t*)twOut, K, out_n, QUANT_FP16);

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer wInBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, twIn, sizeof(uint16_t) * K * proj_n, MEMORY_RAM);
    buffer wOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, twOut, sizeof(uint16_t) * K * out_n, MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * M * n_qk * dim, MEMORY_VRAM);
    buffer kOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kOut, sizeof(float) * M * n_qk * dim, MEMORY_VRAM);
    buffer vOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vOut, sizeof(float) * M * n_v * dim, MEMORY_VRAM);
    buffer zOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, zOut, sizeof(float) * M * n_v * dim, MEMORY_VRAM);
    buffer aOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, aOut, sizeof(float) * M * n_v, MEMORY_VRAM);
    buffer bOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, bOut, sizeof(float) * M * n_v, MEMORY_VRAM);
    buffer sBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, Sbuf, sizeof(float) * smat, MEMORY_VRAM);
    buffer yGatedBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, yGated, sizeof(float) * M * out_n, MEMORY_VRAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * out_n, MEMORY_VRAM);
    buffer wnBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, wn, sizeof(float) * dim, MEMORY_RAM);
    buffer aLogBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, aLog, sizeof(float) * n_v, MEMORY_RAM);
    buffer dtBiasBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, dtBias, sizeof(float) * n_v, MEMORY_RAM);
    buffer bufs[] = {xBuffer, gammaBuffer, wInBuffer, wOutBuffer, qOutBuffer, kOutBuffer, vOutBuffer, zOutBuffer, aOutBuffer, bOutBuffer, sBuffer, yGatedBuffer, outBuffer, wnBuffer, aLogBuffer, dtBiasBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 16);
    free(twIn);
    free(twOut);

    operation ops[] = {
        {.shader = "RmsNorm-LinearProj-GEMM-FP16.spv", .buffers = {xBuffer, gammaBuffer, wInBuffer, qOutBuffer, kOutBuffer, vOutBuffer, zOutBuffer, aOutBuffer, bOutBuffer}, .bufferCount = 9,
         .pushConstants = {M, proj_n, K, 2048, 4096, 8192, 12288, 12320}, .pushConstantCount = 8,
         .dispatchX = proj_n / 16, .dispatchY = M / 16, .dispatchZ = 1},
        {.shader = "GatedDeltaNet-GEMM.spv", .buffers = {qOutBuffer, kOutBuffer, vOutBuffer, aOutBuffer, bOutBuffer, sBuffer, yGatedBuffer, zOutBuffer, wnBuffer, aLogBuffer, dtBiasBuffer}, .bufferCount = 11,
         .pushConstants = {M, 32, 16, M}, .pushConstantCount = 4,
         .dispatchX = n_v, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "GEMM-FP16.spv", .buffers = {yGatedBuffer, wOutBuffer, outBuffer}, .bufferCount = 3,
         .pushConstants = {M, out_n, K}, .pushConstantCount = 3,
         .dispatchX = out_n / 16, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 3);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, sBuffer, Sbuf);
    report("GatedDeltaNet-GEMM FP16", 100, out, ref, M * out_n, ms);
    report("GatedDeltaNet-GEMM-S FP16", 100, Sbuf, S_ref, smat, ms);

    destroy_buffers(s, bufs, 16);
    free(w_in_scaled);
    free(S_ref); free(yg_all); free(ref);
    free(wn); free(aLog); free(dtBias);
    free(qOut); free(kOut); free(vOut); free(zOut); free(aOut); free(bOut);
    free(yGated); free(out); free(Sbuf);
}

void validateGatedDeltaNetGEMMINT8(session s, int M, int K, float* input, float* gamma, QuantizedData w_inINT8, QuantizedData woINT8) {
    int proj_n = 12352;
    int out_n = 4096;
    int n_qk = 16;
    int n_v = 32;
    int dim = 128;
    int smat = n_v * dim * dim;
    int inBlocks = (proj_n + 255) / 256;
    int outBlocks = out_n / 256;

    const float wscale = 1.0f / 64.0f;
    QuantizedData w_in_scaled = w_inINT8;
    float* w_in_s8 = (float*)malloc(sizeof(float) * K * inBlocks);
    float* w_in_z8 = (float*)malloc(sizeof(float) * K * inBlocks);
    for (int i = 0; i < K * inBlocks; i++) {
        w_in_s8[i] = w_inINT8.scale[i] * wscale;
        w_in_z8[i] = w_inINT8.z[i] * wscale;
    }
    w_in_scaled.scale = w_in_s8;
    w_in_scaled.z = w_in_z8;

    float* S_ref = (float*)calloc(smat, sizeof(float));
    float* yg_all = (float*)malloc(sizeof(float) * M * out_n);
    float* wn = getData(64051, 1, dim);
    float* aLog = getData(64052, 1, n_v);
    float* dtBias = getData(64053, 1, n_v);
    for (int m = 0; m < M; m++) {
        float* xn = (float*)malloc(sizeof(float) * K);
        rms_norm_apply(input + m * K, gamma, xn, K);
        float* proj = (float*)malloc(sizeof(float) * proj_n);
        gemv_ref_int8(xn, &w_in_scaled, proj, proj_n, K);
        deltanet_ref(proj, aLog, dtBias, wn, S_ref, yg_all + m * out_n, n_qk, n_v, dim);
        free(xn);
        free(proj);
    }
    float* ref = (float*)malloc(sizeof(float) * M * out_n);
    gemm_ref_int8(yg_all, &woINT8, ref, M, out_n, K);

    float* qOut = (float*)calloc(M * n_qk * dim, sizeof(float));
    float* kOut = (float*)calloc(M * n_qk * dim, sizeof(float));
    float* vOut = (float*)calloc(M * n_v * dim, sizeof(float));
    float* aOut = (float*)calloc(M * n_v, sizeof(float));
    float* bOut = (float*)calloc(M * n_v, sizeof(float));
    float* yGated = (float*)calloc(M * out_n, sizeof(float));
    float* out = (float*)calloc(M * out_n, sizeof(float));
    float* Sbuf = (float*)calloc(smat, sizeof(float));

    uint8_t* twIn = (uint8_t*)malloc(sizeof(uint8_t) * K * proj_n);
    uint8_t* twOut = (uint8_t*)malloc(sizeof(uint8_t) * K * out_n);
    transpose_block16(w_inINT8.data, twIn, K, proj_n, QUANT_INT8);
    transpose_block16(woINT8.data, twOut, K, out_n, QUANT_INT8);

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer wInBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, twIn, sizeof(uint8_t) * K * proj_n, MEMORY_RAM);
    buffer wInScale = createBuffer(s.dev.device, s.dev.physicalDevice, w_in_scaled.scale, sizeof(float) * K * inBlocks, MEMORY_RAM);
    buffer wInZero = createBuffer(s.dev.device, s.dev.physicalDevice, w_in_scaled.z, sizeof(float) * K * inBlocks, MEMORY_RAM);
    buffer wOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, twOut, sizeof(uint8_t) * K * out_n, MEMORY_RAM);
    buffer wOutScale = createBuffer(s.dev.device, s.dev.physicalDevice, woINT8.scale, sizeof(float) * K * outBlocks, MEMORY_RAM);
    buffer wOutZero = createBuffer(s.dev.device, s.dev.physicalDevice, woINT8.z, sizeof(float) * K * outBlocks, MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * M * n_qk * dim, MEMORY_VRAM);
    buffer kOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kOut, sizeof(float) * M * n_qk * dim, MEMORY_VRAM);
    buffer vOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vOut, sizeof(float) * M * n_v * dim, MEMORY_VRAM);
    buffer aOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, aOut, sizeof(float) * M * n_v, MEMORY_VRAM);
    buffer bOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, bOut, sizeof(float) * M * n_v, MEMORY_VRAM);
    buffer sBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, Sbuf, sizeof(float) * smat, MEMORY_VRAM);
    buffer yGatedBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, yGated, sizeof(float) * M * out_n, MEMORY_VRAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * out_n, MEMORY_VRAM);
    buffer bufs[] = {xBuffer, gammaBuffer, wInBuffer, wInScale, wInZero, wOutBuffer, wOutScale, wOutZero, qOutBuffer, kOutBuffer, vOutBuffer, aOutBuffer, bOutBuffer, sBuffer, yGatedBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 16);
    free(twIn);
    free(twOut);

    operation ops[] = {
        {.shader = "RmsNorm-LinearProj-GEMM-INT8.spv", .buffers = {xBuffer, gammaBuffer, wInBuffer, qOutBuffer, kOutBuffer, vOutBuffer, aOutBuffer, bOutBuffer, wInScale, wInZero}, .bufferCount = 10,
         .pushConstants = {M, proj_n, K, 2048, 4096, 8192, 12288, 12320}, .pushConstantCount = 8,
         .dispatchX = proj_n / 16, .dispatchY = M / 16, .dispatchZ = 1},
        {.shader = "GatedDeltaNet-GEMM.spv", .buffers = {qOutBuffer, kOutBuffer, vOutBuffer, aOutBuffer, bOutBuffer, sBuffer, yGatedBuffer}, .bufferCount = 7,
         .pushConstants = {M, 32, 16, M}, .pushConstantCount = 4,
         .dispatchX = n_v, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "GEMM-INT8.spv", .buffers = {yGatedBuffer, wOutBuffer, outBuffer, wOutScale, wOutZero}, .bufferCount = 5,
         .pushConstants = {M, out_n, K}, .pushConstantCount = 3,
         .dispatchX = out_n / 16, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 3);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, sBuffer, Sbuf);
    report("GatedDeltaNet-GEMM INT8", 100, out, ref, M * out_n, ms);
    report("GatedDeltaNet-GEMM-S INT8", 100, Sbuf, S_ref, smat, ms);

    destroy_buffers(s, bufs, 16);
    free(w_in_s8); free(w_in_z8);
    free(S_ref); free(yg_all); free(ref);
    free(qOut); free(kOut); free(vOut); free(aOut); free(bOut);
    free(yGated); free(out); free(Sbuf);
}

void validateGatedDeltaNetGEMMINT4(session s, int M, int K, float* input, float* gamma, QuantizedData w_inINT4, QuantizedData woINT4) {
    int proj_n = 12352;
    int out_n = 4096;
    int n_qk = 16;
    int n_v = 32;
    int dim = 128;
    int smat = n_v * dim * dim;
    int inBlocks = (proj_n + 255) / 256;
    int outBlocks = out_n / 256;

    const float wscale = 1.0f / 64.0f;
    QuantizedData w_in_scaled = w_inINT4;
    float* w_in_s4 = (float*)malloc(sizeof(float) * K * inBlocks);
    float* w_in_z4 = (float*)malloc(sizeof(float) * K * inBlocks);
    for (int i = 0; i < K * inBlocks; i++) {
        w_in_s4[i] = w_inINT4.scale[i] * wscale;
        w_in_z4[i] = w_inINT4.z[i] * wscale;
    }
    w_in_scaled.scale = w_in_s4;
    w_in_scaled.z = w_in_z4;

    float* S_ref = (float*)calloc(smat, sizeof(float));
    float* yg_all = (float*)malloc(sizeof(float) * M * out_n);
    float* wn = getData(64051, 1, dim);
    float* aLog = getData(64052, 1, n_v);
    float* dtBias = getData(64053, 1, n_v);
    for (int m = 0; m < M; m++) {
        float* xn = (float*)malloc(sizeof(float) * K);
        rms_norm_apply(input + m * K, gamma, xn, K);
        float* proj = (float*)malloc(sizeof(float) * proj_n);
        gemv_ref_int4(xn, &w_in_scaled, proj, proj_n, K);
        deltanet_ref(proj, aLog, dtBias, wn, S_ref, yg_all + m * out_n, n_qk, n_v, dim);
        free(xn);
        free(proj);
    }
    float* ref = (float*)malloc(sizeof(float) * M * out_n);
    gemm_ref_int4(yg_all, &woINT4, ref, M, out_n, K);

    float* qOut = (float*)calloc(M * n_qk * dim, sizeof(float));
    float* kOut = (float*)calloc(M * n_qk * dim, sizeof(float));
    float* vOut = (float*)calloc(M * n_v * dim, sizeof(float));
    float* aOut = (float*)calloc(M * n_v, sizeof(float));
    float* bOut = (float*)calloc(M * n_v, sizeof(float));
    float* yGated = (float*)calloc(M * out_n, sizeof(float));
    float* out = (float*)calloc(M * out_n, sizeof(float));
    float* Sbuf = (float*)calloc(smat, sizeof(float));

    uint8_t* twIn = (uint8_t*)malloc(sizeof(uint8_t) * K * proj_n / 2);
    uint8_t* twOut = (uint8_t*)malloc(sizeof(uint8_t) * K * out_n / 2);
    transpose_block16(w_inINT4.data, twIn, K, proj_n, QUANT_INT4);
    transpose_block16(woINT4.data, twOut, K, out_n, QUANT_INT4);

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer wInBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, twIn, sizeof(uint8_t) * K * proj_n / 2, MEMORY_RAM);
    buffer wInScale = createBuffer(s.dev.device, s.dev.physicalDevice, w_in_scaled.scale, sizeof(float) * K * inBlocks, MEMORY_RAM);
    buffer wInZero = createBuffer(s.dev.device, s.dev.physicalDevice, w_in_scaled.z, sizeof(float) * K * inBlocks, MEMORY_RAM);
    buffer wOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, twOut, sizeof(uint8_t) * K * out_n / 2, MEMORY_RAM);
    buffer wOutScale = createBuffer(s.dev.device, s.dev.physicalDevice, woINT4.scale, sizeof(float) * K * outBlocks, MEMORY_RAM);
    buffer wOutZero = createBuffer(s.dev.device, s.dev.physicalDevice, woINT4.z, sizeof(float) * K * outBlocks, MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * M * n_qk * dim, MEMORY_VRAM);
    buffer kOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kOut, sizeof(float) * M * n_qk * dim, MEMORY_VRAM);
    buffer vOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vOut, sizeof(float) * M * n_v * dim, MEMORY_VRAM);
    buffer aOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, aOut, sizeof(float) * M * n_v, MEMORY_VRAM);
    buffer bOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, bOut, sizeof(float) * M * n_v, MEMORY_VRAM);
    buffer sBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, Sbuf, sizeof(float) * smat, MEMORY_VRAM);
    buffer yGatedBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, yGated, sizeof(float) * M * out_n, MEMORY_VRAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * out_n, MEMORY_VRAM);
    buffer bufs[] = {xBuffer, gammaBuffer, wInBuffer, wInScale, wInZero, wOutBuffer, wOutScale, wOutZero, qOutBuffer, kOutBuffer, vOutBuffer, aOutBuffer, bOutBuffer, sBuffer, yGatedBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 16);
    free(twIn);
    free(twOut);

    operation ops[] = {
        {.shader = "RmsNorm-LinearProj-GEMM-INT4.spv", .buffers = {xBuffer, gammaBuffer, wInBuffer, qOutBuffer, kOutBuffer, vOutBuffer, aOutBuffer, bOutBuffer, wInScale, wInZero}, .bufferCount = 10,
         .pushConstants = {M, proj_n, K, 2048, 4096, 8192, 12288, 12320}, .pushConstantCount = 8,
         .dispatchX = proj_n / 16, .dispatchY = M / 16, .dispatchZ = 1},
        {.shader = "GatedDeltaNet-GEMM.spv", .buffers = {qOutBuffer, kOutBuffer, vOutBuffer, aOutBuffer, bOutBuffer, sBuffer, yGatedBuffer}, .bufferCount = 7,
         .pushConstants = {M, 32, 16, M}, .pushConstantCount = 4,
         .dispatchX = n_v, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "GEMM-INT4.spv", .buffers = {yGatedBuffer, wOutBuffer, outBuffer, wOutScale, wOutZero}, .bufferCount = 5,
         .pushConstants = {M, out_n, K}, .pushConstantCount = 3,
         .dispatchX = out_n / 16, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 3);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, sBuffer, Sbuf);
    report("GatedDeltaNet-GEMM INT4", 100, out, ref, M * out_n, ms);
    report("GatedDeltaNet-GEMM-S INT4", 100, Sbuf, S_ref, smat, ms);

    destroy_buffers(s, bufs, 16);
    free(w_in_s4); free(w_in_z4);
    free(S_ref); free(yg_all); free(ref);
    free(qOut); free(kOut); free(vOut); free(aOut); free(bOut);
    free(yGated); free(out); free(Sbuf);
}

void validateGemmAddFP16(session s, int M, int N, int K, float* input, float* residual, uint16_t* weightFP16) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint16_t* transposed = (uint16_t*)malloc(sizeof(uint16_t) * K * N);
    transpose_block16((uint8_t*)weightFP16, (uint8_t*)transposed, K, N, QUANT_FP16);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint16_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer residualBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, residual, sizeof(float) * M * N, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, weightBuffer, outBuffer, residualBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 4);
    free(transposed);

    operation ops[] = {
        {.shader = "GEMM-ADD-FP16.spv", .buffers = {inputBuffer, weightBuffer, outBuffer, residualBuffer}, .bufferCount = 4,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = N / 64, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemm_ref_fp16(input, weightFP16, ref, M, N, K);
    for (int i = 0; i < M * N; i++) ref[i] += residual[i];
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMM-ADD FP16", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 4);
    free(out);
    free(ref);
}

void validateGemmAddINT8(session s, int M, int N, int K, float* input, float* residual, QuantizedData weightINT8) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N);
    transpose_block16(weightINT8.data, transposed, K, N, QUANT_INT8);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.scale, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.z, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer residualBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, residual, sizeof(float) * M * N, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer, residualBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 6);
    free(transposed);

    operation ops[] = {
        {.shader = "GEMM-ADD-INT8.spv", .buffers = {inputBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer, residualBuffer}, .bufferCount = 6,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = N / 64, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemm_ref_int8(input, &weightINT8, ref, M, N, K);
    for (int i = 0; i < M * N; i++) ref[i] += residual[i];
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMM-ADD INT8", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 6);
    free(out);
    free(ref);
}

void validateGemmAddINT4(session s, int M, int N, int K, float* input, float* residual, QuantizedData weightINT4) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N / 2);
    transpose_block16(weightINT4.data, transposed, K, N, QUANT_INT4);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N / 2, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.scale, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.z, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer residualBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, residual, sizeof(float) * M * N, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer, residualBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 6);
    free(transposed);

    operation ops[] = {
        {.shader = "GEMM-ADD-INT4.spv", .buffers = {inputBuffer, weightBuffer, outBuffer, scaleBuffer, zeroBuffer, residualBuffer}, .bufferCount = 6,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = N / 64, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    gemm_ref_int4(input, &weightINT4, ref, M, N, K);
    for (int i = 0; i < M * N; i++) ref[i] += residual[i];
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("GEMM-ADD INT4", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 6);
    free(out);
    free(ref);
}

void validateQkvRopeSplitKFP16(session s, int K, int qkv_heads, int qkv_kv_heads, int qkv_dim, float* input, float* gamma, uint16_t* qkv_weightFP16, float* qkv_theta) {
    int k = K;
    int heads = qkv_heads;
    int kv_heads = qkv_kv_heads;
    int dim = qkv_dim;
    int gate_heads = heads;
    int n_total = (heads + gate_heads + 2 * kv_heads) * dim;
    int g_offset = heads * dim;
    int k_offset = (heads + gate_heads) * dim;
    int v_offset = (heads + gate_heads + kv_heads) * dim;
    int rows = kv_heads * dim;
    int ctx = 33;
    int seq_len = ctx + 1;
    uint32_t posVal = (uint32_t)ctx;

    float* xn = (float*)malloc(sizeof(float) * k);
    rms_norm_apply(input, gamma, xn, k);
    float* proj = (float*)malloc(sizeof(float) * n_total);
    gemv_ref_fp16(xn, qkv_weightFP16, proj, n_total, k);
    float* qref = (float*)malloc(sizeof(float) * heads * dim);
    float* kref = (float*)malloc(sizeof(float) * rows);
    float* vref = (float*)malloc(sizeof(float) * rows);
    float* gref = (float*)malloc(sizeof(float) * gate_heads * dim);
    float* qgamma = getData(61001, 1, dim);
    float* kgamma = getData(62001, 1, dim);
    qkv_rope_ref(proj, qkv_theta, qref, kref, vref, gref, n_total, g_offset, k_offset, v_offset, dim, ctx, qgamma, kgamma);

    float* qOut = (float*)calloc(heads * dim, sizeof(float));
    uint16_t* kCache = (uint16_t*)calloc((size_t)rows * 32768, sizeof(uint16_t));
    uint16_t* vCache = (uint16_t*)calloc(rows * seq_len, sizeof(uint16_t));

    uint16_t* transposed = (uint16_t*)malloc(sizeof(uint16_t) * k * n_total);
    transpose_block16((uint8_t*)qkv_weightFP16, (uint8_t*)transposed, k, n_total, QUANT_FP16);
    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * k, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * k, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint16_t) * k * n_total, MEMORY_RAM);
    float* pinit = (float*)calloc(4 * n_total, sizeof(float));
    buffer partialBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, pinit, sizeof(float) * 4 * n_total, MEMORY_RAM);
    buffer thetaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_theta, sizeof(float) * (QKV_ROPE_DIM / 2), MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * heads * dim, MEMORY_VRAM);
    buffer kCacheBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kCache, sizeof(uint16_t) * (size_t)rows * 32768, MEMORY_VRAM);
    buffer vCacheBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vCache, sizeof(uint16_t) * rows * seq_len, MEMORY_VRAM);
    buffer posBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, &posVal, sizeof(uint32_t), MEMORY_VRAM);
    float* gOut = (float*)calloc(gate_heads * dim, sizeof(float));
    buffer qGammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qgamma, sizeof(float) * dim, MEMORY_VRAM);
    buffer kGammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kgamma, sizeof(float) * dim, MEMORY_VRAM);
    buffer gOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gOut, sizeof(float) * gate_heads * dim, MEMORY_VRAM);
    buffer bufs[] = {xBuffer, gammaBuffer, weightBuffer, partialBuffer, thetaBuffer, qOutBuffer, kCacheBuffer, vCacheBuffer, posBuffer, qGammaBuffer, kGammaBuffer, gOutBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 12);
    free(transposed);
    free(pinit);

    operation ops[] = {
        {.shader = "RmsNorm-QKV-SplitK-FP16.spv", .buffers = {xBuffer, gammaBuffer, weightBuffer, partialBuffer}, .bufferCount = 4,
         .pushConstants = {1, n_total, k}, .pushConstantCount = 3,
         .dispatchX = n_total / 256, .dispatchY = 4, .dispatchZ = 1},
        {.shader = "Reduce-Rope-FP16.spv", .buffers = {partialBuffer, qOutBuffer, kCacheBuffer, vCacheBuffer, thetaBuffer, posBuffer, qGammaBuffer, kGammaBuffer, gOutBuffer}, .bufferCount = 9,
         .pushConstants = {n_total, g_offset, k_offset, v_offset, 32768, 256, 64, 16, 4}, .pushConstantCount = 9,
         .dispatchX = n_total / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 2);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, qOutBuffer, qOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kCacheBuffer, kCache);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vCacheBuffer, vCache);
    float* kstored = (float*)malloc(sizeof(float) * rows);
    float* vstored = (float*)malloc(sizeof(float) * rows);
    for (int r = 0; r < rows; r++) kstored[r] = fp16_to_float(kCache[r * 32768 + (seq_len - 1)]);
    for (int r = 0; r < rows; r++) vstored[r] = fp16_to_float(vCache[(seq_len - 1) * rows + r]);
    float* kvOut = (float*)malloc(sizeof(float) * 2 * rows);
    float* kvRef = (float*)malloc(sizeof(float) * 2 * rows);
    for (int r = 0; r < rows; r++) {
        kvOut[r] = kstored[r];
        kvOut[rows + r] = vstored[r];
        kvRef[r] = kref[r];
        kvRef[rows + r] = vref[r];
    }

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, gOutBuffer, gOut);
    report("QKV-Split-Rope FP16", 100, qOut, qref, heads * dim, ms);
    report("QKV-Split-Rope-g FP16", 100, gOut, gref, gate_heads * dim, ms);
    report("QKV-Split-Rope-kv FP16", 100, kvOut, kvRef, 2 * rows, ms);

    destroy_buffers(s, bufs, 12);
    free(xn);
    free(proj);
    free(qref);
    free(kref);
    free(vref);
    free(gref);
    free(qgamma);
    free(kgamma);
    free(gOut);
    free(qOut);
    free(kCache);
    free(vCache);
    free(kstored);
    free(vstored);
    free(kvOut);
    free(kvRef);
}

void validateQkvRopeSplitKINT8(session s, int K, int qkv_heads, int qkv_kv_heads, int qkv_dim, float* input, float* gamma, QuantizedData qkv_weightINT8, float* qkv_theta) {
    int k = K;
    int heads = qkv_heads;
    int kv_heads = qkv_kv_heads;
    int dim = qkv_dim;
    int gate_heads = heads;
    int n_total = (heads + gate_heads + 2 * kv_heads) * dim;
    int g_offset = heads * dim;
    int k_offset = (heads + gate_heads) * dim;
    int v_offset = (heads + gate_heads + kv_heads) * dim;
    int rows = kv_heads * dim;
    int ctx = 33;
    int seq_len = ctx + 1;
    int scaleCount = k * n_total / qkv_weightINT8.group_size;
    uint32_t posVal = (uint32_t)ctx;

    float* xn = (float*)malloc(sizeof(float) * k);
    rms_norm_apply(input, gamma, xn, k);
    float* proj = (float*)malloc(sizeof(float) * n_total);
    gemv_ref_int8(xn, &qkv_weightINT8, proj, n_total, k);
    float* qref = (float*)malloc(sizeof(float) * heads * dim);
    float* kref = (float*)malloc(sizeof(float) * rows);
    float* vref = (float*)malloc(sizeof(float) * rows);
    float* gref = (float*)malloc(sizeof(float) * gate_heads * dim);
    float* qgamma = getData(61001, 1, dim);
    float* kgamma = getData(62001, 1, dim);
    qkv_rope_ref(proj, qkv_theta, qref, kref, vref, gref, n_total, g_offset, k_offset, v_offset, dim, ctx, qgamma, kgamma);

    float* qOut = (float*)calloc(heads * dim, sizeof(float));
    uint8_t* kCache = (uint8_t*)calloc((size_t)rows * 32768, sizeof(uint8_t));
    uint8_t* vCache = (uint8_t*)calloc(rows * seq_len, sizeof(uint8_t));
    float* kScale = (float*)calloc(kv_heads * 32768, sizeof(float));
    float* kZero = (float*)calloc(kv_heads * 32768, sizeof(float));
    float* vScale = (float*)calloc(kv_heads * 32768, sizeof(float));
    float* vZero = (float*)calloc(kv_heads * 32768, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * k * n_total);
    transpose_block16(qkv_weightINT8.data, transposed, k, n_total, QUANT_INT8);
    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * k, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * k, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * k * n_total, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_weightINT8.scale, sizeof(float) * scaleCount, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_weightINT8.z, sizeof(float) * scaleCount, MEMORY_RAM);
    float* pinit = (float*)calloc(4 * n_total, sizeof(float));
    buffer partialBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, pinit, sizeof(float) * 4 * n_total, MEMORY_RAM);
    buffer thetaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_theta, sizeof(float) * (QKV_ROPE_DIM / 2), MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * heads * dim, MEMORY_VRAM);
    buffer kCacheBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kCache, sizeof(uint8_t) * (size_t)rows * 32768, MEMORY_VRAM);
    buffer vCacheBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vCache, sizeof(uint8_t) * rows * seq_len, MEMORY_VRAM);
    buffer kScaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kScale, sizeof(float) * kv_heads * 32768, MEMORY_VRAM);
    buffer kZeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kZero, sizeof(float) * kv_heads * 32768, MEMORY_VRAM);
    buffer vScaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vScale, sizeof(float) * kv_heads * 32768, MEMORY_VRAM);
    buffer vZeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vZero, sizeof(float) * kv_heads * 32768, MEMORY_VRAM);
    buffer posBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, &posVal, sizeof(uint32_t), MEMORY_VRAM);
    float* gOut = (float*)calloc(gate_heads * dim, sizeof(float));
    buffer qGammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qgamma, sizeof(float) * dim, MEMORY_VRAM);
    buffer kGammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kgamma, sizeof(float) * dim, MEMORY_VRAM);
    buffer gOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gOut, sizeof(float) * gate_heads * dim, MEMORY_VRAM);
    buffer bufs[] = {xBuffer, gammaBuffer, weightBuffer, scaleBuffer, zeroBuffer, partialBuffer, thetaBuffer, qOutBuffer, kCacheBuffer, vCacheBuffer, kScaleBuffer, kZeroBuffer, vScaleBuffer, vZeroBuffer, posBuffer, qGammaBuffer, kGammaBuffer, gOutBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 18);
    free(transposed);
    free(pinit);

    operation ops[] = {
        {.shader = "RmsNorm-QKV-SplitK-INT8.spv", .buffers = {xBuffer, gammaBuffer, weightBuffer, scaleBuffer, zeroBuffer, partialBuffer}, .bufferCount = 6,
         .pushConstants = {1, n_total, k}, .pushConstantCount = 3,
         .dispatchX = n_total / 256, .dispatchY = 4, .dispatchZ = 1},
        {.shader = "Reduce-Rope-INT8.spv", .buffers = {partialBuffer, qOutBuffer, kCacheBuffer, vCacheBuffer, kScaleBuffer, kZeroBuffer, vScaleBuffer, vZeroBuffer, thetaBuffer, posBuffer, qGammaBuffer, kGammaBuffer, gOutBuffer}, .bufferCount = 13,
         .pushConstants = {n_total, g_offset, k_offset, v_offset, 32768, 256, 64, 16, 4}, .pushConstantCount = 9,
         .dispatchX = n_total / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 2);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, qOutBuffer, qOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kCacheBuffer, kCache);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vCacheBuffer, vCache);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kScaleBuffer, kScale);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kZeroBuffer, kZero);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vScaleBuffer, vScale);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vZeroBuffer, vZero);
    float* kstored = (float*)malloc(sizeof(float) * rows);
    float* vstored = (float*)malloc(sizeof(float) * rows);
    for (int r = 0; r < rows; r++) {
        int h = r / dim;
        kstored[r] = (float)kCache[r * 32768 + (seq_len - 1)] * kScale[h * 32768 + seq_len - 1] - kZero[h * 32768 + seq_len - 1];
        vstored[r] = (float)vCache[(seq_len - 1) * rows + r] * vScale[h * 32768 + seq_len - 1] - vZero[h * 32768 + seq_len - 1];
    }
    float* kvOut = (float*)malloc(sizeof(float) * 2 * rows);
    float* kvRef = (float*)malloc(sizeof(float) * 2 * rows);
    for (int r = 0; r < rows; r++) {
        kvOut[r] = kstored[r];
        kvOut[rows + r] = vstored[r];
        kvRef[r] = kref[r];
        kvRef[rows + r] = vref[r];
    }

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, gOutBuffer, gOut);
    report("QKV-Split-Rope INT8", 100, qOut, qref, heads * dim, ms);
    report("QKV-Split-Rope-g INT8", 100, gOut, gref, gate_heads * dim, ms);
    report("QKV-Split-Rope-kv INT8", 100, kvOut, kvRef, 2 * rows, ms);

    destroy_buffers(s, bufs, 18);
    free(xn);
    free(proj);
    free(qref);
    free(kref);
    free(vref);
    free(gref);
    free(qgamma);
    free(kgamma);
    free(gOut);
    free(qOut);
    free(kCache);
    free(vCache);
    free(kScale);
    free(kZero);
    free(vScale);
    free(vZero);
    free(kstored);
    free(vstored);
    free(kvOut);
    free(kvRef);
}

void validateQkvRopeSplitKINT4(session s, int K, int qkv_heads, int qkv_kv_heads, int qkv_dim, float* input, float* gamma, QuantizedData qkv_weightINT4, float* qkv_theta) {
    int k = K;
    int heads = qkv_heads;
    int kv_heads = qkv_kv_heads;
    int dim = qkv_dim;
    int gate_heads = heads;
    int n_total = (heads + gate_heads + 2 * kv_heads) * dim;
    int g_offset = heads * dim;
    int k_offset = (heads + gate_heads) * dim;
    int v_offset = (heads + gate_heads + kv_heads) * dim;
    int rows = kv_heads * dim;
    int ctx = 33;
    int seq_len = ctx + 1;
    int scaleCount = k * n_total / qkv_weightINT4.group_size;
    uint32_t posVal = (uint32_t)ctx;

    float* xn = (float*)malloc(sizeof(float) * k);
    rms_norm_apply(input, gamma, xn, k);
    float* proj = (float*)malloc(sizeof(float) * n_total);
    gemv_ref_int4(xn, &qkv_weightINT4, proj, n_total, k);
    float* qref = (float*)malloc(sizeof(float) * heads * dim);
    float* kref = (float*)malloc(sizeof(float) * rows);
    float* vref = (float*)malloc(sizeof(float) * rows);
    float* gref = (float*)malloc(sizeof(float) * gate_heads * dim);
    float* qgamma = getData(61001, 1, dim);
    float* kgamma = getData(62001, 1, dim);
    qkv_rope_ref(proj, qkv_theta, qref, kref, vref, gref, n_total, g_offset, k_offset, v_offset, dim, ctx, qgamma, kgamma);

    float* qOut = (float*)calloc(heads * dim, sizeof(float));
    uint8_t* kCache = (uint8_t*)calloc((size_t)rows * 32768, sizeof(uint8_t));
    uint8_t* vCache = (uint8_t*)calloc(rows * seq_len, sizeof(uint8_t));
    float* kScale = (float*)calloc(kv_heads * 32768, sizeof(float));
    float* kZero = (float*)calloc(kv_heads * 32768, sizeof(float));
    float* vScale = (float*)calloc(kv_heads * 32768, sizeof(float));
    float* vZero = (float*)calloc(kv_heads * 32768, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * k * n_total / 2);
    transpose_block16(qkv_weightINT4.data, transposed, k, n_total, QUANT_INT4);
    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * k, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * k, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * k * n_total / 2, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_weightINT4.scale, sizeof(float) * scaleCount, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_weightINT4.z, sizeof(float) * scaleCount, MEMORY_RAM);
    float* pinit = (float*)calloc(4 * n_total, sizeof(float));
    buffer partialBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, pinit, sizeof(float) * 4 * n_total, MEMORY_RAM);
    buffer thetaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_theta, sizeof(float) * (QKV_ROPE_DIM / 2), MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * heads * dim, MEMORY_VRAM);
    buffer kCacheBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kCache, sizeof(uint8_t) * (size_t)rows * 32768, MEMORY_VRAM);
    buffer vCacheBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vCache, sizeof(uint8_t) * rows * seq_len, MEMORY_VRAM);
    buffer kScaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kScale, sizeof(float) * kv_heads * 32768, MEMORY_VRAM);
    buffer kZeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kZero, sizeof(float) * kv_heads * 32768, MEMORY_VRAM);
    buffer vScaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vScale, sizeof(float) * kv_heads * 32768, MEMORY_VRAM);
    buffer vZeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vZero, sizeof(float) * kv_heads * 32768, MEMORY_VRAM);
    buffer posBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, &posVal, sizeof(uint32_t), MEMORY_VRAM);
    float* gOut = (float*)calloc(gate_heads * dim, sizeof(float));
    buffer qGammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qgamma, sizeof(float) * dim, MEMORY_VRAM);
    buffer kGammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kgamma, sizeof(float) * dim, MEMORY_VRAM);
    buffer gOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gOut, sizeof(float) * gate_heads * dim, MEMORY_VRAM);
    buffer bufs[] = {xBuffer, gammaBuffer, weightBuffer, scaleBuffer, zeroBuffer, partialBuffer, thetaBuffer, qOutBuffer, kCacheBuffer, vCacheBuffer, kScaleBuffer, kZeroBuffer, vScaleBuffer, vZeroBuffer, posBuffer, qGammaBuffer, kGammaBuffer, gOutBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 18);
    free(transposed);
    free(pinit);

    operation ops[] = {
        {.shader = "RmsNorm-QKV-SplitK-INT4.spv", .buffers = {xBuffer, gammaBuffer, weightBuffer, scaleBuffer, zeroBuffer, partialBuffer}, .bufferCount = 6,
         .pushConstants = {1, n_total, k}, .pushConstantCount = 3,
         .dispatchX = n_total / 256, .dispatchY = 4, .dispatchZ = 1},
        {.shader = "Reduce-Rope-INT4.spv", .buffers = {partialBuffer, qOutBuffer, kCacheBuffer, vCacheBuffer, kScaleBuffer, kZeroBuffer, vScaleBuffer, vZeroBuffer, thetaBuffer, posBuffer, qGammaBuffer, kGammaBuffer, gOutBuffer}, .bufferCount = 13,
         .pushConstants = {n_total, g_offset, k_offset, v_offset, 32768, 256, 64, 16, 4}, .pushConstantCount = 9,
         .dispatchX = n_total / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 2);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, qOutBuffer, qOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kCacheBuffer, kCache);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vCacheBuffer, vCache);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kScaleBuffer, kScale);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kZeroBuffer, kZero);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vScaleBuffer, vScale);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vZeroBuffer, vZero);
    float* kstored = (float*)malloc(sizeof(float) * rows);
    float* vstored = (float*)malloc(sizeof(float) * rows);
    for (int r = 0; r < rows; r++) {
        int h = r / dim;
        kstored[r] = (float)kCache[r * 32768 + (seq_len - 1)] * kScale[h * 32768 + seq_len - 1] - kZero[h * 32768 + seq_len - 1];
        vstored[r] = (float)vCache[(seq_len - 1) * rows + r] * vScale[h * 32768 + seq_len - 1] - vZero[h * 32768 + seq_len - 1];
    }
    float* kvOut = (float*)malloc(sizeof(float) * 2 * rows);
    float* kvRef = (float*)malloc(sizeof(float) * 2 * rows);
    for (int r = 0; r < rows; r++) {
        kvOut[r] = kstored[r];
        kvOut[rows + r] = vstored[r];
        kvRef[r] = kref[r];
        kvRef[rows + r] = vref[r];
    }

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, gOutBuffer, gOut);
    report("QKV-Split-Rope INT4", 100, qOut, qref, heads * dim, ms);
    report("QKV-Split-Rope-g INT4", 100, gOut, gref, gate_heads * dim, ms);
    report("QKV-Split-Rope-kv INT4", 100, kvOut, kvRef, 2 * rows, ms);

    destroy_buffers(s, bufs, 18);
    free(xn);
    free(proj);
    free(qref);
    free(kref);
    free(vref);
    free(gref);
    free(qgamma);
    free(kgamma);
    free(gOut);
    free(qOut);
    free(kCache);
    free(vCache);
    free(kScale);
    free(kZero);
    free(vScale);
    free(vZero);
    free(kstored);
    free(vstored);
    free(kvOut);
    free(kvRef);
}

void validateRmsNormSwigluFfnGEMMFP16(session s, int M, int N, int K, float* input, float* gamma, uint16_t* weightFP16, uint16_t* weight2FP16) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint16_t* transposed = (uint16_t*)malloc(sizeof(uint16_t) * K * N);
    uint16_t* transposed2 = (uint16_t*)malloc(sizeof(uint16_t) * K * N);
    transpose_block16((uint8_t*)weightFP16, (uint8_t*)transposed, K, N, QUANT_FP16);
    transpose_block16((uint8_t*)weight2FP16, (uint8_t*)transposed2, K, N, QUANT_FP16);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer gateBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint16_t) * K * N, MEMORY_RAM);
    buffer upBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed2, sizeof(uint16_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 5);
    free(transposed);
    free(transposed2);

    operation ops[] = {
        {.shader = "RmsNorm-swiglu-ffn-GEMM-FP16.spv", .buffers = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, outBuffer}, .bufferCount = 5,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = N / 64, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    for (int m = 0; m < M; m++)
        swiglu_ref_fp16(input + m * K, gamma, weightFP16, weight2FP16, ref + m * N, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("RmsNorm-swiglu-ffn-GEMM-FP16", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 5);
    free(out);
    free(ref);
}

void validateRmsNormSwigluFfnGEMMINT8(session s, int M, int N, int K, float* input, float* gamma, QuantizedData weightINT8, QuantizedData weight2INT8) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N);
    uint8_t* transposed2 = (uint8_t*)malloc(sizeof(uint8_t) * K * N);
    transpose_block16(weightINT8.data, transposed, K, N, QUANT_INT8);
    transpose_block16(weight2INT8.data, transposed2, K, N, QUANT_INT8);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer gateBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N, MEMORY_RAM);
    buffer upBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed2, sizeof(uint8_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer gateScale = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.scale, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer gateZero = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.z, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer upScale = createBuffer(s.dev.device, s.dev.physicalDevice, weight2INT8.scale, sizeof(float) * K * N / weight2INT8.group_size, MEMORY_RAM);
    buffer upZero = createBuffer(s.dev.device, s.dev.physicalDevice, weight2INT8.z, sizeof(float) * K * N / weight2INT8.group_size, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, outBuffer, gateScale, gateZero, upScale, upZero};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 9);
    free(transposed);
    free(transposed2);

    operation ops[] = {
        {.shader = "RmsNorm-swiglu-ffn-GEMM-INT8.spv", .buffers = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, outBuffer, gateScale, gateZero, upScale, upZero}, .bufferCount = 9,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = N / 64, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    for (int m = 0; m < M; m++)
        swiglu_ref_int8(input + m * K, gamma, &weightINT8, &weight2INT8, ref + m * N, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("RmsNorm-swiglu-ffn-GEMM-INT8", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 9);
    free(out);
    free(ref);
}

void validateRmsNormSwigluFfnGEMMINT4(session s, int M, int N, int K, float* input, float* gamma, QuantizedData weightINT4, QuantizedData weight2INT4) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N / 2);
    uint8_t* transposed2 = (uint8_t*)malloc(sizeof(uint8_t) * K * N / 2);
    transpose_block16(weightINT4.data, transposed, K, N, QUANT_INT4);
    transpose_block16(weight2INT4.data, transposed2, K, N, QUANT_INT4);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer gateBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N / 2, MEMORY_RAM);
    buffer upBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed2, sizeof(uint8_t) * K * N / 2, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer gateScale = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.scale, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer gateZero = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.z, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer upScale = createBuffer(s.dev.device, s.dev.physicalDevice, weight2INT4.scale, sizeof(float) * K * N / weight2INT4.group_size, MEMORY_RAM);
    buffer upZero = createBuffer(s.dev.device, s.dev.physicalDevice, weight2INT4.z, sizeof(float) * K * N / weight2INT4.group_size, MEMORY_RAM);
    buffer bufs[] = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, outBuffer, gateScale, gateZero, upScale, upZero};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 9);
    free(transposed);
    free(transposed2);

    operation ops[] = {
        {.shader = "RmsNorm-swiglu-ffn-GEMM-INT4.spv", .buffers = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, outBuffer, gateScale, gateZero, upScale, upZero}, .bufferCount = 9,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = N / 64, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    for (int m = 0; m < M; m++)
        swiglu_ref_int4(input + m * K, gamma, &weightINT4, &weight2INT4, ref + m * N, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("RmsNorm-swiglu-ffn-GEMM-INT4", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 9);
    free(out);
    free(ref);
}

void validateRmsNormSwigluFfnGEMM2FP16(session s, int M, int N, int K, float* input, float* gamma, uint16_t* weightFP16, uint16_t* weight2FP16) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint16_t* transposed = (uint16_t*)malloc(sizeof(uint16_t) * K * N);
    uint16_t* transposed2 = (uint16_t*)malloc(sizeof(uint16_t) * K * N);
    transpose_block16((uint8_t*)weightFP16, (uint8_t*)transposed, K, N, QUANT_FP16);
    transpose_block16((uint8_t*)weight2FP16, (uint8_t*)transposed2, K, N, QUANT_FP16);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer gateBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint16_t) * K * N, MEMORY_RAM);
    buffer upBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed2, sizeof(uint16_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    float* invInit = (float*)calloc(M, sizeof(float));
    buffer invRmsBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, invInit, sizeof(float) * M, MEMORY_VRAM);
    free(invInit);
    buffer bufs[] = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, outBuffer, invRmsBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 6);
    free(transposed);
    free(transposed2);

    operation ops[] = {
        {.shader = "RmsNorm-Prologue.spv", .buffers = {inputBuffer, invRmsBuffer}, .bufferCount = 2,
         .pushConstants = {K}, .pushConstantCount = 1,
         .dispatchX = M, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "RmsNorm-swiglu-ffn-GEMM2-FP16.spv", .buffers = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, outBuffer, invRmsBuffer}, .bufferCount = 6,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = N / 32, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 2);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    for (int m = 0; m < M; m++)
        swiglu_ref_fp16(input + m * K, gamma, weightFP16, weight2FP16, ref + m * N, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("RmsNorm-swiglu-ffn-GEMM2-FP16", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 6);
    free(out);
    free(ref);
}

void validateRmsNormSwigluFfnGEMM2INT8(session s, int M, int N, int K, float* input, float* gamma, QuantizedData weightINT8, QuantizedData weight2INT8) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N);
    uint8_t* transposed2 = (uint8_t*)malloc(sizeof(uint8_t) * K * N);
    transpose_block16(weightINT8.data, transposed, K, N, QUANT_INT8);
    transpose_block16(weight2INT8.data, transposed2, K, N, QUANT_INT8);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer gateBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N, MEMORY_RAM);
    buffer upBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed2, sizeof(uint8_t) * K * N, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer gateScale = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.scale, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer gateZero = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.z, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer upScale = createBuffer(s.dev.device, s.dev.physicalDevice, weight2INT8.scale, sizeof(float) * K * N / weight2INT8.group_size, MEMORY_RAM);
    buffer upZero = createBuffer(s.dev.device, s.dev.physicalDevice, weight2INT8.z, sizeof(float) * K * N / weight2INT8.group_size, MEMORY_RAM);
    float* invInit = (float*)calloc(M, sizeof(float));
    buffer invRmsBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, invInit, sizeof(float) * M, MEMORY_VRAM);
    free(invInit);
    buffer bufs[] = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, outBuffer, gateScale, gateZero, upScale, upZero, invRmsBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 10);
    free(transposed);
    free(transposed2);

    operation ops[] = {
        {.shader = "RmsNorm-Prologue.spv", .buffers = {inputBuffer, invRmsBuffer}, .bufferCount = 2,
         .pushConstants = {K}, .pushConstantCount = 1,
         .dispatchX = M, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "RmsNorm-swiglu-ffn-GEMM2-INT8.spv", .buffers = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, outBuffer, gateScale, gateZero, upScale, upZero, invRmsBuffer}, .bufferCount = 10,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = N / 32, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 2);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    for (int m = 0; m < M; m++)
        swiglu_ref_int8(input + m * K, gamma, &weightINT8, &weight2INT8, ref + m * N, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("RmsNorm-swiglu-ffn-GEMM2-INT8", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 10);
    free(out);
    free(ref);
}

void validateRmsNormSwigluFfnGEMM2INT4(session s, int M, int N, int K, float* input, float* gamma, QuantizedData weightINT4, QuantizedData weight2INT4) {
    float* out = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N / 2);
    uint8_t* transposed2 = (uint8_t*)malloc(sizeof(uint8_t) * K * N / 2);
    transpose_block16(weightINT4.data, transposed, K, N, QUANT_INT4);
    transpose_block16(weight2INT4.data, transposed2, K, N, QUANT_INT4);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer gateBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N / 2, MEMORY_RAM);
    buffer upBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed2, sizeof(uint8_t) * K * N / 2, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * N, MEMORY_RAM);
    buffer gateScale = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.scale, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer gateZero = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.z, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer upScale = createBuffer(s.dev.device, s.dev.physicalDevice, weight2INT4.scale, sizeof(float) * K * N / weight2INT4.group_size, MEMORY_RAM);
    buffer upZero = createBuffer(s.dev.device, s.dev.physicalDevice, weight2INT4.z, sizeof(float) * K * N / weight2INT4.group_size, MEMORY_RAM);
    float* invInit = (float*)calloc(M, sizeof(float));
    buffer invRmsBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, invInit, sizeof(float) * M, MEMORY_VRAM);
    free(invInit);
    buffer bufs[] = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, outBuffer, gateScale, gateZero, upScale, upZero, invRmsBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 10);
    free(transposed);
    free(transposed2);

    operation ops[] = {
        {.shader = "RmsNorm-Prologue.spv", .buffers = {inputBuffer, invRmsBuffer}, .bufferCount = 2,
         .pushConstants = {K}, .pushConstantCount = 1,
         .dispatchX = M, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "RmsNorm-swiglu-ffn-GEMM2-INT4.spv", .buffers = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, outBuffer, gateScale, gateZero, upScale, upZero, invRmsBuffer}, .bufferCount = 10,
         .pushConstants = {M, N, K}, .pushConstantCount = 3,
         .dispatchX = N / 32, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 2);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    for (int m = 0; m < M; m++)
        swiglu_ref_int4(input + m * K, gamma, &weightINT4, &weight2INT4, ref + m * N, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("RmsNorm-swiglu-ffn-GEMM2-INT4", 100, out, ref, M * N, ms);

    destroy_buffers(s, bufs, 10);
    free(out);
    free(ref);
}

void validateRmsNormSwigluFlatGEMM2INT8(session s, int M, int N, int K, float* input, float* gamma, QuantizedData weightINT8, QuantizedData weight2INT8) {
    float* gOut = (float*)calloc(M * N, sizeof(float));
    float* uOut = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N);
    uint8_t* transposed2 = (uint8_t*)malloc(sizeof(uint8_t) * K * N);
    transpose_block16(weightINT8.data, transposed, K, N, QUANT_INT8);
    transpose_block16(weight2INT8.data, transposed2, K, N, QUANT_INT8);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer gateBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N, MEMORY_RAM);
    buffer upBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed2, sizeof(uint8_t) * K * N, MEMORY_RAM);
    buffer gBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gOut, sizeof(float) * M * N, MEMORY_VRAM);
    buffer uBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, uOut, sizeof(float) * M * N, MEMORY_VRAM);
    buffer gateScale = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.scale, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer gateZero = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT8.z, sizeof(float) * K * N / weightINT8.group_size, MEMORY_RAM);
    buffer upScale = createBuffer(s.dev.device, s.dev.physicalDevice, weight2INT8.scale, sizeof(float) * K * N / weight2INT8.group_size, MEMORY_RAM);
    buffer upZero = createBuffer(s.dev.device, s.dev.physicalDevice, weight2INT8.z, sizeof(float) * K * N / weight2INT8.group_size, MEMORY_RAM);
    float* invInit = (float*)calloc(M, sizeof(float));
    buffer invRmsBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, invInit, sizeof(float) * M, MEMORY_VRAM);
    free(invInit);
    buffer bufs[] = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, gBuffer, uBuffer, gateScale, gateZero, upScale, upZero, invRmsBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 11);
    free(transposed);
    free(transposed2);

    operation ops[] = {
        {.shader = "RmsNorm-Prologue.spv", .buffers = {inputBuffer, invRmsBuffer}, .bufferCount = 2,
         .pushConstants = {K}, .pushConstantCount = 1,
         .dispatchX = M, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "RmsNorm-swiglu-flat-GEMM2-INT8.spv", .buffers = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, gBuffer, uBuffer, gateScale, gateZero, upScale, upZero, invRmsBuffer}, .bufferCount = 11,
         .pushConstants = {M, N, K, N}, .pushConstantCount = 4,
         .dispatchX = (N * 2) / 32, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 2);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    for (int m = 0; m < M; m++)
        swiglu_ref_int8(input + m * K, gamma, &weightINT8, &weight2INT8, ref + m * N, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, gBuffer, gOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, uBuffer, uOut);
    float* comb = (float*)malloc(sizeof(float) * M * N);
    for (int i = 0; i < M * N; i++) {
        float v = gOut[i];
        comb[i] = (v / (1.0f + expf(-v))) * uOut[i];
    }
    report("RmsNorm-swiglu-flat-GEMM2-INT8", 100, comb, ref, M * N, ms);

    destroy_buffers(s, bufs, 11);
    free(gOut);
    free(uOut);
    free(comb);
    free(ref);
}

void validateRmsNormSwigluFlatGEMM2INT4(session s, int M, int N, int K, float* input, float* gamma, QuantizedData weightINT4, QuantizedData weight2INT4) {
    float* gOut = (float*)calloc(M * N, sizeof(float));
    float* uOut = (float*)calloc(M * N, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * K * N / 2);
    uint8_t* transposed2 = (uint8_t*)malloc(sizeof(uint8_t) * K * N / 2);
    transpose_block16(weightINT4.data, transposed, K, N, QUANT_INT4);
    transpose_block16(weight2INT4.data, transposed2, K, N, QUANT_INT4);
    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer gateBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * K * N / 2, MEMORY_RAM);
    buffer upBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed2, sizeof(uint8_t) * K * N / 2, MEMORY_RAM);
    buffer gBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gOut, sizeof(float) * M * N, MEMORY_VRAM);
    buffer uBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, uOut, sizeof(float) * M * N, MEMORY_VRAM);
    buffer gateScale = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.scale, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer gateZero = createBuffer(s.dev.device, s.dev.physicalDevice, weightINT4.z, sizeof(float) * K * N / weightINT4.group_size, MEMORY_RAM);
    buffer upScale = createBuffer(s.dev.device, s.dev.physicalDevice, weight2INT4.scale, sizeof(float) * K * N / weight2INT4.group_size, MEMORY_RAM);
    buffer upZero = createBuffer(s.dev.device, s.dev.physicalDevice, weight2INT4.z, sizeof(float) * K * N / weight2INT4.group_size, MEMORY_RAM);
    float* invInit = (float*)calloc(M, sizeof(float));
    buffer invRmsBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, invInit, sizeof(float) * M, MEMORY_VRAM);
    free(invInit);
    buffer bufs[] = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, gBuffer, uBuffer, gateScale, gateZero, upScale, upZero, invRmsBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 11);
    free(transposed);
    free(transposed2);

    operation ops[] = {
        {.shader = "RmsNorm-Prologue.spv", .buffers = {inputBuffer, invRmsBuffer}, .bufferCount = 2,
         .pushConstants = {K}, .pushConstantCount = 1,
         .dispatchX = M, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "RmsNorm-swiglu-flat-GEMM2-INT4.spv", .buffers = {inputBuffer, gammaBuffer, gateBuffer, upBuffer, gBuffer, uBuffer, gateScale, gateZero, upScale, upZero, invRmsBuffer}, .bufferCount = 11,
         .pushConstants = {M, N, K, N}, .pushConstantCount = 4,
         .dispatchX = (N * 2) / 32, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 2);

    float* ref = (float*)malloc(sizeof(float) * M * N);
    for (int m = 0; m < M; m++)
        swiglu_ref_int4(input + m * K, gamma, &weightINT4, &weight2INT4, ref + m * N, N, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, gBuffer, gOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, uBuffer, uOut);
    float* comb = (float*)malloc(sizeof(float) * M * N);
    for (int i = 0; i < M * N; i++) {
        float v = gOut[i];
        comb[i] = (v / (1.0f + expf(-v))) * uOut[i];
    }
    report("RmsNorm-swiglu-flat-GEMM2-INT4", 100, comb, ref, M * N, ms);

    destroy_buffers(s, bufs, 11);
    free(gOut);
    free(uOut);
    free(comb);
    free(ref);
}

void validateRmsNormLinearProjGEMMFP16(session s, int M, int K, float* input, float* gamma, uint16_t* w_inFP16) {
    int proj_n = 8256;
    float* qref = (float*)malloc(sizeof(float) * M * 2048);
    float* kref = (float*)malloc(sizeof(float) * M * 2048);
    float* vref = (float*)malloc(sizeof(float) * M * 4096);
    for (int m = 0; m < M; m++) {
        float* xn = (float*)malloc(sizeof(float) * K);
        rms_norm_apply(input + m * K, gamma, xn, K);
        float* proj = (float*)malloc(sizeof(float) * proj_n);
        gemv_ref_fp16(xn, w_inFP16, proj, proj_n, K);
        memcpy(qref + m * 2048, proj, sizeof(float) * 2048);
        memcpy(kref + m * 2048, proj + 2048, sizeof(float) * 2048);
        memcpy(vref + m * 4096, proj + 4096, sizeof(float) * 4096);
        free(xn);
        free(proj);
    }

    float* qOut = (float*)calloc(M * 2048, sizeof(float));
    float* kOut = (float*)calloc(M * 2048, sizeof(float));
    float* vOut = (float*)calloc(M * 4096, sizeof(float));
    float* aOut = (float*)calloc(M * 32, sizeof(float));
    float* bOut = (float*)calloc(M * 32, sizeof(float));

    uint16_t* twIn = (uint16_t*)malloc(sizeof(uint16_t) * K * proj_n);
    transpose_block16((uint8_t*)w_inFP16, (uint8_t*)twIn, K, proj_n, QUANT_FP16);

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer wInBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, twIn, sizeof(uint16_t) * K * proj_n, MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * M * 2048, MEMORY_VRAM);
    buffer kOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kOut, sizeof(float) * M * 2048, MEMORY_VRAM);
    buffer vOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vOut, sizeof(float) * M * 4096, MEMORY_VRAM);
    buffer aOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, aOut, sizeof(float) * M * 32, MEMORY_VRAM);
    buffer bOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, bOut, sizeof(float) * M * 32, MEMORY_VRAM);
    buffer bufs[] = {xBuffer, gammaBuffer, wInBuffer, qOutBuffer, kOutBuffer, vOutBuffer, aOutBuffer, bOutBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 8);
    free(twIn);

    operation ops[] = {
        {.shader = "RmsNorm-LinearProj-GEMM-FP16.spv", .buffers = {xBuffer, gammaBuffer, wInBuffer, qOutBuffer, kOutBuffer, vOutBuffer, aOutBuffer, bOutBuffer}, .bufferCount = 8,
         .pushConstants = {M, proj_n, K, 2048, 4096, 8192, 12288, 12320}, .pushConstantCount = 8,
         .dispatchX = proj_n / 16, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, qOutBuffer, qOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vOutBuffer, vOut);
    report("RmsNorm-LinearProj-GEMM-FP16", 100, qOut, qref, M * 2048, ms);
    report("RmsNorm-LinearProj-GEMM-v FP16", 100, vOut, vref, M * 4096, ms);

    destroy_buffers(s, bufs, 8);
    free(qref); free(kref); free(vref);
    free(qOut); free(kOut); free(vOut); free(aOut); free(bOut);
}

void validateRmsNormLinearProjGEMMINT8(session s, int M, int K, float* input, float* gamma, QuantizedData w_inINT8) {
    int proj_n = 8256;
    int inBlocks = (proj_n + 255) / 256;
    float* qref = (float*)malloc(sizeof(float) * M * 2048);
    float* kref = (float*)malloc(sizeof(float) * M * 2048);
    float* vref = (float*)malloc(sizeof(float) * M * 4096);
    for (int m = 0; m < M; m++) {
        float* xn = (float*)malloc(sizeof(float) * K);
        rms_norm_apply(input + m * K, gamma, xn, K);
        float* proj = (float*)malloc(sizeof(float) * proj_n);
        gemv_ref_int8(xn, &w_inINT8, proj, proj_n, K);
        memcpy(qref + m * 2048, proj, sizeof(float) * 2048);
        memcpy(kref + m * 2048, proj + 2048, sizeof(float) * 2048);
        memcpy(vref + m * 4096, proj + 4096, sizeof(float) * 4096);
        free(xn);
        free(proj);
    }

    float* qOut = (float*)calloc(M * 2048, sizeof(float));
    float* kOut = (float*)calloc(M * 2048, sizeof(float));
    float* vOut = (float*)calloc(M * 4096, sizeof(float));
    float* aOut = (float*)calloc(M * 32, sizeof(float));
    float* bOut = (float*)calloc(M * 32, sizeof(float));

    uint8_t* twIn = (uint8_t*)malloc(sizeof(uint8_t) * K * proj_n);
    transpose_block16(w_inINT8.data, twIn, K, proj_n, QUANT_INT8);

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer wInBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, twIn, sizeof(uint8_t) * K * proj_n, MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * M * 2048, MEMORY_VRAM);
    buffer kOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kOut, sizeof(float) * M * 2048, MEMORY_VRAM);
    buffer vOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vOut, sizeof(float) * M * 4096, MEMORY_VRAM);
    buffer aOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, aOut, sizeof(float) * M * 32, MEMORY_VRAM);
    buffer bOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, bOut, sizeof(float) * M * 32, MEMORY_VRAM);
    buffer wInScale = createBuffer(s.dev.device, s.dev.physicalDevice, w_inINT8.scale, sizeof(float) * K * inBlocks, MEMORY_RAM);
    buffer wInZero = createBuffer(s.dev.device, s.dev.physicalDevice, w_inINT8.z, sizeof(float) * K * inBlocks, MEMORY_RAM);
    buffer bufs[] = {xBuffer, gammaBuffer, wInBuffer, qOutBuffer, kOutBuffer, vOutBuffer, aOutBuffer, bOutBuffer, wInScale, wInZero};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 10);
    free(twIn);

    operation ops[] = {
        {.shader = "RmsNorm-LinearProj-GEMM-INT8.spv", .buffers = {xBuffer, gammaBuffer, wInBuffer, qOutBuffer, kOutBuffer, vOutBuffer, aOutBuffer, bOutBuffer, wInScale, wInZero}, .bufferCount = 10,
         .pushConstants = {M, proj_n, K, 2048, 4096, 8192, 12288, 12320}, .pushConstantCount = 8,
         .dispatchX = proj_n / 16, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, qOutBuffer, qOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vOutBuffer, vOut);
    report("RmsNorm-LinearProj-GEMM-INT8", 100, qOut, qref, M * 2048, ms);
    report("RmsNorm-LinearProj-GEMM-v INT8", 100, vOut, vref, M * 4096, ms);

    destroy_buffers(s, bufs, 10);
    free(qref); free(kref); free(vref);
    free(qOut); free(kOut); free(vOut); free(aOut); free(bOut);
}

void validateRmsNormLinearProjGEMMINT4(session s, int M, int K, float* input, float* gamma, QuantizedData w_inINT4) {
    int proj_n = 8256;
    int inBlocks = (proj_n + 255) / 256;
    float* qref = (float*)malloc(sizeof(float) * M * 2048);
    float* kref = (float*)malloc(sizeof(float) * M * 2048);
    float* vref = (float*)malloc(sizeof(float) * M * 4096);
    for (int m = 0; m < M; m++) {
        float* xn = (float*)malloc(sizeof(float) * K);
        rms_norm_apply(input + m * K, gamma, xn, K);
        float* proj = (float*)malloc(sizeof(float) * proj_n);
        gemv_ref_int4(xn, &w_inINT4, proj, proj_n, K);
        memcpy(qref + m * 2048, proj, sizeof(float) * 2048);
        memcpy(kref + m * 2048, proj + 2048, sizeof(float) * 2048);
        memcpy(vref + m * 4096, proj + 4096, sizeof(float) * 4096);
        free(xn);
        free(proj);
    }

    float* qOut = (float*)calloc(M * 2048, sizeof(float));
    float* kOut = (float*)calloc(M * 2048, sizeof(float));
    float* vOut = (float*)calloc(M * 4096, sizeof(float));
    float* aOut = (float*)calloc(M * 32, sizeof(float));
    float* bOut = (float*)calloc(M * 32, sizeof(float));

    uint8_t* twIn = (uint8_t*)malloc(sizeof(uint8_t) * K * proj_n / 2);
    transpose_block16(w_inINT4.data, twIn, K, proj_n, QUANT_INT4);

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer wInBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, twIn, sizeof(uint8_t) * K * proj_n / 2, MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * M * 2048, MEMORY_VRAM);
    buffer kOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kOut, sizeof(float) * M * 2048, MEMORY_VRAM);
    buffer vOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vOut, sizeof(float) * M * 4096, MEMORY_VRAM);
    buffer aOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, aOut, sizeof(float) * M * 32, MEMORY_VRAM);
    buffer bOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, bOut, sizeof(float) * M * 32, MEMORY_VRAM);
    buffer wInScale = createBuffer(s.dev.device, s.dev.physicalDevice, w_inINT4.scale, sizeof(float) * K * inBlocks, MEMORY_RAM);
    buffer wInZero = createBuffer(s.dev.device, s.dev.physicalDevice, w_inINT4.z, sizeof(float) * K * inBlocks, MEMORY_RAM);
    buffer bufs[] = {xBuffer, gammaBuffer, wInBuffer, qOutBuffer, kOutBuffer, vOutBuffer, aOutBuffer, bOutBuffer, wInScale, wInZero};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 10);
    free(twIn);

    operation ops[] = {
        {.shader = "RmsNorm-LinearProj-GEMM-INT4.spv", .buffers = {xBuffer, gammaBuffer, wInBuffer, qOutBuffer, kOutBuffer, vOutBuffer, aOutBuffer, bOutBuffer, wInScale, wInZero}, .bufferCount = 10,
         .pushConstants = {M, proj_n, K, 2048, 4096, 8192, 12288, 12320}, .pushConstantCount = 8,
         .dispatchX = proj_n / 16, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, qOutBuffer, qOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vOutBuffer, vOut);
    report("RmsNorm-LinearProj-GEMM-INT4", 100, qOut, qref, M * 2048, ms);
    report("RmsNorm-LinearProj-GEMM-v INT4", 100, vOut, vref, M * 4096, ms);

    destroy_buffers(s, bufs, 10);
    free(qref); free(kref); free(vref);
    free(qOut); free(kOut); free(vOut); free(aOut); free(bOut);
}

void validateQkvRopeGEMMFP16(session s, int K, int qkv_heads, int qkv_kv_heads, int qkv_dim, float* input, float* gamma, uint16_t* qkv_weightFP16, float* qkv_theta, int M) {
    int k = K;
    int heads = qkv_heads;
    int kv_heads = qkv_kv_heads;
    int dim = qkv_dim;
    int gate_heads = heads;
    int n_total = (heads + gate_heads + 2 * kv_heads) * dim;
    int g_offset = heads * dim;
    int k_offset = (heads + gate_heads) * dim;
    int v_offset = (heads + gate_heads + kv_heads) * dim;
    int rows = kv_heads * dim;
    uint32_t posVal = 0;

    float* qref = (float*)malloc(sizeof(float) * M * heads * dim);
    float* kref = (float*)malloc(sizeof(float) * M * rows);
    float* vref = (float*)malloc(sizeof(float) * M * rows);
    float* gref = (float*)malloc(sizeof(float) * M * gate_heads * dim);
    float* qgamma = getData(61001, 1, dim);
    float* kgamma = getData(62001, 1, dim);
    for (int m = 0; m < M; m++) {
        float* xn = (float*)malloc(sizeof(float) * k);
        rms_norm_apply(input + m * k, gamma, xn, k);
        float* proj = (float*)malloc(sizeof(float) * n_total);
        gemv_ref_fp16(xn, qkv_weightFP16, proj, n_total, k);
        qkv_rope_ref(proj, qkv_theta, qref + m * heads * dim, kref + m * rows, vref + m * rows,
                     gref + m * gate_heads * dim, n_total, g_offset, k_offset, v_offset, dim, m, qgamma, kgamma);
        free(xn);
        free(proj);
    }

    float* qOut = (float*)calloc(M * heads * dim, sizeof(float));
    uint16_t* kCache = (uint16_t*)calloc(M * rows, sizeof(uint16_t));
    uint16_t* vCache = (uint16_t*)calloc(M * rows, sizeof(uint16_t));

    uint16_t* transposed = (uint16_t*)malloc(sizeof(uint16_t) * k * n_total);
    transpose_block16((uint8_t*)qkv_weightFP16, (uint8_t*)transposed, k, n_total, QUANT_FP16);

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * k, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * k, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint16_t) * k * n_total, MEMORY_RAM);
    buffer thetaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_theta, sizeof(float) * (QKV_ROPE_DIM / 2), MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * M * heads * dim, MEMORY_VRAM);
    buffer kCacheBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kCache, sizeof(uint16_t) * M * rows, MEMORY_VRAM);
    buffer vCacheBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vCache, sizeof(uint16_t) * M * rows, MEMORY_VRAM);
    buffer posBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, &posVal, sizeof(uint32_t), MEMORY_VRAM);
    float* gOut = (float*)calloc(M * gate_heads * dim, sizeof(float));
    buffer qGammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qgamma, sizeof(float) * dim, MEMORY_VRAM);
    buffer kGammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kgamma, sizeof(float) * dim, MEMORY_VRAM);
    buffer gOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gOut, sizeof(float) * M * gate_heads * dim, MEMORY_VRAM);
    buffer bufs[] = {xBuffer, gammaBuffer, weightBuffer, thetaBuffer, qOutBuffer, kCacheBuffer, vCacheBuffer, posBuffer, qGammaBuffer, kGammaBuffer, gOutBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 11);
    free(transposed);

    operation ops[] = {
        {.shader = "RmsNorm-QKV-GEMM-FP16.spv", .buffers = {xBuffer, gammaBuffer, weightBuffer, thetaBuffer, qOutBuffer, kCacheBuffer, vCacheBuffer, posBuffer, qGammaBuffer, kGammaBuffer, gOutBuffer}, .bufferCount = 11,
         .pushConstants = {M, n_total, k, 0, g_offset, k_offset, v_offset}, .pushConstantCount = 7,
         .dispatchX = n_total / dim, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, qOutBuffer, qOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kCacheBuffer, kCache);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vCacheBuffer, vCache);
    float* kstored = (float*)malloc(sizeof(float) * M * rows);
    float* vstored = (float*)malloc(sizeof(float) * M * rows);
    for (int i = 0; i < M * rows; i++) kstored[i] = fp16_to_float(kCache[i]);
    for (int i = 0; i < M * rows; i++) vstored[i] = fp16_to_float(vCache[i]);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, gOutBuffer, gOut);
    report("QKV-Rope-GEMM FP16", 100, qOut, qref, M * heads * dim, ms);
    report("QKV-Rope-GEMM-g FP16", 100, gOut, gref, M * gate_heads * dim, ms);
    report("QKV-Rope-GEMM-kv FP16", 100, kstored, kref, M * rows, ms);
    report("QKV-Rope-GEMM-v FP16", 100, vstored, vref, M * rows, ms);
    uint32_t posRead = 0;
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, posBuffer, &posRead);
    printf("QKV-Rope-GEMM-pos FP16: shader[0]= %u ref= %d\n", posRead, M);

    destroy_buffers(s, bufs, 11);
    free(qref); free(kref); free(vref); free(gref); free(qgamma); free(kgamma); free(gOut);
    free(qOut); free(kCache); free(vCache);
    free(kstored); free(vstored);
}

void validateQkvRopeGEMMINT8(session s, int K, int qkv_heads, int qkv_kv_heads, int qkv_dim, float* input, float* gamma, QuantizedData qkv_weightINT8, float* qkv_theta, int M) {
    int k = K;
    int heads = qkv_heads;
    int kv_heads = qkv_kv_heads;
    int dim = qkv_dim;
    int gate_heads = heads;
    int n_total = (heads + gate_heads + 2 * kv_heads) * dim;
    int g_offset = heads * dim;
    int k_offset = (heads + gate_heads) * dim;
    int v_offset = (heads + gate_heads + kv_heads) * dim;
    int rows = kv_heads * dim;
    int scaleCount = k * n_total / qkv_weightINT8.group_size;
    uint32_t posVal = 0;

    float* qref = (float*)malloc(sizeof(float) * M * heads * dim);
    float* kref = (float*)malloc(sizeof(float) * M * rows);
    float* vref = (float*)malloc(sizeof(float) * M * rows);
    float* gref = (float*)malloc(sizeof(float) * M * gate_heads * dim);
    float* qgamma = getData(61001, 1, dim);
    float* kgamma = getData(62001, 1, dim);
    for (int m = 0; m < M; m++) {
        float* xn = (float*)malloc(sizeof(float) * k);
        rms_norm_apply(input + m * k, gamma, xn, k);
        float* proj = (float*)malloc(sizeof(float) * n_total);
        gemv_ref_int8(xn, &qkv_weightINT8, proj, n_total, k);
        qkv_rope_ref(proj, qkv_theta, qref + m * heads * dim, kref + m * rows, vref + m * rows,
                     gref + m * gate_heads * dim, n_total, g_offset, k_offset, v_offset, dim, m, qgamma, kgamma);
        free(xn);
        free(proj);
    }

    float* qOut = (float*)calloc(M * heads * dim, sizeof(float));
    uint8_t* kCache = (uint8_t*)calloc(M * rows, sizeof(uint8_t));
    uint8_t* vCache = (uint8_t*)calloc(M * rows, sizeof(uint8_t));
    float* kScale = (float*)calloc(kv_heads * M, sizeof(float));
    float* kZero = (float*)calloc(kv_heads * M, sizeof(float));
    float* vScale = (float*)calloc(kv_heads * M, sizeof(float));
    float* vZero = (float*)calloc(kv_heads * M, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * k * n_total);
    transpose_block16(qkv_weightINT8.data, transposed, k, n_total, QUANT_INT8);

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * k, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * k, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * k * n_total, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_weightINT8.scale, sizeof(float) * scaleCount, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_weightINT8.z, sizeof(float) * scaleCount, MEMORY_RAM);
    buffer thetaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_theta, sizeof(float) * (QKV_ROPE_DIM / 2), MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * M * heads * dim, MEMORY_VRAM);
    buffer kCacheBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kCache, sizeof(uint8_t) * M * rows, MEMORY_VRAM);
    buffer vCacheBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vCache, sizeof(uint8_t) * M * rows, MEMORY_VRAM);
    buffer kScaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kScale, sizeof(float) * kv_heads * M, MEMORY_VRAM);
    buffer kZeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kZero, sizeof(float) * kv_heads * M, MEMORY_VRAM);
    buffer vScaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vScale, sizeof(float) * kv_heads * M, MEMORY_VRAM);
    buffer vZeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vZero, sizeof(float) * kv_heads * M, MEMORY_VRAM);
    buffer posBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, &posVal, sizeof(uint32_t), MEMORY_VRAM);
    float* gOut = (float*)calloc(M * gate_heads * dim, sizeof(float));
    buffer qGammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qgamma, sizeof(float) * dim, MEMORY_VRAM);
    buffer kGammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kgamma, sizeof(float) * dim, MEMORY_VRAM);
    buffer gOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gOut, sizeof(float) * M * gate_heads * dim, MEMORY_VRAM);
    buffer bufs[] = {xBuffer, gammaBuffer, weightBuffer, scaleBuffer, zeroBuffer, thetaBuffer, qOutBuffer, kCacheBuffer, vCacheBuffer, kScaleBuffer, kZeroBuffer, vScaleBuffer, vZeroBuffer, posBuffer, qGammaBuffer, kGammaBuffer, gOutBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 17);
    free(transposed);

    operation ops[] = {
        {.shader = "RmsNorm-QKV-GEMM-INT8.spv", .buffers = {xBuffer, gammaBuffer, weightBuffer, scaleBuffer, zeroBuffer, thetaBuffer, qOutBuffer, kCacheBuffer, vCacheBuffer, kScaleBuffer, kZeroBuffer, vScaleBuffer, vZeroBuffer, posBuffer, qGammaBuffer, kGammaBuffer, gOutBuffer}, .bufferCount = 17,
         .pushConstants = {M, n_total, k, 0, g_offset, k_offset, v_offset}, .pushConstantCount = 7,
         .dispatchX = n_total / dim, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, qOutBuffer, qOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kCacheBuffer, kCache);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vCacheBuffer, vCache);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kScaleBuffer, kScale);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kZeroBuffer, kZero);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vScaleBuffer, vScale);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vZeroBuffer, vZero);
    float* kstored = (float*)malloc(sizeof(float) * M * rows);
    float* vstored = (float*)malloc(sizeof(float) * M * rows);
    for (int m = 0; m < M; m++) {
        for (int r = 0; r < rows; r++) {
            int h = r / dim;
            kstored[m * rows + r] = (float)kCache[m * rows + r] * kScale[h * M + m] - kZero[h * M + m];
            vstored[m * rows + r] = (float)vCache[m * rows + r] * vScale[h * M + m] - vZero[h * M + m];
        }
    }

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, gOutBuffer, gOut);
    report("QKV-Rope-GEMM INT8", 100, qOut, qref, M * heads * dim, ms);
    report("QKV-Rope-GEMM-g INT8", 100, gOut, gref, M * gate_heads * dim, ms);
    report("QKV-Rope-GEMM-kv INT8", 100, kstored, kref, M * rows, ms);
    report("QKV-Rope-GEMM-v INT8", 100, vstored, vref, M * rows, ms);
    uint32_t posRead = 0;
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, posBuffer, &posRead);
    printf("QKV-Rope-GEMM-pos INT8: shader[0]= %u ref= %d\n", posRead, M);

    destroy_buffers(s, bufs, 17);
    free(qref); free(kref); free(vref); free(gref); free(qgamma); free(kgamma); free(gOut);
    free(qOut); free(kCache); free(vCache);
    free(kScale); free(kZero); free(vScale); free(vZero);
    free(kstored); free(vstored);
}

void validateQkvRopeGEMMINT4(session s, int K, int qkv_heads, int qkv_kv_heads, int qkv_dim, float* input, float* gamma, QuantizedData qkv_weightINT4, float* qkv_theta, int M) {
    int k = K;
    int heads = qkv_heads;
    int kv_heads = qkv_kv_heads;
    int dim = qkv_dim;
    int gate_heads = heads;
    int n_total = (heads + gate_heads + 2 * kv_heads) * dim;
    int g_offset = heads * dim;
    int k_offset = (heads + gate_heads) * dim;
    int v_offset = (heads + gate_heads + kv_heads) * dim;
    int rows = kv_heads * dim;
    int scaleCount = k * n_total / qkv_weightINT4.group_size;
    uint32_t posVal = 0;

    float* qref = (float*)malloc(sizeof(float) * M * heads * dim);
    float* kref = (float*)malloc(sizeof(float) * M * rows);
    float* vref = (float*)malloc(sizeof(float) * M * rows);
    float* gref = (float*)malloc(sizeof(float) * M * gate_heads * dim);
    float* qgamma = getData(61001, 1, dim);
    float* kgamma = getData(62001, 1, dim);
    for (int m = 0; m < M; m++) {
        float* xn = (float*)malloc(sizeof(float) * k);
        rms_norm_apply(input + m * k, gamma, xn, k);
        float* proj = (float*)malloc(sizeof(float) * n_total);
        gemv_ref_int4(xn, &qkv_weightINT4, proj, n_total, k);
        qkv_rope_ref(proj, qkv_theta, qref + m * heads * dim, kref + m * rows, vref + m * rows,
                     gref + m * gate_heads * dim, n_total, g_offset, k_offset, v_offset, dim, m, qgamma, kgamma);
        free(xn);
        free(proj);
    }

    float* qOut = (float*)calloc(M * heads * dim, sizeof(float));
    uint8_t* kCache = (uint8_t*)calloc(M * rows, sizeof(uint8_t));
    uint8_t* vCache = (uint8_t*)calloc(M * rows, sizeof(uint8_t));
    float* kScale = (float*)calloc(kv_heads * M, sizeof(float));
    float* kZero = (float*)calloc(kv_heads * M, sizeof(float));
    float* vScale = (float*)calloc(kv_heads * M, sizeof(float));
    float* vZero = (float*)calloc(kv_heads * M, sizeof(float));

    uint8_t* transposed = (uint8_t*)malloc(sizeof(uint8_t) * k * n_total / 2);
    transpose_block16(qkv_weightINT4.data, transposed, k, n_total, QUANT_INT4);

    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * k, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * k, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint8_t) * k * n_total / 2, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_weightINT4.scale, sizeof(float) * scaleCount, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_weightINT4.z, sizeof(float) * scaleCount, MEMORY_RAM);
    buffer thetaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qkv_theta, sizeof(float) * (QKV_ROPE_DIM / 2), MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * M * heads * dim, MEMORY_VRAM);
    buffer kCacheBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kCache, sizeof(uint8_t) * M * rows, MEMORY_VRAM);
    buffer vCacheBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vCache, sizeof(uint8_t) * M * rows, MEMORY_VRAM);
    buffer kScaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kScale, sizeof(float) * kv_heads * M, MEMORY_VRAM);
    buffer kZeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kZero, sizeof(float) * kv_heads * M, MEMORY_VRAM);
    buffer vScaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vScale, sizeof(float) * kv_heads * M, MEMORY_VRAM);
    buffer vZeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vZero, sizeof(float) * kv_heads * M, MEMORY_VRAM);
    buffer posBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, &posVal, sizeof(uint32_t), MEMORY_VRAM);
    float* gOut = (float*)calloc(M * gate_heads * dim, sizeof(float));
    buffer qGammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qgamma, sizeof(float) * dim, MEMORY_VRAM);
    buffer kGammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kgamma, sizeof(float) * dim, MEMORY_VRAM);
    buffer gOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gOut, sizeof(float) * M * gate_heads * dim, MEMORY_VRAM);
    buffer bufs[] = {xBuffer, gammaBuffer, weightBuffer, scaleBuffer, zeroBuffer, thetaBuffer, qOutBuffer, kCacheBuffer, vCacheBuffer, kScaleBuffer, kZeroBuffer, vScaleBuffer, vZeroBuffer, posBuffer, qGammaBuffer, kGammaBuffer, gOutBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 17);
    free(transposed);

    operation ops[] = {
        {.shader = "RmsNorm-QKV-GEMM-INT4.spv", .buffers = {xBuffer, gammaBuffer, weightBuffer, scaleBuffer, zeroBuffer, thetaBuffer, qOutBuffer, kCacheBuffer, vCacheBuffer, kScaleBuffer, kZeroBuffer, vScaleBuffer, vZeroBuffer, posBuffer, qGammaBuffer, kGammaBuffer, gOutBuffer}, .bufferCount = 17,
         .pushConstants = {M, n_total, k, 0, g_offset, k_offset, v_offset}, .pushConstantCount = 7,
         .dispatchX = n_total / dim, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, qOutBuffer, qOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kCacheBuffer, kCache);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vCacheBuffer, vCache);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kScaleBuffer, kScale);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kZeroBuffer, kZero);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vScaleBuffer, vScale);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vZeroBuffer, vZero);
    float* kstored = (float*)malloc(sizeof(float) * M * rows);
    float* vstored = (float*)malloc(sizeof(float) * M * rows);
    for (int m = 0; m < M; m++) {
        for (int r = 0; r < rows; r++) {
            int h = r / dim;
            kstored[m * rows + r] = (float)kCache[m * rows + r] * kScale[h * M + m] - kZero[h * M + m];
            vstored[m * rows + r] = (float)vCache[m * rows + r] * vScale[h * M + m] - vZero[h * M + m];
        }
    }

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, gOutBuffer, gOut);
    report("QKV-Rope-GEMM INT4", 100, qOut, qref, M * heads * dim, ms);
    report("QKV-Rope-GEMM-g INT4", 100, gOut, gref, M * gate_heads * dim, ms);
    report("QKV-Rope-GEMM-kv INT4", 100, kstored, kref, M * rows, ms);
    report("QKV-Rope-GEMM-v INT4", 100, vstored, vref, M * rows, ms);
    uint32_t posRead = 0;
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, posBuffer, &posRead);
    printf("QKV-Rope-GEMM-pos INT4: shader[0]= %u ref= %d\n", posRead, M);

    destroy_buffers(s, bufs, 17);
    free(qref); free(kref); free(vref); free(gref); free(qgamma); free(kgamma); free(gOut);
    free(qOut); free(kCache); free(vCache);
    free(kScale); free(kZero); free(vScale); free(vZero);
    free(kstored); free(vstored);
}

static int lmhead_argmax_ref_fp16(const float* x, const uint16_t* w, int n, int k) {
    float best = -FLT_MAX;
    int bestIdx = 0;
    for (int j = 0; j < n; j++) {
        float acc = 0.0f;
        for (int i = 0; i < k; i++) acc += x[i] * fp16_to_float(w[i * n + j]);
        if (acc > best) {
            best = acc;
            bestIdx = j;
        }
    }
    return bestIdx;
}

void validateSwigluFfnSplitKFP16(session s, int M, int N, int K, float* input, float* gamma, uint16_t* gateW, uint16_t* upW, uint16_t* downW) {
    float* act = (float*)malloc(sizeof(float) * M * N);
    swiglu_ref_fp16(input, gamma, gateW, upW, act, N, K);
    float* ref = (float*)malloc(sizeof(float) * M * K);
    gemv_ref_fp16(act, downW, ref, K, N);

    float* out = (float*)calloc(M * K, sizeof(float));
    float* resZero = (float*)calloc(K, sizeof(float));

    uint16_t* tg = (uint16_t*)malloc(sizeof(uint16_t) * K * N);
    uint16_t* tu = (uint16_t*)malloc(sizeof(uint16_t) * K * N);
    uint16_t* td = (uint16_t*)malloc(sizeof(uint16_t) * N * K);
    transpose_block16((uint8_t*)gateW, (uint8_t*)tg, K, N, QUANT_FP16);
    transpose_block16((uint8_t*)upW, (uint8_t*)tu, K, N, QUANT_FP16);
    transpose_block16((uint8_t*)downW, (uint8_t*)td, N, K, QUANT_FP16);
    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * M * K, MEMORY_RAM);
    buffer gateBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, tg, sizeof(uint16_t) * K * N, MEMORY_RAM);
    buffer upBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, tu, sizeof(uint16_t) * K * N, MEMORY_RAM);
    float* pinit = (float*)calloc(8 * N, sizeof(float));
    float* ginit = (float*)calloc(4 * K, sizeof(float));
    buffer pgpuBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, pinit, sizeof(float) * 8 * N, MEMORY_RAM);
    buffer downBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, td, sizeof(uint16_t) * N * K, MEMORY_RAM);
    buffer goutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, ginit, sizeof(float) * 4 * K, MEMORY_RAM);
    buffer resBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, resZero, sizeof(float) * K, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * K, MEMORY_RAM);
    buffer bufs[] = {xBuffer, gammaBuffer, gateBuffer, upBuffer, pgpuBuffer, downBuffer, goutBuffer, resBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 9);
    free(tg); free(tu); free(td); free(pinit); free(ginit);

    operation ops[] = {
        {.shader = "RmsNorm-up-ffn-SplitK-FP16.spv", .buffers = {xBuffer, gammaBuffer, gateBuffer, upBuffer, pgpuBuffer}, .bufferCount = 5,
         .pushConstants = {M, N, K, N}, .pushConstantCount = 4,
         .dispatchX = 2 * N / 256, .dispatchY = 4, .dispatchZ = 1},
        {.shader = "FFN-Down-SplitK-FP16.spv", .buffers = {pgpuBuffer, downBuffer, goutBuffer}, .bufferCount = 3,
         .pushConstants = {M, K, N}, .pushConstantCount = 3,
         .dispatchX = K / 256, .dispatchY = 4, .dispatchZ = 1},
        {.shader = "Reduce-GEMV-ADD.spv", .buffers = {goutBuffer, resBuffer, outBuffer}, .bufferCount = 3,
         .pushConstants = {K, 4}, .pushConstantCount = 2,
         .dispatchX = K / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 3);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("SwiGLU-Split FP16", 100, out, ref, M * K, ms);

    destroy_buffers(s, bufs, 9);
    free(act);
    free(ref);
    free(out);
    free(resZero);
}

void validateSwigluFfnSplitKINT8(session s, int M, int N, int K, float* input, float* gamma, QuantizedData gateQ, QuantizedData upQ, QuantizedData downQ) {
    float* act = (float*)malloc(sizeof(float) * M * N);
    swiglu_ref_int8(input, gamma, &gateQ, &upQ, act, N, K);
    float* ref = (float*)malloc(sizeof(float) * M * K);
    gemv_ref_int8(act, &downQ, ref, K, N);

    float* out = (float*)calloc(M * K, sizeof(float));
    float* resZero = (float*)calloc(K, sizeof(float));

    uint8_t* tg = (uint8_t*)malloc(K * N);
    uint8_t* tu = (uint8_t*)malloc(K * N);
    uint8_t* td = (uint8_t*)malloc(N * K);
    transpose_block16(gateQ.data, tg, K, N, QUANT_INT8);
    transpose_block16(upQ.data, tu, K, N, QUANT_INT8);
    transpose_block16(downQ.data, td, N, K, QUANT_INT8);
    int gs = K * N / gateQ.group_size;
    int ds = N * K / downQ.group_size;
    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * M * K, MEMORY_RAM);
    buffer gateBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, tg, K * N, MEMORY_RAM);
    buffer upBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, tu, K * N, MEMORY_RAM);
    float* pinit = (float*)calloc(8 * N, sizeof(float));
    float* ginit = (float*)calloc(4 * K, sizeof(float));
    buffer pgpuBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, pinit, sizeof(float) * 8 * N, MEMORY_RAM);
    buffer gsc = createBuffer(s.dev.device, s.dev.physicalDevice, gateQ.scale, sizeof(float) * gs, MEMORY_RAM);
    buffer gze = createBuffer(s.dev.device, s.dev.physicalDevice, gateQ.z, sizeof(float) * gs, MEMORY_RAM);
    buffer usc = createBuffer(s.dev.device, s.dev.physicalDevice, upQ.scale, sizeof(float) * gs, MEMORY_RAM);
    buffer uze = createBuffer(s.dev.device, s.dev.physicalDevice, upQ.z, sizeof(float) * gs, MEMORY_RAM);
    buffer downBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, td, N * K, MEMORY_RAM);
    buffer dsc = createBuffer(s.dev.device, s.dev.physicalDevice, downQ.scale, sizeof(float) * ds, MEMORY_RAM);
    buffer dze = createBuffer(s.dev.device, s.dev.physicalDevice, downQ.z, sizeof(float) * ds, MEMORY_RAM);
    buffer goutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, ginit, sizeof(float) * 4 * K, MEMORY_RAM);
    buffer resBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, resZero, sizeof(float) * K, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * K, MEMORY_RAM);
    buffer bufs[] = {xBuffer, gammaBuffer, gateBuffer, upBuffer, pgpuBuffer, gsc, gze, usc, uze, downBuffer, dsc, dze, goutBuffer, resBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 15);
    free(tg); free(tu); free(td); free(pinit); free(ginit);

    operation ops[] = {
        {.shader = "RmsNorm-up-ffn-SplitK-INT8.spv", .buffers = {xBuffer, gammaBuffer, gateBuffer, upBuffer, pgpuBuffer, gsc, gze, usc, uze}, .bufferCount = 9,
         .pushConstants = {M, N, K, N}, .pushConstantCount = 4,
         .dispatchX = 2 * N / 256, .dispatchY = 4, .dispatchZ = 1},
        {.shader = "FFN-Down-SplitK-INT8.spv", .buffers = {pgpuBuffer, downBuffer, goutBuffer, dsc, dze}, .bufferCount = 5,
         .pushConstants = {M, K, N}, .pushConstantCount = 3,
         .dispatchX = K / 256, .dispatchY = 4, .dispatchZ = 1},
        {.shader = "Reduce-GEMV-ADD.spv", .buffers = {goutBuffer, resBuffer, outBuffer}, .bufferCount = 3,
         .pushConstants = {K, 4}, .pushConstantCount = 2,
         .dispatchX = K / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 3);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("SwiGLU-Split INT8", 100, out, ref, M * K, ms);

    destroy_buffers(s, bufs, 15);
    free(act);
    free(ref);
    free(out);
    free(resZero);
}

void validateSwigluFfnSplitKINT4(session s, int M, int N, int K, float* input, float* gamma, QuantizedData gateQ, QuantizedData upQ, QuantizedData downQ) {
    float* act = (float*)malloc(sizeof(float) * M * N);
    swiglu_ref_int4(input, gamma, &gateQ, &upQ, act, N, K);
    float* ref = (float*)malloc(sizeof(float) * M * K);
    gemv_ref_int4(act, &downQ, ref, K, N);

    float* out = (float*)calloc(M * K, sizeof(float));
    float* resZero = (float*)calloc(K, sizeof(float));

    uint8_t* tg = (uint8_t*)malloc(K * N / 2);
    uint8_t* tu = (uint8_t*)malloc(K * N / 2);
    uint8_t* td = (uint8_t*)malloc(N * K / 2);
    transpose_block16(gateQ.data, tg, K, N, QUANT_INT4);
    transpose_block16(upQ.data, tu, K, N, QUANT_INT4);
    transpose_block16(downQ.data, td, N, K, QUANT_INT4);
    int gs = K * N / gateQ.group_size;
    int ds = N * K / downQ.group_size;
    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * M * K, MEMORY_RAM);
    buffer gateBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, tg, K * N / 2, MEMORY_RAM);
    buffer upBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, tu, K * N / 2, MEMORY_RAM);
    float* pinit = (float*)calloc(8 * N, sizeof(float));
    float* ginit = (float*)calloc(4 * K, sizeof(float));
    buffer pgpuBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, pinit, sizeof(float) * 8 * N, MEMORY_RAM);
    buffer gsc = createBuffer(s.dev.device, s.dev.physicalDevice, gateQ.scale, sizeof(float) * gs, MEMORY_RAM);
    buffer gze = createBuffer(s.dev.device, s.dev.physicalDevice, gateQ.z, sizeof(float) * gs, MEMORY_RAM);
    buffer usc = createBuffer(s.dev.device, s.dev.physicalDevice, upQ.scale, sizeof(float) * gs, MEMORY_RAM);
    buffer uze = createBuffer(s.dev.device, s.dev.physicalDevice, upQ.z, sizeof(float) * gs, MEMORY_RAM);
    buffer downBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, td, N * K / 2, MEMORY_RAM);
    buffer dsc = createBuffer(s.dev.device, s.dev.physicalDevice, downQ.scale, sizeof(float) * ds, MEMORY_RAM);
    buffer dze = createBuffer(s.dev.device, s.dev.physicalDevice, downQ.z, sizeof(float) * ds, MEMORY_RAM);
    buffer goutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, ginit, sizeof(float) * 4 * K, MEMORY_RAM);
    buffer resBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, resZero, sizeof(float) * K, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * M * K, MEMORY_RAM);
    buffer bufs[] = {xBuffer, gammaBuffer, gateBuffer, upBuffer, pgpuBuffer, gsc, gze, usc, uze, downBuffer, dsc, dze, goutBuffer, resBuffer, outBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 15);
    free(tg); free(tu); free(td); free(pinit); free(ginit);

    operation ops[] = {
        {.shader = "RmsNorm-up-ffn-SplitK-INT4.spv", .buffers = {xBuffer, gammaBuffer, gateBuffer, upBuffer, pgpuBuffer, gsc, gze, usc, uze}, .bufferCount = 9,
         .pushConstants = {M, N, K, N}, .pushConstantCount = 4,
         .dispatchX = 2 * N / 256, .dispatchY = 4, .dispatchZ = 1},
        {.shader = "FFN-Down-SplitK-INT4.spv", .buffers = {pgpuBuffer, downBuffer, goutBuffer, dsc, dze}, .bufferCount = 5,
         .pushConstants = {M, K, N}, .pushConstantCount = 3,
         .dispatchX = K / 256, .dispatchY = 4, .dispatchZ = 1},
        {.shader = "Reduce-GEMV-ADD.spv", .buffers = {goutBuffer, resBuffer, outBuffer}, .bufferCount = 3,
         .pushConstants = {K, 4}, .pushConstantCount = 2,
         .dispatchX = K / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 3);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("SwiGLU-Split INT4", 100, out, ref, M * K, ms);

    destroy_buffers(s, bufs, 15);
    free(act);
    free(ref);
    free(out);
    free(resZero);
}

#define K_OFF2 2048
#define V_OFF2 4096
#define Z_OFF2 8192
#define A_OFF2 12288
#define B_OFF2 12320

static void linear_proj_route_ref(const float* proj, float* q, float* k, float* v, float* z, float* a, float* b) {
    for (int i = 0; i < 2048; i++) q[i] = proj[i];
    for (int i = 0; i < 2048; i++) k[i] = proj[K_OFF2 + i];
    for (int i = 0; i < 4096; i++) v[i] = proj[V_OFF2 + i];
    for (int i = 0; i < 4096; i++) z[i] = proj[Z_OFF2 + i];
    for (int i = 0; i < 32; i++) a[i] = proj[A_OFF2 + i];
    for (int i = 0; i < 32; i++) b[i] = proj[B_OFF2 + i];
}

static double linear_proj_split_run(session s, int M, int K, float* input, float* gamma,
                                    buffer weightBuffer, buffer scaleBuffer, buffer zeroBuffer, int hasScale,
                                    int nTotal, const char* splitShader, const char* reduceShader,
                                    buffer qOut, buffer kOut, buffer vOut, buffer zOut, buffer aOut, buffer bOut) {
    buffer xBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * M * K, MEMORY_RAM);
    float* pinit = (float*)calloc(4 * nTotal, sizeof(float));
    buffer partialBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, pinit, sizeof(float) * 4 * nTotal, MEMORY_RAM);

    operation ops[2] = {0};
    if (hasScale) {
        snprintf(ops[0].shader, sizeof(ops[0].shader), "%s", splitShader);
        ops[0].buffers[0] = xBuffer; ops[0].buffers[1] = gammaBuffer; ops[0].buffers[2] = weightBuffer;
        ops[0].buffers[3] = scaleBuffer; ops[0].buffers[4] = zeroBuffer; ops[0].buffers[5] = partialBuffer;
        ops[0].bufferCount = 6;
    } else {
        snprintf(ops[0].shader, sizeof(ops[0].shader), "%s", splitShader);
        ops[0].buffers[0] = xBuffer; ops[0].buffers[1] = gammaBuffer; ops[0].buffers[2] = weightBuffer;
        ops[0].buffers[3] = partialBuffer;
        ops[0].bufferCount = 4;
    }
    ops[0].pushConstants[0] = M; ops[0].pushConstants[1] = nTotal; ops[0].pushConstants[2] = K;
    ops[0].pushConstantCount = 3;
    ops[0].dispatchX = (nTotal + 255) / 256; ops[0].dispatchY = 4; ops[0].dispatchZ = 1;

    snprintf(ops[1].shader, sizeof(ops[1].shader), "%s", reduceShader);
    ops[1].buffers[0] = partialBuffer; ops[1].buffers[1] = qOut; ops[1].buffers[2] = kOut;
    ops[1].buffers[3] = vOut; ops[1].buffers[4] = zOut; ops[1].buffers[5] = aOut; ops[1].buffers[6] = bOut;
    ops[1].bufferCount = 7;
    ops[1].pushConstants[0] = nTotal;
    ops[1].pushConstants[1] = 2048;
    ops[1].pushConstants[2] = 4096;
    ops[1].pushConstants[3] = 8192;
    ops[1].pushConstants[4] = 12288;
    ops[1].pushConstants[5] = 12320;
    ops[1].pushConstantCount = 6;
    ops[1].dispatchX = (nTotal + 255) / 256; ops[1].dispatchY = 1; ops[1].dispatchZ = 1;

    if (hasScale) {
        buffer bufs[] = {xBuffer, gammaBuffer, weightBuffer, scaleBuffer, zeroBuffer, partialBuffer};
        createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 6);
    } else {
        buffer bufs[] = {xBuffer, gammaBuffer, weightBuffer, partialBuffer};
        createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 4);
    }
    double ms = run_ops(s, ops, 2);
    free(pinit);
    destroyBuffer(s.dev.device, xBuffer);
    destroyBuffer(s.dev.device, gammaBuffer);
    destroyBuffer(s.dev.device, partialBuffer);
    return ms;
}

void validateLinearProjSplitKFP16(session s, int M, int K, float* input, float* gamma, uint16_t* w_inFP16) {
    int nTotal = 12352;
    float* xn = (float*)malloc(sizeof(float) * K);
    rms_norm_apply(input, gamma, xn, K);
    float* proj = (float*)malloc(sizeof(float) * nTotal);
    gemv_ref_fp16(xn, w_inFP16, proj, nTotal, K);
    float* qr = (float*)malloc(sizeof(float) * 2048);
    float* kr = (float*)malloc(sizeof(float) * 2048);
    float* vr = (float*)malloc(sizeof(float) * 4096);
    float* zr = (float*)malloc(sizeof(float) * 4096);
    float* ar = (float*)malloc(sizeof(float) * 32);
    float* br = (float*)malloc(sizeof(float) * 32);
    linear_proj_route_ref(proj, qr, kr, vr, zr, ar, br);

    uint16_t* tw = (uint16_t*)malloc(sizeof(uint16_t) * K * nTotal);
    transpose_block16((uint8_t*)w_inFP16, (uint8_t*)tw, K, nTotal, QUANT_FP16);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, tw, sizeof(uint16_t) * K * nTotal, MEMORY_RAM);
    buffer qO = createBuffer(s.dev.device, s.dev.physicalDevice, qr, sizeof(float) * 2048, MEMORY_RAM);
    buffer kO = createBuffer(s.dev.device, s.dev.physicalDevice, kr, sizeof(float) * 2048, MEMORY_RAM);
    buffer vO = createBuffer(s.dev.device, s.dev.physicalDevice, vr, sizeof(float) * 4096, MEMORY_RAM);
    buffer zO = createBuffer(s.dev.device, s.dev.physicalDevice, zr, sizeof(float) * 4096, MEMORY_RAM);
    buffer aO = createBuffer(s.dev.device, s.dev.physicalDevice, ar, sizeof(float) * 32, MEMORY_RAM);
    buffer bO = createBuffer(s.dev.device, s.dev.physicalDevice, br, sizeof(float) * 32, MEMORY_RAM);

    double ms = linear_proj_split_run(s, M, K, input, gamma, weightBuffer, weightBuffer, weightBuffer, 0,
                                      nTotal, "RmsNorm-LinearProj-SplitK-FP16.spv", "Reduce-LinearProj.spv",
                                      qO, kO, vO, zO, aO, bO);

    float* q = (float*)malloc(sizeof(float) * 2048);
    float* kk = (float*)malloc(sizeof(float) * 2048);
    float* vv = (float*)malloc(sizeof(float) * 4096);
    float* zz = (float*)malloc(sizeof(float) * 4096);
    float* aa = (float*)malloc(sizeof(float) * 32);
    float* bb = (float*)malloc(sizeof(float) * 32);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, qO, q);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kO, kk);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vO, vv);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, zO, zz);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, aO, aa);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, bO, bb);
    report("LP-Split-q FP16", 100, q, qr, 2048, ms);
    report("LP-Split-k FP16", 100, kk, kr, 2048, ms);
    report("LP-Split-v FP16", 100, vv, vr, 4096, ms);
    report("LP-Split-z FP16", 100, zz, zr, 4096, ms);
    report("LP-Split-a FP16", 0, aa, ar, 32, ms);
    report("LP-Split-b FP16", 0, bb, br, 32, ms);

    destroy_buffers(s, (buffer[]){weightBuffer, qO, kO, vO, zO, aO, bO}, 7);
    free(xn); free(proj); free(qr); free(kr); free(vr); free(zr); free(ar); free(br);
    free(q); free(kk); free(vv); free(zz); free(aa); free(bb);
}

void validateLinearProjSplitKINT8(session s, int M, int K, float* input, float* gamma, QuantizedData wQ) {
    int nTotal = 12352;
    float* xn = (float*)malloc(sizeof(float) * K);
    rms_norm_apply(input, gamma, xn, K);
    float* proj = (float*)malloc(sizeof(float) * nTotal);
    gemv_ref_int8(xn, &wQ, proj, nTotal, K);
    float* qr = (float*)malloc(sizeof(float) * 2048);
    float* kr = (float*)malloc(sizeof(float) * 2048);
    float* vr = (float*)malloc(sizeof(float) * 4096);
    float* zr = (float*)malloc(sizeof(float) * 4096);
    float* ar = (float*)malloc(sizeof(float) * 32);
    float* br = (float*)malloc(sizeof(float) * 32);
    linear_proj_route_ref(proj, qr, kr, vr, zr, ar, br);

    uint8_t* tw = (uint8_t*)malloc(K * nTotal);
    transpose_block16(wQ.data, tw, K, nTotal, QUANT_INT8);
    int sc = ((nTotal + 255) / 256) * K;
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, tw, K * nTotal, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, wQ.scale, sizeof(float) * sc, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, wQ.z, sizeof(float) * sc, MEMORY_RAM);
    buffer qO = createBuffer(s.dev.device, s.dev.physicalDevice, qr, sizeof(float) * 2048, MEMORY_RAM);
    buffer kO = createBuffer(s.dev.device, s.dev.physicalDevice, kr, sizeof(float) * 2048, MEMORY_RAM);
    buffer vO = createBuffer(s.dev.device, s.dev.physicalDevice, vr, sizeof(float) * 4096, MEMORY_RAM);
    buffer zO = createBuffer(s.dev.device, s.dev.physicalDevice, zr, sizeof(float) * 4096, MEMORY_RAM);
    buffer aO = createBuffer(s.dev.device, s.dev.physicalDevice, ar, sizeof(float) * 32, MEMORY_RAM);
    buffer bO = createBuffer(s.dev.device, s.dev.physicalDevice, br, sizeof(float) * 32, MEMORY_RAM);

    double ms = linear_proj_split_run(s, M, K, input, gamma, weightBuffer, scaleBuffer, zeroBuffer, 1,
                                      nTotal, "RmsNorm-LinearProj-SplitK-INT8.spv", "Reduce-LinearProj.spv",
                                      qO, kO, vO, zO, aO, bO);

    float* q = (float*)malloc(sizeof(float) * 2048);
    float* kk = (float*)malloc(sizeof(float) * 2048);
    float* vv = (float*)malloc(sizeof(float) * 4096);
    float* zz = (float*)malloc(sizeof(float) * 4096);
    float* aa = (float*)malloc(sizeof(float) * 32);
    float* bb = (float*)malloc(sizeof(float) * 32);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, qO, q);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kO, kk);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vO, vv);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, zO, zz);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, aO, aa);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, bO, bb);
    report("LP-Split-q INT8", 100, q, qr, 2048, ms);
    report("LP-Split-k INT8", 100, kk, kr, 2048, ms);
    report("LP-Split-v INT8", 100, vv, vr, 4096, ms);
    report("LP-Split-z INT8", 100, zz, zr, 4096, ms);
    report("LP-Split-a INT8", 0, aa, ar, 32, ms);
    report("LP-Split-b INT8", 0, bb, br, 32, ms);

    destroy_buffers(s, (buffer[]){weightBuffer, scaleBuffer, zeroBuffer, qO, kO, vO, zO, aO, bO}, 9);
    free(xn); free(proj); free(qr); free(kr); free(vr); free(zr); free(ar); free(br);
    free(q); free(kk); free(vv); free(zz); free(aa); free(bb);
}

void validateLinearProjSplitKINT4(session s, int M, int K, float* input, float* gamma, QuantizedData wQ) {
    int nTotal = 12352;
    float* xn = (float*)malloc(sizeof(float) * K);
    rms_norm_apply(input, gamma, xn, K);
    float* proj = (float*)malloc(sizeof(float) * nTotal);
    gemv_ref_int4(xn, &wQ, proj, nTotal, K);
    float* qr = (float*)malloc(sizeof(float) * 2048);
    float* kr = (float*)malloc(sizeof(float) * 2048);
    float* vr = (float*)malloc(sizeof(float) * 4096);
    float* zr = (float*)malloc(sizeof(float) * 4096);
    float* ar = (float*)malloc(sizeof(float) * 32);
    float* br = (float*)malloc(sizeof(float) * 32);
    linear_proj_route_ref(proj, qr, kr, vr, zr, ar, br);

    uint8_t* tw = (uint8_t*)malloc(K * nTotal / 2);
    transpose_block16(wQ.data, tw, K, nTotal, QUANT_INT4);
    int sc = ((nTotal + 255) / 256) * K;
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, tw, K * nTotal / 2, MEMORY_RAM);
    buffer scaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, wQ.scale, sizeof(float) * sc, MEMORY_RAM);
    buffer zeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, wQ.z, sizeof(float) * sc, MEMORY_RAM);
    buffer qO = createBuffer(s.dev.device, s.dev.physicalDevice, qr, sizeof(float) * 2048, MEMORY_RAM);
    buffer kO = createBuffer(s.dev.device, s.dev.physicalDevice, kr, sizeof(float) * 2048, MEMORY_RAM);
    buffer vO = createBuffer(s.dev.device, s.dev.physicalDevice, vr, sizeof(float) * 4096, MEMORY_RAM);
    buffer zO = createBuffer(s.dev.device, s.dev.physicalDevice, zr, sizeof(float) * 4096, MEMORY_RAM);
    buffer aO = createBuffer(s.dev.device, s.dev.physicalDevice, ar, sizeof(float) * 32, MEMORY_RAM);
    buffer bO = createBuffer(s.dev.device, s.dev.physicalDevice, br, sizeof(float) * 32, MEMORY_RAM);

    double ms = linear_proj_split_run(s, M, K, input, gamma, weightBuffer, scaleBuffer, zeroBuffer, 1,
                                      nTotal, "RmsNorm-LinearProj-SplitK-INT4.spv", "Reduce-LinearProj.spv",
                                      qO, kO, vO, zO, aO, bO);

    float* q = (float*)malloc(sizeof(float) * 2048);
    float* kk = (float*)malloc(sizeof(float) * 2048);
    float* vv = (float*)malloc(sizeof(float) * 4096);
    float* zz = (float*)malloc(sizeof(float) * 4096);
    float* aa = (float*)malloc(sizeof(float) * 32);
    float* bb = (float*)malloc(sizeof(float) * 32);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, qO, q);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kO, kk);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vO, vv);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, zO, zz);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, aO, aa);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, bO, bb);
    report("LP-Split-q INT4", 100, q, qr, 2048, ms);
    report("LP-Split-k INT4", 100, kk, kr, 2048, ms);
    report("LP-Split-v INT4", 100, vv, vr, 4096, ms);
    report("LP-Split-z INT4", 100, zz, zr, 4096, ms);
    report("LP-Split-a INT4", 0, aa, ar, 32, ms);
    report("LP-Split-b INT4", 0, bb, br, 32, ms);

    destroy_buffers(s, (buffer[]){weightBuffer, scaleBuffer, zeroBuffer, qO, kO, vO, zO, aO, bO}, 9);
    free(xn); free(proj); free(qr); free(kr); free(vr); free(zr); free(ar); free(br);
    free(q); free(kk); free(vv); free(zz); free(aa); free(bb);
}

void validateAttentionSplitKFP16(session s, int att_seq, int att_heads, int att_kv_heads, int att_dim, float* att_q, uint16_t* att_k, uint16_t* att_v) {
    int seq = att_seq;
    int heads = att_heads;
    int kv_heads = att_kv_heads;
    int dim = att_dim;
    int rows = kv_heads * dim;
    int kvs = rows * seq;
    float* kf = (float*)malloc(sizeof(float) * kvs);
    float* vf = (float*)malloc(sizeof(float) * kvs);
    for (int i = 0; i < kvs; i++) {
        kf[i] = fp16_to_float(att_k[i]);
        vf[i] = fp16_to_float(att_v[i]);
    }
    uint16_t* att_k_t = (uint16_t*)malloc(sizeof(uint16_t) * kvs);
    uint16_t* att_v_t = (uint16_t*)malloc(sizeof(uint16_t) * kvs);
    for (int s2 = 0; s2 < seq; s2++) {
        for (int r = 0; r < rows; r++) {
            att_k_t[s2 * rows + r] = att_k[r * seq + s2];
            att_v_t[s2 * rows + r] = att_v[r * seq + s2];
        }
    }
    float* out = (float*)calloc(heads * dim, sizeof(float));
    float* partials = (float*)calloc(528384, sizeof(float));

    uint32_t posVal = (uint32_t)(seq - 1);
    buffer keyBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_k_t, sizeof(uint16_t) * kvs, MEMORY_RAM);
    buffer valueBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_v_t, sizeof(uint16_t) * kvs, MEMORY_RAM);
    buffer queryBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_q, sizeof(float) * heads * dim, MEMORY_RAM);
    buffer partialBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, partials, sizeof(float) * 528384, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * heads * dim, MEMORY_RAM);
    buffer posBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, &posVal, sizeof(uint32_t), MEMORY_VRAM);
    buffer bufs[] = {keyBuffer, valueBuffer, queryBuffer, partialBuffer, outBuffer, posBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 6);
    free(att_k_t);
    free(att_v_t);

    operation ops[] = {
        {.shader = "Att-SplitK-FP16.spv", .buffers = {keyBuffer, valueBuffer, queryBuffer, partialBuffer, posBuffer}, .bufferCount = 5,
         .pushConstants = {32768, 4, 1024, 4, 256, 16}, .pushConstantCount = 6,
         .dispatchX = heads, .dispatchY = 32, .dispatchZ = 1},
        {.shader = "Reduce-Att.spv", .buffers = {partialBuffer, outBuffer, posBuffer}, .bufferCount = 3,
         .pushConstants = {16}, .pushConstantCount = 1,
         .dispatchX = heads * dim / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 2);

    float* ref = (float*)malloc(sizeof(float) * heads * dim);
    validate_attention(att_q, kf, vf, ref, seq, heads, kv_heads, dim);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("Attention SplitK FP16", 100, out, ref, heads * dim, ms);

    destroy_buffers(s, bufs, 6);
    free(kf);
    free(vf);
    free(out);
    free(ref);
    free(partials);
}

void validateAttentionSplitKINT8(session s, int att_seq, int att_heads, int att_kv_heads, int att_dim, float* att_q, uint16_t* att_k, uint16_t* att_v) {
    int seq = att_seq;
    int heads = att_heads;
    int kv_heads = att_kv_heads;
    int dim = att_dim;
    int rows = kv_heads * dim;
    int kvs = rows * seq;
    float* kf = (float*)malloc(sizeof(float) * kvs);
    float* vf = (float*)malloc(sizeof(float) * kvs);
    uint8_t* kq = (uint8_t*)malloc(kvs);
    uint8_t* vq = (uint8_t*)malloc(kvs);
    float* kScale = (float*)malloc(sizeof(float) * kv_heads * seq);
    float* kZero = (float*)malloc(sizeof(float) * kv_heads * seq);
    float* vScale = (float*)malloc(sizeof(float) * kv_heads * seq);
    float* vZero = (float*)malloc(sizeof(float) * kv_heads * seq);
    for (int h = 0; h < kv_heads; h++) {
        for (int s2 = 0; s2 < seq; s2++) {
            float mn = FLT_MAX;
            float mx = -FLT_MAX;
            for (int d = 0; d < dim; d++) {
                float kv = fp16_to_float(att_k[(h * dim + d) * seq + s2]);
                mn = fminf(mn, kv);
                mx = fmaxf(mx, kv);
            }
            float ks = (mx - mn) / 255.0f;
            if (ks == 0.0f) ks = 1.0f;
            kScale[h * seq + s2] = ks;
            kZero[h * seq + s2] = -mn;
            for (int d = 0; d < dim; d++) {
                float kv = fp16_to_float(att_k[(h * dim + d) * seq + s2]);
                int q = (int)rintf((kv + kZero[h * seq + s2]) / ks);
                if (q < 0) q = 0;
                if (q > 255) q = 255;
                kq[s2 * rows + (h * dim + d)] = (uint8_t)q;
            }
        }
    }
    for (int h = 0; h < kv_heads; h++) {
        for (int s2 = 0; s2 < seq; s2++) {
            float mn = FLT_MAX;
            float mx = -FLT_MAX;
            for (int d = 0; d < dim; d++) {
                float vv = fp16_to_float(att_v[(h * dim + d) * seq + s2]);
                mn = fminf(mn, vv);
                mx = fmaxf(mx, vv);
            }
            float vs = (mx - mn) / 255.0f;
            if (vs == 0.0f) vs = 1.0f;
            vScale[h * seq + s2] = vs;
            vZero[h * seq + s2] = -mn;
            for (int d = 0; d < dim; d++) {
                float vv = fp16_to_float(att_v[(h * dim + d) * seq + s2]);
                int q = (int)rintf((vv + vZero[h * seq + s2]) / vs);
                if (q < 0) q = 0;
                if (q > 255) q = 255;
                vq[s2 * rows + (h * dim + d)] = (uint8_t)q;
            }
        }
    }
    for (int r = 0; r < rows; r++) {
        int h = r / dim;
        for (int s2 = 0; s2 < seq; s2++) {
            kf[r * seq + s2] = (float)kq[s2 * rows + r] * kScale[h * seq + s2] - kZero[h * seq + s2];
            vf[r * seq + s2] = (float)vq[s2 * rows + r] * vScale[h * seq + s2] - vZero[h * seq + s2];
        }
    }
    float* out = (float*)calloc(heads * dim, sizeof(float));
    float* partials = (float*)calloc(528384, sizeof(float));

    uint32_t posVal = (uint32_t)(seq - 1);
    buffer keyBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kq, sizeof(uint8_t) * kvs, MEMORY_RAM);
    buffer valueBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vq, sizeof(uint8_t) * kvs, MEMORY_RAM);
    buffer queryBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_q, sizeof(float) * heads * dim, MEMORY_RAM);
    buffer kScaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kScale, sizeof(float) * kv_heads * seq, MEMORY_RAM);
    buffer kZeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kZero, sizeof(float) * kv_heads * seq, MEMORY_RAM);
    buffer vScaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vScale, sizeof(float) * kv_heads * seq, MEMORY_RAM);
    buffer vZeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vZero, sizeof(float) * kv_heads * seq, MEMORY_RAM);
    buffer partialBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, partials, sizeof(float) * 528384, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * heads * dim, MEMORY_RAM);
    buffer posBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, &posVal, sizeof(uint32_t), MEMORY_VRAM);
    buffer bufs[] = {keyBuffer, valueBuffer, queryBuffer, kScaleBuffer, kZeroBuffer, vScaleBuffer, vZeroBuffer, partialBuffer, outBuffer, posBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 10);

    operation ops[] = {
        {.shader = "Att-SplitK-INT8.spv", .buffers = {keyBuffer, valueBuffer, queryBuffer, kScaleBuffer, kZeroBuffer, vScaleBuffer, vZeroBuffer, partialBuffer, posBuffer}, .bufferCount = 9,
         .pushConstants = {32768, 4, 1024, 4, 256, 16}, .pushConstantCount = 6,
         .dispatchX = heads, .dispatchY = 32, .dispatchZ = 1},
        {.shader = "Reduce-Att.spv", .buffers = {partialBuffer, outBuffer, posBuffer}, .bufferCount = 3,
.pushConstants = {16}, .pushConstantCount = 1,
         .dispatchX = heads * dim / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 2);

    float* ref = (float*)malloc(sizeof(float) * heads * dim);
    validate_attention(att_q, kf, vf, ref, seq, heads, kv_heads, dim);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("Attention SplitK INT8", 100, out, ref, heads * dim, ms);

    destroy_buffers(s, bufs, 10);
    free(kf);
    free(vf);
    free(kq);
    free(vq);
    free(kScale);
    free(kZero);
    free(vScale);
    free(vZero);
    free(out);
    free(ref);
    free(partials);
}

void validateAttentionSplitKINT4(session s, int att_seq, int att_heads, int att_kv_heads, int att_dim, float* att_q, uint16_t* att_k, uint16_t* att_v) {
    int seq = att_seq;
    int heads = att_heads;
    int kv_heads = att_kv_heads;
    int dim = att_dim;
    int rows = kv_heads * dim;
    int kvs = rows * seq;
    float* kf = (float*)malloc(sizeof(float) * kvs);
    float* vf = (float*)malloc(sizeof(float) * kvs);
    uint8_t* kq = (uint8_t*)malloc(kvs);
    uint8_t* vq = (uint8_t*)malloc(kvs);
    float* kScale = (float*)malloc(sizeof(float) * kv_heads * seq);
    float* kZero = (float*)malloc(sizeof(float) * kv_heads * seq);
    float* vScale = (float*)malloc(sizeof(float) * kv_heads * seq);
    float* vZero = (float*)malloc(sizeof(float) * kv_heads * seq);
    for (int h = 0; h < kv_heads; h++) {
        for (int s2 = 0; s2 < seq; s2++) {
            float mn = FLT_MAX;
            float mx = -FLT_MAX;
            for (int d = 0; d < dim; d++) {
                float kv = fp16_to_float(att_k[(h * dim + d) * seq + s2]);
                mn = fminf(mn, kv);
                mx = fmaxf(mx, kv);
            }
            float ks = (mx - mn) / 15.0f;
            if (ks == 0.0f) ks = 1.0f;
            kScale[h * seq + s2] = ks;
            kZero[h * seq + s2] = -mn;
            for (int d = 0; d < dim; d++) {
                float kv = fp16_to_float(att_k[(h * dim + d) * seq + s2]);
                int q = (int)rintf((kv + kZero[h * seq + s2]) / ks);
                if (q < 0) q = 0;
                if (q > 15) q = 15;
                kq[s2 * rows + (h * dim + d)] = (uint8_t)q;
            }
        }
    }
    for (int h = 0; h < kv_heads; h++) {
        for (int s2 = 0; s2 < seq; s2++) {
            float mn = FLT_MAX;
            float mx = -FLT_MAX;
            for (int d = 0; d < dim; d++) {
                float vv = fp16_to_float(att_v[(h * dim + d) * seq + s2]);
                mn = fminf(mn, vv);
                mx = fmaxf(mx, vv);
            }
            float vs = (mx - mn) / 15.0f;
            if (vs == 0.0f) vs = 1.0f;
            vScale[h * seq + s2] = vs;
            vZero[h * seq + s2] = -mn;
            for (int d = 0; d < dim; d++) {
                float vv = fp16_to_float(att_v[(h * dim + d) * seq + s2]);
                int q = (int)rintf((vv + vZero[h * seq + s2]) / vs);
                if (q < 0) q = 0;
                if (q > 15) q = 15;
                vq[s2 * rows + (h * dim + d)] = (uint8_t)q;
            }
        }
    }
    for (int r = 0; r < rows; r++) {
        int h = r / dim;
        for (int s2 = 0; s2 < seq; s2++) {
            kf[r * seq + s2] = (float)kq[s2 * rows + r] * kScale[h * seq + s2] - kZero[h * seq + s2];
            vf[r * seq + s2] = (float)vq[s2 * rows + r] * vScale[h * seq + s2] - vZero[h * seq + s2];
        }
    }
    float* out = (float*)calloc(heads * dim, sizeof(float));
    float* partials = (float*)calloc(528384, sizeof(float));

    uint32_t posVal = (uint32_t)(seq - 1);
    buffer keyBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kq, sizeof(uint8_t) * kvs, MEMORY_RAM);
    buffer valueBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vq, sizeof(uint8_t) * kvs, MEMORY_RAM);
    buffer queryBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_q, sizeof(float) * heads * dim, MEMORY_RAM);
    buffer kScaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kScale, sizeof(float) * kv_heads * seq, MEMORY_RAM);
    buffer kZeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kZero, sizeof(float) * kv_heads * seq, MEMORY_RAM);
    buffer vScaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vScale, sizeof(float) * kv_heads * seq, MEMORY_RAM);
    buffer vZeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vZero, sizeof(float) * kv_heads * seq, MEMORY_RAM);
    buffer partialBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, partials, sizeof(float) * 528384, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * heads * dim, MEMORY_RAM);
    buffer posBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, &posVal, sizeof(uint32_t), MEMORY_VRAM);
    buffer bufs[] = {keyBuffer, valueBuffer, queryBuffer, kScaleBuffer, kZeroBuffer, vScaleBuffer, vZeroBuffer, partialBuffer, outBuffer, posBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 10);

    operation ops[] = {
        {.shader = "Att-SplitK-INT4.spv", .buffers = {keyBuffer, valueBuffer, queryBuffer, kScaleBuffer, kZeroBuffer, vScaleBuffer, vZeroBuffer, partialBuffer, posBuffer}, .bufferCount = 9,
         .pushConstants = {32768, 4, 1024, 4, 256, 16}, .pushConstantCount = 6,
         .dispatchX = heads, .dispatchY = 32, .dispatchZ = 1},
        {.shader = "Reduce-Att.spv", .buffers = {partialBuffer, outBuffer, posBuffer}, .bufferCount = 3,
.pushConstants = {16}, .pushConstantCount = 1,
         .dispatchX = heads * dim / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 2);

    float* ref = (float*)malloc(sizeof(float) * heads * dim);
    validate_attention(att_q, kf, vf, ref, seq, heads, kv_heads, dim);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("Attention SplitK INT4", 100, out, ref, heads * dim, ms);

    destroy_buffers(s, bufs, 10);
    free(kf);
    free(vf);
    free(kq);
    free(vq);
    free(kScale);
    free(kZero);
    free(vScale);
    free(vZero);
    free(out);
    free(ref);
    free(partials);
}

void validateAttentionSplitK2FP16(session s, int att_seq, int att_heads, int att_kv_heads, int att_dim, float* att_q, uint16_t* att_k, uint16_t* att_v) {
    int seq = att_seq;
    int heads = att_heads;
    int kv_heads = att_kv_heads;
    int dim = att_dim;
    int rows = kv_heads * dim;
    int kvs = rows * seq;
    float* kf = (float*)malloc(sizeof(float) * kvs);
    float* vf = (float*)malloc(sizeof(float) * kvs);
    for (int i = 0; i < kvs; i++) {
        kf[i] = fp16_to_float(att_k[i]);
        vf[i] = fp16_to_float(att_v[i]);
    }
    uint16_t* att_k_t = (uint16_t*)calloc((size_t)rows * 32768, sizeof(uint16_t));
    uint16_t* att_v_t = (uint16_t*)malloc(sizeof(uint16_t) * kvs);
    for (int s2 = 0; s2 < seq; s2++) {
        for (int r = 0; r < rows; r++) {
            att_k_t[r * 32768 + s2] = att_k[r * seq + s2];
            att_v_t[s2 * rows + r] = att_v[r * seq + s2];
        }
    }
    float* out = (float*)calloc(heads * dim, sizeof(float));
    float* partials = (float*)calloc(528384, sizeof(float));

    uint32_t posVal = (uint32_t)(seq - 1);
    buffer keyBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_k_t, sizeof(uint16_t) * (size_t)rows * 32768, MEMORY_RAM);
    buffer valueBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_v_t, sizeof(uint16_t) * kvs, MEMORY_RAM);
    buffer queryBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_q, sizeof(float) * heads * dim, MEMORY_RAM);
    buffer partialBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, partials, sizeof(float) * 528384, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * heads * dim, MEMORY_RAM);
    buffer posBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, &posVal, sizeof(uint32_t), MEMORY_VRAM);
    buffer bufs[] = {keyBuffer, valueBuffer, queryBuffer, partialBuffer, outBuffer, posBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 6);
    free(att_k_t);
    free(att_v_t);

    operation ops[] = {
        {.shader = "Att-SplitK2-FP16.spv", .buffers = {keyBuffer, valueBuffer, queryBuffer, partialBuffer, posBuffer}, .bufferCount = 5,
         .pushConstants = {32768, 4, 1024, 4, 256, 16}, .pushConstantCount = 6,
         .dispatchX = heads, .dispatchY = 128, .dispatchZ = 1},
        {.shader = "Reduce-Att2.spv", .buffers = {partialBuffer, outBuffer, posBuffer}, .bufferCount = 3,
         .pushConstants = {32768, 4, 1024, 4, 256, 16}, .pushConstantCount = 6,
         .dispatchX = heads * dim / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 2);

    float* ref = (float*)malloc(sizeof(float) * heads * dim);
    validate_attention(att_q, kf, vf, ref, seq, heads, kv_heads, dim);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("Attention SplitK2 FP16", 100, out, ref, heads * dim, ms);

    destroy_buffers(s, bufs, 6);
    free(kf);
    free(vf);
    free(out);
    free(ref);
    free(partials);
}

void validateAttentionSplitK2INT8(session s, int att_seq, int att_heads, int att_kv_heads, int att_dim, float* att_q, uint16_t* att_k, uint16_t* att_v) {
    int seq = att_seq;
    int heads = att_heads;
    int kv_heads = att_kv_heads;
    int dim = att_dim;
    int rows = kv_heads * dim;
    int kvs = rows * seq;
    float* kf = (float*)malloc(sizeof(float) * kvs);
    float* vf = (float*)malloc(sizeof(float) * kvs);
    uint8_t* kq = (uint8_t*)malloc(kvs);
    uint8_t* vq = (uint8_t*)malloc(kvs);
    float* kScale = (float*)malloc(sizeof(float) * kv_heads * 32768);
    float* kZero = (float*)malloc(sizeof(float) * kv_heads * 32768);
    float* vScale = (float*)malloc(sizeof(float) * kv_heads * 32768);
    float* vZero = (float*)malloc(sizeof(float) * kv_heads * 32768);
    for (int h = 0; h < kv_heads; h++) {
        for (int s2 = 0; s2 < seq; s2++) {
            float mn = FLT_MAX;
            float mx = -FLT_MAX;
            for (int d = 0; d < dim; d++) {
                float kv = fp16_to_float(att_k[(h * dim + d) * seq + s2]);
                mn = fminf(mn, kv);
                mx = fmaxf(mx, kv);
            }
            float ks = (mx - mn) / 255.0f;
            if (ks == 0.0f) ks = 1.0f;
            kScale[h * 32768 + s2] = ks;
            kZero[h * 32768 + s2] = -mn;
            for (int d = 0; d < dim; d++) {
                float kv = fp16_to_float(att_k[(h * dim + d) * seq + s2]);
                int q = (int)rintf((kv + kZero[h * 32768 + s2]) / ks);
                if (q < 0) q = 0;
                if (q > 255) q = 255;
                kq[s2 * rows + (h * dim + d)] = (uint8_t)q;
            }
        }
    }
    for (int h = 0; h < kv_heads; h++) {
        for (int s2 = 0; s2 < seq; s2++) {
            float mn = FLT_MAX;
            float mx = -FLT_MAX;
            for (int d = 0; d < dim; d++) {
                float vv = fp16_to_float(att_v[(h * dim + d) * seq + s2]);
                mn = fminf(mn, vv);
                mx = fmaxf(mx, vv);
            }
            float vs = (mx - mn) / 255.0f;
            if (vs == 0.0f) vs = 1.0f;
            vScale[h * 32768 + s2] = vs;
            vZero[h * 32768 + s2] = -mn;
            for (int d = 0; d < dim; d++) {
                float vv = fp16_to_float(att_v[(h * dim + d) * seq + s2]);
                int q = (int)rintf((vv + vZero[h * 32768 + s2]) / vs);
                if (q < 0) q = 0;
                if (q > 255) q = 255;
                vq[s2 * rows + (h * dim + d)] = (uint8_t)q;
            }
        }
    }
    for (int r = 0; r < rows; r++) {
        int h = r / dim;
        for (int s2 = 0; s2 < seq; s2++) {
            kf[r * seq + s2] = (float)kq[s2 * rows + r] * kScale[h * 32768 + s2] - kZero[h * 32768 + s2];
            vf[r * seq + s2] = (float)vq[s2 * rows + r] * vScale[h * 32768 + s2] - vZero[h * 32768 + s2];
        }
    }
    float* out = (float*)calloc(heads * dim, sizeof(float));
    float* partials = (float*)calloc(528384, sizeof(float));

    uint32_t posVal = (uint32_t)(seq - 1);
    uint8_t* kq_t = (uint8_t*)calloc((size_t)rows * 32768, sizeof(uint8_t));
    for (int s2 = 0; s2 < seq; s2++)
        for (int r = 0; r < rows; r++)
            kq_t[r * 32768 + s2] = kq[s2 * rows + r];
    buffer keyBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kq_t, sizeof(uint8_t) * (size_t)rows * 32768, MEMORY_RAM);
    free(kq_t);
    buffer valueBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vq, sizeof(uint8_t) * kvs, MEMORY_RAM);
    buffer queryBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_q, sizeof(float) * heads * dim, MEMORY_RAM);
    buffer kScaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kScale, sizeof(float) * kv_heads * 32768, MEMORY_RAM);
    buffer kZeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kZero, sizeof(float) * kv_heads * 32768, MEMORY_RAM);
    buffer vScaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vScale, sizeof(float) * kv_heads * 32768, MEMORY_RAM);
    buffer vZeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vZero, sizeof(float) * kv_heads * 32768, MEMORY_RAM);
    buffer partialBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, partials, sizeof(float) * 528384, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * heads * dim, MEMORY_RAM);
    buffer posBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, &posVal, sizeof(uint32_t), MEMORY_VRAM);
    buffer bufs[] = {keyBuffer, valueBuffer, queryBuffer, kScaleBuffer, kZeroBuffer, vScaleBuffer, vZeroBuffer, partialBuffer, outBuffer, posBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 10);

    operation ops[] = {
        {.shader = "Att-SplitK2-INT8.spv", .buffers = {keyBuffer, valueBuffer, queryBuffer, kScaleBuffer, kZeroBuffer, vScaleBuffer, vZeroBuffer, partialBuffer, posBuffer}, .bufferCount = 9,
         .pushConstants = {32768, 4, 1024, 4, 256, 16}, .pushConstantCount = 6,
         .dispatchX = heads, .dispatchY = 128, .dispatchZ = 1},
        {.shader = "Reduce-Att2.spv", .buffers = {partialBuffer, outBuffer, posBuffer}, .bufferCount = 3,
         .pushConstants = {32768, 4, 1024, 4, 256, 16}, .pushConstantCount = 6,
         .dispatchX = heads * dim / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 2);

    float* ref = (float*)malloc(sizeof(float) * heads * dim);
    validate_attention(att_q, kf, vf, ref, seq, heads, kv_heads, dim);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("Attention SplitK2 INT8", 100, out, ref, heads * dim, ms);

    destroy_buffers(s, bufs, 10);
    free(kf);
    free(vf);
    free(kq);
    free(vq);
    free(kScale);
    free(kZero);
    free(vScale);
    free(vZero);
    free(out);
    free(ref);
    free(partials);
}

void validateAttentionSplitK2INT4(session s, int att_seq, int att_heads, int att_kv_heads, int att_dim, float* att_q, uint16_t* att_k, uint16_t* att_v) {
    int seq = att_seq;
    int heads = att_heads;
    int kv_heads = att_kv_heads;
    int dim = att_dim;
    int rows = kv_heads * dim;
    int kvs = rows * seq;
    float* kf = (float*)malloc(sizeof(float) * kvs);
    float* vf = (float*)malloc(sizeof(float) * kvs);
    uint8_t* kq = (uint8_t*)malloc(kvs);
    uint8_t* vq = (uint8_t*)malloc(kvs);
    float* kScale = (float*)malloc(sizeof(float) * kv_heads * 32768);
    float* kZero = (float*)malloc(sizeof(float) * kv_heads * 32768);
    float* vScale = (float*)malloc(sizeof(float) * kv_heads * 32768);
    float* vZero = (float*)malloc(sizeof(float) * kv_heads * 32768);
    for (int h = 0; h < kv_heads; h++) {
        for (int s2 = 0; s2 < seq; s2++) {
            float mn = FLT_MAX;
            float mx = -FLT_MAX;
            for (int d = 0; d < dim; d++) {
                float kv = fp16_to_float(att_k[(h * dim + d) * seq + s2]);
                mn = fminf(mn, kv);
                mx = fmaxf(mx, kv);
            }
            float ks = (mx - mn) / 15.0f;
            if (ks == 0.0f) ks = 1.0f;
            kScale[h * 32768 + s2] = ks;
            kZero[h * 32768 + s2] = -mn;
            for (int d = 0; d < dim; d++) {
                float kv = fp16_to_float(att_k[(h * dim + d) * seq + s2]);
                int q = (int)rintf((kv + kZero[h * 32768 + s2]) / ks);
                if (q < 0) q = 0;
                if (q > 15) q = 15;
                kq[s2 * rows + (h * dim + d)] = (uint8_t)q;
            }
        }
    }
    for (int h = 0; h < kv_heads; h++) {
        for (int s2 = 0; s2 < seq; s2++) {
            float mn = FLT_MAX;
            float mx = -FLT_MAX;
            for (int d = 0; d < dim; d++) {
                float vv = fp16_to_float(att_v[(h * dim + d) * seq + s2]);
                mn = fminf(mn, vv);
                mx = fmaxf(mx, vv);
            }
            float vs = (mx - mn) / 15.0f;
            if (vs == 0.0f) vs = 1.0f;
            vScale[h * 32768 + s2] = vs;
            vZero[h * 32768 + s2] = -mn;
            for (int d = 0; d < dim; d++) {
                float vv = fp16_to_float(att_v[(h * dim + d) * seq + s2]);
                int q = (int)rintf((vv + vZero[h * 32768 + s2]) / vs);
                if (q < 0) q = 0;
                if (q > 15) q = 15;
                vq[s2 * rows + (h * dim + d)] = (uint8_t)q;
            }
        }
    }
    for (int r = 0; r < rows; r++) {
        int h = r / dim;
        for (int s2 = 0; s2 < seq; s2++) {
            kf[r * seq + s2] = (float)kq[s2 * rows + r] * kScale[h * 32768 + s2] - kZero[h * 32768 + s2];
            vf[r * seq + s2] = (float)vq[s2 * rows + r] * vScale[h * 32768 + s2] - vZero[h * 32768 + s2];
        }
    }
    float* out = (float*)calloc(heads * dim, sizeof(float));
    float* partials = (float*)calloc(528384, sizeof(float));

    uint32_t posVal = (uint32_t)(seq - 1);
    uint8_t* kq_t = (uint8_t*)calloc((size_t)rows * 32768, sizeof(uint8_t));
    for (int s2 = 0; s2 < seq; s2++)
        for (int r = 0; r < rows; r++)
            kq_t[r * 32768 + s2] = kq[s2 * rows + r];
    buffer keyBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kq_t, sizeof(uint8_t) * (size_t)rows * 32768, MEMORY_RAM);
    free(kq_t);
    buffer valueBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vq, sizeof(uint8_t) * kvs, MEMORY_RAM);
    buffer queryBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, att_q, sizeof(float) * heads * dim, MEMORY_RAM);
    buffer kScaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kScale, sizeof(float) * kv_heads * 32768, MEMORY_RAM);
    buffer kZeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kZero, sizeof(float) * kv_heads * 32768, MEMORY_RAM);
    buffer vScaleBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vScale, sizeof(float) * kv_heads * 32768, MEMORY_RAM);
    buffer vZeroBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vZero, sizeof(float) * kv_heads * 32768, MEMORY_RAM);
    buffer partialBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, partials, sizeof(float) * 528384, MEMORY_RAM);
    buffer outBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, out, sizeof(float) * heads * dim, MEMORY_RAM);
    buffer posBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, &posVal, sizeof(uint32_t), MEMORY_VRAM);
    buffer bufs[] = {keyBuffer, valueBuffer, queryBuffer, kScaleBuffer, kZeroBuffer, vScaleBuffer, vZeroBuffer, partialBuffer, outBuffer, posBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 10);

    operation ops[] = {
        {.shader = "Att-SplitK2-INT4.spv", .buffers = {keyBuffer, valueBuffer, queryBuffer, kScaleBuffer, kZeroBuffer, vScaleBuffer, vZeroBuffer, partialBuffer, posBuffer}, .bufferCount = 9,
         .pushConstants = {32768, 4, 1024, 4, 256, 16}, .pushConstantCount = 6,
         .dispatchX = heads, .dispatchY = 128, .dispatchZ = 1},
        {.shader = "Reduce-Att2.spv", .buffers = {partialBuffer, outBuffer, posBuffer}, .bufferCount = 3,
         .pushConstants = {32768, 4, 1024, 4, 256, 16}, .pushConstantCount = 6,
         .dispatchX = heads * dim / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 2);

    float* ref = (float*)malloc(sizeof(float) * heads * dim);
    validate_attention(att_q, kf, vf, ref, seq, heads, kv_heads, dim);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, outBuffer, out);
    report("Attention SplitK2 INT4", 100, out, ref, heads * dim, ms);

    destroy_buffers(s, bufs, 10);
    free(kf);
    free(vf);
    free(kq);
    free(vq);
    free(kScale);
    free(kZero);
    free(vScale);
    free(vZero);
    free(out);
    free(ref);
    free(partials);
}

void validateLmHeadArgMaxFP16(session s, int vocabSize, int K, float* input, float* gamma, uint16_t* lmHeadFP16) {
    int numGroups = (vocabSize + 255) / 256;
    float* maxValues = (float*)calloc(numGroups, sizeof(float));
    uint32_t* maxIndices = (uint32_t*)calloc(numGroups, sizeof(uint32_t));
    uint32_t* result = (uint32_t*)calloc(1, sizeof(uint32_t));
    uint32_t posVal = 41;
    uint32_t tokenVal = 0;

    uint16_t* transposed = (uint16_t*)malloc(sizeof(uint16_t) * K * vocabSize);
    transpose_block16((uint8_t*)lmHeadFP16, (uint8_t*)transposed, K, vocabSize, QUANT_FP16);

    float* logitsZero = (float*)calloc(vocabSize, sizeof(float));
    sample_params spZero = {0};
    uint32_t* histZero = (uint32_t*)calloc(MAX_PENALTY_LEN, sizeof(uint32_t));
    uint32_t rngZero = 1;

    buffer inputBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, input, sizeof(float) * K, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, transposed, sizeof(uint16_t) * K * vocabSize, MEMORY_RAM);
    buffer maxValueBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, maxValues, sizeof(float) * numGroups, MEMORY_VRAM);
    buffer maxIndexBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, maxIndices, sizeof(uint32_t) * numGroups, MEMORY_VRAM);
    buffer resultBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, result, sizeof(uint32_t), MEMORY_VRAM);
    buffer posBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, &posVal, sizeof(uint32_t), MEMORY_VRAM);
    buffer tokenIdsBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, &tokenVal, sizeof(uint32_t), MEMORY_VRAM);
    buffer logitsBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, logitsZero, sizeof(float) * vocabSize, MEMORY_VRAM);
    buffer sampleParamsBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, &spZero, sizeof(sample_params), MEMORY_RAM);
    buffer historyBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, histZero, sizeof(uint32_t) * MAX_PENALTY_LEN, MEMORY_VRAM);
    buffer rngBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, &rngZero, sizeof(uint32_t), MEMORY_RAM);
    buffer bufs[] = {inputBuffer, gammaBuffer, weightBuffer, maxValueBuffer, maxIndexBuffer, resultBuffer, posBuffer, tokenIdsBuffer, logitsBuffer, sampleParamsBuffer, historyBuffer, rngBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 12);
    free(transposed);
    free(logitsZero);
    free(histZero);

    operation ops[] = {
        {.shader = "LMHead-GEMV-ArgMax-FP16.spv", .buffers = {inputBuffer, weightBuffer, maxValueBuffer, maxIndexBuffer, gammaBuffer}, .bufferCount = 5,
         .pushConstants = {1, vocabSize, K}, .pushConstantCount = 3,
         .dispatchX = numGroups, .dispatchY = 1, .dispatchZ = 1},
        {.shader = "ArgMax-Reduce.spv", .buffers = {maxValueBuffer, maxIndexBuffer, logitsBuffer, sampleParamsBuffer, historyBuffer, rngBuffer, resultBuffer, posBuffer, tokenIdsBuffer}, .bufferCount = 9,
         .pushConstants = {vocabSize, 1, 0, 0}, .pushConstantCount = 4,
         .dispatchX = 1, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 2);

    float* xn = (float*)malloc(sizeof(float) * K);
    rms_norm_apply(input, gamma, xn, K);
    int ref = lmhead_argmax_ref_fp16(xn, lmHeadFP16, vocabSize, K);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, resultBuffer, result);
    uint32_t posRead = 0;
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, posBuffer, &posRead);
    printf("LMHead-ArgMax FP16: shader[0]= %u ref= %d\n", result[0], ref);
    printf("LMHead-ArgMax-pos FP16: shader[0]= %u ref= %d\n", posRead, posVal + 1);
    printf("LMHead-ArgMax FP16 time: %.3f ms\n", ms);

    destroy_buffers(s, bufs, 12);
    free(maxValues);
    free(maxIndices);
    free(result);
    free(xn);
}

static uint32_t xorshift32_ref(uint32_t x) {
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    return x;
}

typedef struct {
    double a;
    double p;
} sampler_pair;

static int sampler_pair_desc(const void* pa, const void* pb) {
    double da = ((const sampler_pair*)pa)->a;
    double db = ((const sampler_pair*)pb)->a;
    if (da > db) return -1;
    if (da < db) return 1;
    return 0;
}

static uint32_t sampler_ref(float* logits, int V, const sample_params* sp, uint32_t* history, uint32_t* pos, uint32_t* rng, uint32_t acceptTok, double tol, int* tolOut) {
    int penLen = (int)sp->penaltyLength;
    if (penLen > MAX_PENALTY_LEN) penLen = MAX_PENALTY_LEN;
    double* a = (double*)malloc(sizeof(double) * V);
    double* p = (double*)malloc(sizeof(double) * V);
    unsigned char* seen = (unsigned char*)calloc(V, 1);
    for (int j = 0; j < V; j++) a[j] = (double)logits[j] / (double)sp->temperature;
    if (penLen > 0 && (sp->repPenalty != 1.0f || sp->presencePenalty != 0.0f)) {
        for (int h = 0; h < penLen; h++) {
            uint32_t id = history[h];
            if (id < (uint32_t)V && !seen[id]) {
                seen[id] = 1;
                if (sp->repPenalty != 1.0f) {
                    a[id] = (a[id] > 0.0) ? (a[id] / (double)sp->repPenalty) : (a[id] * (double)sp->repPenalty);
                }
                a[id] -= (double)sp->presencePenalty / (double)sp->temperature;
            }
        }
    }

    double mx = -1e300;
    for (int j = 0; j < V; j++) if (a[j] > mx) mx = a[j];
    for (int j = 0; j < V; j++) {
        p[j] = exp(a[j] - mx);
        logits[j] = (float)p[j];
    }

    sampler_pair* arr = (sampler_pair*)malloc(sizeof(sampler_pair) * V);
    for (int j = 0; j < V; j++) {
        arr[j].a = a[j];
        arr[j].p = p[j];
    }
    qsort(arr, V, sizeof(sampler_pair), sampler_pair_desc);

    int topK = (int)sp->topK;
    if (topK > V) topK = V;
    double thrL = -1e300;
    int kept = V;
    if (topK > 0) {
        thrL = arr[topK - 1].a;
        kept = 0;
        while (kept < V && arr[kept].a >= thrL) kept++;
    }

    if (sp->topP < 1.0f && sp->topP > 0.0f) {
        double Zk = 0.0;
        for (int i = 0; i < kept; i++) Zk += arr[i].p;
        double target = (double)sp->topP * Zk;
        double acc = 0.0;
        int i = 0;
        for (; i < kept; i++) {
            acc += arr[i].p;
            if (acc >= target) break;
        }
        if (i >= kept) i = kept - 1;
        thrL = arr[i].a;
        kept = 0;
        while (kept < V && arr[kept].a >= thrL) kept++;
    }

    if (sp->minP > 0.0f) {
        double thrMin = mx + log((double)sp->minP);
        if (thrMin > thrL) {
            thrL = thrMin;
            kept = 0;
            while (kept < V && arr[kept].a >= thrL) kept++;
        }
    }

    double Zkept = 0.0;
    for (int i = 0; i < kept; i++) Zkept += arr[i].p;

    uint32_t s = *rng;
    if (s == 0u) s = 0x9E3779B9u;
    double u = (double)(s >> 8) / 16777216.0;
    double target = u * Zkept;
    double acc = 0.0;
    uint32_t chosen = 0;
    int found = 0;
    double qLo = -1.0;
    double qHi = -1.0;
    for (int j = 0; j < V; j++) {
        if (a[j] >= thrL) {
            if ((uint32_t)j == acceptTok) {
                qLo = acc;
                qHi = acc + p[j];
            }
            acc += p[j];
            if (!found) {
                chosen = (uint32_t)j;
                if (target < acc) found = 1;
            }
            if (found && (uint32_t)j >= acceptTok) break;
        }
    }
    if (!found) {
        for (int j = V - 1; j >= 0; j--) {
            if (a[j] >= thrL) { chosen = (uint32_t)j; break; }
        }
    }
    if (tolOut != NULL) *tolOut = 0;
    if (acceptTok != 0xFFFFFFFFu && acceptTok != chosen && qLo >= 0.0) {
        double dist = target > qHi ? target - qHi : (target < qLo ? qLo - target : 0.0);
        if (dist <= tol * Zkept) {
            chosen = acceptTok;
            if (tolOut != NULL) *tolOut = 1;
        }
    }

    if (penLen > 0) history[*pos % (uint32_t)penLen] = chosen;
    *rng = xorshift32_ref(s);
    *pos = *pos + 1u;

    free(a);
    free(p);
    free(seen);
    free(arr);
    return chosen;
}

void validateArgMaxSampler(session s, int vocabSize) {
    int numGroups = (vocabSize + 255) / 256;
    int drawsPerCase = 3;
    double uTol = 5e-4;

    float* logitsHost = getData(777001, 1, vocabSize);
    uint32_t* histInit = (uint32_t*)malloc(sizeof(uint32_t) * MAX_PENALTY_LEN);
    for (int h = 0; h < MAX_PENALTY_LEN; h++) histInit[h] = 0xFFFFFFFFu;
    for (int h = 0; h < 256; h++) {
        histInit[h] = (uint32_t)((h * 4099 + 17) % vocabSize);
        if ((h & 7) == 3) histInit[h] = histInit[h - 1];
        if ((h & 31) == 5) histInit[h] = 0xFFFFFFFFu;
    }

    float* dummyMax = (float*)calloc(numGroups, sizeof(float));
    uint32_t* dummyIdx = (uint32_t*)calloc(numGroups, sizeof(uint32_t));
    uint32_t posInit = 7;
    uint32_t rngInit = 0x2545F491u;
    uint32_t resultVal = 0;
    uint32_t tokenVal = 0;
    sample_params spCase[4] = {
        {0.1f, 1.2f, 256u, 40u, 0.9f, 0.0f, 0.0f},
        {0.1f, 1.0f, 0u, 0u, 1.0f, 0.0f, 0.0f},
        {0.1f, 1.0f, 0u, 0u, 1.0f, 0.001f, 0.0f},
        {0.1f, 1.0f, 256u, 40u, 0.9f, 0.0f, 0.3f}
    };
    uint32_t rngSeed[4] = {0x2545F491u, 0x9E3779B9u, 0x85EBCA6Bu, 0x7F4A7C15u};

    buffer maxValueBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, dummyMax, sizeof(float) * numGroups, MEMORY_VRAM);
    buffer maxIndexBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, dummyIdx, sizeof(uint32_t) * numGroups, MEMORY_VRAM);
    buffer logitsBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, logitsHost, sizeof(float) * vocabSize, MEMORY_VRAM);
    buffer sampleParamsBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, &spCase[0], sizeof(sample_params), MEMORY_RAM);
    buffer historyBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, histInit, sizeof(uint32_t) * MAX_PENALTY_LEN, MEMORY_VRAM);
    buffer rngBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, &rngInit, sizeof(uint32_t), MEMORY_RAM);
    buffer resultBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, &resultVal, sizeof(uint32_t), MEMORY_VRAM);
    buffer posBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, &posInit, sizeof(uint32_t), MEMORY_VRAM);
    buffer tokenIdsBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, &tokenVal, sizeof(uint32_t), MEMORY_VRAM);
    buffer bufs[] = {maxValueBuffer, maxIndexBuffer, logitsBuffer, sampleParamsBuffer, historyBuffer, rngBuffer, resultBuffer, posBuffer, tokenIdsBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 9);
    free(dummyMax);
    free(dummyIdx);

    uint32_t* refHist = (uint32_t*)malloc(sizeof(uint32_t) * MAX_PENALTY_LEN);
    memcpy(refHist, histInit, sizeof(uint32_t) * MAX_PENALTY_LEN);
    uint32_t posRef = posInit;
    uint32_t rngRef = rngSeed[0];

    for (int c = 0; c < 4; c++) {
        const char* name = c == 0 ? "ArgMax-Sampler-A" : (c == 1 ? "ArgMax-Sampler-B" : (c == 2 ? "ArgMax-Sampler-C" : "ArgMax-Sampler-D"));
        memcpy(sampleParamsBuffer.mappedMemory, &spCase[c], sizeof(sample_params));
        rngRef = rngSeed[c];
        memcpy(rngBuffer.mappedMemory, &rngRef, sizeof(uint32_t));
        double ms = 0.0;

        for (int d = 0; d < drawsPerCase; d++) {
            operation op = {
                .shader = "ArgMax-Reduce.spv",
                .buffers = {maxValueBuffer, maxIndexBuffer, logitsBuffer, sampleParamsBuffer, historyBuffer, rngBuffer, resultBuffer, posBuffer, tokenIdsBuffer},
                .bufferCount = 9,
                .pushConstants = {vocabSize, 1, 0, 1},
                .pushConstantCount = 4,
                .dispatchX = 1, .dispatchY = 1, .dispatchZ = 1
            };
            ms += run_ops(s, &op, 1);
            readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, resultBuffer, &resultVal);

            int tol = 0;
            uint32_t ref = sampler_ref(logitsHost, vocabSize, &spCase[c], refHist, &posRef, &rngRef, resultVal, uTol, &tol);
            int match = (resultVal == ref);
            printf("%s draw %d: shader= %u ref= %u match= %d%s\n", name, d, resultVal, ref, match, tol ? " (tolerance)" : "");
        }
        printf("%s time: %.3f ms\n", name, ms);
    }

    uint32_t posRead = 0;
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, posBuffer, &posRead);
    printf("ArgMax-Sampler-pos: shader= %u ref= %u match= %d\n", posRead, posRef, posRead == posRef);

    uint32_t* histRead = (uint32_t*)malloc(sizeof(uint32_t) * MAX_PENALTY_LEN);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, historyBuffer, histRead);
    int histErr = 0;
    for (int h = 0; h < MAX_PENALTY_LEN; h++) {
        if (histRead[h] != refHist[h]) histErr++;
    }
    printf("ArgMax-Sampler-history: mismatches= %d\n", histErr);

    destroy_buffers(s, bufs, 9);
    free(logitsHost);
    free(histInit);
    free(refHist);
    free(histRead);
}

void validateEmbedRmsNormLinearProjFP16(session s, int vocabSize, int K, uint32_t token, float* gamma, uint16_t* lmHeadFP16, uint16_t* w_inFP16) {
    int proj_n = 8256;
    float* emb = (float*)malloc(sizeof(float) * K);
    for (int i = 0; i < K; i++) emb[i] = fp16_to_float(lmHeadFP16[i * vocabSize + token]);
    float* xn = (float*)malloc(sizeof(float) * K);
    rms_norm_apply(emb, gamma, xn, K);
    float* proj = (float*)malloc(sizeof(float) * proj_n);
    gemv_ref_fp16(xn, w_inFP16, proj, proj_n, K);

    float* qref = (float*)malloc(sizeof(float) * 2048);
    float* kref = (float*)malloc(sizeof(float) * 2048);
    float* vref = (float*)malloc(sizeof(float) * 4096);
    float* aref = (float*)malloc(sizeof(float) * 32);
    float* bref = (float*)malloc(sizeof(float) * 32);
    memcpy(qref, proj, sizeof(float) * 2048);
    memcpy(kref, proj + 2048, sizeof(float) * 2048);
    memcpy(vref, proj + 4096, sizeof(float) * 4096);
    memcpy(aref, proj + 8192, sizeof(float) * 32);
    memcpy(bref, proj + 8224, sizeof(float) * 32);

    float* qOut = (float*)calloc(2048, sizeof(float));
    float* kOut = (float*)calloc(2048, sizeof(float));
    float* vOut = (float*)calloc(4096, sizeof(float));
    float* aOut = (float*)calloc(32, sizeof(float));
    float* bOut = (float*)calloc(32, sizeof(float));
    float* embOut = (float*)calloc(K, sizeof(float));

    uint16_t* twIn = (uint16_t*)malloc(sizeof(uint16_t) * K * proj_n);
    transpose_block16((uint8_t*)w_inFP16, (uint8_t*)twIn, K, proj_n, QUANT_FP16);
    uint16_t* tlm = (uint16_t*)malloc(sizeof(uint16_t) * K * vocabSize);
    transpose_block16((uint8_t*)lmHeadFP16, (uint8_t*)tlm, K, vocabSize, QUANT_FP16);

    buffer tokenBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, &token, sizeof(uint32_t), MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, tlm, sizeof(uint16_t) * K * vocabSize, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer wInBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, twIn, sizeof(uint16_t) * K * proj_n, MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * 2048, MEMORY_VRAM);
    buffer kOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kOut, sizeof(float) * 2048, MEMORY_VRAM);
    buffer vOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vOut, sizeof(float) * 4096, MEMORY_VRAM);
    buffer aOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, aOut, sizeof(float) * 32, MEMORY_VRAM);
    buffer bOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, bOut, sizeof(float) * 32, MEMORY_VRAM);
    buffer embOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, embOut, sizeof(float) * K, MEMORY_VRAM);
    buffer bufs[] = {tokenBuffer, weightBuffer, gammaBuffer, wInBuffer, qOutBuffer, kOutBuffer, vOutBuffer, aOutBuffer, bOutBuffer, embOutBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 10);
    free(twIn);
    free(tlm);

    operation ops[] = {
        {.shader = "Embed-RmsNorm-LinearProj-FP16.spv", .buffers = {tokenBuffer, weightBuffer, gammaBuffer, wInBuffer, qOutBuffer, kOutBuffer, vOutBuffer, aOutBuffer, bOutBuffer, embOutBuffer}, .bufferCount = 10,
         .pushConstants = {1, proj_n, K, 2048, 4096, 8192, 12288, 12320, vocabSize}, .pushConstantCount = 9,
         .dispatchX = (proj_n + 255) / 256, .dispatchY = 1, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, qOutBuffer, qOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kOutBuffer, kOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vOutBuffer, vOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, aOutBuffer, aOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, bOutBuffer, bOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, embOutBuffer, embOut);
    report("Embed-RmsNorm-LinearProj FP16", 100, qOut, qref, 2048, ms);
    report("Embed-RmsNorm-LinearProj-k FP16", 100, kOut, kref, 2048, ms);
    report("Embed-RmsNorm-LinearProj-v FP16", 100, vOut, vref, 4096, ms);
    report("Embed-RmsNorm-LinearProj-a FP16", 0, aOut, aref, 32, ms);
    report("Embed-RmsNorm-LinearProj-b FP16", 0, bOut, bref, 32, ms);
    report("Embed-RmsNorm-LinearProj-emb FP16", 100, embOut, emb, K, ms);

    destroy_buffers(s, bufs, 10);
    free(emb);
    free(xn);
    free(proj);
    free(qref); free(kref); free(vref); free(aref); free(bref);
    free(qOut); free(kOut); free(vOut); free(aOut); free(bOut);
    free(embOut);
}

void validateEmbedRmsNormLinearProjGEMMFP16(session s, int M, int vocabSize, int K, uint32_t* tokens, float* gamma, uint16_t* lmHeadFP16, uint16_t* w_inFP16) {
    int proj_n = 8256;
    float* qref = (float*)malloc(sizeof(float) * M * 2048);
    float* kref = (float*)malloc(sizeof(float) * M * 2048);
    float* vref = (float*)malloc(sizeof(float) * M * 4096);
    float* aref = (float*)malloc(sizeof(float) * M * 32);
    float* bref = (float*)malloc(sizeof(float) * M * 32);
    float* embRef = (float*)malloc(sizeof(float) * M * K);
    for (int m = 0; m < M; m++) {
        float* emb = (float*)malloc(sizeof(float) * K);
        for (int i = 0; i < K; i++) {
            emb[i] = fp16_to_float(lmHeadFP16[i * vocabSize + tokens[m]]);
            embRef[m * K + i] = emb[i];
        }
        float* xn = (float*)malloc(sizeof(float) * K);
        rms_norm_apply(emb, gamma, xn, K);
        float* proj = (float*)malloc(sizeof(float) * proj_n);
        gemv_ref_fp16(xn, w_inFP16, proj, proj_n, K);
        memcpy(qref + m * 2048, proj, sizeof(float) * 2048);
        memcpy(kref + m * 2048, proj + 2048, sizeof(float) * 2048);
        memcpy(vref + m * 4096, proj + 4096, sizeof(float) * 4096);
        memcpy(aref + m * 32, proj + 8192, sizeof(float) * 32);
        memcpy(bref + m * 32, proj + 8224, sizeof(float) * 32);
        free(emb);
        free(xn);
        free(proj);
    }

    float* qOut = (float*)calloc(M * 2048, sizeof(float));
    float* kOut = (float*)calloc(M * 2048, sizeof(float));
    float* vOut = (float*)calloc(M * 4096, sizeof(float));
    float* aOut = (float*)calloc(M * 32, sizeof(float));
    float* bOut = (float*)calloc(M * 32, sizeof(float));
    float* embOut = (float*)calloc(M * K, sizeof(float));

    uint16_t* twIn = (uint16_t*)malloc(sizeof(uint16_t) * K * proj_n);
    transpose_block16((uint8_t*)w_inFP16, (uint8_t*)twIn, K, proj_n, QUANT_FP16);
    uint16_t* tlm = (uint16_t*)malloc(sizeof(uint16_t) * K * vocabSize);
    transpose_block16((uint8_t*)lmHeadFP16, (uint8_t*)tlm, K, vocabSize, QUANT_FP16);

    buffer tokenBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, tokens, sizeof(uint32_t) * M, MEMORY_RAM);
    buffer weightBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, tlm, sizeof(uint16_t) * K * vocabSize, MEMORY_RAM);
    buffer gammaBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer wInBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, twIn, sizeof(uint16_t) * K * proj_n, MEMORY_RAM);
    buffer qOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, qOut, sizeof(float) * M * 2048, MEMORY_VRAM);
    buffer kOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, kOut, sizeof(float) * M * 2048, MEMORY_VRAM);
    buffer vOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, vOut, sizeof(float) * M * 4096, MEMORY_VRAM);
    buffer aOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, aOut, sizeof(float) * M * 32, MEMORY_VRAM);
    buffer bOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, bOut, sizeof(float) * M * 32, MEMORY_VRAM);
    buffer embOutBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, embOut, sizeof(float) * M * K, MEMORY_VRAM);
    buffer bufs[] = {tokenBuffer, weightBuffer, gammaBuffer, wInBuffer, qOutBuffer, kOutBuffer, vOutBuffer, aOutBuffer, bOutBuffer, embOutBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 10);
    free(twIn);
    free(tlm);

    operation ops[] = {
        {.shader = "Embed-RmsNorm-LinearProj-GEMM-FP16.spv", .buffers = {tokenBuffer, weightBuffer, gammaBuffer, wInBuffer, qOutBuffer, kOutBuffer, vOutBuffer, aOutBuffer, bOutBuffer, embOutBuffer}, .bufferCount = 10,
         .pushConstants = {M, proj_n, K, 2048, 4096, 8192, 12288, 12320, vocabSize}, .pushConstantCount = 9,
         .dispatchX = proj_n / 16, .dispatchY = M / 16, .dispatchZ = 1}
    };
    double ms = run_ops(s, ops, 1);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, qOutBuffer, qOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kOutBuffer, kOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vOutBuffer, vOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, aOutBuffer, aOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, bOutBuffer, bOut);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, embOutBuffer, embOut);
    report("Embed-RmsNorm-LinearProj-GEMM FP16", 100, qOut, qref, M * 2048, ms);
    report("Embed-RmsNorm-LinearProj-GEMM-k FP16", 100, kOut, kref, M * 2048, ms);
    report("Embed-RmsNorm-LinearProj-GEMM-v FP16", 100, vOut, vref, M * 4096, ms);
    report("Embed-RmsNorm-LinearProj-GEMM-a FP16", 0, aOut, aref, M * 32, ms);
    report("Embed-RmsNorm-LinearProj-GEMM-b FP16", 0, bOut, bref, M * 32, ms);
    report("Embed-RmsNorm-LinearProj-GEMM-emb FP16", 100, embOut, embRef, M * K, ms);

    destroy_buffers(s, bufs, 10);
    free(qref); free(kref); free(vref); free(aref); free(bref);
    free(embRef);
    free(qOut); free(kOut); free(vOut); free(aOut); free(bOut);
    free(embOut);
}

static void conv_split(const float* z, float* q, float* k, float* v, int M) {
    for (int m = 0; m < M; m++) {
        memcpy(q + m * 2048, z + m * 8192, sizeof(float) * 2048);
        memcpy(k + m * 2048, z + m * 8192 + 2048, sizeof(float) * 2048);
        memcpy(v + m * 4096, z + m * 8192 + 4096, sizeof(float) * 4096);
    }
}

static void conv_join(float* z, const float* q, const float* k, const float* v, int M) {
    for (int m = 0; m < M; m++) {
        memcpy(z + m * 8192, q + m * 2048, sizeof(float) * 2048);
        memcpy(z + m * 8192 + 2048, k + m * 2048, sizeof(float) * 2048);
        memcpy(z + m * 8192 + 4096, v + m * 4096, sizeof(float) * 4096);
    }
}

static void conv_silu_ref(const float* z, const float* conv, float* out, float* hist, int M) {
    for (int c = 0; c < 8192; c++) {
        float w0 = conv[c * 4 + 0];
        float w1 = conv[c * 4 + 1];
        float w2 = conv[c * 4 + 2];
        float w3 = conv[c * 4 + 3];
        float h0 = hist[0 * 8192 + c];
        float h1 = hist[1 * 8192 + c];
        float h2 = hist[2 * 8192 + c];
        for (int m = 0; m < M; m++) {
            float cur = z[m * 8192 + c];
            float zz = w0 * h2 + w1 * h1 + w2 * h0 + w3 * cur;
            out[m * 8192 + c] = zz / (1.0f + expf(-zz));
            h2 = h1;
            h1 = h0;
            h0 = cur;
        }
        hist[0 * 8192 + c] = h0;
        hist[1 * 8192 + c] = h1;
        hist[2 * 8192 + c] = h2;
    }
}

static void conv_silu_run(session s, const float* z, const float* conv, buffer histB,
                          float* out, float* histOut, int M, double* ms) {
    float* q = (float*)malloc(sizeof(float) * M * 2048);
    float* k = (float*)malloc(sizeof(float) * M * 2048);
    float* v = (float*)malloc(sizeof(float) * M * 4096);
    conv_split(z, q, k, v, M);

    buffer qB = createBuffer(s.dev.device, s.dev.physicalDevice, q, sizeof(float) * M * 2048, MEMORY_VRAM);
    buffer kB = createBuffer(s.dev.device, s.dev.physicalDevice, k, sizeof(float) * M * 2048, MEMORY_VRAM);
    buffer vB = createBuffer(s.dev.device, s.dev.physicalDevice, v, sizeof(float) * M * 4096, MEMORY_VRAM);
    buffer convB = createBuffer(s.dev.device, s.dev.physicalDevice, (void*)conv, sizeof(float) * 8192 * 4, MEMORY_VRAM);
    buffer bufs[] = {qB, kB, vB, convB};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 4);

    operation ops[] = {{
        .shader = "Conv-SiLU.spv",
        .buffers = {qB, kB, vB, convB, histB}, .bufferCount = 5,
        .pushConstants = {M, 2048, 4096, 8192, M}, .pushConstantCount = 5,
        .dispatchX = 8192 / 256, .dispatchY = 1, .dispatchZ = 1}};
    *ms = run_ops(s, ops, 1);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, qB, q);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, kB, k);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, vB, v);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, histB, histOut);
    conv_join(out, q, k, v, M);

    destroy_buffers(s, bufs, 4);
    free(q);
    free(k);
    free(v);
}

void validateConvSilu(session s, int M) {
    float* z = getData(7123, M, 8192);
    float* conv = getData(7245, 8192, 4);
    float* hinit = getData(7367, 3, 8192);

    float* refOut = (float*)malloc(sizeof(float) * M * 8192);
    float* refHist = (float*)malloc(sizeof(float) * 3 * 8192);
    memcpy(refHist, hinit, sizeof(float) * 3 * 8192);
    conv_silu_ref(z, conv, refOut, refHist, M);

    buffer histB = createBuffer(s.dev.device, s.dev.physicalDevice, hinit, sizeof(float) * 3 * 8192, MEMORY_VRAM);
    createTransferAndCopy(s.dev.device, s.dev.queue, &histB, 1);

    float* out = (float*)malloc(sizeof(float) * M * 8192);
    float* histOut = (float*)malloc(sizeof(float) * 3 * 8192);
    double ms = 0.0;
    conv_silu_run(s, z, conv, histB, out, histOut, M, &ms);

    report("Conv-SiLU", M * 8192 / 2, out, refOut, M * 8192, ms);
    report("Conv-SiLU-hist", 0, histOut, refHist, 3 * 8192, ms);

    destroyBuffer(s.dev.device, histB);
    free(z);
    free(conv);
    free(hinit);
    free(refOut);
    free(refHist);
    free(out);
    free(histOut);
}

void validateConvSiluChunked(session s, int M) {
    float* z1 = getData(7123, M, 8192);
    float* z2 = getData(8123, M, 8192);
    float* conv = getData(7245, 8192, 4);

    float* refOut = (float*)malloc(sizeof(float) * 2 * M * 8192);
    float* refHist = (float*)calloc(3 * 8192, sizeof(float));
    float* zAll = (float*)malloc(sizeof(float) * 2 * M * 8192);
    memcpy(zAll, z1, sizeof(float) * M * 8192);
    memcpy(zAll + M * 8192, z2, sizeof(float) * M * 8192);
    conv_silu_ref(zAll, conv, refOut, refHist, 2 * M);

    float* zeroHist = (float*)calloc(3 * 8192, sizeof(float));
    buffer histB = createBuffer(s.dev.device, s.dev.physicalDevice, zeroHist, sizeof(float) * 3 * 8192, MEMORY_VRAM);
    createTransferAndCopy(s.dev.device, s.dev.queue, &histB, 1);

    float* out1 = (float*)malloc(sizeof(float) * M * 8192);
    float* out2 = (float*)malloc(sizeof(float) * M * 8192);
    float* histOut = (float*)calloc(3 * 8192, sizeof(float));
    double ms = 0.0;
    conv_silu_run(s, z1, conv, histB, out1, histOut, M, &ms);
    conv_silu_run(s, z2, conv, histB, out2, histOut, M, &ms);

    float* outAll = (float*)malloc(sizeof(float) * 2 * M * 8192);
    memcpy(outAll, out1, sizeof(float) * M * 8192);
    memcpy(outAll + M * 8192, out2, sizeof(float) * M * 8192);

    report("Conv-SiLU-chunk", 2 * M * 8192 / 2, outAll, refOut, 2 * M * 8192, ms);
    report("Conv-SiLU-hist-chunk", 0, histOut, refHist, 3 * 8192, ms);

    destroyBuffer(s.dev.device, histB);
    free(z1);
    free(z2);
    free(conv);
    free(zAll);
    free(refOut);
    free(refHist);
    free(zeroHist);
    free(out1);
    free(out2);
    free(outAll);
    free(histOut);
}

void validateGateSigmoid(session s, int count, float* g, float* o) {
    float* ref = (float*)malloc(sizeof(float) * count);
    for (int i = 0; i < count; i++) {
        ref[i] = o[i] * (1.0f / (1.0f + expf(-g[i])));
    }
    buffer gBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, g, sizeof(float) * count, MEMORY_RAM);
    buffer oBuffer = createBuffer(s.dev.device, s.dev.physicalDevice, o, sizeof(float) * count, MEMORY_VRAM);
    buffer bufs[] = {gBuffer, oBuffer};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 2);
    operation ops[] = {{
        .shader = "Gate-Sigmoid.spv",
        .buffers = {gBuffer, oBuffer}, .bufferCount = 2,
        .pushConstants = {count}, .pushConstantCount = 1,
        .dispatchX = (count + 255) / 256, .dispatchY = 1, .dispatchZ = 1}};
    double ms = run_ops(s, ops, 1);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, oBuffer, o);
    report("Gate-Sigmoid", count / 2, o, ref, count, ms);
    destroy_buffers(s, bufs, 2);
    free(ref);
}


#define MOE_E 16
#define MOE_POOL 17
#define MOE_TOPK 8
#define MOE_SLOTS 9
#define MOE_K 2048
#define MOE_I 512
#define MOE_N 2048

static float moe_dq_int4(const QuantizedData* q, int k, int j) {
    uint8_t b = q->data[k * (q->N / 2) + j / 2];
    int nib = (j & 1) ? (b & 0x0F) : (b >> 4);
    int bj = j / q->group_size;
    return (float)nib * q->scale[bj * q->M + k] - q->z[bj * q->M + k];
}

static void moe_dequant_int4(const QuantizedData* q, float* out) {
    for (int j = 0; j < q->N; j++)
        for (int k = 0; k < q->M; k++)
            out[(size_t)j * q->M + k] = moe_dq_int4(q, k, j);
}

static void moe_router_ref(const float* xn, const uint16_t* w, const uint16_t* sgw, int M, int K, int E, int slots,
                           uint32_t* ids, float* weights, float* sharedW) {
    for (int m = 0; m < M; m++) {
        float logits[256];
        for (int e = 0; e < E; e++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += xn[m * K + k] * fp16_to_float(w[k * E + e]);
            logits[e] = acc;
        }
        float mx = logits[0];
        for (int e = 1; e < E; e++) if (logits[e] > mx) mx = logits[e];
        float probs[256];
        for (int e = 0; e < E; e++) probs[e] = expf(logits[e] - mx);
        float selProb[MOE_TOPK];
        uint32_t selIdx[MOE_TOPK];
        float selSum = 0.0f;
        for (int s = 0; s < MOE_TOPK; s++) {
            int best = 0;
            for (int e = 1; e < E; e++) if (probs[e] > probs[best]) best = e;
            selIdx[s] = (uint32_t)best;
            selProb[s] = probs[best];
            selSum += probs[best];
            probs[best] = -1.0f;
        }
        for (int s = 0; s < MOE_TOPK; s++) {
            ids[m * slots + s] = selIdx[s];
            weights[m * slots + s] = selProb[s] / selSum;
        }
        float sg = 0.0f;
        for (int k = 0; k < K; k++) sg += xn[m * K + k] * fp16_to_float(sgw[k]);
        sharedW[m] = 1.0f / (1.0f + expf(-sg));
    }
}

static void moe_swiglu_ref(const float* x, float** guF, const uint32_t* ids, int M, int slots, int K, int N, float* h) {
    for (int g = 0; g < M * slots; g++) {
        int e = (int)ids[g];
        const float* gf = guF[e];
        const float* uf = guF[e] + (size_t)N * K;
        const float* xg = x + (size_t)(g / slots) * K;
        for (int j = 0; j < N; j++) {
            const float* gr = gf + (size_t)j * K;
            const float* ur = uf + (size_t)j * K;
            float gate = 0.0f;
            float up = 0.0f;
            for (int k = 0; k < K; k++) {
                gate += xg[k] * gr[k];
                up += xg[k] * ur[k];
            }
            gate = gate / (1.0f + exp2f(-gate * 1.44269504f));
            h[g * N + j] = gate * up;
        }
    }
}

static void moe_down_ref(const float* h, float** dnF, const uint32_t* ids, int M, int slots, int K, int N, float* p) {
    for (int g = 0; g < M * slots; g++) {
        int e = (int)ids[g];
        const float* df = dnF[e];
        const float* hg = h + (size_t)g * K;
        for (int j = 0; j < N; j++) {
            const float* dr = df + (size_t)j * K;
            float acc = 0.0f;
            for (int k = 0; k < K; k++) acc += hg[k] * dr[k];
            p[g * N + j] = acc;
        }
    }
}

static void moe_combine_ref(const float* p, const float* weights, const float* sharedW, const float* h, int M, int slots, int N, float* y) {
    for (int m = 0; m < M; m++) {
        for (int c = 0; c < N; c++) {
            float acc = sharedW[m] * p[(size_t)(m * slots + slots - 1) * N + c];
            for (int s = 0; s < slots - 1; s++) {
                acc += weights[m * slots + s] * p[(size_t)(m * slots + s) * N + c];
            }
            y[m * N + c] = acc + h[m * N + c];
        }
    }
}

typedef struct {
    uint8_t* vramData;
    float* vramScale;
    float* vramZero;
    uint8_t* ramData;
    float* ramScale;
    float* ramZero;
    int vramCount;
} moe_pool;

static void moe_split_pool(moe_pool* out, const uint8_t* data, const float* scale, const float* zero,
                           int rows, int cols, int experts, int vramRouted) {
    size_t dataStride = (size_t)rows * cols / 2;
    size_t scaleStride = (size_t)(cols / 256) * rows;
    int vramCount = vramRouted + 1;
    out->vramCount = vramCount;
    out->vramData = (uint8_t*)malloc(dataStride * vramCount);
    out->vramScale = (float*)malloc(sizeof(float) * scaleStride * vramCount);
    out->vramZero = (float*)malloc(sizeof(float) * scaleStride * vramCount);
    memcpy(out->vramData, data, dataStride * vramRouted);
    memcpy(out->vramData + dataStride * vramRouted, data + dataStride * (experts - 1), dataStride);
    memcpy(out->vramScale, scale, sizeof(float) * scaleStride * vramRouted);
    memcpy(out->vramScale + scaleStride * vramRouted, scale + scaleStride * (experts - 1), sizeof(float) * scaleStride);
    memcpy(out->vramZero, zero, sizeof(float) * scaleStride * vramRouted);
    memcpy(out->vramZero + scaleStride * vramRouted, zero + scaleStride * (experts - 1), sizeof(float) * scaleStride);

    int ramCount = experts - 1 - vramRouted;
    out->ramData = (uint8_t*)malloc(dataStride * ramCount);
    out->ramScale = (float*)malloc(sizeof(float) * scaleStride * ramCount);
    out->ramZero = (float*)malloc(sizeof(float) * scaleStride * ramCount);
    memcpy(out->ramData, data + dataStride * vramRouted, dataStride * ramCount);
    memcpy(out->ramScale, scale + scaleStride * vramRouted, sizeof(float) * scaleStride * ramCount);
    memcpy(out->ramZero, zero + scaleStride * vramRouted, sizeof(float) * scaleStride * ramCount);
}

static void moe_free_pool(moe_pool pool) {
    free(pool.vramData);
    free(pool.vramScale);
    free(pool.vramZero);
    free(pool.ramData);
    free(pool.ramScale);
    free(pool.ramZero);
}

void validateRouterTopK(session s, int M) {
    int E = MOE_E;
    int K = MOE_K;
    int slots = MOE_TOPK;

    float* x = getData(9001, M, K);
    float* gamma = getData(9004, 1, K);
    float* xn = (float*)malloc(sizeof(float) * M * K);
    for (int m = 0; m < M; m++) rms_norm_apply(x + m * K, gamma, xn + m * K, K);

    uint16_t* w = getDataFP16(9002, K, E);
    uint16_t* sgw = getDataFP16(9003, K, 1);
    uint16_t* wT = (uint16_t*)malloc(sizeof(uint16_t) * K * E);
    uint16_t* sgT = (uint16_t*)malloc(sizeof(uint16_t) * K);
    transpose_block16((uint8_t*)w, (uint8_t*)wT, K, E, QUANT_FP16);
    transpose_block16((uint8_t*)sgw, (uint8_t*)sgT, K, 1, QUANT_FP16);

    uint32_t* refIds = (uint32_t*)calloc(M * slots, sizeof(uint32_t));
    float* refW = (float*)calloc(M * slots, sizeof(float));
    float* refSw = (float*)calloc(M, sizeof(float));
    moe_router_ref(xn, w, sgw, M, K, E, slots, refIds, refW, refSw);

    uint32_t* outIds = (uint32_t*)calloc(M * slots, sizeof(uint32_t));
    float* outW = (float*)calloc(M * slots, sizeof(float));
    float* outSw = (float*)calloc(M, sizeof(float));
    float* outXn = (float*)calloc(M * K, sizeof(float));

    buffer xB = createBuffer(s.dev.device, s.dev.physicalDevice, x, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaB = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer wB = createBuffer(s.dev.device, s.dev.physicalDevice, wT, sizeof(uint16_t) * K * E, MEMORY_RAM);
    buffer sgB = createBuffer(s.dev.device, s.dev.physicalDevice, sgT, sizeof(uint16_t) * K, MEMORY_RAM);
    buffer idsB = createBuffer(s.dev.device, s.dev.physicalDevice, outIds, sizeof(uint32_t) * M * slots, MEMORY_RAM);
    buffer wtsB = createBuffer(s.dev.device, s.dev.physicalDevice, outW, sizeof(float) * M * slots, MEMORY_RAM);
    buffer swB = createBuffer(s.dev.device, s.dev.physicalDevice, outSw, sizeof(float) * M, MEMORY_RAM);
    buffer xnB = createBuffer(s.dev.device, s.dev.physicalDevice, outXn, sizeof(float) * M * K, MEMORY_RAM);
    buffer bufs[] = {xB, gammaB, wB, sgB, idsB, wtsB, swB, xnB};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 8);

    operation op = {.shader = "Router-TopK.spv", .buffers = {xB, gammaB, wB, sgB, idsB, wtsB, swB, xnB}, .bufferCount = 8,
                    .pushConstants = {M, E, K, slots}, .pushConstantCount = 4,
                    .dispatchX = M, .dispatchY = 1, .dispatchZ = 1};
    double ms = run_ops(s, &op, 1);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, idsB, outIds);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, wtsB, outW);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, swB, outSw);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, xnB, outXn);

    int mism = 0;
    for (int i = 0; i < M * slots; i++) if (outIds[i] != refIds[i]) mism++;
    printf("Router-TopK ids: %d mismatches out of %d\n", mism, M * slots);
    report("Router-TopK weights", 0, outW, refW, M * slots, ms);
    report("Router-TopK sharedW", 0, outSw, refSw, M, ms);
    report("Router-TopK xn", 100, outXn, xn, M * K, ms);

    destroy_buffers(s, bufs, 8);
    free(x);
    free(gamma);
    free(xn);
    free(w);
    free(sgw);
    free(wT);
    free(sgT);
    free(refIds);
    free(refW);
    free(refSw);
    free(outIds);
    free(outW);
    free(outSw);
    free(outXn);
}

void validateExpertSwigluINT4(session s, int M) {
    int experts = MOE_POOL;
    int vramRouted = 10;
    int slots = MOE_TOPK;
    int K = MOE_K;
    int N = MOE_I;
    int sharedId = MOE_E;

    float* x = getData(9101, M, K);
    QuantizedData gu[MOE_POOL];
    for (int e = 0; e < experts; e++) gu[e] = getDataINT4(9200 + e, K, 2 * N);

    size_t dataStride = (size_t)K * 2 * N / 2;
    size_t scaleStride = (size_t)(2 * N / 256) * K;
    uint8_t* poolData = (uint8_t*)malloc(dataStride * experts);
    float* poolScale = (float*)malloc(sizeof(float) * scaleStride * experts);
    float* poolZero = (float*)malloc(sizeof(float) * scaleStride * experts);
    for (int e = 0; e < experts; e++) {
        transpose_block16(gu[e].data, poolData + e * dataStride, K, 2 * N, QUANT_INT4);
        memcpy(poolScale + e * scaleStride, gu[e].scale, sizeof(float) * scaleStride);
        memcpy(poolZero + e * scaleStride, gu[e].z, sizeof(float) * scaleStride);
    }
    moe_pool poolGpu;
    moe_split_pool(&poolGpu, poolData, poolScale, poolZero, K, 2 * N, experts, vramRouted);

    uint32_t* ids = (uint32_t*)malloc(sizeof(uint32_t) * M * slots);
    for (int g = 0; g < M * slots; g++) {
        int r = (g * 5 + 7) % experts;
        ids[g] = (uint32_t)r;
        if (g % 9 == 8) ids[g] = (uint32_t)sharedId;
    }

    float* guF[MOE_POOL];
    for (int e = 0; e < experts; e++) {
        guF[e] = (float*)malloc(sizeof(float) * (size_t)(2 * N) * K);
        moe_dequant_int4(&gu[e], guF[e]);
    }
    float* refH = (float*)malloc(sizeof(float) * M * slots * N);
    moe_swiglu_ref(x, guF, ids, M, slots, K, N, refH);

    float* outH = (float*)calloc(M * slots * N, sizeof(float));

    buffer xB = createBuffer(s.dev.device, s.dev.physicalDevice, x, sizeof(float) * M * K, MEMORY_RAM);
    buffer idsB = createBuffer(s.dev.device, s.dev.physicalDevice, ids, sizeof(uint32_t) * M * slots, MEMORY_RAM);
    buffer vramWB = createBuffer(s.dev.device, s.dev.physicalDevice, poolGpu.vramData, (int64_t)(dataStride * poolGpu.vramCount), MEMORY_RAM);
    buffer vramScaleB = createBuffer(s.dev.device, s.dev.physicalDevice, poolGpu.vramScale, sizeof(float) * scaleStride * poolGpu.vramCount, MEMORY_RAM);
    buffer vramZeroB = createBuffer(s.dev.device, s.dev.physicalDevice, poolGpu.vramZero, sizeof(float) * scaleStride * poolGpu.vramCount, MEMORY_RAM);
    buffer hB = createBuffer(s.dev.device, s.dev.physicalDevice, outH, sizeof(float) * M * slots * N, MEMORY_RAM);
    buffer ramWB = createBuffer(s.dev.device, s.dev.physicalDevice, poolGpu.ramData, (int64_t)(dataStride * (experts - poolGpu.vramCount)), MEMORY_RAM);
    buffer ramScaleB = createBuffer(s.dev.device, s.dev.physicalDevice, poolGpu.ramScale, sizeof(float) * scaleStride * (experts - poolGpu.vramCount), MEMORY_RAM);
    buffer ramZeroB = createBuffer(s.dev.device, s.dev.physicalDevice, poolGpu.ramZero, sizeof(float) * scaleStride * (experts - poolGpu.vramCount), MEMORY_RAM);
    buffer bufs[] = {xB, idsB, vramWB, vramScaleB, vramZeroB, hB, ramWB, ramScaleB, ramZeroB};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 9);

    operation op = {.shader = "Expert-Swiglu-INT4.spv", .buffers = {xB, idsB, vramWB, vramScaleB, vramZeroB, hB, ramWB, ramScaleB, ramZeroB}, .bufferCount = 9,
                    .pushConstants = {K, N, slots, vramRouted + 1, experts}, .pushConstantCount = 5,
                    .dispatchX = N / 256, .dispatchY = M * slots, .dispatchZ = 1};
    double ms = run_ops(s, &op, 1);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, hB, outH);
    report("Expert-Swiglu-INT4", 100, outH, refH, M * slots * N, ms);

    destroy_buffers(s, bufs, 9);
    moe_free_pool(poolGpu);
    free(poolData);
    free(poolScale);
    free(poolZero);
    for (int e = 0; e < experts; e++) {
        free_quantized_data(gu[e]);
        free(guF[e]);
    }
    free(x);
    free(ids);
    free(refH);
    free(outH);
}

void validateExpertDownINT4(session s, int M) {
    int experts = MOE_POOL;
    int vramRouted = 10;
    int slots = MOE_TOPK;
    int K = MOE_I;
    int N = MOE_N;
    int sharedId = MOE_E;

    float* h = getData(9301, M * slots, K);
    QuantizedData dn[MOE_POOL];
    for (int e = 0; e < experts; e++) dn[e] = getDataINT4(9400 + e, K, N);

    size_t dataStride = (size_t)K * N / 2;
    size_t scaleStride = (size_t)(N / 256) * K;
    uint8_t* poolData = (uint8_t*)malloc(dataStride * experts);
    float* poolScale = (float*)malloc(sizeof(float) * scaleStride * experts);
    float* poolZero = (float*)malloc(sizeof(float) * scaleStride * experts);
    for (int e = 0; e < experts; e++) {
        transpose_block16(dn[e].data, poolData + e * dataStride, K, N, QUANT_INT4);
        memcpy(poolScale + e * scaleStride, dn[e].scale, sizeof(float) * scaleStride);
        memcpy(poolZero + e * scaleStride, dn[e].z, sizeof(float) * scaleStride);
    }
    moe_pool poolGpu;
    moe_split_pool(&poolGpu, poolData, poolScale, poolZero, K, N, experts, vramRouted);

    uint32_t* ids = (uint32_t*)malloc(sizeof(uint32_t) * M * slots);
    for (int g = 0; g < M * slots; g++) {
        int r = (g * 3 + 1) % experts;
        ids[g] = (uint32_t)r;
        if (g % 9 == 8) ids[g] = (uint32_t)sharedId;
    }

    float* dnF[MOE_POOL];
    for (int e = 0; e < experts; e++) {
        dnF[e] = (float*)malloc(sizeof(float) * (size_t)N * K);
        moe_dequant_int4(&dn[e], dnF[e]);
    }
    float* refP = (float*)malloc(sizeof(float) * M * slots * N);
    moe_down_ref(h, dnF, ids, M, slots, K, N, refP);

    float* outP = (float*)calloc(M * slots * N, sizeof(float));

    buffer hB = createBuffer(s.dev.device, s.dev.physicalDevice, h, sizeof(float) * M * slots * K, MEMORY_RAM);
    buffer idsB = createBuffer(s.dev.device, s.dev.physicalDevice, ids, sizeof(uint32_t) * M * slots, MEMORY_RAM);
    buffer vramWB = createBuffer(s.dev.device, s.dev.physicalDevice, poolGpu.vramData, (int64_t)(dataStride * poolGpu.vramCount), MEMORY_RAM);
    buffer vramScaleB = createBuffer(s.dev.device, s.dev.physicalDevice, poolGpu.vramScale, sizeof(float) * scaleStride * poolGpu.vramCount, MEMORY_RAM);
    buffer vramZeroB = createBuffer(s.dev.device, s.dev.physicalDevice, poolGpu.vramZero, sizeof(float) * scaleStride * poolGpu.vramCount, MEMORY_RAM);
    buffer pB = createBuffer(s.dev.device, s.dev.physicalDevice, outP, sizeof(float) * M * slots * N, MEMORY_RAM);
    buffer ramWB = createBuffer(s.dev.device, s.dev.physicalDevice, poolGpu.ramData, (int64_t)(dataStride * (experts - poolGpu.vramCount)), MEMORY_RAM);
    buffer ramScaleB = createBuffer(s.dev.device, s.dev.physicalDevice, poolGpu.ramScale, sizeof(float) * scaleStride * (experts - poolGpu.vramCount), MEMORY_RAM);
    buffer ramZeroB = createBuffer(s.dev.device, s.dev.physicalDevice, poolGpu.ramZero, sizeof(float) * scaleStride * (experts - poolGpu.vramCount), MEMORY_RAM);
    buffer bufs[] = {hB, idsB, vramWB, vramScaleB, vramZeroB, pB, ramWB, ramScaleB, ramZeroB};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 9);

    operation op = {.shader = "Expert-Down-INT4.spv", .buffers = {hB, idsB, vramWB, vramScaleB, vramZeroB, pB, ramWB, ramScaleB, ramZeroB}, .bufferCount = 9,
                    .pushConstants = {K, N, slots, vramRouted + 1, experts}, .pushConstantCount = 5,
                    .dispatchX = N / 256, .dispatchY = M * slots, .dispatchZ = 1};
    double ms = run_ops(s, &op, 1);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, pB, outP);
    report("Expert-Down-INT4", 100, outP, refP, M * slots * N, ms);

    destroy_buffers(s, bufs, 9);
    moe_free_pool(poolGpu);
    free(poolData);
    free(poolScale);
    free(poolZero);
    for (int e = 0; e < experts; e++) {
        free_quantized_data(dn[e]);
        free(dnF[e]);
    }
    free(h);
    free(ids);
    free(refP);
    free(outP);
}

void validateMoeCombine(session s, int M) {
    int slots = MOE_TOPK;
    int N = MOE_N;

    float* p = getData(9501, M * slots, N);
    float* weights = getData(9502, M, slots);
    float* sharedW = getData(9503, M, 1);
    float* h = getData(9504, M, N);

    float* refY = (float*)malloc(sizeof(float) * M * N);
    moe_combine_ref(p, weights, sharedW, h, M, slots, N, refY);

    float* outY = (float*)calloc(M * N, sizeof(float));

    buffer pB = createBuffer(s.dev.device, s.dev.physicalDevice, p, sizeof(float) * M * slots * N, MEMORY_RAM);
    buffer wB = createBuffer(s.dev.device, s.dev.physicalDevice, weights, sizeof(float) * M * slots, MEMORY_RAM);
    buffer swB = createBuffer(s.dev.device, s.dev.physicalDevice, sharedW, sizeof(float) * M, MEMORY_RAM);
    buffer hB = createBuffer(s.dev.device, s.dev.physicalDevice, h, sizeof(float) * M * N, MEMORY_RAM);
    buffer yB = createBuffer(s.dev.device, s.dev.physicalDevice, outY, sizeof(float) * M * N, MEMORY_RAM);
    buffer bufs[] = {pB, wB, swB, hB, yB};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 5);

    operation op = {.shader = "Moe-Combine.spv", .buffers = {pB, wB, swB, hB, yB}, .bufferCount = 5,
                    .pushConstants = {M, N, slots}, .pushConstantCount = 3,
                    .dispatchX = (M * N + 255) / 256, .dispatchY = 1, .dispatchZ = 1};
    double ms = run_ops(s, &op, 1);

    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, yB, outY);
    report("Moe-Combine", 100, outY, refY, M * N, ms);

    destroy_buffers(s, bufs, 5);
    free(p);
    free(weights);
    free(sharedW);
    free(h);
    free(refY);
    free(outY);
}

void validateMoeLayerINT4(session s, int M) {
    int E = MOE_E;
    int experts = MOE_POOL;
    int vramRouted = 10;
    int slots = MOE_SLOTS;
    int K = MOE_K;
    int I = MOE_I;
    int N = MOE_N;
    int sharedId = MOE_E;

    float* x = getData(9601, M, K);
    float* gamma = getData(9602, 1, K);
    float* xn = (float*)malloc(sizeof(float) * M * K);
    for (int m = 0; m < M; m++) rms_norm_apply(x + m * K, gamma, xn + m * K, K);

    uint16_t* w = getDataFP16(9603, K, E);
    uint16_t* sgw = getDataFP16(9604, K, 1);
    uint16_t* wT = (uint16_t*)malloc(sizeof(uint16_t) * K * E);
    uint16_t* sgT = (uint16_t*)malloc(sizeof(uint16_t) * K);
    transpose_block16((uint8_t*)w, (uint8_t*)wT, K, E, QUANT_FP16);
    transpose_block16((uint8_t*)sgw, (uint8_t*)sgT, K, 1, QUANT_FP16);

    QuantizedData gu[MOE_POOL];
    QuantizedData dn[MOE_POOL];
    for (int e = 0; e < experts; e++) {
        gu[e] = getDataINT4(9700 + e, K, 2 * I);
        dn[e] = getDataINT4(9800 + e, I, N);
    }

    size_t guDataStride = (size_t)K * 2 * I / 2;
    size_t guScaleStride = (size_t)(2 * I / 256) * K;
    size_t dnDataStride = (size_t)I * N / 2;
    size_t dnScaleStride = (size_t)(N / 256) * I;
    uint8_t* guData = (uint8_t*)malloc(guDataStride * experts);
    float* guScale = (float*)malloc(sizeof(float) * guScaleStride * experts);
    float* guZero = (float*)malloc(sizeof(float) * guScaleStride * experts);
    uint8_t* dnData = (uint8_t*)malloc(dnDataStride * experts);
    float* dnScale = (float*)malloc(sizeof(float) * dnScaleStride * experts);
    float* dnZero = (float*)malloc(sizeof(float) * dnScaleStride * experts);
    for (int e = 0; e < experts; e++) {
        transpose_block16(gu[e].data, guData + e * guDataStride, K, 2 * I, QUANT_INT4);
        memcpy(guScale + e * guScaleStride, gu[e].scale, sizeof(float) * guScaleStride);
        memcpy(guZero + e * guScaleStride, gu[e].z, sizeof(float) * guScaleStride);
        transpose_block16(dn[e].data, dnData + e * dnDataStride, I, N, QUANT_INT4);
        memcpy(dnScale + e * dnScaleStride, dn[e].scale, sizeof(float) * dnScaleStride);
        memcpy(dnZero + e * dnScaleStride, dn[e].z, sizeof(float) * dnScaleStride);
    }
    moe_pool guPool;
    moe_split_pool(&guPool, guData, guScale, guZero, K, 2 * I, experts, vramRouted);
    moe_pool dnPool;
    moe_split_pool(&dnPool, dnData, dnScale, dnZero, I, N, experts, vramRouted);

    uint32_t* refIds = (uint32_t*)calloc(M * slots, sizeof(uint32_t));
    float* refW = (float*)calloc(M * slots, sizeof(float));
    float* refSw = (float*)calloc(M, sizeof(float));
    moe_router_ref(xn, w, sgw, M, K, E, slots, refIds, refW, refSw);
    for (int m = 0; m < M; m++) refIds[m * slots + slots - 1] = sharedId;

    float* guF[MOE_POOL];
    float* dnF[MOE_POOL];
    for (int e = 0; e < experts; e++) {
        guF[e] = (float*)malloc(sizeof(float) * (size_t)(2 * I) * K);
        moe_dequant_int4(&gu[e], guF[e]);
        dnF[e] = (float*)malloc(sizeof(float) * (size_t)N * I);
        moe_dequant_int4(&dn[e], dnF[e]);
    }
    float* refH = (float*)malloc(sizeof(float) * M * slots * I);
    moe_swiglu_ref(xn, guF, refIds, M, slots, K, I, refH);
    float* refP = (float*)malloc(sizeof(float) * M * slots * N);
    moe_down_ref(refH, dnF, refIds, M, slots, I, N, refP);
    float* refY = (float*)malloc(sizeof(float) * M * N);
    moe_combine_ref(refP, refW, refSw, x, M, slots, N, refY);

    uint32_t* idsH = (uint32_t*)calloc(M * slots, sizeof(uint32_t));
    for (int m = 0; m < M; m++) idsH[m * slots + slots - 1] = sharedId;
    uint32_t* outIds = (uint32_t*)calloc(M * slots, sizeof(uint32_t));
    float* outW = (float*)calloc(M * slots, sizeof(float));
    float* outSw = (float*)calloc(M, sizeof(float));
    float* outXn = (float*)calloc(M * K, sizeof(float));
    float* outH = (float*)calloc(M * slots * I, sizeof(float));
    float* outP = (float*)calloc(M * slots * N, sizeof(float));
    float* outY = (float*)calloc(M * N, sizeof(float));

    buffer xB = createBuffer(s.dev.device, s.dev.physicalDevice, x, sizeof(float) * M * K, MEMORY_RAM);
    buffer gammaB = createBuffer(s.dev.device, s.dev.physicalDevice, gamma, sizeof(float) * K, MEMORY_RAM);
    buffer wB = createBuffer(s.dev.device, s.dev.physicalDevice, wT, sizeof(uint16_t) * K * E, MEMORY_RAM);
    buffer sgB = createBuffer(s.dev.device, s.dev.physicalDevice, sgT, sizeof(uint16_t) * K, MEMORY_RAM);
    buffer idsB = createBuffer(s.dev.device, s.dev.physicalDevice, idsH, sizeof(uint32_t) * M * slots, MEMORY_RAM);
    buffer wtsB = createBuffer(s.dev.device, s.dev.physicalDevice, outW, sizeof(float) * M * slots, MEMORY_RAM);
    buffer swB = createBuffer(s.dev.device, s.dev.physicalDevice, outSw, sizeof(float) * M, MEMORY_RAM);
    buffer xnB = createBuffer(s.dev.device, s.dev.physicalDevice, outXn, sizeof(float) * M * K, MEMORY_RAM);
    buffer guVWB = createBuffer(s.dev.device, s.dev.physicalDevice, guPool.vramData, (int64_t)(guDataStride * guPool.vramCount), MEMORY_RAM);
    buffer guVScaleB = createBuffer(s.dev.device, s.dev.physicalDevice, guPool.vramScale, sizeof(float) * guScaleStride * guPool.vramCount, MEMORY_RAM);
    buffer guVZeroB = createBuffer(s.dev.device, s.dev.physicalDevice, guPool.vramZero, sizeof(float) * guScaleStride * guPool.vramCount, MEMORY_RAM);
    buffer guRWB = createBuffer(s.dev.device, s.dev.physicalDevice, guPool.ramData, (int64_t)(guDataStride * (experts - guPool.vramCount)), MEMORY_RAM);
    buffer guRScaleB = createBuffer(s.dev.device, s.dev.physicalDevice, guPool.ramScale, sizeof(float) * guScaleStride * (experts - guPool.vramCount), MEMORY_RAM);
    buffer guRZeroB = createBuffer(s.dev.device, s.dev.physicalDevice, guPool.ramZero, sizeof(float) * guScaleStride * (experts - guPool.vramCount), MEMORY_RAM);
    buffer dnVWB = createBuffer(s.dev.device, s.dev.physicalDevice, dnPool.vramData, (int64_t)(dnDataStride * dnPool.vramCount), MEMORY_RAM);
    buffer dnVScaleB = createBuffer(s.dev.device, s.dev.physicalDevice, dnPool.vramScale, sizeof(float) * dnScaleStride * dnPool.vramCount, MEMORY_RAM);
    buffer dnVZeroB = createBuffer(s.dev.device, s.dev.physicalDevice, dnPool.vramZero, sizeof(float) * dnScaleStride * dnPool.vramCount, MEMORY_RAM);
    buffer dnRWB = createBuffer(s.dev.device, s.dev.physicalDevice, dnPool.ramData, (int64_t)(dnDataStride * (experts - dnPool.vramCount)), MEMORY_RAM);
    buffer dnRScaleB = createBuffer(s.dev.device, s.dev.physicalDevice, dnPool.ramScale, sizeof(float) * dnScaleStride * (experts - dnPool.vramCount), MEMORY_RAM);
    buffer dnRZeroB = createBuffer(s.dev.device, s.dev.physicalDevice, dnPool.ramZero, sizeof(float) * dnScaleStride * (experts - dnPool.vramCount), MEMORY_RAM);
    buffer hB = createBuffer(s.dev.device, s.dev.physicalDevice, outH, sizeof(float) * M * slots * I, MEMORY_RAM);
    buffer pB = createBuffer(s.dev.device, s.dev.physicalDevice, outP, sizeof(float) * M * slots * N, MEMORY_RAM);
    buffer yB = createBuffer(s.dev.device, s.dev.physicalDevice, outY, sizeof(float) * M * N, MEMORY_RAM);
    buffer bufs[] = {xB, gammaB, wB, sgB, idsB, wtsB, swB, xnB, guVWB, guVScaleB, guVZeroB, guRWB, guRScaleB, guRZeroB, dnVWB, dnVScaleB, dnVZeroB, dnRWB, dnRScaleB, dnRZeroB, hB, pB, yB};
    createTransferAndCopy(s.dev.device, s.dev.queue, bufs, 23);

    operation routerOp = {.shader = "Router-TopK.spv", .buffers = {xB, gammaB, wB, sgB, idsB, wtsB, swB, xnB}, .bufferCount = 8,
                          .pushConstants = {M, E, K, slots}, .pushConstantCount = 4,
                          .dispatchX = M, .dispatchY = 1, .dispatchZ = 1};
    operation swigluOp = {.shader = "Expert-Swiglu-INT4.spv", .buffers = {xnB, idsB, guVWB, guVScaleB, guVZeroB, hB, guRWB, guRScaleB, guRZeroB}, .bufferCount = 9,
                          .pushConstants = {K, I, slots, vramRouted + 1, experts}, .pushConstantCount = 5,
                          .dispatchX = I / 256, .dispatchY = M * slots, .dispatchZ = 1};
    operation downOp = {.shader = "Expert-Down-INT4.spv", .buffers = {hB, idsB, dnVWB, dnVScaleB, dnVZeroB, pB, dnRWB, dnRScaleB, dnRZeroB}, .bufferCount = 9,
                        .pushConstants = {I, N, slots, vramRouted + 1, experts}, .pushConstantCount = 5,
                        .dispatchX = N / 256, .dispatchY = M * slots, .dispatchZ = 1};
    operation combineOp = {.shader = "Moe-Combine.spv", .buffers = {pB, wtsB, swB, xB, yB}, .bufferCount = 5,
                           .pushConstants = {M, N, slots}, .pushConstantCount = 3,
                           .dispatchX = (M * N + 255) / 256, .dispatchY = 1, .dispatchZ = 1};

    double ms1 = run_ops(s, &routerOp, 1);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, idsB, outIds);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, wtsB, outW);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, swB, outSw);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, xnB, outXn);
    int mism = 0;
    for (int i = 0; i < M * slots; i++) if (outIds[i] != refIds[i]) mism++;
    printf("Moe-Layer-INT4 ids: %d mismatches out of %d\n", mism, M * slots);
    report("Moe-Layer router weights", 0, outW, refW, M * slots, ms1);
    report("Moe-Layer sharedW", 0, outSw, refSw, M, ms1);
    report("Moe-Layer xn", 100, outXn, xn, M * K, ms1);

    double ms2 = run_ops(s, &swigluOp, 1);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, hB, outH);
    report("Moe-Layer-INT4 h", 100, outH, refH, M * slots * I, ms2);

    double ms3 = run_ops(s, &downOp, 1);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, pB, outP);
    report("Moe-Layer-INT4 partials", 100, outP, refP, M * slots * N, ms3);

    double ms4 = run_ops(s, &combineOp, 1);
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, yB, outY);
    report("Moe-Layer-INT4 y", 100, outY, refY, M * N, ms4);

    destroy_buffers(s, bufs, 23);
    moe_free_pool(guPool);
    moe_free_pool(dnPool);
    free(guData);
    free(guScale);
    free(guZero);
    free(dnData);
    free(dnScale);
    free(dnZero);
    for (int e = 0; e < experts; e++) {
        free_quantized_data(gu[e]);
        free_quantized_data(dn[e]);
        free(guF[e]);
        free(dnF[e]);
    }
    free(x);
    free(gamma);
    free(xn);
    free(w);
    free(sgw);
    free(wT);
    free(sgT);
    free(refIds);
    free(refW);
    free(refSw);
    free(refH);
    free(refP);
    free(refY);
    free(idsH);
    free(outIds);
    free(outW);
    free(outSw);
    free(outXn);
    free(outH);
    free(outP);
    free(outY);
}

void validation(void) {
    session s = createSession();

    int M = 1;
    int N = 12288;
    int K = 4096;
    float* input = getData(4321, M, K);
    float* input2 = getData(5555, M, K);
    float* gamma = getData(58923, M, K);
    float* weight = getData(936, K, N);
    uint16_t* weightFP16 = getDataFP16(936, K, N);
    QuantizedData weightINT8 = getDataINT8(936, K, N);
    QuantizedData weightINT4 = getDataINT4(936, K, N);
    uint16_t* weight2FP16 = getDataFP16(1348, K, N);
    QuantizedData weight2INT8 = getDataINT8(1348, K, N);
    QuantizedData weight2INT4 = getDataINT4(1348, K, N);

    int softmax_n = 1000;
    float* softmax_x = getData(2335, 1, softmax_n);
    float* softmax_v = getData(6346, softmax_n, 256);

    int att_seq = 2048;
    int att_heads = 16;
    int att_kv_heads = 4;
    int att_dim = 256;
    float* att_q = getData(7777, 1, att_heads * att_dim);
    uint16_t* att_k = getDataFP16(8100, att_kv_heads * att_dim, att_seq);
    uint16_t* att_v = getDataFP16(9100, att_kv_heads * att_dim, att_seq);

    int gate_count = 4096;
    float* gateG = getData(63001, 1, gate_count);
    float* gateO = getData(64001, 1, gate_count);
    validateGateSigmoid(s, gate_count, gateG, gateO);
    free(gateG);
    free(gateO);

    int qkv_heads = 16;
    int qkv_kv_heads = 4;
    int qkv_dim = 256;
    int qkv_gate_heads = qkv_heads;
    int qkv_n = (qkv_heads + qkv_gate_heads + 2 * qkv_kv_heads) * qkv_dim;
    float* qkv_weight = getData(2468, K, qkv_n);
    uint16_t* qkv_weightFP16 = getDataFP16(2468, K, qkv_n);
    QuantizedData qkv_weightINT8 = getDataINT8(2468, K, qkv_n);
    QuantizedData qkv_weightINT4 = getDataINT4(2468, K, qkv_n);
    float* qkv_theta = (float*)malloc(sizeof(float) * (QKV_ROPE_DIM / 2));
    for (int i = 0; i < QKV_ROPE_DIM / 2; i++) {
        qkv_theta[i] = pow(1e6, -((double)i) / (QKV_ROPE_DIM / 2));
    }

    int w_in_n = 12352;
    int wo_n = 4096;
    uint16_t* w_inFP16 = getDataFP16(3579, K, w_in_n);
    QuantizedData w_inINT8 = getDataINT8(3579, K, w_in_n);
    QuantizedData w_inINT4 = getDataINT4(3579, K, w_in_n);
    uint16_t* woFP16 = getDataFP16(8642, K, wo_n);
    QuantizedData woINT8 = getDataINT8(8642, K, wo_n);
    QuantizedData woINT4 = getDataINT4(8642, K, wo_n);

    // validateGEMV(s, M, N, K, input, weight);
    // validateGEMVFP16(s, M, N, K, input, weightFP16);
    // validateGEMVINT8(s, M, N, K, input, weightINT8);
    // validateGEMVINT4(s, M, N, K, input, weightINT4);

    int vocab_size = MODEL_VOCAB;
    uint16_t* lmHeadFP16 = getDataFP16(15001, K, vocab_size);
    validateLmHeadArgMaxFP16(s, vocab_size, K, input, gamma, lmHeadFP16);
    validateArgMaxSampler(s, vocab_size);

    int Mg = 64;
    // uint32_t token = 12345 % vocab_size;
    // validateEmbedRmsNormLinearProjFP16(s, vocab_size, K, token, gamma, lmHeadFP16, w_inFP16);
    // uint32_t tokensM[64];
    // for (int i = 0; i < Mg; i++) tokensM[i] = (uint32_t)((i * 1237 + 555) % vocab_size);
    // validateEmbedRmsNormLinearProjGEMMFP16(s, Mg, vocab_size, K, tokensM, gamma, lmHeadFP16, w_inFP16);
    free(lmHeadFP16);

    float* inputM = getData(4321, Mg, K);
    // validateGEMMFP16(s, Mg, N, K, inputM, weightFP16);
    // validateGEMMINT8(s, Mg, N, K, inputM, weightINT8);
    // validateGEMMINT4(s, Mg, N, K, inputM, weightINT4);

    float* residualM = getData(5555, Mg, wo_n);
    // validateGemmAddFP16(s, Mg, wo_n, K, inputM, residualM, woFP16);
    // validateGemmAddINT8(s, Mg, wo_n, K, inputM, residualM, woINT8);
    // validateGemmAddINT4(s, Mg, wo_n, K, inputM, residualM, woINT4);
    free(residualM);

    int ffn_n = 12288;
    float* ffnDownInput = getData(9111, 1, ffn_n);
    float* ffnDownResidual = getData(9222, 1, K);
    uint16_t* ffnDownFP16 = getDataFP16(9333, ffn_n, K);
    QuantizedData ffnDownINT8 = getDataINT8(9333, ffn_n, K);
    QuantizedData ffnDownINT4 = getDataINT4(9333, ffn_n, K);
    // validateGemvAddFP16(s, 1, K, ffn_n, ffnDownInput, ffnDownResidual, ffnDownFP16);
    // validateGemvAddINT8(s, 1, K, ffn_n, ffnDownInput, ffnDownResidual, ffnDownINT8);
    // validateGemvAddINT4(s, 1, K, ffn_n, ffnDownInput, ffnDownResidual, ffnDownINT4);
    validateGemvSplitKINT4(s, 1, K, ffn_n, ffnDownInput, ffnDownResidual, ffnDownINT4);
    validateSwigluFfnSplitKFP16(s, 1, ffn_n, K, input, gamma, weightFP16, weight2FP16, ffnDownFP16);
    validateSwigluFfnSplitKINT8(s, 1, ffn_n, K, input, gamma, weightINT8, weight2INT8, ffnDownINT8);
    validateSwigluFfnSplitKINT4(s, 1, ffn_n, K, input, gamma, weightINT4, weight2INT4, ffnDownINT4);
    float* ffnDownInputM = getData(9111, Mg, ffn_n);
    float* ffnDownResidualM = getData(9222, Mg, K);
    // validateGemmAddFP16(s, Mg, K, ffn_n, ffnDownInputM, ffnDownResidualM, ffnDownFP16);
    // validateGemmAddINT8(s, Mg, K, ffn_n, ffnDownInputM, ffnDownResidualM, ffnDownINT8);
    // validateGemmAddINT4(s, Mg, K, ffn_n, ffnDownInputM, ffnDownResidualM, ffnDownINT4);
    free(ffnDownInput);
    free(ffnDownResidual);
    free(ffnDownFP16);
    free_quantized_data(ffnDownINT8);
    free_quantized_data(ffnDownINT4);
    free(ffnDownInputM);
    free(ffnDownResidualM);

    // validateRmsNormSwigluFfnGEMMFP16(s, Mg, N, K, inputM, gamma, weightFP16, weight2FP16);
    // validateRmsNormSwigluFfnGEMMINT8(s, Mg, N, K, inputM, gamma, weightINT8, weight2INT8);
    // validateRmsNormSwigluFfnGEMMINT4(s, Mg, N, K, inputM, gamma, weightINT4, weight2INT4);
    validateRmsNormSwigluFfnGEMM2FP16(s, Mg, N, K, inputM, gamma, weightFP16, weight2FP16);
    validateRmsNormSwigluFfnGEMM2INT8(s, Mg, N, K, inputM, gamma, weightINT8, weight2INT8);
    validateRmsNormSwigluFfnGEMM2INT4(s, Mg, N, K, inputM, gamma, weightINT4, weight2INT4);
    validateRmsNormSwigluFlatGEMM2INT8(s, Mg, N, K, inputM, gamma, weightINT8, weight2INT8);
    validateRmsNormSwigluFlatGEMM2INT4(s, Mg, N, K, inputM, gamma, weightINT4, weight2INT4);

    // validateRmsNormLinearProjGEMMFP16(s, Mg, K, inputM, gamma, w_inFP16);
    // validateRmsNormLinearProjGEMMINT8(s, Mg, K, inputM, gamma, w_inINT8);
    // validateRmsNormLinearProjGEMMINT4(s, Mg, K, inputM, gamma, w_inINT4);

    // validateQkvRopeGEMMFP16(s, K, qkv_heads, qkv_kv_heads, qkv_dim, inputM, gamma, qkv_weightFP16, qkv_theta, Mg);
    validateQkvRopeGEMMINT8(s, K, qkv_heads, qkv_kv_heads, qkv_dim, inputM, gamma, qkv_weightINT8, qkv_theta, Mg);
    // validateQkvRopeGEMMINT4(s, K, qkv_heads, qkv_kv_heads, qkv_dim, inputM, gamma, qkv_weightINT4, qkv_theta, Mg);

    // validateAttentionGEMMFP16(s, Mg, 16, 4, 256);
    validateAttentionGEMMINT8(s, Mg, 16, 4, 256);
    // validateAttentionGEMMINT4(s, Mg, 16, 4, 256);

    validateGatedDeltaNetGEMMFP16(s, Mg, K, inputM, gamma, w_inFP16, woFP16);
    // validateGatedDeltaNetGEMMINT8(s, Mg, K, inputM, gamma, w_inINT8, woINT8);
    // validateGatedDeltaNetGEMMINT4(s, Mg, K, inputM, gamma, w_inINT4, woINT4);

    free(inputM);
    // validateRmsNormGEMVFP16(s, M, N, K, input, gamma, weightFP16);
    // validateRmsNormGEMVINT8(s, M, N, K, input, gamma, weightINT8);
    // validateRmsNormGEMVINT4(s, M, N, K, input, gamma, weightINT4);
    // validateGemvAddFP16(s, M, wo_n, K, input2, input, woFP16);
    // validateGemvAddINT8(s, M, wo_n, K, input2, input, woINT8);
    // validateGemvAddINT4(s, M, wo_n, K, input2, input, woINT4);
    validateGemvSplitKFP16(s, M, wo_n, K, input2, input, woFP16);
    validateGemvSplitKINT8(s, M, wo_n, K, input2, input, woINT8);
    validateGemvSplitKINT4(s, M, wo_n, K, input2, input, woINT4);
    // validateRmsNormSwigluFfn(s, M, N, K, input, gamma, weight);
    // validateRmsNormSwigluFfnFP16(s, M, N, K, input, gamma, weightFP16, weight2FP16);
    // validateRmsNormSwigluFfnINT8(s, M, N, K, input, gamma, weightINT8, weight2INT8);
    // validateRmsNormSwigluFfnINT4(s, M, N, K, input, gamma, weightINT4, weight2INT4);
    // validateOnlineSoftmax(s, softmax_n, softmax_x, softmax_v);
    // validateAttentionFP16(s, att_seq, att_heads, att_kv_heads, att_dim, att_q, att_k, att_v);
    validateAttentionINT8(s, att_seq, att_heads, att_kv_heads, att_dim, att_q, att_k, att_v);
    // validateAttentionINT4(s, att_seq, att_heads, att_kv_heads, att_dim, att_q, att_k, att_v);
    validateAttentionSplitKFP16(s, att_seq, att_heads, att_kv_heads, att_dim, att_q, att_k, att_v);
    validateAttentionSplitKINT8(s, att_seq, att_heads, att_kv_heads, att_dim, att_q, att_k, att_v);
    validateAttentionSplitKINT4(s, att_seq, att_heads, att_kv_heads, att_dim, att_q, att_k, att_v);
    validateAttentionSplitK2FP16(s, att_seq, att_heads, att_kv_heads, att_dim, att_q, att_k, att_v);
    validateAttentionSplitK2INT8(s, att_seq, att_heads, att_kv_heads, att_dim, att_q, att_k, att_v);
    validateAttentionSplitK2INT4(s, att_seq, att_heads, att_kv_heads, att_dim, att_q, att_k, att_v);
    // validateQkvRopeFP16(s, K, qkv_heads, qkv_kv_heads, qkv_dim, input, gamma, qkv_weightFP16, qkv_theta);
    validateQkvRopeINT8(s, K, qkv_heads, qkv_kv_heads, qkv_dim, input, gamma, qkv_weightINT8, qkv_theta);
    // validateQkvRopeINT4(s, K, qkv_heads, qkv_kv_heads, qkv_dim, input, gamma, qkv_weightINT4, qkv_theta);
    validateQkvRopeSplitKFP16(s, K, qkv_heads, qkv_kv_heads, qkv_dim, input, gamma, qkv_weightFP16, qkv_theta);
    validateQkvRopeSplitKINT8(s, K, qkv_heads, qkv_kv_heads, qkv_dim, input, gamma, qkv_weightINT8, qkv_theta);
    validateQkvRopeSplitKINT4(s, K, qkv_heads, qkv_kv_heads, qkv_dim, input, gamma, qkv_weightINT4, qkv_theta);
    validateLinearProjSplitKFP16(s, M, K, input, gamma, w_inFP16);
    validateLinearProjSplitKINT8(s, M, K, input, gamma, w_inINT8);
    validateLinearProjSplitKINT4(s, M, K, input, gamma, w_inINT4);
    validateConvSilu(s, 1);
    validateConvSilu(s, 64);
    validateConvSiluChunked(s, 64);
    // validateGatedDeltaNetFP16(s, K, input, input2, gamma, w_inFP16, woFP16);
    validateGatedDeltaNetINT8(s, K, input, input2, gamma, w_inINT8, woINT8);
    validateGatedDeltaNetINT4(s, K, input, input2, gamma, w_inINT4, woINT4);

    validateRouterTopK(s, 8);
    validateExpertSwigluINT4(s, 8);
    validateExpertDownINT4(s, 8);
    validateMoeCombine(s, 8);
    validateMoeLayerINT4(s, 1);
    validateMoeLayerINT4(s, 8);

    free(input);
    free(input2);
    free(gamma);
    free(weight);
    free(weightFP16);
    free_quantized_data(weightINT8);
    free_quantized_data(weightINT4);
    free(weight2FP16);
    free_quantized_data(weight2INT8);
    free_quantized_data(weight2INT4);
    free(softmax_x);
    free(softmax_v);
    free(att_q);
    free(att_k);
    free(att_v);
    free(qkv_weight);
    free(qkv_weightFP16);
    free_quantized_data(qkv_weightINT8);
    free_quantized_data(qkv_weightINT4);
    free(qkv_theta);
    free(w_inFP16);
    free_quantized_data(w_inINT8);
    free_quantized_data(w_inINT4);
    free(woFP16);
    free_quantized_data(woINT8);
    free_quantized_data(woINT4);

    destroySession(s);
}
