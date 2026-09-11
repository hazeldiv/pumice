#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <direct.h>
#include <windows.h>
#include "weights.h"
#include "safetensors.h"
#include "gguf.h"

static int64_t weightBytes = 0;
static int verboseWeights = 0;

static const char* SPINNER = "-\\|/";
static int spinnerIdx = 0;

static void printProgress(void) {
    if (verboseWeights) return;
    fprintf(stderr, "\r[%c] loading weights: %.2f MB      ",
            SPINNER[spinnerIdx++ % 4],
            (double)weightBytes / (1024.0 * 1024.0));
}

#define TENSOR_CACHE_MAX 512
#define TENSOR_FILE_MAGIC 0x54454E53

typedef struct {
    char name[64];
    QuantType q;
    int rows;
    int cols;
    uint8_t* data;
    int dataBytes;
    float* scale;
    float* zero;
    int scaleCount;
} cachedTensor;

static cachedTensor tensorCache[TENSOR_CACHE_MAX];
static int tensorCacheCount = 0;
static char cacheDir[256] = "weights";

static const char* quantSuffix(QuantType q) {
    if (q == QUANT_FP16) return "FP16";
    if (q == QUANT_INT8) return "INT8";
    return "INT4";
}

static void fatal(const char* msg) {
    fprintf(stderr, "weight load: %s\n", msg);
    exit(1);
}

#define MAX_WEIGHT_BUFS 2560
#define WEIGHT_FLUSH_BATCH 12

static session g_wbufSession;
static buffer* g_wbufs[MAX_WEIGHT_BUFS];
static int g_wbufsCount = 0;
static buffer g_wbufSmall[MAX_WEIGHT_BUFS];
static int g_wbufSmallCount = 0;

static void weightFlush(void) {
    if (g_wbufsCount + g_wbufSmallCount == 0) return;
    buffer tmp[WEIGHT_FLUSH_BATCH * 2];
    int n = 0;
    for (int i = 0; i < g_wbufsCount; i++) tmp[n++] = *g_wbufs[i];
    for (int i = 0; i < g_wbufSmallCount; i++) tmp[n++] = g_wbufSmall[i];
    createTransferAndCopy(g_wbufSession.dev.device, g_wbufSession.dev.queue, tmp, n);
    for (int i = 0; i < g_wbufsCount; i++) releaseStaging(g_wbufSession.dev.device, g_wbufs[i]);
    g_wbufsCount = 0;
    g_wbufSmallCount = 0;
}

static void registerWeightBuffer(buffer* b) {
    if (g_wbufsCount < MAX_WEIGHT_BUFS) {
        g_wbufs[g_wbufsCount++] = b;
    }
    if (g_wbufsCount + g_wbufSmallCount >= WEIGHT_FLUSH_BATCH) weightFlush();
}

static void registerWeightBufferSmall(buffer b) {
    if (g_wbufSmallCount < MAX_WEIGHT_BUFS) {
        g_wbufSmall[g_wbufSmallCount++] = b;
    }
    if (g_wbufsCount + g_wbufSmallCount >= WEIGHT_FLUSH_BATCH) weightFlush();
}

static cachedTensor* cacheFind(const char* name, QuantType q) {
    for (int i = 0; i < tensorCacheCount; i++) {
        if (tensorCache[i].q == q && strcmp(tensorCache[i].name, name) == 0) {
            return &tensorCache[i];
        }
    }
    return NULL;
}

static cachedTensor* cacheStore(const char* name, QuantType q, int rows, int cols, uint8_t* data, int dataBytes, float* scale, float* zero, int scaleCount) {
    if (tensorCacheCount >= TENSOR_CACHE_MAX) fatal("tensor cache overflow");
    cachedTensor* ct = &tensorCache[tensorCacheCount++];
    memset(ct, 0, sizeof(cachedTensor));
    snprintf(ct->name, sizeof(ct->name), "%s", name);
    ct->q = q;
    ct->rows = rows;
    ct->cols = cols;
    ct->data = data;
    ct->dataBytes = dataBytes;
    ct->scale = scale;
    ct->zero = zero;
    ct->scaleCount = scaleCount;
    return ct;
}

static void cacheClear(void) {
    for (int i = 0; i < tensorCacheCount; i++) {
        free(tensorCache[i].data);
        free(tensorCache[i].scale);
        free(tensorCache[i].zero);
    }
    tensorCacheCount = 0;
}

static void cacheRelease(cachedTensor* ct) {
    if (ct == NULL) return;
    free(ct->data);
    free(ct->scale);
    free(ct->zero);
    ct->data = NULL;
    ct->scale = NULL;
    ct->zero = NULL;
}

static void tensorWriteFile(const char* path, QuantType q, int rows, int cols, const uint8_t* data, int dataBytes, const float* scale, const float* zero, int scaleCount) {
    FILE* f = fopen(path, "wb");
    if (!f) return;
    int header[4] = {TENSOR_FILE_MAGIC, rows, cols, (int)q};
    fwrite(header, sizeof(int), 4, f);
    fwrite(data, 1, dataBytes, f);
    if (q != QUANT_FP16) {
        fwrite(scale, sizeof(float), scaleCount, f);
        fwrite(zero, sizeof(float), scaleCount, f);
    }
    fclose(f);
}

static cachedTensor* tensorLoadFile(const char* path, const char* name, QuantType q, int rows, int cols, int blocks) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    int header[4];
    if (fread(header, sizeof(int), 4, f) != 4) {
        fclose(f);
        return NULL;
    }
    if (header[0] != TENSOR_FILE_MAGIC || header[1] != rows || header[2] != cols || header[3] != (int)q) {
        fclose(f);
        return NULL;
    }
    int dataBytes;
    if (q == QUANT_FP16) dataBytes = rows * cols * 2;
    else if (q == QUANT_INT8) dataBytes = rows * cols;
    else dataBytes = rows * cols / 2;
    int scaleCount = rows * blocks;
    uint8_t* data = (uint8_t*)malloc(dataBytes);
    float* scale = NULL;
    float* zero = NULL;
    if (fread(data, 1, dataBytes, f) != (size_t)dataBytes) {
        free(data);
        fclose(f);
        return NULL;
    }
    if (q != QUANT_FP16) {
        scale = (float*)malloc(sizeof(float) * scaleCount);
        zero = (float*)malloc(sizeof(float) * scaleCount);
        if (fread(scale, sizeof(float), scaleCount, f) != (size_t)scaleCount ||
            fread(zero, sizeof(float), scaleCount, f) != (size_t)scaleCount) {
            free(data);
            free(scale);
            free(zero);
            fclose(f);
            return NULL;
        }
    }
    fclose(f);
    return cacheStore(name, q, rows, cols, data, dataBytes, scale, zero, scaleCount);
}

static cachedTensor* cacheGet(const char* name, QuantType q, int rows, int cols) {
    cachedTensor* ct = cacheFind(name, q);
    if (ct != NULL) return ct;
    int blocks = (cols + 255) / 256;
    char path[384];
    snprintf(path, sizeof(path), "%s/%s_%s.bin", cacheDir, name, quantSuffix(q));
    return tensorLoadFile(path, name, q, rows, cols, blocks);
}

static cachedTensor* tensorBuild(const char* path, const char* name, QuantType q, int rows, int cols, float wscale, const float* mat) {
    int blocks = (cols + 255) / 256;
    int scaleCount = rows * blocks;
    int64_t total = (int64_t)rows * cols;
    cachedTensor* ct;

    if (q == QUANT_FP16) {
        uint16_t* w = (uint16_t*)malloc(sizeof(uint16_t) * total);
        for (int64_t i = 0; i < total; i++) w[i] = float_to_fp16(mat[i] * wscale);
        uint16_t* tw = (uint16_t*)malloc(sizeof(uint16_t) * total);
        transpose_block16((uint8_t*)w, (uint8_t*)tw, rows, cols, QUANT_FP16);
        free(w);
        ct = cacheStore(name, q, rows, cols, (uint8_t*)tw, (int)(total * 2), NULL, NULL, 0);
        tensorWriteFile(path, q, rows, cols, ct->data, ct->dataBytes, NULL, NULL, 0);
    } else {
        QuantizedData qd = (q == QUANT_INT8) ? quantizeDataINT8(mat, rows, cols) : quantizeDataINT4(mat, rows, cols);
        if (wscale != 1.0f) {
            for (int i = 0; i < scaleCount; i++) {
                qd.scale[i] *= wscale;
                qd.z[i] *= wscale;
            }
        }
        int dataBytes = (q == QUANT_INT8) ? rows * cols : rows * cols / 2;
        uint8_t* tw = (uint8_t*)malloc(dataBytes);
        transpose_block16(qd.data, tw, rows, cols, q);
        free(qd.data);
        ct = cacheStore(name, q, rows, cols, tw, dataBytes, qd.scale, qd.z, scaleCount);
        tensorWriteFile(path, q, rows, cols, ct->data, ct->dataBytes, ct->scale, ct->zero, ct->scaleCount);
    }
    return ct;
}

static void countBuffer(const char* name, int layer, buffer b) {
    weightBytes += b.size;
    if (verboseWeights) {
        fprintf(stderr, "%s[%d]: %lld bytes (%.2f MB) | Total: %lld bytes (%.2f MB)\n",
                name, layer, (long long)b.size, (double)b.size / (1024.0 * 1024.0),
                (long long)weightBytes, (double)weightBytes / (1024.0 * 1024.0));
    } else {
        printProgress();
    }
}

static void loadTensorInto(session s, tensor* t, const char* name, int layer, int rows, int cols, QuantType q, float wscale, const float* mat) {
    t->q = q;
    t->rows = rows;
    t->cols = cols;

    cachedTensor* ct = cacheGet(name, q, rows, cols);
    if (ct == NULL) {
        char path[384];
        snprintf(path, sizeof(path), "%s/%s_%s.bin", cacheDir, name, quantSuffix(q));
        ct = tensorBuild(path, name, q, rows, cols, wscale, mat);
    }

    t->data = createBufferNamed(s.dev.device, s.dev.physicalDevice, ct->data, ct->dataBytes, MEMORY_VRAM, name);
    countBuffer(name, layer, t->data);
    registerWeightBuffer(&t->data);
    if (q != QUANT_FP16) {
        char label[80];
        snprintf(label, sizeof(label), "%s-scale", name);
        t->scale = createBufferNamed(s.dev.device, s.dev.physicalDevice, ct->scale, sizeof(float) * ct->scaleCount, MEMORY_VRAM, label);
        countBuffer(label, layer, t->scale);
        registerWeightBuffer(&t->scale);
        snprintf(label, sizeof(label), "%s-zero", name);
        t->zero = createBufferNamed(s.dev.device, s.dev.physicalDevice, ct->zero, sizeof(float) * ct->scaleCount, MEMORY_VRAM, label);
        countBuffer(label, layer, t->zero);
        registerWeightBuffer(&t->zero);
    }
    cacheRelease(ct);
}

static void destroyTensor(session s, tensor* t) {
    if (t->data.buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, t->data);
    if (t->scale.buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, t->scale);
    if (t->zero.buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, t->zero);
}

static void lname(char* buf, size_t cap, int L, const char* sub) {
    snprintf(buf, cap, "model.language_model.layers.%d.%s", L, sub);
}

static const sa_tensor* require(const safetensors* sf, const char* name) {
    const sa_tensor* t = sf ? safetensors_find(sf, name) : NULL;
    if (!t) {
        char msg[320];
        if (sf == NULL) {
            snprintf(msg, sizeof(msg), "cache miss for %s and safetensors not available", name);
        } else {
            snprintf(msg, sizeof(msg), "missing tensor %s", name);
        }
        fatal(msg);
    }
    return t;
}

#define VEC_FILE_MAGIC 0x56454353

static int findShards(const char* dir, char out[][512], int max);

static char g_weightDir[512];
static safetensors g_shards;
static int g_shardState = 0;
static int g_gguf = 0;

static const safetensors* shardSource(void) {
    if (g_shardState != 0) return g_shardState == 1 ? &g_shards : NULL;
    if (gguf_path_is_file(g_weightDir)) {
        gguf g;
        if (gguf_open(&g, g_weightDir) != 0) {
            g_shardState = -1;
            return NULL;
        }
        int rc = gguf_as_safetensors(&g, &g_shards);
        gguf_close(&g);
        if (rc != 0) {
            g_shardState = -1;
            return NULL;
        }
        g_shardState = 1;
        return &g_shards;
    }
    char shardPaths[SA_MAX_FILES][512];
    const char* shardPtrs[SA_MAX_FILES];
    int shardCount = findShards(g_weightDir, shardPaths, SA_MAX_FILES);
    if (shardCount == 0) {
        g_shardState = -1;
        return NULL;
    }
    for (int i = 0; i < shardCount; i++) shardPtrs[i] = shardPaths[i];
    if (safetensors_open(&g_shards, shardPtrs, shardCount) != 0) {
        g_shardState = -1;
        return NULL;
    }
    g_shardState = 1;
    return &g_shards;
}

static void shardSourceClose(void) {
    if (g_shardState == 1) safetensors_close(&g_shards);
    g_shardState = 0;
}

static float* loadVecRaw(const char* cacheName, int len) {
    char path[384];
    snprintf(path, sizeof(path), "%s/%s.bin", cacheDir, cacheName);
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    int header[2];
    if (fread(header, sizeof(int), 2, f) != 2 || header[0] != VEC_FILE_MAGIC || header[1] != len) {
        fclose(f);
        return NULL;
    }
    float* v = (float*)malloc(sizeof(float) * len);
    if (fread(v, sizeof(float), len, f) != (size_t)len) {
        free(v);
        fclose(f);
        return NULL;
    }
    fclose(f);
    return v;
}

static float* loadVecRawAny(const char* cacheName, int64_t* outLen) {
    char path[384];
    snprintf(path, sizeof(path), "%s/%s.bin", cacheDir, cacheName);
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    int header[2];
    if (fread(header, sizeof(int), 2, f) != 2 || header[0] != VEC_FILE_MAGIC || header[1] <= 0) {
        fclose(f);
        return NULL;
    }
    int len = header[1];
    float* v = (float*)malloc(sizeof(float) * len);
    if (fread(v, sizeof(float), len, f) != (size_t)len) {
        free(v);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *outLen = len;
    return v;
}

static void saveVecRaw(const char* cacheName, const float* v, int len) {
    char path[384];
    snprintf(path, sizeof(path), "%s/%s.bin", cacheDir, cacheName);
    FILE* f = fopen(path, "wb");
    if (!f) return;
    int header[2] = {VEC_FILE_MAGIC, len};
    fwrite(header, sizeof(int), 2, f);
    fwrite(v, sizeof(float), len, f);
    fclose(f);
}

static buffer loadVecBuffer(session s, const char* hfName, int len, const char* label, int layer, int addOne) {
    char cacheName[80];
    snprintf(cacheName, sizeof(cacheName), "vec_%s_%d", label, layer);
    float* v = loadVecRaw(cacheName, len);
    if (v == NULL) {
        const safetensors* sf = shardSource();
        const sa_tensor* t = require(sf, hfName);
        int64_t n = 0;
        v = safetensors_load_f32(sf, t, &n);
        if (!v || n != len) fatal("vector length mismatch");
        if (addOne && !g_gguf) {
            for (int i = 0; i < len; i++) v[i] += 1.0f;
        }
        if (g_gguf && strcmp(label, "aLog") == 0) {
            for (int i = 0; i < len; i++) v[i] = logf(-v[i]);
        }
        saveVecRaw(cacheName, v, len);
    }
    buffer b = createBufferNamed(s.dev.device, s.dev.physicalDevice, v, sizeof(float) * len, MEMORY_VRAM, label);
    countBuffer(label, layer, b);
    registerWeightBufferSmall(b);
    free(v);
    return b;
}

static float* buildEngineMatrix(const safetensors* sf, const char** hfNames, int hfCount, int engineRows, int* outCols) {
    int total = 0;
    int sizes[32];
    if (hfCount > 32) fatal("too many concat sources");
    for (int i = 0; i < hfCount; i++) {
        const sa_tensor* t = require(sf, hfNames[i]);
        if (t->ndim < 2) fatal("matrix tensor expected");
        sizes[i] = (int)t->shape[0];
        total += sizes[i];
    }
    float* hf = (float*)malloc(sizeof(float) * (size_t)total * engineRows);
    int off = 0;
    for (int i = 0; i < hfCount; i++) {
        const sa_tensor* t = safetensors_find(sf, hfNames[i]);
        int64_t n = 0;
        float* src = safetensors_load_f32(sf, t, &n);
        if (!src || n != (int64_t)sizes[i] * engineRows) {
            char buf[256];
            snprintf(buf, sizeof(buf), "matrix length mismatch: %s got=%lld want=%lld", hfNames[i], (long long)n, (long long)sizes[i] * engineRows);
            fatal(buf);
        }
        memcpy(hf + (size_t)off * engineRows, src, sizeof(float) * n);
        free(src);
        off += sizes[i];
    }
    float* eng = (float*)malloc(sizeof(float) * (size_t)total * engineRows);
    transpose(hf, eng, total, engineRows);
    free(hf);
    *outCols = total;
    return eng;
}

static float* buildQkvMatrix(const safetensors* sf, const char* qn, const char* kn, const char* vn, int engineRows, int headDim, int heads, int* outCols) {
    const sa_tensor* tq = require(sf, qn);
    const sa_tensor* tk = require(sf, kn);
    const sa_tensor* tv = require(sf, vn);
    int qRows = (int)tq->shape[0];
    int kRows = (int)tk->shape[0];
    int vRows = (int)tv->shape[0];
    int total = qRows + kRows + vRows;
    int hd = headDim;
    int qPart = heads * hd;

    float* hf = (float*)malloc(sizeof(float) * (size_t)total * engineRows);

    int64_t n = 0;
    float* src = safetensors_load_f32(sf, tq, &n);
    if (!src || n != (int64_t)qRows * engineRows) fatal("q proj length mismatch");
    for (int c = 0; c < qRows; c++) {
        int head, dim, srcRow;
        if (c < qPart) {
            head = c / hd;
            dim = c % hd;
            srcRow = head * (2 * hd) + dim;
        } else {
            head = (c - qPart) / hd;
            dim = (c - qPart) % hd;
            srcRow = head * (2 * hd) + hd + dim;
        }
        memcpy(hf + (size_t)c * engineRows, src + (size_t)srcRow * engineRows, sizeof(float) * engineRows);
    }
    free(src);

    float* sk = safetensors_load_f32(sf, tk, &n);
    if (!sk || n != (int64_t)kRows * engineRows) fatal("k proj length mismatch");
    memcpy(hf + (size_t)qRows * engineRows, sk, sizeof(float) * n);
    free(sk);

    float* sv = safetensors_load_f32(sf, tv, &n);
    if (!sv || n != (int64_t)vRows * engineRows) fatal("v proj length mismatch");
    memcpy(hf + (size_t)(qRows + kRows) * engineRows, sv, sizeof(float) * n);
    free(sv);

    float* eng = (float*)malloc(sizeof(float) * (size_t)total * engineRows);
    transpose(hf, eng, total, engineRows);
    free(hf);
    *outCols = total;
    return eng;
}

static void loadEmbedLike(session s, const char* const* candPaths, int candCount, const char* hfName, const char* name, int V, int K, buffer* out) {
    char path[384];
    snprintf(path, sizeof(path), "%s/%s_%d_FP16.bin", cacheDir, name, V);
    cachedTensor* ct = tensorLoadFile(path, name, QUANT_FP16, K, V, (V + 255) / 256);
    if (ct == NULL) {
        const sa_tensor* t = NULL;
        safetensors sfCand;
        int opened = 0;
        for (int i = 0; i < candCount && t == NULL; i++) {
            if (candPaths[i] == NULL) continue;
            const char* p = candPaths[i];
            if (safetensors_open(&sfCand, &p, 1) == 0) {
                t = safetensors_find(&sfCand, hfName);
                if (t != NULL) {
                    opened = 1;
                } else {
                    safetensors_close(&sfCand);
                }
            }
        }
        const safetensors* sfUse = opened ? &sfCand : shardSource();
        if (t == NULL && sfUse != NULL) t = safetensors_find(sfUse, hfName);
        if (t == NULL) t = require(sfUse, hfName);
        if (t->ndim < 2 || t->shape[0] != V || t->shape[1] != K) fatal("embedding shape mismatch");
        uint16_t* eng = (uint16_t*)malloc(sizeof(uint16_t) * (size_t)K * V);
        if (t->dtype == SA_DTYPE_BF16) {
            uint16_t* raw = (uint16_t*)malloc(sizeof(uint16_t) * (size_t)V * K);
            FILE* f = sfUse->files[t->fileIndex];
            _fseeki64(f, t->offset, SEEK_SET);
            if (fread(raw, sizeof(uint16_t), (size_t)V * K, f) != (size_t)V * K) fatal("embedding read error");
            for (int v = 0; v < V; v++) {
                for (int k = 0; k < K; k++) {
                    eng[(size_t)k * V + v] = float_to_fp16(bf16_to_float(raw[(size_t)v * K + k]));
                }
            }
            free(raw);
        } else {
            int64_t n = 0;
            float* src = safetensors_load_f32(sfUse, t, &n);
            if (!src || n != (int64_t)V * K) fatal("embedding read error");
            for (int v = 0; v < V; v++) {
                for (int k = 0; k < K; k++) {
                    eng[(size_t)k * V + v] = float_to_fp16(src[(size_t)v * K + k]);
                }
            }
            free(src);
        }
        uint16_t* tw = (uint16_t*)malloc(sizeof(uint16_t) * (size_t)K * V);
        transpose_block16((uint8_t*)eng, (uint8_t*)tw, K, V, QUANT_FP16);
        free(eng);
        ct = cacheStore(name, QUANT_FP16, K, V, (uint8_t*)tw, K * V * 2, NULL, NULL, 0);
        tensorWriteFile(path, QUANT_FP16, K, V, ct->data, ct->dataBytes, NULL, NULL, 0);
        if (opened) safetensors_close(&sfCand);
    }
    *out = createBufferNamed(s.dev.device, s.dev.physicalDevice, ct->data, ct->dataBytes, MEMORY_VRAM, name);
    countBuffer(name, -1, *out);
    registerWeightBuffer(out);
    cacheRelease(ct);
}

static buffer loadConv(session s, const char* name, int layer) {
    char cacheName[80];
    snprintf(cacheName, sizeof(cacheName), "conv_%d", layer);
    float* v;
    int64_t n = 0;
    v = loadVecRawAny(cacheName, &n);
    if (v == NULL) {
        const safetensors* sf = shardSource();
        const sa_tensor* t = require(sf, name);
        v = safetensors_load_f32(sf, t, &n);
        if (!v) fatal("conv read error");
        saveVecRaw(cacheName, v, (int)n);
    }
    buffer b = createBufferNamed(s.dev.device, s.dev.physicalDevice, v, sizeof(float) * n, MEMORY_VRAM, cacheName);
    countBuffer("conv", layer, b);
    registerWeightBufferSmall(b);
    free(v);
    return b;
}

static buffer loadRouterFp16(session s, const char* hfName, int K, int N, const char* cacheName, int layer) {
    char path[384];
    snprintf(path, sizeof(path), "%s/%s.bin", cacheDir, cacheName);
    int64_t bytes = (int64_t)K * N * 2;
    uint16_t* tw = NULL;

    FILE* f = fopen(path, "rb");
    if (f != NULL) {
        int header[4];
        if (fread(header, sizeof(int), 4, f) == 4 &&
            header[0] == TENSOR_FILE_MAGIC && header[1] == K && header[2] == N && header[3] == (int)QUANT_FP16) {
            tw = (uint16_t*)malloc((size_t)bytes);
            if (fread(tw, 1, (size_t)bytes, f) != (size_t)bytes) {
                free(tw);
                tw = NULL;
            }
        }
        fclose(f);
    }

    if (tw == NULL) {
        const safetensors* sf = shardSource();
        const sa_tensor* t = require(sf, hfName);
        if (t->ndim != 2 || t->shape[0] != N || t->shape[1] != K) fatal("router shape mismatch");
        int64_t n = 0;
        float* src = safetensors_load_f32(sf, t, &n);
        if (!src || n != (int64_t)N * K) fatal("router length mismatch");
        uint16_t* eng = (uint16_t*)malloc((size_t)bytes);
        for (int e = 0; e < N; e++) {
            for (int k = 0; k < K; k++) {
                eng[(size_t)k * N + e] = float_to_fp16(src[(size_t)e * K + k]);
            }
        }
        free(src);
        tw = (uint16_t*)malloc((size_t)bytes);
        transpose_block16((uint8_t*)eng, (uint8_t*)tw, K, N, QUANT_FP16);
        free(eng);
        tensorWriteFile(path, QUANT_FP16, K, N, (uint8_t*)tw, (int)bytes, NULL, NULL, 0);
    }

    buffer b = createBufferNamed(s.dev.device, s.dev.physicalDevice, tw, bytes, MEMORY_VRAM, cacheName);
    countBuffer(cacheName, layer, b);
    registerWeightBufferSmall(b);
    free(tw);
    return b;
}

typedef struct expert_pool_build {
    cachedTensor* ct;
    char name[64];
    int rows;
    int cols;
    int experts;
} expert_pool_build;

static void expertPoolSplit(session s, expert_pool* p, const expert_pool_build* b, int vramExperts, int layer, const char* label) {
    int rows = b->rows;
    int cols = b->cols;
    int experts = b->experts;
    if (vramExperts > experts - 1) vramExperts = experts - 1;
    if (vramExperts < 1) vramExperts = 1;
    int vramCount = vramExperts + 1;
    int ramCount = experts - 1 - vramExperts;
    int64_t dataStride = (int64_t)rows * cols / 2;
    int64_t scaleStride = (int64_t)sizeof(float) * rows * (cols / 256);

    uint8_t* vramData;
    float* vramScale;
    float* vramZero;
    if (ramCount == 0) {
        vramData = b->ct->data;
        vramScale = b->ct->scale;
        vramZero = b->ct->zero;
    } else {
        vramData = (uint8_t*)malloc((size_t)(dataStride * vramCount));
        vramScale = (float*)malloc((size_t)(scaleStride * vramCount));
        vramZero = (float*)malloc((size_t)(scaleStride * vramCount));
        memcpy(vramData, b->ct->data, (size_t)(dataStride * vramExperts));
        memcpy(vramData + dataStride * vramExperts, (uint8_t*)b->ct->data + dataStride * (experts - 1), (size_t)dataStride);
        memcpy(vramScale, b->ct->scale, (size_t)(scaleStride * vramExperts));
        memcpy((uint8_t*)vramScale + scaleStride * vramExperts, (uint8_t*)b->ct->scale + scaleStride * (experts - 1), (size_t)scaleStride);
        memcpy(vramZero, b->ct->zero, (size_t)(scaleStride * vramExperts));
        memcpy((uint8_t*)vramZero + scaleStride * vramExperts, (uint8_t*)b->ct->zero + scaleStride * (experts - 1), (size_t)scaleStride);
    }

    char name[80];
    snprintf(name, sizeof(name), "%s%d-vram", label, layer);
    p->vramData = createBufferNamed(s.dev.device, s.dev.physicalDevice, vramData, dataStride * vramCount, MEMORY_VRAM, name);
    snprintf(name, sizeof(name), "%s%d-vram-scale", label, layer);
    p->vramScale = createBufferNamed(s.dev.device, s.dev.physicalDevice, vramScale, scaleStride * vramCount, MEMORY_VRAM, name);
    snprintf(name, sizeof(name), "%s%d-vram-zero", label, layer);
    p->vramZero = createBufferNamed(s.dev.device, s.dev.physicalDevice, vramZero, scaleStride * vramCount, MEMORY_VRAM, name);
    registerWeightBuffer(&p->vramData);
    registerWeightBuffer(&p->vramScale);
    registerWeightBuffer(&p->vramZero);

    if (ramCount > 0) {
        snprintf(name, sizeof(name), "%s%d-ram", label, layer);
        p->ramData = createBufferNamed(s.dev.device, s.dev.physicalDevice,
                                       (uint8_t*)b->ct->data + dataStride * vramExperts,
                                       dataStride * ramCount, MEMORY_RAM, name);
        snprintf(name, sizeof(name), "%s%d-ram-scale", label, layer);
        p->ramScale = createBufferNamed(s.dev.device, s.dev.physicalDevice,
                                        (uint8_t*)b->ct->scale + scaleStride * vramExperts,
                                        scaleStride * ramCount, MEMORY_RAM, name);
        snprintf(name, sizeof(name), "%s%d-ram-zero", label, layer);
        p->ramZero = createBufferNamed(s.dev.device, s.dev.physicalDevice,
                                       (uint8_t*)b->ct->zero + scaleStride * vramExperts,
                                       scaleStride * ramCount, MEMORY_RAM, name);
    }

    if (vramData != b->ct->data) {
        free(vramData);
        free(vramScale);
        free(vramZero);
    }

    p->vramExperts = vramCount;
    p->ramBase = vramCount;
    p->expertCount = experts;
    countBuffer(label, layer, p->vramData);
}

static void destroyExpertPool(session s, expert_pool* p) {
    destroyBuffer(s.dev.device, p->vramData);
    destroyBuffer(s.dev.device, p->vramScale);
    destroyBuffer(s.dev.device, p->vramZero);
    if (p->ramData.buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, p->ramData);
    if (p->ramScale.buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, p->ramScale);
    if (p->ramZero.buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, p->ramZero);
}

static cachedTensor* expertPoolLoadFile(const char* cacheName, QuantType q, int rows, int cols, int experts) {
    char path[384];
    snprintf(path, sizeof(path), "%s/%s_%s.bin", cacheDir, cacheName, quantSuffix(q));
    int blocks = (cols + 255) / 256;
    int scaleCount = rows * blocks;
    int64_t dataBytes = (int64_t)rows * cols / 2;
    int64_t scaleBytes = (int64_t)sizeof(float) * scaleCount * experts;
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    int header[5];
    int ok = (fread(header, sizeof(int), 5, f) == 5 &&
              header[0] == TENSOR_FILE_MAGIC && header[1] == rows && header[2] == cols &&
              header[3] == (int)q && header[4] == experts);
    if (!ok) {
        fclose(f);
        return NULL;
    }
    uint8_t* data = (uint8_t*)malloc((size_t)(dataBytes * experts));
    float* scale = (float*)malloc((size_t)scaleBytes);
    float* zero = (float*)malloc((size_t)scaleBytes);
    if (fread(data, 1, (size_t)(dataBytes * experts), f) != (size_t)(dataBytes * experts) ||
        fread(scale, 1, (size_t)scaleBytes, f) != (size_t)scaleBytes ||
        fread(zero, 1, (size_t)scaleBytes, f) != (size_t)scaleBytes) {
        free(data);
        free(scale);
        free(zero);
        fclose(f);
        return NULL;
    }
    fclose(f);
    return cacheStore(cacheName, q, rows, cols, data, (int)(dataBytes * experts), scale, zero, scaleCount * experts);
}

static void expertPoolWriteFile(const char* cacheName, QuantType q, int rows, int cols, int experts, const uint8_t* data, const float* scale, const float* zero, int scaleCount) {
    char path[384];
    snprintf(path, sizeof(path), "%s/%s_%s.bin", cacheDir, cacheName, quantSuffix(q));
    FILE* f = fopen(path, "wb");
    if (!f) return;
    int64_t dataBytes = (int64_t)rows * cols / 2 * experts;
    int64_t scaleBytes = (int64_t)sizeof(float) * scaleCount;
    int header[5] = {TENSOR_FILE_MAGIC, rows, cols, (int)q, experts};
    fwrite(header, sizeof(int), 5, f);
    fwrite(data, 1, (size_t)dataBytes, f);
    fwrite(scale, 1, (size_t)scaleBytes, f);
    fwrite(zero, 1, (size_t)scaleBytes, f);
    fclose(f);
}

static void expertPoolBuildLayer(const safetensors* sf, const char* hfName, int rows, int cols, int experts,
                                 const int* hfSrcRows, int hfSrcCount, const char* sharedName, const char* sharedUpName,
                                 const char* cacheName, expert_pool_build* out) {
    int blocks = (cols + 255) / 256;
    int scaleCount = rows * blocks;
    int64_t dataBytes = (int64_t)rows * cols / 2;
    uint8_t* poolData = (uint8_t*)malloc((size_t)(dataBytes * experts));
    float* poolScale = (float*)malloc(sizeof(float) * (size_t)scaleCount * experts);
    float* poolZero = (float*)malloc(sizeof(float) * (size_t)scaleCount * experts);

    for (int e = 0; e < experts; e++) {
        float* eng = NULL;
        if (e < hfSrcCount) {
            const sa_tensor* t = require(sf, hfName);
            if (t->ndim != 3 || t->shape[0] != hfSrcCount || t->shape[1] != cols || t->shape[2] != rows) fatal("expert tensor shape mismatch");
            FILE* f = sf->files[t->fileIndex];
            _fseeki64(f, t->offset + (int64_t)hfSrcRows[e] * cols * rows * 2, SEEK_SET);
            int64_t n = (int64_t)cols * rows;
            uint16_t* raw = (uint16_t*)malloc((size_t)n * 2);
            if (fread(raw, 2, (size_t)n, f) != (size_t)n) fatal("expert read error");
            eng = (float*)malloc(sizeof(float) * (size_t)n);
            for (int64_t i = 0; i < n; i++) eng[i] = bf16_to_float(raw[i]);
            free(raw);
        } else if (sharedUpName != NULL) {
            const sa_tensor* tg = require(sf, sharedName);
            const sa_tensor* tu = require(sf, sharedUpName);
            if (tg->ndim != 2 || tg->shape[0] != cols / 2 || tg->shape[1] != rows) fatal("shared expert gate shape mismatch");
            if (tu->ndim != 2 || tu->shape[0] != cols / 2 || tu->shape[1] != rows) fatal("shared expert up shape mismatch");
            int64_t half = (int64_t)(cols / 2) * rows;
            int64_t got = 0;
            float* gate = safetensors_load_f32(sf, tg, &got);
            if (!gate || got != half) fatal("shared expert gate length mismatch");
            float* up = safetensors_load_f32(sf, tu, &got);
            if (!up || got != half) fatal("shared expert up length mismatch");
            eng = (float*)malloc(sizeof(float) * (size_t)cols * rows);
            memcpy(eng, gate, sizeof(float) * (size_t)half);
            memcpy(eng + half, up, sizeof(float) * (size_t)half);
            free(gate);
            free(up);
        } else {
            const sa_tensor* t = require(sf, sharedName);
            if (t->ndim != 2 || t->shape[0] != cols || t->shape[1] != rows) fatal("shared expert shape mismatch");
            int64_t got = 0;
            eng = safetensors_load_f32(sf, t, &got);
            if (!eng || got != (int64_t)cols * rows) fatal("shared expert length mismatch");
        }
        float* hf = eng;
        eng = (float*)malloc(sizeof(float) * (size_t)rows * cols);
        transpose(hf, eng, cols, rows);
        free(hf);
        QuantizedData qd = quantizeDataINT4(eng, rows, cols);
        free(eng);
        transpose_block16(qd.data, poolData + (size_t)e * dataBytes, rows, cols, QUANT_INT4);
        free(qd.data);
        memcpy(poolScale + (size_t)e * scaleCount, qd.scale, sizeof(float) * scaleCount);
        memcpy(poolZero + (size_t)e * scaleCount, qd.z, sizeof(float) * scaleCount);
        free(qd.scale);
        free(qd.z);
    }

    expertPoolWriteFile(cacheName, QUANT_INT4, rows, cols, experts, poolData, poolScale, poolZero, scaleCount * experts);
    out->ct = cacheStore(cacheName, QUANT_INT4, rows, cols, poolData, (int)(dataBytes * experts), poolScale, poolZero, scaleCount * experts);
    snprintf(out->name, sizeof(out->name), "%s", cacheName);
    out->rows = rows;
    out->cols = cols;
    out->experts = experts;
}

static int findShards(const char* dir, char out[][512], int max) {
    char pattern[512];
    snprintf(pattern, sizeof(pattern), "%s/model*.safetensors", dir);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE) return 0;
    int count = 0;
    do {
        if (count >= max) break;
        snprintf(out[count], 512, "%s/%s", dir, fd.cFileName);
        count++;
    } while (FindNextFileA(h, &fd));
    FindClose(h);

    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (strcmp(out[i], out[j]) > 0) {
                char tmp[512];
                strcpy(tmp, out[i]);
                strcpy(out[i], out[j]);
                strcpy(out[j], tmp);
            }
        }
    }
    return count;
}

static int findVocabFile(const char* dir, const char* prefix, int V, char* out, size_t cap) {
    char pattern[512];
    snprintf(pattern, sizeof(pattern), "%s/vocab/%s.%d.safetensors", dir, prefix, V);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE) return 0;
    snprintf(out, cap, "%s/vocab/%s", dir, fd.cFileName);
    FindClose(h);
    return 1;
}

static int cacheFileExists(const char* name, QuantType q, int rows, int cols) {
    char path[384];
    snprintf(path, sizeof(path), "%s/%s_%s.bin", cacheDir, name, quantSuffix(q));
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    int header[4];
    int ok = (fread(header, sizeof(int), 4, f) == 4 &&
              header[0] == TENSOR_FILE_MAGIC && header[1] == rows && header[2] == cols && header[3] == (int)q);
    fclose(f);
    return ok;
}

static int vecCacheExists(const char* cacheName, int len) {
    char path[384];
    snprintf(path, sizeof(path), "%s/%s.bin", cacheDir, cacheName);
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    int header[2];
    int ok = (fread(header, sizeof(int), 2, f) == 2 &&
              header[0] == VEC_FILE_MAGIC && header[1] == len);
    fclose(f);
    return ok;
}

static int expertPoolCacheExists(const char* cacheName, int rows, int cols, int experts) {
    char path[384];
    snprintf(path, sizeof(path), "%s/%s_INT4.bin", cacheDir, cacheName);
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    int header[5];
    int ok = (fread(header, sizeof(int), 5, f) == 5 &&
              header[0] == TENSOR_FILE_MAGIC && header[1] == rows && header[2] == cols &&
              header[3] == (int)QUANT_INT4 && header[4] == experts);
    fclose(f);
    return ok;
}

static int cacheComplete(const model_config* spec) {
    const model_dims* d = &spec->dims;
    char name[80];
    if (!vecCacheExists("vec_gammaFinal_-1", d->K)) return 0;
    snprintf(name, sizeof(name), "embed_%d", d->vocab);
    if (!cacheFileExists(name, QUANT_FP16, d->K, d->vocab)) return 0;
    if (!d->tied) {
        snprintf(name, sizeof(name), "lmHead_%d", d->vocab);
        if (!cacheFileExists(name, QUANT_FP16, d->K, d->vocab)) return 0;
    }
    for (int L = 0; L < d->layerCount; L++) {
        const layer* ly = &spec->layers[L];
        QuantType q = ly->attn.q;
        QuantType f = ly->ffn.q;
        char vecName[80];
        snprintf(vecName, sizeof(vecName), "vec_gammaIn_%d", L);
        if (!vecCacheExists(vecName, d->K)) return 0;
        snprintf(vecName, sizeof(vecName), "vec_gammaF_%d", L);
        if (!vecCacheExists(vecName, d->K)) return 0;
        if (ly->attn.type == ATTENTION_FULL) {
            snprintf(name, sizeof(name), "proj_%d", L);
            if (!cacheFileExists(name, q, d->K, d->qkvN)) return 0;
            snprintf(vecName, sizeof(vecName), "vec_qNorm_%d", L);
            if (!vecCacheExists(vecName, d->headDim)) return 0;
            snprintf(vecName, sizeof(vecName), "vec_kNorm_%d", L);
            if (!vecCacheExists(vecName, d->headDim)) return 0;
        } else {
            snprintf(name, sizeof(name), "proj_%d", L);
            if (!cacheFileExists(name, q, d->K, d->projN)) return 0;
            snprintf(vecName, sizeof(vecName), "conv_%d", L);
            if (!vecCacheExists(vecName, 4 * d->zqkvN)) return 0;
            snprintf(vecName, sizeof(vecName), "vec_aLog_%d", L);
            if (!vecCacheExists(vecName, d->nV)) return 0;
            snprintf(vecName, sizeof(vecName), "vec_dtBias_%d", L);
            if (!vecCacheExists(vecName, d->nV)) return 0;
            snprintf(vecName, sizeof(vecName), "vec_attnNorm_%d", L);
            if (!vecCacheExists(vecName, d->dim)) return 0;
        }
        snprintf(name, sizeof(name), "out_%d", L);
        if (ly->attn.type == ATTENTION_FULL) {
            if (!cacheFileExists(name, q, d->qOff, d->K)) return 0;
        } else {
            if (!cacheFileExists(name, q, d->nV * d->dim, d->K)) return 0;
        }
        if (ly->ffn.type == FFN_SWIGLU) {
            snprintf(name, sizeof(name), "gate_%d", L);
            if (!cacheFileExists(name, f, d->K, d->ffnN)) return 0;
            snprintf(name, sizeof(name), "up_%d", L);
            if (!cacheFileExists(name, f, d->K, d->ffnN)) return 0;
            snprintf(name, sizeof(name), "down_%d", L);
            if (!cacheFileExists(name, f, d->ffnN, d->K)) return 0;
        } else if (ly->ffn.type == FFN_MOE) {
            snprintf(name, sizeof(name), "guPool_%d", L);
            if (!expertPoolCacheExists(name, d->K, 2 * d->moeI, d->experts + 1)) return 0;
            snprintf(name, sizeof(name), "dnPool_%d", L);
            if (!expertPoolCacheExists(name, d->moeI, d->K, d->experts + 1)) return 0;
            snprintf(name, sizeof(name), "router_%d", L);
            if (!cacheFileExists(name, QUANT_FP16, d->K, d->experts)) return 0;
            snprintf(name, sizeof(name), "sgGate_%d", L);
            if (!cacheFileExists(name, QUANT_FP16, d->K, 1)) return 0;
        }
    }
    return 1;
}

model_weights createWeights(session s, const model_config* spec, const char* weightDir, int verbose) {
    model_weights w = {0};
    const model_dims* d = &spec->dims;
    weightBytes = 0;
    verboseWeights = verbose;
    g_wbufsCount = 0;
    g_wbufSession = s;
    cacheClear();
    snprintf(cacheDir, sizeof(cacheDir), "weights/%s", spec->name);
    snprintf(g_weightDir, sizeof(g_weightDir), "%s", weightDir);
    g_gguf = gguf_path_is_file(weightDir);
    char modelDir[512];
    snprintf(modelDir, sizeof(modelDir), "%s", weightDir);
    if (gguf_path_is_file(weightDir)) gguf_dir_of(weightDir, modelDir, sizeof(modelDir));
    _mkdir("weights");
    _mkdir(cacheDir);

    char shardProbe[SA_MAX_FILES][512];
    int shardCount = findShards(weightDir, shardProbe, SA_MAX_FILES);
    if (!gguf_path_is_file(weightDir) && shardCount == 0 && !cacheComplete(spec)) {
        fatal("no safetensors found and weight cache is incomplete");
    }

    char headPath[512];
    char embedPath[512];
    int hasHead = 0;
    int hasEmbed = 0;
    headPath[0] = '\0';
    embedPath[0] = '\0';

    int rotaryHalf = d->rotaryDim / 2;
    float* theta = (float*)malloc(sizeof(float) * rotaryHalf);
    for (int i = 0; i < rotaryHalf; i++) {
        theta[i] = (float)pow(d->ropeTheta, -((double)i) / rotaryHalf);
    }
    w.theta = createBufferNamed(s.dev.device, s.dev.physicalDevice, theta, sizeof(float) * rotaryHalf, MEMORY_VRAM, "theta");
    countBuffer("theta", -1, w.theta);
    registerWeightBuffer(&w.theta);
    free(theta);

    w.gammaFinal = loadVecBuffer(s, "model.language_model.norm.weight", d->K, "gammaFinal", -1, 1);

    int V = d->vocab;
    w.vocab = V;
    w.layerCount = d->layerCount;

    hasHead = findVocabFile(modelDir, "lm_head", V, headPath, sizeof(headPath));
    hasEmbed = findVocabFile(modelDir, "embed_tokens", V, embedPath, sizeof(embedPath));
    if (d->tied) hasHead = 0;
    if (d->tied && !hasEmbed) {
        hasEmbed = findVocabFile(modelDir, "lm_head", V, embedPath, sizeof(embedPath));
    }

    w.layerBufs = (buffer*)calloc((size_t)d->layerCount * 13, sizeof(buffer));
    w.gammaIn = w.layerBufs + 0 * d->layerCount;
    w.gammaF = w.layerBufs + 1 * d->layerCount;
    w.qNorm = w.layerBufs + 2 * d->layerCount;
    w.kNorm = w.layerBufs + 3 * d->layerCount;
    w.conv = w.layerBufs + 4 * d->layerCount;
    w.aLog = w.layerBufs + 5 * d->layerCount;
    w.dtBias = w.layerBufs + 6 * d->layerCount;
    w.attnNorm = w.layerBufs + 7 * d->layerCount;
    w.tensorBufs = (tensor*)calloc((size_t)d->layerCount * 5, sizeof(tensor));
    w.proj = w.tensorBufs + 0 * d->layerCount;
    w.out = w.tensorBufs + 1 * d->layerCount;
    w.gate = w.tensorBufs + 2 * d->layerCount;
    w.up = w.tensorBufs + 3 * d->layerCount;
    w.down = w.tensorBufs + 4 * d->layerCount;

    int isMoe = 0;
    for (int L = 0; L < d->layerCount; L++) {
        if (spec->layers[L].ffn.type == FFN_MOE) {
            isMoe = 1;
            break;
        }
    }
    if (isMoe) {
        w.poolBufs = (expert_pool*)calloc((size_t)d->layerCount * 2, sizeof(expert_pool));
        w.guPool = w.poolBufs + 0 * d->layerCount;
        w.dnPool = w.poolBufs + 1 * d->layerCount;
        w.router = (buffer*)calloc((size_t)d->layerCount, sizeof(buffer));
        w.sharedGate = (buffer*)calloc((size_t)d->layerCount, sizeof(buffer));
    }

    const char* embedCands[2];
    int embedCandCount = 0;
    if (hasEmbed) embedCands[embedCandCount++] = embedPath;

    const char* headCands[2];
    int headCandCount = 0;
    if (hasHead) headCands[headCandCount++] = headPath;
    if (hasEmbed) headCands[headCandCount++] = embedPath;

    loadEmbedLike(s, embedCands, embedCandCount, "model.language_model.embed_tokens.weight", "embed", V, d->K, &w.embed);
    if (d->tied) {
        w.lmHead = w.embed;
    } else {
        loadEmbedLike(s, headCands, headCandCount, "lm_head.weight", "lmHead", V, d->K, &w.lmHead);
    }

    char n1[256], n2[256], n3[256], n4[256];

    for (int L = 0; L < spec->dims.layerCount; L++) {
        const layer* ly = &spec->layers[L];
        QuantType q = ly->attn.q;
        QuantType f = ly->ffn.q;

        lname(n1, sizeof(n1), L, "input_layernorm.weight");
        w.gammaIn[L] = loadVecBuffer(s, n1, d->K, "gammaIn", L, 1);
        lname(n1, sizeof(n1), L, "post_attention_layernorm.weight");
        w.gammaF[L] = loadVecBuffer(s, n1, d->K, "gammaF", L, 1);

        char projName[64], outName[64];
        snprintf(projName, sizeof(projName), "proj_%d", L);
        snprintf(outName, sizeof(outName), "out_%d", L);

        if (ly->attn.type == ATTENTION_FULL) {
            lname(n1, sizeof(n1), L, "self_attn.q_norm.weight");
            w.qNorm[L] = loadVecBuffer(s, n1, d->headDim, "qNorm", L, 1);
            lname(n1, sizeof(n1), L, "self_attn.k_norm.weight");
            w.kNorm[L] = loadVecBuffer(s, n1, d->headDim, "kNorm", L, 1);

            lname(n1, sizeof(n1), L, "self_attn.q_proj.weight");
            lname(n2, sizeof(n2), L, "self_attn.k_proj.weight");
            lname(n3, sizeof(n3), L, "self_attn.v_proj.weight");
            int cols = 0;
            float* mat = NULL;
            if (cacheGet(projName, q, d->K, d->qkvN) == NULL) {
                mat = buildQkvMatrix(shardSource(), n1, n2, n3, d->K, d->headDim, d->heads, &cols);
                if (cols != d->qkvN) fatal("qkv projection width mismatch");
            }
            loadTensorInto(s, &w.proj[L], projName, L, d->K, d->qkvN, q, 1.0f, mat);
            free(mat);

            lname(n1, sizeof(n1), L, "self_attn.o_proj.weight");
            const char* on[1] = {n1};
            mat = NULL;
            if (cacheGet(outName, q, d->qOff, d->K) == NULL) {
                mat = buildEngineMatrix(shardSource(), on, 1, d->qOff, &cols);
                if (cols != d->K) fatal("o_proj width mismatch");
            }
            loadTensorInto(s, &w.out[L], outName, L, d->qOff, d->K, q, 1.0f, mat);
            free(mat);
        } else {
            lname(n1, sizeof(n1), L, "linear_attn.conv1d.weight");
            w.conv[L] = loadConv(s, n1, L);
            lname(n1, sizeof(n1), L, "linear_attn.A_log");
            w.aLog[L] = loadVecBuffer(s, n1, d->nV, "aLog", L, 0);
            lname(n1, sizeof(n1), L, "linear_attn.dt_bias");
            w.dtBias[L] = loadVecBuffer(s, n1, d->nV, "dtBias", L, 0);
            lname(n1, sizeof(n1), L, "linear_attn.norm.weight");
            w.attnNorm[L] = loadVecBuffer(s, n1, d->dim, "attnNorm", L, 0);

            lname(n1, sizeof(n1), L, "linear_attn.in_proj_qkv.weight");
            lname(n2, sizeof(n2), L, "linear_attn.in_proj_z.weight");
            lname(n3, sizeof(n3), L, "linear_attn.in_proj_a.weight");
            lname(n4, sizeof(n4), L, "linear_attn.in_proj_b.weight");
            const char* pn[4] = {n1, n2, n3, n4};
            int cols = 0;
            float* mat = NULL;
            if (cacheGet(projName, q, d->K, d->projN) == NULL) {
                mat = buildEngineMatrix(shardSource(), pn, 4, d->K, &cols);
                if (cols != d->projN) fatal("delta projection width mismatch");
            }
            loadTensorInto(s, &w.proj[L], projName, L, d->K, d->projN, q, 1.0f, mat);
            free(mat);

            lname(n1, sizeof(n1), L, "linear_attn.out_proj.weight");
            const char* on[1] = {n1};
            mat = NULL;
            int deltaOutRows = d->nV * d->dim;
            if (cacheGet(outName, q, deltaOutRows, d->K) == NULL) {
                mat = buildEngineMatrix(shardSource(), on, 1, deltaOutRows, &cols);
                if (cols != d->K) fatal("out_proj width mismatch");
            }
            loadTensorInto(s, &w.out[L], outName, L, deltaOutRows, d->K, q, 1.0f, mat);
            free(mat);
        }

        if (ly->ffn.type == FFN_SWIGLU) {
            char gateName[64], upName[64], downName[64];
            snprintf(gateName, sizeof(gateName), "gate_%d", L);
            snprintf(upName, sizeof(upName), "up_%d", L);
            snprintf(downName, sizeof(downName), "down_%d", L);

            const char* gn[1] = {n1};
            int cols = 0;
            lname(n1, sizeof(n1), L, "mlp.gate_proj.weight");
            float* mat = NULL;
            if (cacheGet(gateName, f, d->K, d->ffnN) == NULL) {
                mat = buildEngineMatrix(shardSource(), gn, 1, d->K, &cols);
                if (cols != d->ffnN) fatal("gate width mismatch");
            }
            loadTensorInto(s, &w.gate[L], gateName, L, d->K, d->ffnN, f, 1.0f, mat);
            free(mat);

            lname(n1, sizeof(n1), L, "mlp.up_proj.weight");
            mat = NULL;
            if (cacheGet(upName, f, d->K, d->ffnN) == NULL) {
                mat = buildEngineMatrix(shardSource(), gn, 1, d->K, &cols);
            }
            loadTensorInto(s, &w.up[L], upName, L, d->K, d->ffnN, f, 1.0f, mat);
            free(mat);

            lname(n1, sizeof(n1), L, "mlp.down_proj.weight");
            mat = NULL;
            if (cacheGet(downName, f, d->ffnN, d->K) == NULL) {
                mat = buildEngineMatrix(shardSource(), gn, 1, d->ffnN, &cols);
                if (cols != d->K) fatal("down projection width mismatch");
            }
            loadTensorInto(s, &w.down[L], downName, L, d->ffnN, d->K, f, 1.0f, mat);
            free(mat);
        } else if (ly->ffn.type == FFN_MOE) {
            char guName[64], dnName[64], rtName[64], sgName[64];
            snprintf(guName, sizeof(guName), "guPool_%d", L);
            snprintf(dnName, sizeof(dnName), "dnPool_%d", L);
            snprintf(rtName, sizeof(rtName), "router_%d", L);
            snprintf(sgName, sizeof(sgName), "sgGate_%d", L);

            int poolExperts = d->experts + 1;
            int* srcRows = (int*)malloc(sizeof(int) * d->experts);
            for (int e = 0; e < d->experts; e++) srcRows[e] = e;

            expert_pool_build gu, dn;
            gu.ct = expertPoolLoadFile(guName, QUANT_INT4, d->K, 2 * d->moeI, poolExperts);
            if (gu.ct == NULL) {
                lname(n1, sizeof(n1), L, "mlp.experts.gate_up_proj");
                lname(n2, sizeof(n2), L, "mlp.shared_expert.gate_proj.weight");
                lname(n3, sizeof(n3), L, "mlp.shared_expert.up_proj.weight");
                expertPoolBuildLayer(shardSource(), n1, d->K, 2 * d->moeI, poolExperts, srcRows, d->experts, n2, n3, guName, &gu);
            } else {
                gu.rows = d->K;
                gu.cols = 2 * d->moeI;
                gu.experts = poolExperts;
            }

            dn.ct = expertPoolLoadFile(dnName, QUANT_INT4, d->moeI, d->K, poolExperts);
            if (dn.ct == NULL) {
                lname(n1, sizeof(n1), L, "mlp.experts.down_proj");
                lname(n2, sizeof(n2), L, "mlp.shared_expert.down_proj.weight");
                expertPoolBuildLayer(shardSource(), n1, d->moeI, d->K, poolExperts, srcRows, d->experts, n2, NULL, dnName, &dn);
            } else {
                dn.rows = d->moeI;
                dn.cols = d->K;
                dn.experts = poolExperts;
            }
            free(srcRows);

            expertPoolSplit(s, &w.guPool[L], &gu, spec->expertsVram, L, "guPool_");
            expertPoolSplit(s, &w.dnPool[L], &dn, spec->expertsVram, L, "dnPool_");
            cacheRelease(gu.ct);
            cacheRelease(dn.ct);

            lname(n1, sizeof(n1), L, "mlp.gate.weight");
            w.router[L] = loadRouterFp16(s, n1, d->K, d->experts, rtName, L);
            lname(n1, sizeof(n1), L, "mlp.shared_expert_gate.weight");
            w.sharedGate[L] = loadRouterFp16(s, n1, d->K, 1, sgName, L);
        }
    }

    if (g_shardState == 0) fprintf(stderr, "weights: resolved fully from cache\n");
    shardSourceClose();
    cacheClear();

    weightFlush();

    if (!verboseWeights) {
        fprintf(stderr, "\r[OK] loaded weights: %.2f MB             \n",
                (double)weightBytes / (1024.0 * 1024.0));
    }
    fprintf(stderr, "total weights: %lld bytes (%.2f MB, %.2f GB)\n",
            (long long)weightBytes,
            (double)weightBytes / (1024.0 * 1024.0),
            (double)weightBytes / (1024.0 * 1024.0 * 1024.0));

    return w;
}

void destroyWeights(session s, model_weights* w) {
    for (int L = 0; L < w->layerCount; L++) {
        destroyTensor(s, &w->proj[L]);
        destroyTensor(s, &w->out[L]);
        destroyTensor(s, &w->gate[L]);
        destroyTensor(s, &w->up[L]);
        destroyTensor(s, &w->down[L]);
        destroyBuffer(s.dev.device, w->gammaIn[L]);
        destroyBuffer(s.dev.device, w->gammaF[L]);
        if (w->qNorm[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, w->qNorm[L]);
        if (w->kNorm[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, w->kNorm[L]);
        if (w->conv[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, w->conv[L]);
        if (w->aLog[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, w->aLog[L]);
        if (w->dtBias[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, w->dtBias[L]);
        if (w->attnNorm[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, w->attnNorm[L]);
    }
    if (w->poolBufs != NULL) {
        for (int L = 0; L < w->layerCount; L++) {
            destroyExpertPool(s, &w->guPool[L]);
            destroyExpertPool(s, &w->dnPool[L]);
            destroyBuffer(s.dev.device, w->router[L]);
            destroyBuffer(s.dev.device, w->sharedGate[L]);
        }
        free(w->poolBufs);
        free(w->router);
        free(w->sharedGate);
    }
    free(w->layerBufs);
    free(w->tensorBufs);
    destroyBuffer(s.dev.device, w->theta);
    destroyBuffer(s.dev.device, w->gammaFinal);
    destroyBuffer(s.dev.device, w->lmHead);
    if (w->lmHead.buffer != w->embed.buffer) {
        destroyBuffer(s.dev.device, w->embed);
    }
}

