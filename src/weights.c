#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <direct.h>
#include <windows.h>
#include "weights.h"
#include "safetensors.h"
#include "gguf.h"
#include "hqm.h"

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

typedef struct {
    char name[64];
    QuantType q;
    int rows;
    int cols;
    uint8_t* data;
    int dataBytes;
    void* scale;
    void* zero;
    int scaleCount;
} cachedTensor;

static cachedTensor tensorCache[TENSOR_CACHE_MAX];
static int tensorCacheCount = 0;

static hqm g_hqm;
static int g_hqmOpen = 0;
static hqm_writer* g_hqmWriter = NULL;
static int g_export = 1;

void weightsSetExport(int enabled) {
    g_export = enabled;
}

static void fatal(const char* msg) {
    fprintf(stderr, "weight load: %s\n", msg);
    exit(1);
}

static void permuteValueHeads(float* data, int64_t headStride, int nQk, int group) {
    if (group <= 1) return;
    int nHeads = nQk * group;
    size_t bytes = sizeof(float) * (size_t)headStride;
    float* tmp = (float*)malloc(bytes * (size_t)nHeads);
    if (tmp == NULL) return;
    memcpy(tmp, data, bytes * (size_t)nHeads);
    for (int k = 0; k < nQk; k++) {
        for (int g = 0; g < group; g++) {
            memcpy(data + (size_t)(k * group + g) * (size_t)headStride,
                   tmp + (size_t)(g * nQk + k) * (size_t)headStride, bytes);
        }
    }
    free(tmp);
}

static void permuteValueHeadColumns(float* mat, int rows, int cols, int colStart, int headDim, int nQk, int group) {
    if (group <= 1) return;
    int nHeads = nQk * group;
    size_t span = (size_t)nHeads * (size_t)headDim;
    float* tmp = (float*)malloc(sizeof(float) * span);
    if (tmp == NULL) return;
    for (int r = 0; r < rows; r++) {
        float* row = mat + (size_t)r * (size_t)cols + (size_t)colStart;
        memcpy(tmp, row, sizeof(float) * span);
        for (int k = 0; k < nQk; k++) {
            for (int g = 0; g < group; g++) {
                memcpy(row + (size_t)(k * group + g) * (size_t)headDim,
                       tmp + (size_t)(g * nQk + k) * (size_t)headDim,
                       sizeof(float) * (size_t)headDim);
            }
        }
    }
    free(tmp);
}

#define MAX_WEIGHT_BUFS 2560
#define WEIGHT_FLUSH_BATCH 12

static session g_wbufSession;
static buffer* g_wbufs[MAX_WEIGHT_BUFS];
static int g_wbufsCount = 0;
static buffer* g_wbufSmall[MAX_WEIGHT_BUFS];
static int g_wbufSmallCount = 0;

static void weightFlush(void) {
    if (g_wbufsCount + g_wbufSmallCount == 0) return;
    buffer tmp[WEIGHT_FLUSH_BATCH * 2];
    int n = 0;
    for (int i = 0; i < g_wbufsCount; i++) tmp[n++] = *g_wbufs[i];
    for (int i = 0; i < g_wbufSmallCount; i++) tmp[n++] = *g_wbufSmall[i];
    createTransferAndCopy(g_wbufSession.dev.device, g_wbufSession.dev.queue, tmp, n);
    for (int i = 0; i < g_wbufsCount; i++) releaseStaging(g_wbufSession.dev.device, g_wbufs[i]);
    for (int i = 0; i < g_wbufSmallCount; i++) releaseStaging(g_wbufSession.dev.device, g_wbufSmall[i]);
    g_wbufsCount = 0;
    g_wbufSmallCount = 0;
}

static void registerWeightBuffer(buffer* b) {
    if (g_wbufsCount < MAX_WEIGHT_BUFS) {
        g_wbufs[g_wbufsCount++] = b;
    }
    if (g_wbufsCount + g_wbufSmallCount >= WEIGHT_FLUSH_BATCH) weightFlush();
}

static void registerWeightBufferSmall(buffer* b) {
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

static cachedTensor* cacheStore(const char* name, QuantType q, int rows, int cols, uint8_t* data, int dataBytes, void* scale, void* zero, int scaleCount) {
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

static int tensorHave(const char* name, QuantType q, int rows, int cols) {
    (void)rows;
    (void)cols;
    if (cacheFind(name, q) != NULL) return 1;
    return g_hqmOpen;
}

static cachedTensor* hqmReadQuant(const char* name, QuantType q, int rows, int cols, int experts) {
    const hqm_tensor* t = hqm_tensor_find(&g_hqm, name);
    if (t == NULL) fatal("hqm missing tensor");
    int block = quant_is_q4(q) ? quant_block(q) : 256;
    int blocks = (cols + block - 1) / block;
    int64_t dataBytes = (q == QUANT_FP16 ? (int64_t)rows * cols * 2 :
                         q == QUANT_INT8 ? (int64_t)rows * cols :
                         (int64_t)rows * cols / 2) * experts;
    int scaleCount = rows * blocks * experts;
    uint8_t* data = (uint8_t*)hqm_tensor_read(&g_hqm, t, NULL);
    if (data == NULL) fatal("hqm tensor read error");
    void* scale = NULL;
    void* zero = NULL;
    if (q != QUANT_FP16) {
        int64_t scaleBytes = (int64_t)quant_scale_bytes(q) * scaleCount;
        char sub[96];
        int64_t got = 0;
        snprintf(sub, sizeof(sub), "%s.scale", name);
        const hqm_tensor* ts = hqm_tensor_find(&g_hqm, sub);
        if (ts == NULL) fatal("hqm missing scale");
        scale = hqm_tensor_read(&g_hqm, ts, &got);
        if (scale == NULL || got != scaleBytes) fatal("hqm scale size mismatch, re-export model");
        snprintf(sub, sizeof(sub), "%s.zero", name);
        const hqm_tensor* tz = hqm_tensor_find(&g_hqm, sub);
        if (tz == NULL) fatal("hqm missing zero");
        zero = hqm_tensor_read(&g_hqm, tz, &got);
        if (zero == NULL || got != scaleBytes) fatal("hqm zero size mismatch, re-export model");
    }
    return cacheStore(name, q, rows, cols, data, (int)dataBytes, scale, zero, scaleCount);
}

static void hqmExportTensor(const cachedTensor* ct) {
    if (g_hqmWriter == NULL) return;
    int type = ct->q == QUANT_FP16 ? HQM_T_FP16 : ct->q == QUANT_INT8 ? HQM_T_INT8 : HQM_T_INT4;
    int64_t dims[2] = {ct->rows, ct->cols};
    hqm_writer_tensor(g_hqmWriter, ct->name, type, ct->data, ct->dataBytes, dims, 2);
    if (ct->q != QUANT_FP16) {
        char sub[96];
        int64_t sd[1] = {ct->scaleCount};
        int scaleBytes = quant_scale_bytes(ct->q);
        int scaleType = scaleBytes == 2 ? HQM_T_FP16 : HQM_T_F32;
        int64_t total = (int64_t)scaleBytes * ct->scaleCount;
        snprintf(sub, sizeof(sub), "%s.scale", ct->name);
        hqm_writer_tensor(g_hqmWriter, sub, scaleType, ct->scale, total, sd, 1);
        snprintf(sub, sizeof(sub), "%s.zero", ct->name);
        hqm_writer_tensor(g_hqmWriter, sub, scaleType, ct->zero, total, sd, 1);
    }
}

static void hqmExportPool(const cachedTensor* ct, int experts) {
    if (g_hqmWriter == NULL) return;
    int64_t dims[3] = {experts, ct->rows, ct->cols};
    hqm_writer_tensor(g_hqmWriter, ct->name, HQM_T_INT4, ct->data, ct->dataBytes, dims, 3);
    char sub[96];
    int64_t sd[1] = {ct->scaleCount};
    int scaleBytes = quant_scale_bytes(ct->q);
    int scaleType = scaleBytes == 2 ? HQM_T_FP16 : HQM_T_F32;
    int64_t total = (int64_t)scaleBytes * ct->scaleCount;
    snprintf(sub, sizeof(sub), "%s.scale", ct->name);
    hqm_writer_tensor(g_hqmWriter, sub, scaleType, ct->scale, total, sd, 1);
    snprintf(sub, sizeof(sub), "%s.zero", ct->name);
    hqm_writer_tensor(g_hqmWriter, sub, scaleType, ct->zero, total, sd, 1);
}

static uint16_t* scaleToFp16(const float* src, int count) {
    uint16_t* dst = (uint16_t*)malloc(sizeof(uint16_t) * count);
    for (int i = 0; i < count; i++) dst[i] = float_to_fp16(src[i]);
    return dst;
}

static cachedTensor* tensorBuild(const char* name, QuantType q, int rows, int cols, float wscale, const float* mat) {
    int block = quant_is_q4(q) ? quant_block(q) : 256;
    int blocks = (cols + block - 1) / block;
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
    } else {
        QuantizedData qd = (q == QUANT_INT8) ? quantizeDataINT8(mat, rows, cols) : quantizeDataQ4(mat, rows, cols, q);
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
        void* scale = qd.scale;
        void* zero = qd.z;
        if (quant_is_q4(q)) {
            scale = scaleToFp16(qd.scale, scaleCount);
            zero = scaleToFp16(qd.z, scaleCount);
            free(qd.scale);
            free(qd.z);
        }
        ct = cacheStore(name, q, rows, cols, tw, dataBytes, scale, zero, scaleCount);
    }
    hqmExportTensor(ct);
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

    cachedTensor* ct = cacheFind(name, q);
    if (ct == NULL && g_hqmOpen) ct = hqmReadQuant(name, q, rows, cols, 1);
    if (ct == NULL) ct = tensorBuild(name, q, rows, cols, wscale, mat);

    t->data = createBufferNamed(s.dev.device, s.dev.physicalDevice, ct->data, ct->dataBytes, MEMORY_VRAM, name);
    countBuffer(name, layer, t->data);
    registerWeightBuffer(&t->data);
    if (q != QUANT_FP16) {
        int scaleBytes = quant_scale_bytes(q);
        char label[80];
        snprintf(label, sizeof(label), "%s-scale", name);
        t->scale = createBufferNamed(s.dev.device, s.dev.physicalDevice, ct->scale, scaleBytes * ct->scaleCount, MEMORY_VRAM, label);
        countBuffer(label, layer, t->scale);
        registerWeightBuffer(&t->scale);
        snprintf(label, sizeof(label), "%s-zero", name);
        t->zero = createBufferNamed(s.dev.device, s.dev.physicalDevice, ct->zero, scaleBytes * ct->scaleCount, MEMORY_VRAM, label);
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

static int findShards(const char* dir, char out[][512], int max);

static char g_weightDir[512];
static char g_ggufPath[512];
static safetensors g_shards;
static int g_shardState = 0;
static int g_gguf = 0;

static const safetensors* shardSource(void) {
    if (g_shardState != 0) return g_shardState == 1 ? &g_shards : NULL;
    const char* ggufPath = g_ggufPath[0] ? g_ggufPath : (gguf_path_is_file(g_weightDir) ? g_weightDir : NULL);
    if (ggufPath != NULL) {
        gguf g;
        if (gguf_open(&g, ggufPath, NULL, 0) != 0) {
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

static float* hqmReadVec(const char* name, int64_t* outLen) {
    const hqm_tensor* t = hqm_tensor_find(&g_hqm, name);
    if (t == NULL) fatal("hqm missing vector");
    int64_t bytes = 0;
    float* v = (float*)hqm_tensor_read(&g_hqm, t, &bytes);
    if (v == NULL) fatal("hqm vector read error");
    *outLen = bytes / 4;
    return v;
}

static buffer loadVecBuffer(session s, const char* hfName, int len, const char* label, int layer, int addOne, int nQk, int group) {
    char cacheName[80];
    snprintf(cacheName, sizeof(cacheName), "vec_%s_%d", label, layer);
    float* v = NULL;
    if (g_hqmOpen) {
        int64_t n = 0;
        v = hqmReadVec(cacheName, &n);
        if (n != len) fatal("hqm vector length mismatch");
    } else {
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
        if (g_gguf) permuteValueHeads(v, 1, nQk, group);
        if (g_hqmWriter != NULL) {
            int64_t dims[1] = {len};
            hqm_writer_tensor(g_hqmWriter, cacheName, HQM_T_F32, v, (int64_t)sizeof(float) * len, dims, 1);
        }
    }
    buffer b = createBufferNamed(s.dev.device, s.dev.physicalDevice, v, sizeof(float) * len, MEMORY_VRAM, label);
    countBuffer(label, layer, b);
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
    cachedTensor* ct = cacheFind(name, QUANT_FP16);
    if (ct == NULL && g_hqmOpen) ct = hqmReadQuant(name, QUANT_FP16, K, V, 1);
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
        hqmExportTensor(ct);
        if (opened) safetensors_close(&sfCand);
    }
    *out = createBufferNamed(s.dev.device, s.dev.physicalDevice, ct->data, ct->dataBytes, MEMORY_VRAM, name);
    countBuffer(name, -1, *out);
    registerWeightBuffer(out);
    cacheRelease(ct);
}

static buffer loadConv(session s, const char* name, int layer, const model_dims* d) {
    char cacheName[80];
    snprintf(cacheName, sizeof(cacheName), "conv_%d", layer);
    float* v = NULL;
    int64_t n = 0;
    if (g_hqmOpen) {
        v = hqmReadVec(cacheName, &n);
    } else {
        const safetensors* sf = shardSource();
        const sa_tensor* t = require(sf, name);
        v = safetensors_load_f32(sf, t, &n);
        if (!v) fatal("conv read error");
        if (g_gguf) {
            int taps = d->convHist + 1;
            permuteValueHeads(v + (size_t)(2 * d->nQk) * (size_t)d->dim * (size_t)taps,
                              (int64_t)d->dim * taps, d->nQk, d->nV / d->nQk);
        }
        if (g_hqmWriter != NULL) {
            int64_t dims[1] = {n};
            hqm_writer_tensor(g_hqmWriter, cacheName, HQM_T_F32, v, n * 4, dims, 1);
        }
    }
    buffer b = createBufferNamed(s.dev.device, s.dev.physicalDevice, v, sizeof(float) * n, MEMORY_VRAM, cacheName);
    countBuffer("conv", layer, b);
    free(v);
    return b;
}

static buffer loadRouterFp16(session s, const char* hfName, int K, int N, const char* cacheName, int layer) {
    int64_t bytes = (int64_t)K * N * 2;
    cachedTensor* ct = cacheFind(cacheName, QUANT_FP16);
    if (ct == NULL && g_hqmOpen) ct = hqmReadQuant(cacheName, QUANT_FP16, K, N, 1);
    if (ct == NULL) {
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
        uint16_t* tw = (uint16_t*)malloc((size_t)bytes);
        transpose_block16((uint8_t*)eng, (uint8_t*)tw, K, N, QUANT_FP16);
        free(eng);
        ct = cacheStore(cacheName, QUANT_FP16, K, N, (uint8_t*)tw, (int)bytes, NULL, NULL, 0);
        hqmExportTensor(ct);
    }
    buffer b = createBufferNamed(s.dev.device, s.dev.physicalDevice, ct->data, ct->dataBytes, MEMORY_VRAM, cacheName);
    countBuffer(cacheName, layer, b);
    cacheRelease(ct);
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
    int block = quant_is_q4(b->ct->q) ? quant_block(b->ct->q) : 256;
    int64_t dataStride = (int64_t)rows * cols / 2;
    int64_t scaleStride = (int64_t)quant_scale_bytes(b->ct->q) * rows * ((cols + block - 1) / block);

    uint8_t* vramData;
    uint16_t* vramScale;
    uint16_t* vramZero;
    if (ramCount == 0) {
        vramData = b->ct->data;
        vramScale = (uint16_t*)b->ct->scale;
        vramZero = (uint16_t*)b->ct->zero;
    } else {
        vramData = (uint8_t*)malloc((size_t)(dataStride * vramCount));
        vramScale = (uint16_t*)malloc((size_t)(scaleStride * vramCount));
        vramZero = (uint16_t*)malloc((size_t)(scaleStride * vramCount));
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

static cachedTensor* expertPoolAcquire(const char* cacheName, QuantType q, int rows, int cols, int experts) {
    cachedTensor* ct = cacheFind(cacheName, q);
    if (ct != NULL) return ct;
    if (g_hqmOpen) return hqmReadQuant(cacheName, q, rows, cols, experts);
    return NULL;
}

static void readBf16Row(const safetensors* sf, const sa_tensor* t, int64_t row, int64_t count, float* dst) {
    FILE* f = sf->files[t->fileIndex];
    _fseeki64(f, t->offset + row * count * 2, SEEK_SET);
    uint16_t* raw = (uint16_t*)malloc((size_t)count * 2);
    if (fread(raw, 2, (size_t)count, f) != (size_t)count) fatal("expert read error");
    for (int64_t i = 0; i < count; i++) dst[i] = bf16_to_float(raw[i]);
    free(raw);
}

static void expertPoolBuildLayer(const safetensors* sf, const char* hfName, const char* hfUpName, int rows, int cols, int experts, QuantType q,
                                 const int* hfSrcRows, int hfSrcCount, const char* sharedName, const char* sharedUpName,
                                 const char* cacheName, expert_pool_build* out) {
    int block = quant_block(q);
    int blocks = (cols + block - 1) / block;
    int scaleCount = rows * blocks;
    int64_t dataBytes = (int64_t)rows * cols / 2;
    uint8_t* poolData = (uint8_t*)malloc((size_t)(dataBytes * experts));
    uint16_t* poolScale = (uint16_t*)malloc(sizeof(uint16_t) * (size_t)scaleCount * experts);
    uint16_t* poolZero = (uint16_t*)malloc(sizeof(uint16_t) * (size_t)scaleCount * experts);

    for (int e = 0; e < experts; e++) {
        float* eng = NULL;
        if (e < hfSrcCount) {
            if (hfUpName != NULL) {
                const sa_tensor* tg = require(sf, hfName);
                const sa_tensor* tu = require(sf, hfUpName);
                int halfCols = cols / 2;
                if (tg->ndim != 3 || tg->shape[0] != hfSrcCount || tg->shape[1] != halfCols || tg->shape[2] != rows) fatal("expert gate shape mismatch");
                if (tu->ndim != 3 || tu->shape[0] != hfSrcCount || tu->shape[1] != halfCols || tu->shape[2] != rows) fatal("expert up shape mismatch");
                int64_t half = (int64_t)halfCols * rows;
                eng = (float*)malloc(sizeof(float) * (size_t)cols * rows);
                readBf16Row(sf, tg, hfSrcRows[e], half, eng);
                readBf16Row(sf, tu, hfSrcRows[e], half, eng + half);
            } else {
                const sa_tensor* t = require(sf, hfName);
                if (t->ndim != 3 || t->shape[0] != hfSrcCount || t->shape[1] != cols || t->shape[2] != rows) fatal("expert tensor shape mismatch");
                eng = (float*)malloc(sizeof(float) * (size_t)cols * rows);
                readBf16Row(sf, t, hfSrcRows[e], (int64_t)cols * rows, eng);
            }
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
        QuantizedData qd = quantizeDataQ4(eng, rows, cols, q);
        free(eng);
        transpose_block16(qd.data, poolData + (size_t)e * dataBytes, rows, cols, q);
        free(qd.data);
        for (int i = 0; i < scaleCount; i++) {
            poolScale[(size_t)e * scaleCount + i] = float_to_fp16(qd.scale[i]);
            poolZero[(size_t)e * scaleCount + i] = float_to_fp16(qd.z[i]);
        }
        free(qd.scale);
        free(qd.z);
    }

    out->ct = cacheStore(cacheName, q, rows, cols, poolData, (int)(dataBytes * experts), poolScale, poolZero, scaleCount * experts);
    hqmExportPool(out->ct, experts);
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

static int readWholeFile(const char* path, uint8_t** out, int64_t* outLen) {
    FILE* f = fopen(path, "rb");
    if (f == NULL) return 0;
    _fseeki64(f, 0, SEEK_END);
    int64_t n = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
    uint8_t* buf = (uint8_t*)malloc((size_t)n + 1);
    if (buf == NULL || (n > 0 && fread(buf, 1, (size_t)n, f) != (size_t)n)) {
        free(buf);
        fclose(f);
        return 0;
    }
    fclose(f);
    *out = buf;
    *outLen = n;
    return 1;
}

static void exportBytes(hqm_writer* w, const char* dir, const char* name) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    uint8_t* buf = NULL;
    int64_t n = 0;
    if (!readWholeFile(path, &buf, &n)) return;
    int64_t dims[1] = {n};
    hqm_writer_tensor(w, name, HQM_T_BYTES, buf, n, dims, 1);
    free(buf);
}

static void exportTokenizer(hqm_writer* w, const model_config* spec, const char* modelDir, const char* weightDir) {
    char dir[512];
    if (spec->pruned) snprintf(dir, sizeof(dir), "%s/vocab", modelDir);
    else snprintf(dir, sizeof(dir), "%s", modelDir);

    char probe[512];
    snprintf(probe, sizeof(probe), "%s/tokenizer.json", dir);
    if (fopen(probe, "rb") != NULL) {
        exportBytes(w, dir, "tokenizer.json");
        exportBytes(w, dir, "tokenizer_config.json");
        exportBytes(w, dir, "vocab.json");
        if (spec->pruned) exportBytes(w, dir, "mapping.npy");
        return;
    }

    const char* ggufPath = g_ggufPath[0] ? g_ggufPath : (gguf_path_is_file(weightDir) ? weightDir : NULL);
    if (ggufPath == NULL) return;
    gguf g;
    if (gguf_open(&g, ggufPath, NULL, 0) != 0) return;
    char** toks = NULL;
    char** merges = NULL;
    int32_t* ttypes = NULL;
    int64_t nTok = 0, nMerge = 0, nType = 0;
    if (gguf_meta_str_array(&g, "tokenizer.ggml.tokens", &toks, &nTok)) {
        hqm_writer_str_array(w, "tokenizer.tokens", (const char* const*)toks, nTok);
    }
    if (gguf_meta_str_array(&g, "tokenizer.ggml.merges", &merges, &nMerge)) {
        hqm_writer_str_array(w, "tokenizer.merges", (const char* const*)merges, nMerge);
    }
    if (gguf_meta_i32_array(&g, "tokenizer.ggml.token_type", &ttypes, &nType)) {
        hqm_writer_i32(w, "tokenizer.token_type", ttypes, nType);
    }
    gguf_str_array_free(toks, nTok);
    gguf_str_array_free(merges, nMerge);
    free(ttypes);
    gguf_close(&g);
}

model_weights createWeights(session s, const model_config* spec, const char* weightDir, int verbose) {
    model_weights w = {0};
    const model_dims* d = &spec->dims;
    weightBytes = 0;
    verboseWeights = verbose;
    g_wbufsCount = 0;
    g_wbufSession = s;
    cacheClear();
    snprintf(g_weightDir, sizeof(g_weightDir), "%s", weightDir);
    snprintf(g_ggufPath, sizeof(g_ggufPath), "%s", spec->ggufPath);
    g_gguf = (spec->ggufPath[0] != '\0') || gguf_path_is_file(weightDir);
    char modelDir[512];
    snprintf(modelDir, sizeof(modelDir), "%s", weightDir);
    if (gguf_path_is_file(weightDir)) gguf_dir_of(weightDir, modelDir, sizeof(modelDir));

    char hqmPath[512];
    g_hqmOpen = 0;
    g_hqmWriter = NULL;
    if (hqm_resolve(spec, weightDir, hqmPath, sizeof(hqmPath))) {
        if (hqm_open(&g_hqm, hqmPath) != 0) fatal("cannot open hqm");
        g_hqmOpen = 1;
        fprintf(stderr, "weights: loading from %s\n", hqmPath);
    } else if (g_export) {
        hqm_model_path(spec, weightDir, hqmPath, sizeof(hqmPath));
        g_hqmWriter = hqm_writer_open(hqmPath);
        if (g_hqmWriter != NULL) {
            hqm_write_config(g_hqmWriter, spec);
        } else {
            fprintf(stderr, "weights: cannot export %s\n", hqmPath);
        }
    }

    if (!g_hqmOpen) {
        char shardProbe[SA_MAX_FILES][512];
        int shardCount = findShards(weightDir, shardProbe, SA_MAX_FILES);
        if (!g_gguf && shardCount == 0) {
            fatal("no model source found");
        }
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

    w.gammaFinal = loadVecBuffer(s, "model.language_model.norm.weight", d->K, "gammaFinal", -1, 1, 0, 1);
    registerWeightBufferSmall(&w.gammaFinal);

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
    if (w.layerBufs == NULL) {
        bufferAllocFail("out of host memory: weight layer table");
        return w;
    }
    w.gammaIn = w.layerBufs + 0 * d->layerCount;
    w.gammaF = w.layerBufs + 1 * d->layerCount;
    w.qNorm = w.layerBufs + 2 * d->layerCount;
    w.kNorm = w.layerBufs + 3 * d->layerCount;
    w.conv = w.layerBufs + 4 * d->layerCount;
    w.aLog = w.layerBufs + 5 * d->layerCount;
    w.dtBias = w.layerBufs + 6 * d->layerCount;
    w.attnNorm = w.layerBufs + 7 * d->layerCount;
    w.tensorBufs = (tensor*)calloc((size_t)d->layerCount * 5, sizeof(tensor));
    if (w.tensorBufs == NULL) {
        bufferAllocFail("out of host memory: weight tensor table");
        return w;
    }
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
        w.router = (buffer*)calloc((size_t)d->layerCount, sizeof(buffer));
        w.sharedGate = (buffer*)calloc((size_t)d->layerCount, sizeof(buffer));
        if (w.poolBufs == NULL || w.router == NULL || w.sharedGate == NULL) {
            bufferAllocFail("out of host memory: expert pool table");
            return w;
        }
        w.guPool = w.poolBufs + 0 * d->layerCount;
        w.dnPool = w.poolBufs + 1 * d->layerCount;
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
    int vGroup = d->nQk > 0 ? d->nV / d->nQk : 1;

    for (int L = 0; L < spec->dims.layerCount; L++) {
        const layer* ly = &spec->layers[L];
        QuantType q = ly->attn.q;
        QuantType f = ly->ffn.q;

        lname(n1, sizeof(n1), L, "input_layernorm.weight");
        w.gammaIn[L] = loadVecBuffer(s, n1, d->K, "gammaIn", L, 1, 0, 1);
        registerWeightBufferSmall(&w.gammaIn[L]);
        lname(n1, sizeof(n1), L, "post_attention_layernorm.weight");
        w.gammaF[L] = loadVecBuffer(s, n1, d->K, "gammaF", L, 1, 0, 1);
        registerWeightBufferSmall(&w.gammaF[L]);

        char projName[64], outName[64];
        snprintf(projName, sizeof(projName), "proj_%d", L);
        snprintf(outName, sizeof(outName), "out_%d", L);

        if (ly->attn.type == ATTENTION_FULL) {
            lname(n1, sizeof(n1), L, "self_attn.q_norm.weight");
            w.qNorm[L] = loadVecBuffer(s, n1, d->headDim, "qNorm", L, 1, 0, 1);
            registerWeightBufferSmall(&w.qNorm[L]);
            lname(n1, sizeof(n1), L, "self_attn.k_norm.weight");
            w.kNorm[L] = loadVecBuffer(s, n1, d->headDim, "kNorm", L, 1, 0, 1);
            registerWeightBufferSmall(&w.kNorm[L]);

            lname(n1, sizeof(n1), L, "self_attn.q_proj.weight");
            lname(n2, sizeof(n2), L, "self_attn.k_proj.weight");
            lname(n3, sizeof(n3), L, "self_attn.v_proj.weight");
            int cols = 0;
            float* mat = NULL;
            if (!tensorHave(projName, q, d->K, d->qkvN)) {
                mat = buildQkvMatrix(shardSource(), n1, n2, n3, d->K, d->headDim, d->heads, &cols);
                if (cols != d->qkvN) fatal("qkv projection width mismatch");
            }
            loadTensorInto(s, &w.proj[L], projName, L, d->K, d->qkvN, q, 1.0f, mat);
            free(mat);

            lname(n1, sizeof(n1), L, "self_attn.o_proj.weight");
            const char* on[1] = {n1};
            mat = NULL;
            if (!tensorHave(outName, q, d->qOff, d->K)) {
                mat = buildEngineMatrix(shardSource(), on, 1, d->qOff, &cols);
                if (cols != d->K) fatal("o_proj width mismatch");
            }
            loadTensorInto(s, &w.out[L], outName, L, d->qOff, d->K, q, 1.0f, mat);
            free(mat);
        } else {
            lname(n1, sizeof(n1), L, "linear_attn.conv1d.weight");
            w.conv[L] = loadConv(s, n1, L, d);
            registerWeightBufferSmall(&w.conv[L]);
            lname(n1, sizeof(n1), L, "linear_attn.A_log");
            w.aLog[L] = loadVecBuffer(s, n1, d->nV, "aLog", L, 0, d->nQk, vGroup);
            registerWeightBufferSmall(&w.aLog[L]);
            lname(n1, sizeof(n1), L, "linear_attn.dt_bias");
            w.dtBias[L] = loadVecBuffer(s, n1, d->nV, "dtBias", L, 0, d->nQk, vGroup);
            registerWeightBufferSmall(&w.dtBias[L]);
            lname(n1, sizeof(n1), L, "linear_attn.norm.weight");
            w.attnNorm[L] = loadVecBuffer(s, n1, d->dim, "attnNorm", L, 0, 0, 1);
            registerWeightBufferSmall(&w.attnNorm[L]);

            lname(n1, sizeof(n1), L, "linear_attn.in_proj_qkv.weight");
            lname(n2, sizeof(n2), L, "linear_attn.in_proj_z.weight");
            lname(n3, sizeof(n3), L, "linear_attn.in_proj_a.weight");
            lname(n4, sizeof(n4), L, "linear_attn.in_proj_b.weight");
            const char* pn[4] = {n1, n2, n3, n4};
            int cols = 0;
            float* mat = NULL;
            if (!tensorHave(projName, q, d->K, d->projN)) {
                mat = buildEngineMatrix(shardSource(), pn, 4, d->K, &cols);
                if (cols != d->projN) fatal("delta projection width mismatch");
                if (g_gguf) {
                    permuteValueHeadColumns(mat, d->K, d->projN, d->projVOff, d->dim, d->nQk, vGroup);
                    permuteValueHeadColumns(mat, d->K, d->projN, d->projZOff, d->dim, d->nQk, vGroup);
                    permuteValueHeadColumns(mat, d->K, d->projN, d->projAOff, 1, d->nQk, vGroup);
                    permuteValueHeadColumns(mat, d->K, d->projN, d->projBOff, 1, d->nQk, vGroup);
                }
            }
            loadTensorInto(s, &w.proj[L], projName, L, d->K, d->projN, q, 1.0f, mat);
            free(mat);

            lname(n1, sizeof(n1), L, "linear_attn.out_proj.weight");
            const char* on[1] = {n1};
            mat = NULL;
            int deltaOutRows = d->nV * d->dim;
            if (!tensorHave(outName, q, deltaOutRows, d->K)) {
                mat = buildEngineMatrix(shardSource(), on, 1, deltaOutRows, &cols);
                if (cols != d->K) fatal("out_proj width mismatch");
                if (g_gguf) permuteValueHeads(mat, (int64_t)d->dim * d->K, d->nQk, vGroup);
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
            if (!tensorHave(gateName, f, d->K, d->ffnN)) {
                mat = buildEngineMatrix(shardSource(), gn, 1, d->K, &cols);
                if (cols != d->ffnN) fatal("gate width mismatch");
            }
            loadTensorInto(s, &w.gate[L], gateName, L, d->K, d->ffnN, f, 1.0f, mat);
            free(mat);

            lname(n1, sizeof(n1), L, "mlp.up_proj.weight");
            mat = NULL;
            if (!tensorHave(upName, f, d->K, d->ffnN)) {
                mat = buildEngineMatrix(shardSource(), gn, 1, d->K, &cols);
            }
            loadTensorInto(s, &w.up[L], upName, L, d->K, d->ffnN, f, 1.0f, mat);
            free(mat);

            lname(n1, sizeof(n1), L, "mlp.down_proj.weight");
            mat = NULL;
            if (!tensorHave(downName, f, d->ffnN, d->K)) {
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
            QuantType eq = quant_is_q4(f) ? f : QUANT_Q4_256;

            expert_pool_build gu, dn;
            gu.ct = expertPoolAcquire(guName, eq, d->K, 2 * d->moeI, poolExperts);
            if (gu.ct == NULL) {
                char guGate[128], guUp[128];
                lname(n1, sizeof(n1), L, "mlp.experts.gate_up_proj");
                lname(n2, sizeof(n2), L, "mlp.shared_expert.gate_proj.weight");
                lname(n3, sizeof(n3), L, "mlp.shared_expert.up_proj.weight");
                lname(guGate, sizeof(guGate), L, "mlp.experts.gate_proj.weight");
                lname(guUp, sizeof(guUp), L, "mlp.experts.up_proj.weight");
                const safetensors* sf = shardSource();
                int split = safetensors_find(sf, n1) == NULL;
                expertPoolBuildLayer(sf, split ? guGate : n1, split ? guUp : NULL, d->K, 2 * d->moeI, poolExperts, eq, srcRows,
                                     d->experts, n2, n3, guName, &gu);
            } else {
                gu.rows = d->K;
                gu.cols = 2 * d->moeI;
                gu.experts = poolExperts;
            }

            dn.ct = expertPoolAcquire(dnName, eq, d->moeI, d->K, poolExperts);
            if (dn.ct == NULL) {
                lname(n1, sizeof(n1), L, "mlp.experts.down_proj");
                lname(n2, sizeof(n2), L, "mlp.shared_expert.down_proj.weight");
                expertPoolBuildLayer(shardSource(), n1, NULL, d->moeI, d->K, poolExperts, eq, srcRows, d->experts, n2, NULL, dnName, &dn);
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
            registerWeightBufferSmall(&w.router[L]);
            lname(n1, sizeof(n1), L, "mlp.shared_expert_gate.weight");
            w.sharedGate[L] = loadRouterFp16(s, n1, d->K, 1, sgName, L);
            registerWeightBufferSmall(&w.sharedGate[L]);
        }
    }

    shardSourceClose();
    cacheClear();

    if (g_hqmWriter != NULL) {
        exportTokenizer(g_hqmWriter, spec, modelDir, weightDir);
        hqm_writer_finish(g_hqmWriter);
        g_hqmWriter = NULL;
    }
    if (g_hqmOpen) {
        hqm_close(&g_hqm);
        g_hqmOpen = 0;
    }

    weightFlush();
    if (d->tied) w.lmHead = w.embed;

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
    if (w->layerBufs != NULL && w->tensorBufs != NULL) {
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
    }
    if (w->poolBufs != NULL && w->router != NULL && w->sharedGate != NULL) {
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

