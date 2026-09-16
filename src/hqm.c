#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <windows.h>
#include "hqm.h"
#include "gguf.h"

enum {
    HQM_V_U32 = 4,
    HQM_V_BOOL = 7,
    HQM_V_STR = 8,
    HQM_V_U64 = 10,
    HQM_V_F64 = 12
};

struct hqm_writer {
    FILE* f;
    char path[512];
    char tmp[520];
    int64_t running;
    int64_t dataOffset;
    uint8_t* hdr;
    int64_t hdrLen;
    int64_t hdrCap;
    uint64_t kvCount;
    uint64_t tensorCount;
};

static int hqm_rd(FILE* f, void* buf, size_t n) {
    return fread(buf, 1, n, f) == n;
}

static int hqm_rd_u32(FILE* f, uint32_t* v) {
    return hqm_rd(f, v, 4);
}

static int hqm_rd_u64(FILE* f, uint64_t* v) {
    return hqm_rd(f, v, 8);
}

static int hqm_rd_str(FILE* f, char* out, int cap) {
    uint64_t n;
    if (!hqm_rd_u64(f, &n)) return 0;
    if (n >= (uint64_t)cap) return 0;
    if (n > 0 && !hqm_rd(f, out, (size_t)n)) return 0;
    out[n] = '\0';
    return 1;
}

int hqm_path_is_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (f == NULL) return 0;
    char magic[3];
    int ok = (fread(magic, 1, 3, f) == 3 && memcmp(magic, HQM_MAGIC, 3) == 0);
    fclose(f);
    return ok;
}

void hqm_dir_of(const char* path, char* out, size_t cap) {
    snprintf(out, cap, "%s", path);
    char* slash = strrchr(out, '/');
    char* bslash = strrchr(out, '\\');
    char* sep = slash > bslash ? slash : bslash;
    if (sep != NULL) *sep = '\0';
}

static char g_exportDir[512] = "";

void hqm_set_export_dir(const char* dir) {
    snprintf(g_exportDir, sizeof(g_exportDir), "%s", dir != NULL ? dir : "");
}

static int64_t hqm_type_bytes(int type, const int64_t* dims, int ndim) {
    int64_t elems = 1;
    for (int i = 0; i < ndim; i++) elems *= dims[i];
    switch (type) {
        case HQM_T_F32: return elems * 4;
        case HQM_T_FP16: return elems * 2;
        case HQM_T_INT8: return elems;
        case HQM_T_INT4: return elems / 2;
        case HQM_T_BYTES: return elems;
        case HQM_T_I32: return elems * 4;
        default: return -1;
    }
}

int hqm_open(hqm* h, const char* path) {
    memset(h, 0, sizeof(*h));
    snprintf(h->path, sizeof(h->path), "%s", path);
    h->f = fopen(path, "rb");
    if (h->f == NULL) return -1;

    char magic[4];
    uint32_t version;
    uint64_t tensorCount, kvCount;
    if (!hqm_rd(h->f, magic, 4) || memcmp(magic, HQM_MAGIC, 3) != 0 ||
        !hqm_rd_u32(h->f, &version) || version != HQM_VERSION ||
        !hqm_rd_u64(h->f, &tensorCount) || !hqm_rd_u64(h->f, &kvCount) ||
        tensorCount > (1u << 24) || kvCount > (1u << 20)) {
        hqm_close(h);
        return -1;
    }

    h->kvs = (hqm_kv*)calloc((size_t)kvCount, sizeof(hqm_kv));
    h->tensors = (hqm_tensor*)calloc((size_t)tensorCount, sizeof(hqm_tensor));
    if (h->kvs == NULL || h->tensors == NULL) {
        hqm_close(h);
        return -1;
    }
    h->kvCount = (int)kvCount;
    h->tensorCount = (int)tensorCount;

    for (uint64_t i = 0; i < kvCount; i++) {
        hqm_kv* kv = &h->kvs[i];
        uint32_t type;
        if (!hqm_rd_str(h->f, kv->key, (int)sizeof(kv->key)) || !hqm_rd_u32(h->f, &type)) {
            hqm_close(h);
            return -1;
        }
        kv->type = (int)type;
        if (type == HQM_V_U32 || type == HQM_V_BOOL) {
            uint32_t v;
            if (!hqm_rd_u32(h->f, &v)) { hqm_close(h); return -1; }
            kv->ival = v;
        } else if (type == HQM_V_U64) {
            uint64_t v;
            if (!hqm_rd_u64(h->f, &v)) { hqm_close(h); return -1; }
            kv->ival = (int64_t)v;
        } else if (type == HQM_V_F64) {
            if (!hqm_rd(h->f, &kv->fval, 8)) { hqm_close(h); return -1; }
            kv->ival = (int64_t)kv->fval;
        } else if (type == HQM_V_STR) {
            uint64_t n;
            if (!hqm_rd_u64(h->f, &n) || n > (1u << 24)) { hqm_close(h); return -1; }
            kv->str = (char*)malloc((size_t)n + 1);
            if (kv->str == NULL) { hqm_close(h); return -1; }
            if (n > 0 && !hqm_rd(h->f, kv->str, (size_t)n)) { hqm_close(h); return -1; }
            kv->str[n] = '\0';
        } else {
            hqm_close(h);
            return -1;
        }
    }

    for (int i = 0; i < (int)tensorCount; i++) {
        hqm_tensor* t = &h->tensors[i];
        uint32_t ndim, type;
        uint64_t offset;
        if (!hqm_rd_str(h->f, t->name, (int)sizeof(t->name)) || !hqm_rd_u32(h->f, &ndim) ||
            ndim > HQM_MAX_DIMS) {
            hqm_close(h);
            return -1;
        }
        t->ndim = (int)ndim;
        for (uint32_t j = 0; j < ndim; j++) {
            uint64_t d;
            if (!hqm_rd_u64(h->f, &d)) { hqm_close(h); return -1; }
            t->dims[j] = (int64_t)d;
        }
        if (!hqm_rd_u32(h->f, &type) || !hqm_rd_u64(h->f, &offset)) {
            hqm_close(h);
            return -1;
        }
        t->type = (int)type;
        t->offset = (int64_t)offset;
    }

    h->dataOffset = hqm_meta_int(h, "hqm.data_offset", -1);
    if (h->dataOffset < 0) {
        hqm_close(h);
        return -1;
    }
    return 0;
}

void hqm_close(hqm* h) {
    if (h->f != NULL) fclose(h->f);
    for (int i = 0; i < h->kvCount; i++) free(h->kvs[i].str);
    free(h->kvs);
    free(h->tensors);
    memset(h, 0, sizeof(*h));
}

int64_t hqm_meta_int(const hqm* h, const char* key, int64_t def) {
    for (int i = 0; i < h->kvCount; i++) {
        if (strcmp(h->kvs[i].key, key) == 0) return h->kvs[i].ival;
    }
    return def;
}

double hqm_meta_num(const hqm* h, const char* key, double def) {
    for (int i = 0; i < h->kvCount; i++) {
        if (strcmp(h->kvs[i].key, key) == 0) {
            return h->kvs[i].type == HQM_V_F64 ? h->kvs[i].fval : (double)h->kvs[i].ival;
        }
    }
    return def;
}

const char* hqm_meta_str(const hqm* h, const char* key, const char* def) {
    for (int i = 0; i < h->kvCount; i++) {
        if (strcmp(h->kvs[i].key, key) == 0 && h->kvs[i].type == HQM_V_STR) return h->kvs[i].str;
    }
    return def;
}

const hqm_tensor* hqm_tensor_find(const hqm* h, const char* name) {
    for (int i = 0; i < h->tensorCount; i++) {
        if (strcmp(h->tensors[i].name, name) == 0) return &h->tensors[i];
    }
    return NULL;
}

void* hqm_tensor_read(const hqm* h, const hqm_tensor* t, int64_t* outBytes) {
    int64_t bytes = hqm_type_bytes(t->type, t->dims, t->ndim);
    if (bytes <= 0) return NULL;
    void* buf = malloc((size_t)bytes);
    if (buf == NULL) return NULL;
    _fseeki64(h->f, h->dataOffset + t->offset, SEEK_SET);
    if (fread(buf, 1, (size_t)bytes, h->f) != (size_t)bytes) {
        free(buf);
        return NULL;
    }
    if (outBytes != NULL) *outBytes = bytes;
    return buf;
}

static int64_t quant_bytes(QuantType q, int rows, int cols) {
    if (q == QUANT_FP16) return (int64_t)rows * cols * 2;
    int block = quant_is_q4(q) ? quant_block(q) : 256;
    int64_t blocks = (cols + block - 1) / block;
    int64_t scaleBytes = (int64_t)quant_scale_bytes(q) * rows * blocks;
    if (q == QUANT_INT8) return (int64_t)rows * cols + 2 * scaleBytes;
    return (int64_t)rows * cols / 2 + 2 * scaleBytes;
}

static int64_t hqm_vram_bytes(const model_config* spec) {
    const model_dims* d = &spec->dims;
    int64_t total = 0;
    total += (int64_t)(d->rotaryDim / 2) * 4;
    total += (int64_t)d->K * 4;
    total += (int64_t)d->K * d->vocab * 2;
    if (!d->tied) total += (int64_t)d->K * d->vocab * 2;
    for (int L = 0; L < d->layerCount; L++) {
        const layer* ly = &spec->layers[L];
        total += (int64_t)d->K * 4 * 2;
        if (ly->attn.type == ATTENTION_FULL) {
            total += (int64_t)d->headDim * 4 * 2;
            total += quant_bytes(ly->attn.q, d->K, d->qkvN);
            total += quant_bytes(ly->attn.q, d->qOff, d->K);
        } else {
            total += (int64_t)d->zqkvN * 4 * 4;
            total += (int64_t)d->nV * 4 * 2;
            total += (int64_t)d->dim * 4;
            total += quant_bytes(ly->attn.q, d->K, d->projN);
            total += quant_bytes(ly->attn.q, d->nV * d->dim, d->K);
        }
        if (ly->ffn.type == FFN_SWIGLU) {
            total += quant_bytes(ly->ffn.q, d->K, d->ffnN);
            total += quant_bytes(ly->ffn.q, d->K, d->ffnN);
            total += quant_bytes(ly->ffn.q, d->ffnN, d->K);
        } else if (ly->ffn.type == FFN_MOE) {
            int vramExperts = spec->expertsVram;
            if (vramExperts > d->experts - 1) vramExperts = d->experts - 1;
            if (vramExperts < 1) vramExperts = 1;
            int vramCount = vramExperts + 1;
            QuantType eq = quant_is_q4(ly->ffn.q) ? ly->ffn.q : QUANT_Q4_256;
            total += vramCount * (quant_bytes(eq, d->K, 2 * d->moeI) + quant_bytes(eq, d->moeI, d->K));
            total += (int64_t)d->K * d->experts * 2;
            total += (int64_t)d->K * 2;
        }
    }
    return total;
}

uint64_t hqm_fingerprint(const model_config* spec) {
    const model_dims* d = &spec->dims;
    uint64_t h = 1469598103934665603ull;
    uint64_t vals[24];
    int n = 0;
    vals[n++] = (uint64_t)d->K;
    vals[n++] = (uint64_t)d->layerCount;
    vals[n++] = (uint64_t)d->ffnN;
    vals[n++] = (uint64_t)d->heads;
    vals[n++] = (uint64_t)d->kvHeads;
    vals[n++] = (uint64_t)d->headDim;
    vals[n++] = (uint64_t)d->rotaryDim;
    vals[n++] = (uint64_t)d->nQk;
    vals[n++] = (uint64_t)d->nV;
    vals[n++] = (uint64_t)d->dim;
    vals[n++] = (uint64_t)d->convHist;
    vals[n++] = (uint64_t)d->experts;
    vals[n++] = (uint64_t)d->expertsPerTok;
    vals[n++] = (uint64_t)d->moeI;
    vals[n++] = (uint64_t)d->vocab;
    vals[n++] = (uint64_t)d->tied;
    vals[n++] = (uint64_t)spec->pruned;
    vals[n++] = (uint64_t)spec->embedQ;
    vals[n++] = (uint64_t)spec->lmHeadQ;
    for (int i = 0; i < n; i++) {
        uint64_t x = vals[i];
        for (int b = 0; b < 8; b++) {
            h ^= (x & 0xff);
            h *= 1099511628211ull;
            x >>= 8;
        }
    }
    uint64_t bits;
    memcpy(&bits, &d->ropeTheta, sizeof(bits));
    h ^= bits;
    h *= 1099511628211ull;
    for (int i = 0; i < d->layerCount; i++) {
        uint64_t lv = ((uint64_t)spec->layers[i].attn.type << 24) |
                      ((uint64_t)spec->layers[i].attn.q << 16) |
                      ((uint64_t)spec->layers[i].ffn.type << 8) |
                      (uint64_t)spec->layers[i].ffn.q;
        h ^= lv;
        h *= 1099511628211ull;
    }
    return h;
}

void hqm_model_path(const model_config* spec, const char* weightDir, char* out, size_t cap) {
    char dir[512];
    if (g_exportDir[0] != '\0') {
        snprintf(dir, sizeof(dir), "%s", g_exportDir);
    } else {
        snprintf(dir, sizeof(dir), "%s", weightDir);
        if (gguf_path_is_file(weightDir) || hqm_path_is_file(weightDir)) {
            hqm_dir_of(weightDir, dir, sizeof(dir));
        }
    }
    double gb = (double)hqm_vram_bytes(spec) / (1024.0 * 1024.0 * 1024.0);
    snprintf(out, cap, "%s/%s%s-%.1fgb.hqm", dir, spec->name, spec->pruned ? "-pruned" : "", gb);
}

int hqm_file_matches(const char* path, const model_config* spec) {
    if (!hqm_path_is_file(path)) return 0;
    hqm h;
    if (hqm_open(&h, path) != 0) return 0;
    int ok = ((uint64_t)hqm_meta_int(&h, "hqm.fingerprint", 0) == hqm_fingerprint(spec));
    hqm_close(&h);
    return ok;
}

int hqm_resolve(const model_config* spec, const char* weightDir, char* out, size_t cap) {
    if (hqm_path_is_file(weightDir)) {
        snprintf(out, cap, "%s", weightDir);
        return 1;
    }
    hqm_model_path(spec, weightDir, out, cap);
    return hqm_file_matches(out, spec);
}

static void hdr_ensure(hqm_writer* w, int64_t extra) {
    if (w->hdrLen + extra <= w->hdrCap) return;
    int64_t cap = w->hdrCap == 0 ? 4096 : w->hdrCap;
    while (cap < w->hdrLen + extra) cap *= 2;
    w->hdr = (uint8_t*)realloc(w->hdr, (size_t)cap);
    w->hdrCap = cap;
}

static void hdr_bytes(hqm_writer* w, const void* p, int64_t n) {
    hdr_ensure(w, n);
    memcpy(w->hdr + w->hdrLen, p, (size_t)n);
    w->hdrLen += n;
}

static void hdr_u32(hqm_writer* w, uint32_t v) {
    hdr_bytes(w, &v, 4);
}

static void hdr_u64(hqm_writer* w, uint64_t v) {
    hdr_bytes(w, &v, 8);
}

static void hdr_f64(hqm_writer* w, double v) {
    hdr_bytes(w, &v, 8);
}

static void hdr_str(hqm_writer* w, const char* s) {
    uint64_t n = strlen(s);
    hdr_u64(w, n);
    hdr_bytes(w, s, (int64_t)n);
}

static void kv_begin(hqm_writer* w, const char* key, int type) {
    hdr_str(w, key);
    hdr_u32(w, (uint32_t)type);
    w->kvCount++;
}

void hqm_writer_u32(hqm_writer* w, const char* key, uint32_t v) {
    kv_begin(w, key, HQM_V_U32);
    hdr_u32(w, v);
}

void hqm_writer_u64(hqm_writer* w, const char* key, uint64_t v) {
    kv_begin(w, key, HQM_V_U64);
    hdr_u64(w, v);
}

void hqm_writer_f64(hqm_writer* w, const char* key, double v) {
    kv_begin(w, key, HQM_V_F64);
    hdr_f64(w, v);
}

void hqm_writer_bool(hqm_writer* w, const char* key, int v) {
    kv_begin(w, key, HQM_V_BOOL);
    hdr_u32(w, (uint32_t)(v ? 1 : 0));
}

void hqm_writer_str(hqm_writer* w, const char* key, const char* s) {
    kv_begin(w, key, HQM_V_STR);
    hdr_str(w, s);
}

hqm_writer* hqm_writer_open(const char* path) {
    hqm_writer* w = (hqm_writer*)calloc(1, sizeof(hqm_writer));
    if (w == NULL) return NULL;
    snprintf(w->path, sizeof(w->path), "%s", path);
    snprintf(w->tmp, sizeof(w->tmp), "%s.tmp", path);
    w->f = fopen(w->tmp, "wb+");
    if (w->f == NULL) {
        char dir[512];
        hqm_dir_of(path, dir, sizeof(dir));
        if (dir[0] != '\0') CreateDirectoryA(dir, NULL);
        w->f = fopen(w->tmp, "wb+");
    }
    if (w->f == NULL) {
        free(w);
        return NULL;
    }
    w->dataOffset = HQM_HEADER_RESERVE;
    w->running = w->dataOffset;
    hqm_writer_u64(w, "hqm.data_offset", (uint64_t)w->dataOffset);
    return w;
}

void hqm_writer_tensor(hqm_writer* w, const char* name, int type, const void* data, int64_t bytes, const int64_t* dims, int ndim) {
    w->running = (w->running + 31) & ~(int64_t)31;
    int64_t offset = w->running - w->dataOffset;
    hdr_str(w, name);
    hdr_u32(w, (uint32_t)ndim);
    for (int i = 0; i < ndim; i++) hdr_u64(w, (uint64_t)dims[i]);
    hdr_u32(w, (uint32_t)type);
    hdr_u64(w, (uint64_t)offset);
    w->tensorCount++;
    _fseeki64(w->f, w->running, SEEK_SET);
    if (bytes > 0 && data != NULL) fwrite(data, 1, (size_t)bytes, w->f);
    w->running += bytes;
}

void hqm_writer_str_array(hqm_writer* w, const char* name, const char* const* arr, int64_t count) {
    int64_t total = 4;
    for (int64_t i = 0; i < count; i++) total += 4 + (int64_t)strlen(arr[i]);
    uint8_t* buf = (uint8_t*)malloc((size_t)total);
    if (buf == NULL) return;
    uint8_t* p = buf;
    uint32_t n32 = (uint32_t)count;
    memcpy(p, &n32, 4);
    p += 4;
    for (int64_t i = 0; i < count; i++) {
        uint32_t len = (uint32_t)strlen(arr[i]);
        memcpy(p, &len, 4);
        p += 4;
        memcpy(p, arr[i], len);
        p += len;
    }
    int64_t dims[1] = {total};
    hqm_writer_tensor(w, name, HQM_T_BYTES, buf, total, dims, 1);
    free(buf);
}

void hqm_writer_i32(hqm_writer* w, const char* name, const int32_t* data, int64_t count) {
    int64_t dims[1] = {count};
    hqm_writer_tensor(w, name, HQM_T_I32, data, count * 4, dims, 1);
}

void hqm_writer_finish(hqm_writer* w) {
    if (w == NULL) return;
    int64_t total = 24 + w->hdrLen;
    if (total > HQM_HEADER_RESERVE) {
        fprintf(stderr, "hqm: header too large (%lld bytes)\n", (long long)total);
        hqm_writer_abort(w);
        return;
    }
    _fseeki64(w->f, 0, SEEK_SET);
    uint8_t prefix[24];
    memcpy(prefix, HQM_MAGIC, 4);
    uint32_t version = HQM_VERSION;
    memcpy(prefix + 4, &version, 4);
    memcpy(prefix + 8, &w->tensorCount, 8);
    memcpy(prefix + 16, &w->kvCount, 8);
    fwrite(prefix, 1, 24, w->f);
    fwrite(w->hdr, 1, (size_t)w->hdrLen, w->f);
    fclose(w->f);
    MoveFileExA(w->tmp, w->path, MOVEFILE_REPLACE_EXISTING);
    free(w->hdr);
    free(w);
}

void hqm_writer_abort(hqm_writer* w) {
    if (w == NULL) return;
    if (w->f != NULL) fclose(w->f);
    DeleteFileA(w->tmp);
    free(w->hdr);
    free(w);
}

void hqm_write_config(hqm_writer* w, const model_config* spec) {
    const model_dims* d = &spec->dims;
    hqm_writer_str(w, "general.name", spec->name);
    hqm_writer_str(w, "hqm.shader_dir", spec->shaderDir);
    hqm_writer_u32(w, "hqm.K", (uint32_t)d->K);
    hqm_writer_u32(w, "hqm.layer_count", (uint32_t)d->layerCount);
    hqm_writer_u32(w, "hqm.ffn", (uint32_t)d->ffnN);
    hqm_writer_u32(w, "hqm.heads", (uint32_t)d->heads);
    hqm_writer_u32(w, "hqm.kv_heads", (uint32_t)d->kvHeads);
    hqm_writer_u32(w, "hqm.head_dim", (uint32_t)d->headDim);
    hqm_writer_u32(w, "hqm.rotary_dim", (uint32_t)d->rotaryDim);
    hqm_writer_f64(w, "hqm.rope_theta", d->ropeTheta);
    hqm_writer_u32(w, "hqm.n_qk", (uint32_t)d->nQk);
    hqm_writer_u32(w, "hqm.n_v", (uint32_t)d->nV);
    hqm_writer_u32(w, "hqm.dim", (uint32_t)d->dim);
    hqm_writer_u32(w, "hqm.conv_hist", (uint32_t)d->convHist);
    hqm_writer_u32(w, "hqm.vocab", (uint32_t)d->vocab);
    hqm_writer_u32(w, "hqm.eos", (uint32_t)d->eos);
    hqm_writer_bool(w, "hqm.tied", d->tied);
    hqm_writer_bool(w, "hqm.pruned", spec->pruned);
    hqm_writer_u32(w, "hqm.max_ctx", (uint32_t)d->maxCtx);
    hqm_writer_u32(w, "hqm.prefill_chunk", (uint32_t)d->prefillChunk);
    hqm_writer_u32(w, "hqm.experts", (uint32_t)d->experts);
    hqm_writer_u32(w, "hqm.experts_per_tok", (uint32_t)d->expertsPerTok);
    hqm_writer_u32(w, "hqm.moe_i", (uint32_t)d->moeI);
    hqm_writer_u32(w, "hqm.experts_vram", (uint32_t)spec->expertsVram);
    hqm_writer_u32(w, "hqm.embed_quant", (uint32_t)spec->embedQ);
    hqm_writer_u32(w, "hqm.lm_head_quant", (uint32_t)spec->lmHeadQ);
    hqm_writer_u64(w, "hqm.fingerprint", hqm_fingerprint(spec));

    int32_t* tmp = (int32_t*)malloc(sizeof(int32_t) * (size_t)d->layerCount);
    if (tmp == NULL) return;
    for (int i = 0; i < d->layerCount; i++) tmp[i] = spec->layers[i].attn.type;
    hqm_writer_i32(w, "config.layer_type", tmp, d->layerCount);
    for (int i = 0; i < d->layerCount; i++) tmp[i] = (int32_t)spec->layers[i].attn.q;
    hqm_writer_i32(w, "config.layer_attn_quant", tmp, d->layerCount);
    for (int i = 0; i < d->layerCount; i++) tmp[i] = (int32_t)spec->layers[i].ffn.q;
    hqm_writer_i32(w, "config.layer_ffn_quant", tmp, d->layerCount);
    free(tmp);
}
