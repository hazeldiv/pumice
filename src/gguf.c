#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "gguf.h"

enum {
    GGUF_UINT8 = 0,
    GGUF_INT8 = 1,
    GGUF_UINT16 = 2,
    GGUF_INT16 = 3,
    GGUF_UINT32 = 4,
    GGUF_INT32 = 5,
    GGUF_FLOAT32 = 6,
    GGUF_BOOL = 7,
    GGUF_STRING = 8,
    GGUF_ARRAY = 9,
    GGUF_UINT64 = 10,
    GGUF_INT64 = 11,
    GGUF_FLOAT64 = 12
};

enum {
    GGUF_TENSOR_F32 = 0,
    GGUF_TENSOR_F16 = 1,
    GGUF_TENSOR_BF16 = 30
};

static int rd(FILE* f, void* buf, size_t n) {
    return fread(buf, 1, n, f) == n;
}

static int rd_u32(FILE* f, uint32_t* v) {
    return rd(f, v, 4);
}

static int rd_u64(FILE* f, uint64_t* v) {
    return rd(f, v, 8);
}

static int rd_string(FILE* f, char* out, int cap) {
    uint64_t n;
    if (!rd_u64(f, &n)) return 0;
    if (out == NULL || cap <= 0) {
        return _fseeki64(f, (int64_t)n, SEEK_CUR) == 0;
    }
    size_t take = n < (uint64_t)(cap - 1) ? (size_t)n : (size_t)(cap - 1);
    if (take > 0 && !rd(f, out, take)) return 0;
    out[take] = '\0';
    if (take < n) return _fseeki64(f, (int64_t)(n - take), SEEK_CUR) == 0;
    return 1;
}

static int scalar_size(int type) {
    switch (type) {
        case GGUF_UINT8:
        case GGUF_INT8:
        case GGUF_BOOL:
            return 1;
        case GGUF_UINT16:
        case GGUF_INT16:
            return 2;
        case GGUF_UINT32:
        case GGUF_INT32:
        case GGUF_FLOAT32:
            return 4;
        case GGUF_UINT64:
        case GGUF_INT64:
        case GGUF_FLOAT64:
            return 8;
        default:
            return 0;
    }
}

static int skip_array(FILE* f, int* elemType, int64_t* count, int64_t* offset) {
    uint32_t et;
    uint64_t n;
    if (!rd_u32(f, &et) || !rd_u64(f, &n)) return 0;
    *elemType = (int)et;
    *count = (int64_t)n;
    *offset = _ftelli64(f);
    if (et == GGUF_STRING) {
        for (uint64_t i = 0; i < n; i++) {
            if (!rd_string(f, NULL, 0)) return 0;
        }
        return 1;
    }
    int sz = scalar_size((int)et);
    if (sz == 0) return 0;
    return _fseeki64(f, (int64_t)n * sz, SEEK_CUR) == 0;
}

static int read_value(FILE* f, int type, gguf_kv* kv) {
    switch (type) {
        case GGUF_UINT8: {
            uint8_t v;
            if (!rd(f, &v, 1)) return 0;
            kv->ival = v;
            return 1;
        }
        case GGUF_INT8: {
            int8_t v;
            if (!rd(f, &v, 1)) return 0;
            kv->ival = v;
            return 1;
        }
        case GGUF_UINT16: {
            uint16_t v;
            if (!rd(f, &v, 2)) return 0;
            kv->ival = v;
            return 1;
        }
        case GGUF_INT16: {
            int16_t v;
            if (!rd(f, &v, 2)) return 0;
            kv->ival = v;
            return 1;
        }
        case GGUF_UINT32: {
            uint32_t v;
            if (!rd(f, &v, 4)) return 0;
            kv->ival = v;
            return 1;
        }
        case GGUF_INT32: {
            int32_t v;
            if (!rd(f, &v, 4)) return 0;
            kv->ival = v;
            return 1;
        }
        case GGUF_FLOAT32: {
            float v;
            if (!rd(f, &v, 4)) return 0;
            kv->fval = v;
            kv->ival = (int64_t)v;
            return 1;
        }
        case GGUF_BOOL: {
            uint8_t v;
            if (!rd(f, &v, 1)) return 0;
            kv->ival = v;
            return 1;
        }
        case GGUF_STRING:
            return rd_string(f, kv->str, (int)sizeof(kv->str));
        case GGUF_ARRAY:
            return skip_array(f, &kv->arrType, &kv->arrCount, &kv->arrOffset);
        case GGUF_UINT64: {
            uint64_t v;
            if (!rd(f, &v, 8)) return 0;
            kv->ival = (int64_t)v;
            return 1;
        }
        case GGUF_INT64: {
            int64_t v;
            if (!rd(f, &v, 8)) return 0;
            kv->ival = v;
            return 1;
        }
        case GGUF_FLOAT64: {
            double v;
            if (!rd(f, &v, 8)) return 0;
            kv->fval = v;
            kv->ival = (int64_t)v;
            return 1;
        }
        default:
            return 0;
    }
}

int gguf_path_is_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (f == NULL) return 0;
    char magic[4];
    int ok = (fread(magic, 1, 4, f) == 4 && memcmp(magic, "GGUF", 4) == 0);
    fclose(f);
    return ok;
}

void gguf_dir_of(const char* path, char* out, size_t cap) {
    snprintf(out, cap, "%s", path);
    char* slash = strrchr(out, '/');
    char* bslash = strrchr(out, '\\');
    char* sep = slash > bslash ? slash : bslash;
    if (sep != NULL) *sep = '\0';
}

int gguf_open(gguf* g, const char* path) {
    memset(g, 0, sizeof(*g));
    snprintf(g->path, sizeof(g->path), "%s", path);

    FILE* f = fopen(path, "rb");
    if (f == NULL) return -1;

    char magic[4];
    uint32_t version;
    uint64_t tensorCount;
    uint64_t kvCount;
    if (!rd(f, magic, 4) || memcmp(magic, "GGUF", 4) != 0 ||
        !rd_u32(f, &version) || !rd_u64(f, &tensorCount) || !rd_u64(f, &kvCount) ||
        version < 2 || version > 3 || tensorCount > (1u << 24)) {
        fclose(f);
        return -1;
    }

    g->kvs = (gguf_kv*)calloc((size_t)kvCount, sizeof(gguf_kv));
    if (g->kvs == NULL) {
        fclose(f);
        return -1;
    }
    g->kvCount = (int)kvCount;

    int64_t alignment = 32;
    for (uint64_t i = 0; i < kvCount; i++) {
        gguf_kv* kv = &g->kvs[i];
        uint32_t type;
        if (!rd_string(f, kv->key, (int)sizeof(kv->key)) || !rd_u32(f, &type)) {
            gguf_close(g);
            fclose(f);
            return -1;
        }
        kv->type = (int)type;
        if (!read_value(f, (int)type, kv)) {
            gguf_close(g);
            fclose(f);
            return -1;
        }
        if (strcmp(kv->key, "general.alignment") == 0 && kv->ival > 0) {
            alignment = kv->ival;
        }
    }

    g->tensors = (gguf_tensor*)calloc((size_t)tensorCount, sizeof(gguf_tensor));
    if (g->tensors == NULL) {
        gguf_close(g);
        fclose(f);
        return -1;
    }
    g->tensorCount = (int)tensorCount;

    for (uint64_t i = 0; i < tensorCount; i++) {
        gguf_tensor* t = &g->tensors[i];
        uint32_t nDims;
        uint32_t type;
        uint64_t offset;
        if (!rd_string(f, t->name, (int)sizeof(t->name)) || !rd_u32(f, &nDims) ||
            nDims > GGUF_MAX_DIMS) {
            gguf_close(g);
            fclose(f);
            return -1;
        }
        t->nDims = (int)nDims;
        for (uint32_t j = 0; j < nDims; j++) {
            uint64_t d;
            if (!rd_u64(f, &d)) {
                gguf_close(g);
                fclose(f);
                return -1;
            }
            t->dims[j] = (int64_t)d;
        }
        if (!rd_u32(f, &type) || !rd_u64(f, &offset)) {
            gguf_close(g);
            fclose(f);
            return -1;
        }
        t->type = (int)type;
        t->offset = (int64_t)offset;
    }

    int64_t pos = _ftelli64(f);
    if (pos < 0) {
        gguf_close(g);
        fclose(f);
        return -1;
    }
    g->dataStart = ((pos + alignment - 1) / alignment) * alignment;
    fclose(f);
    return 0;
}

void gguf_close(gguf* g) {
    free(g->kvs);
    free(g->tensors);
    g->kvs = NULL;
    g->tensors = NULL;
    g->kvCount = 0;
    g->tensorCount = 0;
}

const gguf_kv* gguf_kv_find(const gguf* g, const char* key) {
    for (int i = 0; i < g->kvCount; i++) {
        if (strcmp(g->kvs[i].key, key) == 0) return &g->kvs[i];
    }
    return NULL;
}

int64_t gguf_meta_int(const gguf* g, const char* key, int64_t def) {
    const gguf_kv* kv = gguf_kv_find(g, key);
    if (kv == NULL) return def;
    if (kv->type == GGUF_FLOAT32 || kv->type == GGUF_FLOAT64) return (int64_t)kv->fval;
    return kv->ival;
}

double gguf_meta_num(const gguf* g, const char* key, double def) {
    const gguf_kv* kv = gguf_kv_find(g, key);
    if (kv == NULL) return def;
    if (kv->type == GGUF_FLOAT32 || kv->type == GGUF_FLOAT64) return kv->fval;
    return (double)kv->ival;
}

const char* gguf_meta_str(const gguf* g, const char* key, const char* def) {
    const gguf_kv* kv = gguf_kv_find(g, key);
    if (kv == NULL || kv->type != GGUF_STRING) return def;
    return kv->str;
}

int64_t gguf_meta_arr_count(const gguf* g, const char* key, int64_t def) {
    const gguf_kv* kv = gguf_kv_find(g, key);
    if (kv == NULL || kv->type != GGUF_ARRAY) return def;
    return kv->arrCount;
}

void gguf_str_array_free(char** arr, int64_t count) {
    if (arr == NULL) return;
    for (int64_t i = 0; i < count; i++) free(arr[i]);
    free(arr);
}

int gguf_meta_str_array(const gguf* g, const char* key, char*** out, int64_t* count) {
    const gguf_kv* kv = gguf_kv_find(g, key);
    if (kv == NULL || kv->type != GGUF_ARRAY || kv->arrType != GGUF_STRING || kv->arrCount <= 0) return 0;
    FILE* f = fopen(g->path, "rb");
    if (f == NULL) return 0;
    if (_fseeki64(f, kv->arrOffset, SEEK_SET) != 0) {
        fclose(f);
        return 0;
    }
    char** arr = (char**)calloc((size_t)kv->arrCount, sizeof(char*));
    if (arr == NULL) {
        fclose(f);
        return 0;
    }
    for (int64_t i = 0; i < kv->arrCount; i++) {
        uint64_t n;
        if (!rd_u64(f, &n) || n > (1u << 24)) {
            gguf_str_array_free(arr, i);
            fclose(f);
            return 0;
        }
        arr[i] = (char*)malloc((size_t)n + 1);
        if (arr[i] == NULL) {
            gguf_str_array_free(arr, i);
            fclose(f);
            return 0;
        }
        if (n > 0 && !rd(f, arr[i], (size_t)n)) {
            free(arr[i]);
            arr[i] = NULL;
            gguf_str_array_free(arr, i);
            fclose(f);
            return 0;
        }
        arr[i][n] = '\0';
    }
    fclose(f);
    *out = arr;
    *count = kv->arrCount;
    return 1;
}

int gguf_meta_i32_array(const gguf* g, const char* key, int32_t** out, int64_t* count) {
    const gguf_kv* kv = gguf_kv_find(g, key);
    if (kv == NULL || kv->type != GGUF_ARRAY || kv->arrCount <= 0) return 0;
    if (kv->arrType != GGUF_INT32 && kv->arrType != GGUF_UINT32) return 0;
    FILE* f = fopen(g->path, "rb");
    if (f == NULL) return 0;
    if (_fseeki64(f, kv->arrOffset, SEEK_SET) != 0) {
        fclose(f);
        return 0;
    }
    int32_t* arr = (int32_t*)malloc(sizeof(int32_t) * (size_t)kv->arrCount);
    if (arr == NULL) {
        fclose(f);
        return 0;
    }
    if (fread(arr, 4, (size_t)kv->arrCount, f) != (size_t)kv->arrCount) {
        free(arr);
        fclose(f);
        return 0;
    }
    fclose(f);
    *out = arr;
    *count = kv->arrCount;
    return 1;
}

const char* gguf_arch(const gguf* g) {
    return gguf_meta_str(g, "general.architecture", "");
}

static void arch_key(const gguf* g, const char* suffix, char* out, size_t cap) {
    snprintf(out, cap, "%s.%s", gguf_arch(g), suffix);
}

int gguf_meta_int_arch(const gguf* g, const char* suffix, int64_t def) {
    char key[256];
    arch_key(g, suffix, key, sizeof(key));
    return gguf_meta_int(g, key, def);
}

double gguf_meta_num_arch(const gguf* g, const char* suffix, double def) {
    char key[256];
    arch_key(g, suffix, key, sizeof(key));
    return gguf_meta_num(g, key, def);
}

const gguf_tensor* gguf_find(const gguf* g, const char* name) {
    for (int i = 0; i < g->tensorCount; i++) {
        if (strcmp(g->tensors[i].name, name) == 0) return &g->tensors[i];
    }
    return NULL;
}

static const char* suffix_hf(const char* s) {
    static const struct {
        const char* gguf;
        const char* hf;
    } map[] = {
        {"attn_norm.weight", "input_layernorm.weight"},
        {"post_attention_norm.weight", "post_attention_layernorm.weight"},
        {"attn_q.weight", "self_attn.q_proj.weight"},
        {"attn_k.weight", "self_attn.k_proj.weight"},
        {"attn_v.weight", "self_attn.v_proj.weight"},
        {"attn_output.weight", "self_attn.o_proj.weight"},
        {"attn_q_norm.weight", "self_attn.q_norm.weight"},
        {"attn_k_norm.weight", "self_attn.k_norm.weight"},
        {"attn_qkv.weight", "linear_attn.in_proj_qkv.weight"},
        {"attn_gate.weight", "linear_attn.in_proj_z.weight"},
        {"ssm_alpha.weight", "linear_attn.in_proj_a.weight"},
        {"ssm_beta.weight", "linear_attn.in_proj_b.weight"},
        {"ssm_out.weight", "linear_attn.out_proj.weight"},
        {"ssm_conv1d.weight", "linear_attn.conv1d.weight"},
        {"ssm_a", "linear_attn.A_log"},
        {"ssm_dt.bias", "linear_attn.dt_bias"},
        {"ssm_norm.weight", "linear_attn.norm.weight"},
        {"ffn_gate.weight", "mlp.gate_proj.weight"},
        {"ffn_up.weight", "mlp.up_proj.weight"},
        {"ffn_down.weight", "mlp.down_proj.weight"},
        {"ffn_gate_inp.weight", "mlp.gate.weight"},
        {"ffn_gate_shexp.weight", "mlp.shared_expert.gate_proj.weight"},
        {"ffn_up_shexp.weight", "mlp.shared_expert.up_proj.weight"},
        {"ffn_down_shexp.weight", "mlp.shared_expert.down_proj.weight"},
        {"ffn_gate_inp_s.weight", "mlp.shared_expert_gate.weight"},
        {NULL, NULL}
    };
    for (int i = 0; map[i].gguf != NULL; i++) {
        if (strcmp(map[i].gguf, s) == 0) return map[i].hf;
    }
    return NULL;
}

static int gguf_hf_name(const char* name, char* out, size_t cap) {
    if (strcmp(name, "token_embd.weight") == 0) {
        snprintf(out, cap, "model.language_model.embed_tokens.weight");
        return 1;
    }
    if (strcmp(name, "output_norm.weight") == 0) {
        snprintf(out, cap, "model.language_model.norm.weight");
        return 1;
    }
    if (strcmp(name, "output.weight") == 0) {
        snprintf(out, cap, "lm_head.weight");
        return 1;
    }
    if (strncmp(name, "blk.", 4) != 0) return 0;
    char* end = NULL;
    long layer = strtol(name + 4, &end, 10);
    if (end == name + 4 || end == NULL || *end != '.') return 0;
    const char* hf = suffix_hf(end + 1);
    if (hf == NULL) return 0;
    snprintf(out, cap, "model.language_model.layers.%ld.%s", layer, hf);
    return 1;
}

static sa_dtype gguf_dtype(int type) {
    if (type == GGUF_TENSOR_F32) return SA_DTYPE_F32;
    if (type == GGUF_TENSOR_F16) return SA_DTYPE_F16;
    if (type == GGUF_TENSOR_BF16) return SA_DTYPE_BF16;
    return SA_DTYPE_UNKNOWN;
}

static int sa_type_size(sa_dtype d) {
    if (d == SA_DTYPE_F32) return 4;
    if (d == SA_DTYPE_F16 || d == SA_DTYPE_BF16) return 2;
    if (d == SA_DTYPE_I8) return 1;
    return 0;
}

int gguf_as_safetensors(const gguf* g, safetensors* sf) {
    memset(sf, 0, sizeof(*sf));
    sf->files[0] = fopen(g->path, "rb");
    if (sf->files[0] == NULL) return -1;
    sf->fileCount = 1;
    sf->tensors = (sa_tensor*)calloc((size_t)g->tensorCount, sizeof(sa_tensor));
    if (sf->tensors == NULL) {
        safetensors_close(sf);
        return -1;
    }
    sf->tensorCap = g->tensorCount;

    for (int i = 0; i < g->tensorCount; i++) {
        const gguf_tensor* gt = &g->tensors[i];
        sa_dtype dt = gguf_dtype(gt->type);
        if (dt == SA_DTYPE_UNKNOWN) continue;
        char hf[256];
        if (!gguf_hf_name(gt->name, hf, sizeof(hf))) continue;

        sa_tensor* st = &sf->tensors[sf->tensorCount];
        memset(st, 0, sizeof(*st));
        snprintf(st->name, sizeof(st->name), "%s", hf);
        st->dtype = dt;
        st->ndim = gt->nDims;
        int64_t elems = 1;
        for (int j = 0; j < gt->nDims; j++) elems *= gt->dims[j];
        for (int j = 0; j < gt->nDims; j++) st->shape[j] = gt->dims[gt->nDims - 1 - j];
        st->offset = g->dataStart + gt->offset;
        st->length = elems * sa_type_size(dt);
        st->fileIndex = 0;
        sf->tensorCount++;
    }
    return 0;
}
