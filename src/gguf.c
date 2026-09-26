#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <windows.h>
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

static void setErr(char* err, size_t cap, const char* msg) {
    if (err != NULL && cap > 0) snprintf(err, cap, "%s", msg);
}

static int endsWithGguf(const char* name) {
    size_t n = strlen(name);
    if (n < 5) return 0;
    const char* e = name + n - 5;
    return tolower((unsigned char)e[0]) == '.' && tolower((unsigned char)e[1]) == 'g' &&
           tolower((unsigned char)e[2]) == 'g' && tolower((unsigned char)e[3]) == 'u' &&
           tolower((unsigned char)e[4]) == 'f';
}

static int allDigits(const char* s, size_t n) {
    if (n == 0) return 0;
    for (size_t i = 0; i < n; i++) {
        if (s[i] < '0' || s[i] > '9') return 0;
    }
    return 1;
}

static int shardPattern(const char* name, char* prefix, size_t prefixCap, char* totalStr, size_t totalCap,
                        int* index, int* total) {
    if (!endsWithGguf(name)) return 0;
    size_t stem = strlen(name) - 5;
    const char* of = NULL;
    for (size_t i = stem; i >= 4; i--) {
        if (name[i - 4] == '-' && name[i - 3] == 'o' && name[i - 2] == 'f' && name[i - 1] == '-') {
            of = name + i;
            break;
        }
    }
    if (of == NULL) return 0;
    size_t sep = (size_t)(of - name);
    if (sep < 5 || !allDigits(of, stem - sep)) return 0;
    const char* idxStart = NULL;
    for (size_t i = sep - 4; i >= 1; i--) {
        if (name[i - 1] == '-') {
            idxStart = name + i;
            break;
        }
    }
    if (idxStart == NULL || !allDigits(idxStart, (sep - 4) - (size_t)(idxStart - name))) return 0;
    size_t totalLen = stem - sep;
    size_t prefixLen = (size_t)(idxStart - name) - 1;
    if (totalLen >= totalCap || prefixLen == 0 || prefixLen >= prefixCap) return 0;
    memcpy(prefix, name, prefixLen);
    prefix[prefixLen] = '\0';
    memcpy(totalStr, of, totalLen);
    totalStr[totalLen] = '\0';
    *index = atoi(idxStart);
    *total = atoi(of);
    return *index > 0 && *total > 0 && *index <= *total;
}

static void baseName(const char* path, const char** out) {
    const char* name = path;
    for (const char* p = path; *p; p++) {
        if (*p == '/' || *p == '\\') name = p + 1;
    }
    *out = name;
}

static int groupShards(const char* file, char shards[][512], int max, int* count, int* canonical, char* err,
                       size_t errCap) {
    char dir[512];
    gguf_dir_of(file, dir, sizeof(dir));
    const char* name = NULL;
    baseName(file, &name);

    char prefix[256];
    char totalStr[32];
    int index = 0;
    int total = 0;
    if (!shardPattern(name, prefix, sizeof(prefix), totalStr, sizeof(totalStr), &index, &total)) {
        snprintf(shards[0], 512, "%s", file);
        *count = 1;
        *canonical = 0;
        return 1;
    }
    if (total > max) {
        char msg[128];
        snprintf(msg, sizeof(msg), "too many gguf shards (%d, max %d)", total, max);
        setErr(err, errCap, msg);
        return -1;
    }

    int width = (int)strlen(totalStr);
    char missing[512] = {0};
    for (int i = 1; i <= total; i++) {
        snprintf(shards[i - 1], 512, "%s/%s-%0*d-of-%s.gguf", dir, prefix, width, i, totalStr);
        if (GetFileAttributesA(shards[i - 1]) == INVALID_FILE_ATTRIBUTES) {
            char entry[320];
            snprintf(entry, sizeof(entry), "%s-%0*d-of-%s.gguf", prefix, width, i, totalStr);
            if (missing[0]) strncat(missing, ", ", sizeof(missing) - strlen(missing) - 1);
            strncat(missing, entry, sizeof(missing) - strlen(missing) - 1);
        }
    }
    if (missing[0]) {
        char msg[768];
        snprintf(msg, sizeof(msg), "missing shards: %s", missing);
        setErr(err, errCap, msg);
        return -1;
    }
    *count = total;
    *canonical = 0;
    return 1;
}

static int parseShard(const char* path, gguf_kv** kvsOut, int* kvCountOut, gguf_tensor** tensorsOut,
                      int* tensorCountOut, int64_t* dataStartOut) {
    *kvsOut = NULL;
    *tensorsOut = NULL;
    FILE* f = fopen(path, "rb");
    if (f == NULL) return -1;

    char magic[4];
    uint32_t version;
    uint64_t tensorCount;
    uint64_t kvCount;
    if (!rd(f, magic, 4) || memcmp(magic, "GGUF", 4) != 0 || !rd_u32(f, &version) || !rd_u64(f, &tensorCount) ||
        !rd_u64(f, &kvCount) || version < 2 || version > 3 || tensorCount > (1u << 24)) {
        fclose(f);
        return -1;
    }

    gguf_kv* kvs = (gguf_kv*)calloc((size_t)kvCount, sizeof(gguf_kv));
    if (kvs == NULL) {
        fclose(f);
        return -1;
    }
    int64_t alignment = 32;
    for (uint64_t i = 0; i < kvCount; i++) {
        gguf_kv* kv = &kvs[i];
        uint32_t type;
        if (!rd_string(f, kv->key, (int)sizeof(kv->key)) || !rd_u32(f, &type)) {
            free(kvs);
            fclose(f);
            return -1;
        }
        kv->type = (int)type;
        if (!read_value(f, (int)type, kv)) {
            free(kvs);
            fclose(f);
            return -1;
        }
        if (strcmp(kv->key, "general.alignment") == 0 && kv->ival > 0) alignment = kv->ival;
    }

    gguf_tensor* tensors = (gguf_tensor*)calloc((size_t)tensorCount, sizeof(gguf_tensor));
    if (tensors == NULL) {
        free(kvs);
        fclose(f);
        return -1;
    }
    for (uint64_t i = 0; i < tensorCount; i++) {
        gguf_tensor* t = &tensors[i];
        uint32_t nDims;
        uint32_t type;
        uint64_t offset;
        if (!rd_string(f, t->name, (int)sizeof(t->name)) || !rd_u32(f, &nDims) || nDims > GGUF_MAX_DIMS) {
            free(tensors);
            free(kvs);
            fclose(f);
            return -1;
        }
        t->nDims = (int)nDims;
        for (uint32_t j = 0; j < nDims; j++) {
            uint64_t d;
            if (!rd_u64(f, &d)) {
                free(tensors);
                free(kvs);
                fclose(f);
                return -1;
            }
            t->dims[j] = (int64_t)d;
        }
        if (!rd_u32(f, &type) || !rd_u64(f, &offset)) {
            free(tensors);
            free(kvs);
            fclose(f);
            return -1;
        }
        t->type = (int)type;
        t->offset = (int64_t)offset;
    }

    int64_t pos = _ftelli64(f);
    if (pos < 0) {
        free(tensors);
        free(kvs);
        fclose(f);
        return -1;
    }
    fclose(f);
    *kvsOut = kvs;
    *kvCountOut = (int)kvCount;
    *tensorsOut = tensors;
    *tensorCountOut = (int)tensorCount;
    *dataStartOut = ((pos + alignment - 1) / alignment) * alignment;
    return 0;
}

int gguf_open(gguf* g, const char* path, char* err, size_t errCap) {
    memset(g, 0, sizeof(*g));

    char shards[GGUF_MAX_SHARDS][512];
    int count = 0;
    int canonical = 0;
    if (groupShards(path, shards, GGUF_MAX_SHARDS, &count, &canonical, err, errCap) != 1) return -1;

    g->shardCount = count;
    for (int i = 0; i < count; i++) snprintf(g->shards[i], sizeof(g->shards[i]), "%s", shards[i]);

    gguf_tensor** tensorLists = (gguf_tensor**)calloc((size_t)count, sizeof(gguf_tensor*));
    int* tensorCounts = (int*)calloc((size_t)count, sizeof(int));
    if (tensorLists == NULL || tensorCounts == NULL) {
        free(tensorLists);
        free(tensorCounts);
        setErr(err, errCap, "out of memory");
        return -1;
    }

    int total = 0;
    for (int i = 0; i < count; i++) {
        gguf_kv* kvs = NULL;
        gguf_tensor* tensors = NULL;
        int kvCount = 0;
        int tensorCount = 0;
        int64_t dataStart = 0;
        if (parseShard(shards[i], &kvs, &kvCount, &tensors, &tensorCount, &dataStart) != 0) {
            for (int j = 0; j < i; j++) free(tensorLists[j]);
            free(tensorLists);
            free(tensorCounts);
            setErr(err, errCap, "cannot parse gguf shard");
            return -1;
        }
        tensorLists[i] = tensors;
        tensorCounts[i] = tensorCount;
        g->shardDataStart[i] = dataStart;
        total += tensorCount;
        if (i == canonical) {
            g->kvs = kvs;
            g->kvCount = kvCount;
            g->dataStart = dataStart;
        } else {
            free(kvs);
        }
    }
    snprintf(g->path, sizeof(g->path), "%s", shards[canonical]);

    g->tensors = (gguf_tensor*)calloc((size_t)total, sizeof(gguf_tensor));
    g->shardOf = (int*)calloc((size_t)total, sizeof(int));
    if (g->tensors == NULL || g->shardOf == NULL) {
        for (int i = 0; i < count; i++) free(tensorLists[i]);
        free(tensorLists);
        free(tensorCounts);
        gguf_close(g);
        setErr(err, errCap, "out of memory");
        return -1;
    }
    int n = 0;
    for (int i = 0; i < count; i++) {
        for (int j = 0; j < tensorCounts[i]; j++) {
            g->tensors[n] = tensorLists[i][j];
            g->shardOf[n] = i;
            n++;
        }
        free(tensorLists[i]);
    }
    g->tensorCount = total;
    free(tensorLists);
    free(tensorCounts);
    return 0;
}

void gguf_close(gguf* g) {
    free(g->kvs);
    free(g->tensors);
    free(g->shardOf);
    g->kvs = NULL;
    g->tensors = NULL;
    g->shardOf = NULL;
    g->kvCount = 0;
    g->tensorCount = 0;
}

static void groupKey(const char* name, char* out, size_t cap) {    char prefix[256];
    char totalStr[32];
    int index = 0;
    int total = 0;
    if (shardPattern(name, prefix, sizeof(prefix), totalStr, sizeof(totalStr), &index, &total)) {
        snprintf(out, cap, "%s", prefix);
    } else {
        snprintf(out, cap, "%s", name);
    }
}

static void collectGroups(const char* dir, char firstFiles[][512], char keys[][512], int max, int* groupCount) {
    char pattern[512];
    snprintf(pattern, sizeof(pattern), "%s/*.gguf", dir);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    do {
        char key[256];
        groupKey(fd.cFileName, key, sizeof(key));
        int seen = 0;
        for (int i = 0; i < *groupCount; i++) {
            if (strcmp(keys[i], key) == 0) {
                seen = 1;
                break;
            }
        }
        if (!seen && *groupCount < max) {
            snprintf(firstFiles[*groupCount], 512, "%s/%s", dir, fd.cFileName);
            snprintf(keys[*groupCount], 512, "%s", key);
            (*groupCount)++;
        }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
}

static void listSubdirs(const char* dir, char out[][512], int max, int* count) {
    char pattern[512];
    snprintf(pattern, sizeof(pattern), "%s/*", dir);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    do {
        if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) continue;
        if (strcmp(fd.cFileName, ".") == 0 || strcmp(fd.cFileName, "..") == 0) continue;
        if (*count >= max) break;
        char cfg[512];
        snprintf(cfg, sizeof(cfg), "%s/%s/config.json", dir, fd.cFileName);
        if (GetFileAttributesA(cfg) != INVALID_FILE_ATTRIBUTES) continue;
        snprintf(out[*count], 512, "%s/%s", dir, fd.cFileName);
        (*count)++;
    } while (FindNextFileA(h, &fd));
    FindClose(h);
}

int gguf_resolve(const char* path, char* out, size_t cap, char* err, size_t errCap) {
    if (gguf_path_is_file(path)) {
        char shards[GGUF_MAX_SHARDS][512];
        int count = 0;
        int canonical = 0;
        if (groupShards(path, shards, GGUF_MAX_SHARDS, &count, &canonical, err, errCap) != 1) return -1;
        snprintf(out, cap, "%s", shards[canonical]);
        return 1;
    }

    DWORD attr = GetFileAttributesA(path);
    if (attr == INVALID_FILE_ATTRIBUTES || !(attr & FILE_ATTRIBUTE_DIRECTORY)) return 0;

    char cfg[512];
    snprintf(cfg, sizeof(cfg), "%s/config.json", path);
    if (GetFileAttributesA(cfg) != INVALID_FILE_ATTRIBUTES) return 0;

    char firsts[GGUF_MAX_SHARDS][512];
    char keys[GGUF_MAX_SHARDS][512];
    int n = 0;
    collectGroups(path, firsts, keys, GGUF_MAX_SHARDS, &n);

    char subdirs[GGUF_MAX_SHARDS][512];
    int subCount = 0;
    listSubdirs(path, subdirs, GGUF_MAX_SHARDS, &subCount);
    for (int s = 0; s < subCount; s++) {
        char subFirst[GGUF_MAX_SHARDS][512];
        char subKeys[GGUF_MAX_SHARDS][512];
        int m = 0;
        collectGroups(subdirs[s], subFirst, subKeys, GGUF_MAX_SHARDS, &m);
        for (int i = 0; i < m && n < GGUF_MAX_SHARDS; i++) {
            snprintf(firsts[n], 512, "%s", subFirst[i]);
            snprintf(keys[n], 512, "%s/%s", subdirs[s], subKeys[i]);
            n++;
        }
    }

    if (n == 0) return 0;
    if (n > 1) {
        char list[512] = {0};
        for (int i = 0; i < n; i++) {
            const char* base = NULL;
            baseName(keys[i], &base);
            if (list[0]) strncat(list, ", ", sizeof(list) - strlen(list) - 1);
            strncat(list, base, sizeof(list) - strlen(list) - 1);
        }
        char msg[768];
        snprintf(msg, sizeof(msg), "multiple gguf models in %s: %s", path, list);
        setErr(err, errCap, msg);
        return -1;
    }

    char shards[GGUF_MAX_SHARDS][512];
    int count = 0;
    int canonical = 0;
    if (groupShards(firsts[0], shards, GGUF_MAX_SHARDS, &count, &canonical, err, errCap) != 1) return -1;
    snprintf(out, cap, "%s", shards[canonical]);
    return 1;
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
        {"ffn_gate_exps.weight", "mlp.experts.gate_proj.weight"},
        {"ffn_up_exps.weight", "mlp.experts.up_proj.weight"},
        {"ffn_down_exps.weight", "mlp.experts.down_proj"},
        {"ffn_gate_inp.weight", "mlp.gate.weight"},
        {"ffn_gate_shexp.weight", "mlp.shared_expert.gate_proj.weight"},
        {"ffn_up_shexp.weight", "mlp.shared_expert.up_proj.weight"},
        {"ffn_down_shexp.weight", "mlp.shared_expert.down_proj.weight"},
        {"ffn_gate_inp_s.weight", "mlp.shared_expert_gate.weight"},
        {"ffn_gate_inp_shexp.weight", "mlp.shared_expert_gate.weight"},
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
    if (g->shardCount > SA_MAX_FILES) return -1;
    for (int i = 0; i < g->shardCount; i++) {
        sf->files[i] = fopen(g->shards[i], "rb");
        if (sf->files[i] == NULL) {
            safetensors_close(sf);
            return -1;
        }
    }
    sf->fileCount = g->shardCount;
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
        st->offset = g->shardDataStart[g->shardOf[i]] + gt->offset;
        st->length = elems * sa_type_size(dt);
        st->fileIndex = g->shardOf[i];
        sf->tensorCount++;
    }
    return 0;
}

void ggufInfo(const char* path) {
    char resolved[512] = {0};
    char err[512] = {0};
    int rc = gguf_resolve(path, resolved, sizeof(resolved), err, sizeof(err));
    if (rc < 0) {
        fprintf(stderr, "ggufinfo: %s\n", err);
        return;
    }
    if (rc == 0) {
        fprintf(stderr, "ggufinfo: not a gguf model: %s\n", path);
        return;
    }

    gguf g;
    if (gguf_open(&g, resolved, err, sizeof(err)) != 0) {
        fprintf(stderr, "ggufinfo: %s\n", err);
        return;
    }

    printf("resolved: %s\n", resolved);
    printf("arch: %s\n", gguf_arch(&g));
    printf("shards: %d\n", g.shardCount);
    for (int i = 0; i < g.shardCount; i++) {
        printf("  shard %d: %s\n", i + 1, g.shards[i]);
    }
    printf("tensors: %d\n", g.tensorCount);
    printf("tied: %s\n", gguf_find(&g, "output.weight") == NULL ? "yes" : "no");

    const char* probes[] = {"token_embd.weight", "output.weight", "blk.0.attn_qkv.weight",
                            "blk.40.ffn_gate_exps.weight", NULL};
    for (int i = 0; probes[i] != NULL; i++) {
        const gguf_tensor* t = gguf_find(&g, probes[i]);
        if (t == NULL) {
            printf("  %s: missing\n", probes[i]);
            continue;
        }
        int index = (int)(t - g.tensors);
        printf("  %s: shard %d\n", probes[i], g.shardOf[index] + 1);
    }

    safetensors sf;
    if (gguf_as_safetensors(&g, &sf) == 0) {
        printf("mapped tensors: %d across %d files\n", sf.tensorCount, sf.fileCount);
        const sa_tensor* head = safetensors_find(&sf, "lm_head.weight");
        if (head != NULL) {
            printf("  lm_head.weight: file %d offset %lld length %lld\n", head->fileIndex, (long long)head->offset,
                   (long long)head->length);
        }
        safetensors_close(&sf);
    }
    gguf_close(&g);
}
