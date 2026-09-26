#ifndef gguf_h
#define gguf_h

#include <stdint.h>
#include <stdio.h>
#include "safetensors.h"

#define GGUF_MAX_DIMS 8
#define GGUF_MAX_SHARDS 64

typedef struct {
    char key[128];
    int type;
    int64_t ival;
    double fval;
    char str[256];
    int arrType;
    int64_t arrCount;
    int64_t arrOffset;
} gguf_kv;

typedef struct {
    char name[256];
    int nDims;
    int64_t dims[GGUF_MAX_DIMS];
    int type;
    int64_t offset;
} gguf_tensor;

typedef struct {
    char path[512];
    gguf_kv* kvs;
    int kvCount;
    gguf_tensor* tensors;
    int tensorCount;
    int64_t dataStart;
    char shards[GGUF_MAX_SHARDS][512];
    int64_t shardDataStart[GGUF_MAX_SHARDS];
    int shardCount;
    int* shardOf;
} gguf;

int gguf_open(gguf* g, const char* path, char* err, size_t errCap);
int gguf_resolve(const char* path, char* out, size_t cap, char* err, size_t errCap);
void gguf_close(gguf* g);
int gguf_path_is_file(const char* path);
void gguf_dir_of(const char* path, char* out, size_t cap);

const gguf_kv* gguf_kv_find(const gguf* g, const char* key);
int64_t gguf_meta_int(const gguf* g, const char* key, int64_t def);
double gguf_meta_num(const gguf* g, const char* key, double def);
const char* gguf_meta_str(const gguf* g, const char* key, const char* def);
int64_t gguf_meta_arr_count(const gguf* g, const char* key, int64_t def);
int gguf_meta_str_array(const gguf* g, const char* key, char*** out, int64_t* count);
void gguf_str_array_free(char** arr, int64_t count);
int gguf_meta_i32_array(const gguf* g, const char* key, int32_t** out, int64_t* count);
int gguf_meta_int_arch(const gguf* g, const char* suffix, int64_t def);
double gguf_meta_num_arch(const gguf* g, const char* suffix, double def);
const char* gguf_arch(const gguf* g);
const gguf_tensor* gguf_find(const gguf* g, const char* name);
int gguf_as_safetensors(const gguf* g, safetensors* sf);

#endif
