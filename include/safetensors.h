#ifndef safetensors_h
#define safetensors_h

#include <stdint.h>
#include <stdio.h>

typedef enum {
    SA_DTYPE_BF16,
    SA_DTYPE_F32,
    SA_DTYPE_F16,
    SA_DTYPE_I8,
    SA_DTYPE_UNKNOWN
} sa_dtype;

typedef struct {
    char name[256];
    int64_t offset;
    int64_t length;
    sa_dtype dtype;
    int ndim;
    int64_t shape[8];
    int fileIndex;
} sa_tensor;

#define SA_MAX_FILES 32

typedef struct {
    FILE* files[SA_MAX_FILES];
    int fileCount;
    sa_tensor* tensors;
    int tensorCount;
    int tensorCap;
} safetensors;

int safetensors_open(safetensors* sf, const char** paths, int count);
const sa_tensor* safetensors_find(const safetensors* sf, const char* name);
float* safetensors_load_f32(const safetensors* sf, const sa_tensor* t, int64_t* outCount);
void safetensors_close(safetensors* sf);

#endif