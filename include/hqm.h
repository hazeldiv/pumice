#ifndef hqm_h
#define hqm_h

#include <stdint.h>
#include <stdio.h>
#include "model.h"

#define HQM_MAGIC "HQM"
#define HQM_VERSION 1
#define HQM_HEADER_RESERVE (16 * 1024 * 1024)

enum {
    HQM_T_F32 = 0,
    HQM_T_FP16 = 1,
    HQM_T_INT8 = 2,
    HQM_T_INT4 = 3,
    HQM_T_BYTES = 4,
    HQM_T_I32 = 5
};

#define HQM_MAX_DIMS 4

typedef struct hqm_writer hqm_writer;

typedef struct {
    char key[64];
    int type;
    int64_t ival;
    double fval;
    char* str;
} hqm_kv;

typedef struct {
    char name[64];
    int type;
    int ndim;
    int64_t dims[HQM_MAX_DIMS];
    int64_t offset;
} hqm_tensor;

typedef struct {
    FILE* f;
    char path[512];
    int64_t dataOffset;
    hqm_kv* kvs;
    int kvCount;
    hqm_tensor* tensors;
    int tensorCount;
} hqm;

int hqm_path_is_file(const char* path);
int hqm_open(hqm* h, const char* path);
void hqm_close(hqm* h);
int64_t hqm_meta_int(const hqm* h, const char* key, int64_t def);
double hqm_meta_num(const hqm* h, const char* key, double def);
const char* hqm_meta_str(const hqm* h, const char* key, const char* def);
const hqm_tensor* hqm_tensor_find(const hqm* h, const char* name);
void* hqm_tensor_read(const hqm* h, const hqm_tensor* t, int64_t* outBytes);

void hqm_dir_of(const char* path, char* out, size_t cap);
void hqm_model_path(const model_config* spec, const char* weightDir, char* out, size_t cap);
int hqm_resolve(const model_config* spec, const char* weightDir, char* out, size_t cap);
int hqm_file_matches(const char* path, const model_config* spec);
uint64_t hqm_fingerprint(const model_config* spec);
void hqm_write_config(hqm_writer* w, const model_config* spec);

hqm_writer* hqm_writer_open(const char* path);
void hqm_writer_u32(hqm_writer* w, const char* key, uint32_t v);
void hqm_writer_u64(hqm_writer* w, const char* key, uint64_t v);
void hqm_writer_f64(hqm_writer* w, const char* key, double v);
void hqm_writer_bool(hqm_writer* w, const char* key, int v);
void hqm_writer_str(hqm_writer* w, const char* key, const char* s);
void hqm_writer_tensor(hqm_writer* w, const char* name, int type, const void* data, int64_t bytes, const int64_t* dims, int ndim);
void hqm_writer_str_array(hqm_writer* w, const char* name, const char* const* arr, int64_t count);
void hqm_writer_i32(hqm_writer* w, const char* name, const int32_t* data, int64_t count);
void hqm_writer_finish(hqm_writer* w);
void hqm_writer_abort(hqm_writer* w);

#endif
