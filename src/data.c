#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "../include/data.h"

float get_pseudo_random(int i, int j, int seed) {
    unsigned int hash = (unsigned int)(i * 73129 + j * 95121 + seed * 15473);
    hash = (hash ^ 61) ^ (hash >> 16);
    hash += (hash << 3);
    hash ^= (hash >> 4);
    hash *= 0x27d4eb2d;
    hash ^= (hash >> 15);
    return ((float)(hash % 2000001) / 1000000.0f) - 1.0f;
}

float* getData(int seed, int M, int N) {
    float* A = (float*)malloc(sizeof(float) * M * N);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = get_pseudo_random(i, j, seed);
        }
    }

    return A;
}

uint16_t float_to_fp16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xff) - 127;
    uint32_t mant = x & 0x7FFFFF;
    
    if (exp <= -15) {
        if (exp < -24) {
            return sign;
        }
        mant = mant | 0x800000;
        return sign | (mant >> (1 - exp));
    }
    
    if (exp >= 16) {
        return sign | 0x7C00;
    }
    
    return sign | ((exp + 15) << 10) | (mant >> 13);
}

float fp16_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    int32_t exp = (h >> 10) & 0x001F;
    uint32_t mant = h & 0x03FF;
    
    uint32_t res_bits;

    if (exp == 0) {
        if (mant == 0) {
            res_bits = sign;
        } else {
            int32_t e = -14;
            while (!(mant & 0x0400)) {
                mant <<= 1;
                e--;
            }
            mant &= ~0x0400;
            res_bits = sign | ((e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        res_bits = sign | (0xFF << 23) | (mant << 13);
    } else {
        res_bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &res_bits, sizeof(f));
    return f;
}

uint16_t* getDataFP16(int seed, int M, int N) {
    uint16_t* A = (uint16_t*)malloc(sizeof(uint16_t) * M * N);

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float val = get_pseudo_random(i, j, seed);
            A[i * N + j] = float_to_fp16(val);
        }
    }
    return A;
}

QuantizedData getDataINT8(int seed, int M, int N) {
    QuantizedData q = {0};
    q.M          = M;
    q.N          = N;
    q.group_size = 256;
    q.type       = QUANT_INT8;

    int blocks_per_row = (N + q.group_size - 1) / q.group_size;
    int blocks_count   = M * blocks_per_row;

    q.data   = (uint8_t*)malloc(sizeof(int8_t) * M * N);
    q.scale  = (float*)malloc(sizeof(float) * blocks_count);
    q.z      = (float*)malloc(sizeof(float) * blocks_count);

    for (int i = 0; i < M; i++) {
        for (int bj = 0; bj < blocks_per_row; bj++) {
            float min_val = 1e9f;
            float max_val = -1e9f;

            int base_j = bj * q.group_size;
            for (int k = 0; k < q.group_size; k++) {
                int j = base_j + k;
                if (j >= N) break;
                float val = get_pseudo_random(i, j, seed);
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
            }
            int block_idx = bj * M + i;
            q.scale[block_idx] = (max_val - min_val) / 255.0f;
            q.z[block_idx] = -min_val;
            
            for (int k = 0; k < q.group_size; k++) {
                int j = base_j + k;
                if (j >= N) break;
                float val = get_pseudo_random(i, j, seed);
                uint8_t qv = (uint8_t)roundf((val + q.z[block_idx]) / q.scale[block_idx]);
                q.data[i * N + j] = qv;
            }
        }
    }

    return q;
}

QuantizedData getDataINT4(int seed, int M, int N) {
    QuantizedData q = {0};
    q.M          = M;
    q.N          = N;
    q.group_size = 256;
    q.type       = QUANT_INT4;

    int blocks_per_row = (N + q.group_size - 1) / q.group_size;
    int blocks_count   = M * blocks_per_row;

    q.data   = (uint8_t*)malloc(sizeof(uint8_t) * M * N / 2);
    q.scale  = (float*)malloc(sizeof(float) * blocks_count);
    q.z      = (float*)malloc(sizeof(float) * blocks_count);

    for (int i = 0; i < M; i++) {
        for (int bj = 0; bj < blocks_per_row; bj++) {
            float min_val = 1e9f;
            float max_val = -1e9f;

            int base_j = bj * q.group_size;
            for (int k = 0; k < q.group_size; k++) {
                int j = base_j + k;
                if (j >= N) break;
                float val = get_pseudo_random(i, j, seed);
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
            }
            
            int block_idx = bj * M + i;
            q.scale[block_idx] = (max_val - min_val) / 15.0f;
            q.z[block_idx] = -min_val;

            for (int k = 0; k < q.group_size; k += 2) {
                int j0 = base_j + k;
                int j1 = base_j + k + 1;

                uint8_t q0 = (uint8_t)roundf((get_pseudo_random(i, j0, seed) + q.z[block_idx]) / q.scale[block_idx]);
                uint8_t q1 = (j1 < N) ? (uint8_t)roundf((get_pseudo_random(i, j1, seed) + q.z[block_idx]) / q.scale[block_idx]) : 0;

                int packed_idx = i * (N / 2) + (bj * q.group_size + k) / 2;
                q.data[packed_idx] = ((q0 & 0x0F) << 4) | ((q1 & 0x0F) << 0);
            }
        }
    }

    return q;
}

float bf16_to_float(uint16_t h) {
    uint32_t u = ((uint32_t)h) << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

uint16_t float_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return (uint16_t)(u >> 16);
}

static float clampf(float v, float lo, float hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

QuantizedData quantizeDataINT8(const float* A, int M, int N) {
    QuantizedData q = {0};
    q.M = M;
    q.N = N;
    q.group_size = 256;
    q.type = QUANT_INT8;

    int blocks_per_row = (N + q.group_size - 1) / q.group_size;
    int blocks_count = M * blocks_per_row;

    q.data = (uint8_t*)malloc(sizeof(uint8_t) * M * N);
    q.scale = (float*)malloc(sizeof(float) * blocks_count);
    q.z = (float*)malloc(sizeof(float) * blocks_count);

    for (int i = 0; i < M; i++) {
        for (int bj = 0; bj < blocks_per_row; bj++) {
            float min_val = 1e9f;
            float max_val = -1e9f;
            int base_j = bj * q.group_size;
            for (int k = 0; k < q.group_size; k++) {
                int j = base_j + k;
                if (j >= N) break;
                float val = A[i * N + j];
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
            }
            int block_idx = bj * M + i;
            q.scale[block_idx] = (max_val - min_val) / 255.0f;
            q.z[block_idx] = -min_val;

            for (int k = 0; k < q.group_size; k++) {
                int j = base_j + k;
                if (j >= N) break;
                float v = clampf((A[i * N + j] + q.z[block_idx]) / q.scale[block_idx], 0.0f, 255.0f);
                q.data[i * N + j] = (uint8_t)roundf(v);
            }
        }
    }

    return q;
}

QuantizedData quantizeDataINT4(const float* A, int M, int N) {
    QuantizedData q = {0};
    q.M = M;
    q.N = N;
    q.group_size = 256;
    q.type = QUANT_INT4;

    int blocks_per_row = (N + q.group_size - 1) / q.group_size;
    int blocks_count = M * blocks_per_row;

    q.data = (uint8_t*)malloc(sizeof(uint8_t) * M * N / 2);
    q.scale = (float*)malloc(sizeof(float) * blocks_count);
    q.z = (float*)malloc(sizeof(float) * blocks_count);

    for (int i = 0; i < M; i++) {
        for (int bj = 0; bj < blocks_per_row; bj++) {
            float min_val = 1e9f;
            float max_val = -1e9f;
            int base_j = bj * q.group_size;
            for (int k = 0; k < q.group_size; k++) {
                int j = base_j + k;
                if (j >= N) break;
                float val = A[i * N + j];
                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
            }
            int block_idx = bj * M + i;
            q.scale[block_idx] = (max_val - min_val) / 15.0f;
            q.z[block_idx] = -min_val;

            for (int k = 0; k < q.group_size; k += 2) {
                int j0 = base_j + k;
                int j1 = base_j + k + 1;
                uint8_t v0 = (uint8_t)clampf(roundf((A[i * N + j0] + q.z[block_idx]) / q.scale[block_idx]), 0.0f, 15.0f);
                uint8_t v1 = (j1 < N) ? (uint8_t)clampf(roundf((A[i * N + j1] + q.z[block_idx]) / q.scale[block_idx]), 0.0f, 15.0f) : 0;
                int packed_idx = i * (N / 2) + (bj * q.group_size + k) / 2;
                q.data[packed_idx] = ((v0 & 0x0F) << 4) | (v1 & 0x0F);
            }
        }
    }

    return q;
}

void transpose(const float* src, float* dest, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int src_idx = i * n + j;
            int dest_idx = j * m + i;
            
            dest[dest_idx] = src[src_idx];
        }
    }
}

void transpose_block16(const uint8_t *input, uint8_t *output, int M, int N, int data_type) {
    int out_N = N * 16;
    for (int b = 0; b < M; b += 128/data_type) {
        int out_row = b / (128/data_type);
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < 128/data_type; k++) {
                if (data_type == 4) {
                    int in_idx = (b + k) * (N / 2) + j/2;
                    int out_idx = out_row * out_N + (j * 16) + (k / 2);
                    if (k%2==0) {
                        output[out_idx] = j%2==0 ? ((input[in_idx] & 0xF0) >> 4) : (input[in_idx] & 0x0F);
                    } else {
                        output[out_idx] = output[out_idx] | (j%2==0 ? (input[in_idx] & 0xF0) : ((input[in_idx] & 0x0F) << 4));
                    }
                } else {
                    for (int l=0; l < data_type/8; l++) {
                        int in_idx = (b + k) * (N * (data_type/8)) + j*(data_type/8) + l;
                        int out_idx = out_row * out_N + (j * 16) + (k*(data_type/8)) + l;
                        output[out_idx] = input[in_idx];
                    }
                }
            }
        }
    }
}

void free_quantized_data(QuantizedData q) {
    if (q.data) {
        free(q.data);
    }
    if (q.scale) {
        free(q.scale);
    }
    if (q.z) {
        free(q.z);
    }
}