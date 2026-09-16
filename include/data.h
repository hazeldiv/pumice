#ifndef data_h
#define data_h

#include <stdlib.h>
#include <stdint.h>

typedef enum {
    QUANT_Q4_32 = 4,
    QUANT_Q4_64 = 5,
    QUANT_Q4_128 = 6,
    QUANT_Q4_256 = 7,
    QUANT_INT8 = 8,
    QUANT_FP16 = 16,
    QUANT_FP32 = 32
} QuantType;

static inline int quant_is_q4(QuantType q) { return q >= QUANT_Q4_32 && q <= QUANT_Q4_256; }
static inline int quant_block(QuantType q) { return quant_is_q4(q) ? 32 << (q - QUANT_Q4_32) : 0; }
static inline int quant_bits(QuantType q) { return quant_is_q4(q) ? 4 : (int)q; }
static inline int quant_scale_bytes(QuantType q) { return quant_is_q4(q) ? 2 : 4; }

typedef struct {
    uint8_t*   data;
    float*  scale;
    float*  z; 
    int     group_size;
    int     M;
    int     N;
    QuantType type;
} QuantizedData;

QuantizedData getDataINT8(int seed, int M, int N);
QuantizedData getDataQ4(int seed, int M, int N);

float* getData(int seed, int M, int N);
uint16_t* getDataFP16(int seed, int M, int N);
float fp16_to_float(uint16_t h);
uint16_t float_to_fp16(float f);
float bf16_to_float(uint16_t h);
uint16_t float_to_bf16(float f);
QuantizedData quantizeDataINT8(const float* src, int M, int N);
QuantizedData quantizeDataQ4(const float* src, int M, int N, QuantType q);
void transpose(const float* src, float* dest, int m, int n);
void transpose_block16(const uint8_t *input, uint8_t *output, int M, int N, QuantType q);
void free_quantized_data(QuantizedData q);

#endif
