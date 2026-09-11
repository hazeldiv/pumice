#ifndef data_h
#define data_h

#include <stdlib.h>
#include <stdint.h>

typedef enum {
    QUANT_INT4 = 4,
    QUANT_INT8 = 8,
    QUANT_FP16 = 16,
    QUANT_FP32 = 32
} QuantType;

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
QuantizedData getDataINT4(int seed, int M, int N);

float* getData(int seed, int M, int N);
uint16_t* getDataFP16(int seed, int M, int N);
float fp16_to_float(uint16_t h);
uint16_t float_to_fp16(float f);
float bf16_to_float(uint16_t h);
uint16_t float_to_bf16(float f);
QuantizedData quantizeDataINT8(const float* src, int M, int N);
QuantizedData quantizeDataINT4(const float* src, int M, int N);
void transpose(const float* src, float* dest, int m, int n);
void transpose_block16(const uint8_t *input, uint8_t *output, int M, int N, int data_type);
void free_quantized_data(QuantizedData q);

#endif
