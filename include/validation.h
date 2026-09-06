#ifndef validation_h
#define validation_h

#include "session.h"
#include "data.h"

void validateGEMV(session s, int M, int N, int K, float* input, float* weight);
void validateGEMVFP16(session s, int M, int N, int K, float* input, uint16_t* weightFP16);
void validateGEMVINT8(session s, int M, int N, int K, float* input, QuantizedData weightINT8);
void validateGEMVINT4(session s, int M, int N, int K, float* input, QuantizedData weightINT4);
void validateGEMMFP16(session s, int M, int N, int K, float* input, uint16_t* weightFP16);
void validateGEMMINT8(session s, int M, int N, int K, float* input, QuantizedData weightINT8);
void validateGEMMINT4(session s, int M, int N, int K, float* input, QuantizedData weightINT4);
void validateRmsNormGEMVFP16(session s, int M, int N, int K, float* input, float* gamma, uint16_t* weightFP16);
void validateRmsNormGEMVINT8(session s, int M, int N, int K, float* input, float* gamma, QuantizedData weightINT8);
void validateRmsNormGEMVINT4(session s, int M, int N, int K, float* input, float* gamma, QuantizedData weightINT4);
void validateGemvAddFP16(session s, int M, int N, int K, float* input, float* residual, uint16_t* weightFP16);
void validateGemvAddINT8(session s, int M, int N, int K, float* input, float* residual, QuantizedData weightINT8);
void validateGemvAddINT4(session s, int M, int N, int K, float* input, float* residual, QuantizedData weightINT4);
void validateGemvSplitKFP16(session s, int M, int N, int K, float* input, float* residual, uint16_t* weightFP16);
void validateGemvSplitKINT8(session s, int M, int N, int K, float* input, float* residual, QuantizedData weightINT8);
void validateGemvSplitKINT4(session s, int M, int N, int K, float* input, float* residual, QuantizedData weightINT4);
void validateQkvRopeSplitKFP16(session s, int K, int qkv_heads, int qkv_kv_heads, int qkv_dim, float* input, float* gamma, uint16_t* qkv_weightFP16, float* qkv_theta);
void validateQkvRopeSplitKINT8(session s, int K, int qkv_heads, int qkv_kv_heads, int qkv_dim, float* input, float* gamma, QuantizedData qkv_weightINT8, float* qkv_theta);
void validateQkvRopeSplitKINT4(session s, int K, int qkv_heads, int qkv_kv_heads, int qkv_dim, float* input, float* gamma, QuantizedData qkv_weightINT4, float* qkv_theta);
void validateSwigluFfnSplitKFP16(session s, int M, int N, int K, float* input, float* gamma, uint16_t* gateW, uint16_t* upW, uint16_t* downW);
void validateSwigluFfnSplitKINT8(session s, int M, int N, int K, float* input, float* gamma, QuantizedData gateQ, QuantizedData upQ, QuantizedData downQ);
void validateSwigluFfnSplitKINT4(session s, int M, int N, int K, float* input, float* gamma, QuantizedData gateQ, QuantizedData upQ, QuantizedData downQ);
void validateLinearProjSplitKFP16(session s, int M, int K, float* input, float* gamma, uint16_t* w_inFP16);
void validateLinearProjSplitKINT8(session s, int M, int K, float* input, float* gamma, QuantizedData w_inINT8);
void validateLinearProjSplitKINT4(session s, int M, int K, float* input, float* gamma, QuantizedData w_inINT4);
void validateGemmAddFP16(session s, int M, int N, int K, float* input, float* residual, uint16_t* weightFP16);
void validateGemmAddINT8(session s, int M, int N, int K, float* input, float* residual, QuantizedData weightINT8);
void validateGemmAddINT4(session s, int M, int N, int K, float* input, float* residual, QuantizedData weightINT4);
void validateRmsNormSwigluFfn(session s, int M, int N, int K, float* input, float* gamma, float* weight);
void validateRmsNormSwigluFfnFP16(session s, int M, int N, int K, float* input, float* gamma, uint16_t* weightFP16, uint16_t* weight2FP16);
void validateRmsNormSwigluFfnINT8(session s, int M, int N, int K, float* input, float* gamma, QuantizedData weightINT8, QuantizedData weight2INT8);
void validateRmsNormSwigluFfnINT4(session s, int M, int N, int K, float* input, float* gamma, QuantizedData weightINT4, QuantizedData weight2INT4);
void validateRmsNormSwigluFfnGEMMFP16(session s, int M, int N, int K, float* input, float* gamma, uint16_t* weightFP16, uint16_t* weight2FP16);
void validateRmsNormSwigluFfnGEMMINT8(session s, int M, int N, int K, float* input, float* gamma, QuantizedData weightINT8, QuantizedData weight2INT8);
void validateRmsNormSwigluFfnGEMMINT4(session s, int M, int N, int K, float* input, float* gamma, QuantizedData weightINT4, QuantizedData weight2INT4);
void validateOnlineSoftmax(session s, int softmax_n, float* softmax_x, float* softmax_v);
void validateAttentionFP16(session s, int att_seq, int att_heads, int att_kv_heads, int att_dim, float* att_q, uint16_t* att_k, uint16_t* att_v);
void validateAttentionINT8(session s, int att_seq, int att_heads, int att_kv_heads, int att_dim, float* att_q, uint16_t* att_k, uint16_t* att_v);
void validateAttentionINT4(session s, int att_seq, int att_heads, int att_kv_heads, int att_dim, float* att_q, uint16_t* att_k, uint16_t* att_v);
void validateAttentionSplitKFP16(session s, int att_seq, int att_heads, int att_kv_heads, int att_dim, float* att_q, uint16_t* att_k, uint16_t* att_v);
void validateAttentionSplitKINT8(session s, int att_seq, int att_heads, int att_kv_heads, int att_dim, float* att_q, uint16_t* att_k, uint16_t* att_v);
void validateAttentionSplitKINT4(session s, int att_seq, int att_heads, int att_kv_heads, int att_dim, float* att_q, uint16_t* att_k, uint16_t* att_v);
void validateQkvRopeFP16(session s, int K, int qkv_heads, int qkv_kv_heads, int qkv_dim, float* input, float* gamma, uint16_t* qkv_weightFP16, float* qkv_theta);
void validateQkvRopeINT8(session s, int K, int qkv_heads, int qkv_kv_heads, int qkv_dim, float* input, float* gamma, QuantizedData qkv_weightINT8, float* qkv_theta);
void validateQkvRopeINT4(session s, int K, int qkv_heads, int qkv_kv_heads, int qkv_dim, float* input, float* gamma, QuantizedData qkv_weightINT4, float* qkv_theta);
void validateGatedDeltaNetFP16(session s, int K, float* input, float* input2, float* gamma, uint16_t* w_inFP16, uint16_t* woFP16);
void validateGatedDeltaNetINT8(session s, int K, float* input, float* input2, float* gamma, QuantizedData w_inINT8, QuantizedData woINT8);
void validateGatedDeltaNetINT4(session s, int K, float* input, float* input2, float* gamma, QuantizedData w_inINT4, QuantizedData woINT4);
void validateRmsNormLinearProjGEMMFP16(session s, int M, int K, float* input, float* gamma, uint16_t* w_inFP16);
void validateRmsNormLinearProjGEMMINT8(session s, int M, int K, float* input, float* gamma, QuantizedData w_inINT8);
void validateRmsNormLinearProjGEMMINT4(session s, int M, int K, float* input, float* gamma, QuantizedData w_inINT4);
void validateQkvRopeGEMMFP16(session s, int K, int qkv_heads, int qkv_kv_heads, int qkv_dim, float* input, float* gamma, uint16_t* qkv_weightFP16, float* qkv_theta, int M);
void validateQkvRopeGEMMINT8(session s, int K, int qkv_heads, int qkv_kv_heads, int qkv_dim, float* input, float* gamma, QuantizedData qkv_weightINT8, float* qkv_theta, int M);
void validateQkvRopeGEMMINT4(session s, int K, int qkv_heads, int qkv_kv_heads, int qkv_dim, float* input, float* gamma, QuantizedData qkv_weightINT4, float* qkv_theta, int M);
void validateAttentionGEMMFP16(session s, int seq, int heads, int kv_heads, int dim);
void validateAttentionGEMMINT8(session s, int seq, int heads, int kv_heads, int dim);
void validateAttentionGEMMINT4(session s, int seq, int heads, int kv_heads, int dim);
void validateGatedDeltaNetGEMMFP16(session s, int M, int K, float* input, float* gamma, uint16_t* w_inFP16, uint16_t* woFP16);
void validateGatedDeltaNetGEMMINT8(session s, int M, int K, float* input, float* gamma, QuantizedData w_inINT8, QuantizedData woINT8);
void validateGatedDeltaNetGEMMINT4(session s, int M, int K, float* input, float* gamma, QuantizedData w_inINT4, QuantizedData woINT4);
void validateLmHeadArgMaxFP16(session s, int vocabSize, int K, float* input, float* gamma, uint16_t* lmHeadFP16);
void validateArgMaxSampler(session s, int vocabSize);
void validateEmbedRmsNormLinearProjFP16(session s, int vocabSize, int K, uint32_t token, float* gamma, uint16_t* lmHeadFP16, uint16_t* w_inFP16);
void validateEmbedRmsNormLinearProjGEMMFP16(session s, int M, int vocabSize, int K, uint32_t* tokens, float* gamma, uint16_t* lmHeadFP16, uint16_t* w_inFP16);
void validateConvSilu(session s, int M);
void validateConvSiluChunked(session s, int M);
void validateRouterTopK(session s, int M);
void validateExpertSwigluINT4(session s, int M);
void validateExpertDownINT4(session s, int M);
void validateMoeCombine(session s, int M);
void validateMoeLayerINT4(session s, int M);

void validation(void);

#endif
