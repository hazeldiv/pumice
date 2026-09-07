#ifndef model_h
#define model_h

#include "data.h"

#define MODEL_MAX_LAYERS 64
#define MODEL_EOS 81896
#define MODEL_MAX_OPS 2048
#define MAX_PENALTY_LEN 1024

typedef enum {
    ATTENTION_NONE,
    ATTENTION_FULL,
    ATTENTION_DELTA
} attention_type;

typedef struct attention {
    attention_type type;
    QuantType q;
} attention;

typedef enum {
    FFN_NONE,
    FFN_SWIGLU,
    FFN_MOE
} ffn_type;

typedef struct ffn {
    ffn_type type;
    QuantType q;
} ffn;

typedef struct layer {
    attention attn;
    ffn ffn;
} layer;

typedef struct model_dims {
    int layerCount;
    int K;
    int qkvN;
    int projN;
    int ffnN;
    int heads;
    int kvHeads;
    int headDim;
    int rotaryDim;
    int rotaryHalf;
    int nQk;
    int nV;
    int dim;
    int kvRows;
    int qOff;
    int gOff;
    int kOff;
    int vOff;
    int projKOff;
    int projVOff;
    int projZOff;
    int projAOff;
    int projBOff;
    int zqkvN;
    int convHist;
    int experts;
    int expertsPerTok;
    int moeI;
    int maxCtx;
    int vocab;
    int eos;
    int prefillChunk;
    double ropeTheta;
    double partialRotary;
    int tied;
} model_dims;

typedef struct model_config {
    char name[128];
    char shaderDir[160];
    model_dims dims;
    layer layers[MODEL_MAX_LAYERS];
    QuantType embedQ;
    QuantType lmHeadQ;
    int expertsVram;
} model_config;

int loadModelConfig(model_config* cfg, const char* modelDir, int maxCtxOverride, int pruned);
int parseEos(model_dims* d, const char* modelDir, int pruned);
const char* model_shader(const char* base, QuantType q);

#endif
