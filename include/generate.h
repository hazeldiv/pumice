#ifndef generate_h
#define generate_h

#include <stdio.h>
#include "session.h"
#include "model.h"
#include "weights.h"
#include "state.h"
#include "dispatch.h"
#include "kvcache.h"

#define DECODE_GROUP 4

typedef struct generator {
    session s;
    model_weights w;
    model_state st;
    const model_config* spec;
    const model_dims* dims;
    int maxM;
    int maxCtx;
    int eos;
    int vocab;
    int layerCount;
    operation groupOps[MODEL_MAX_OPS];
    int groupOpCount;
    operation groupOpsShort[MODEL_MAX_OPS];
    int groupOpCountShort;
    operation prefillOps[MODEL_MAX_OPS];
    int prefillOpCount;
    operation finalOps[MODEL_MAX_OPS];
    int finalOpCount;
    uint32_t nextPos;
    char dumpDir[256];
    int dumpLayers;
    int sampling;
    char dumpHiddenDir[256];
    int dumpHiddenReq;
    char dumpTopPPath[512];
    FILE* dumpTopPFile;
    uint32_t dumpTopPStep;
    volatile int stop;
    int skipFinal;
    int finishReason;
    void (*boundaryHook)(void* ctx, int pos);
    void* boundaryHookCtx;
    int boundaryInterval;
} generator;

generator* createGenerator(session s, const model_config* spec, const char* weightDir, int verboseWeights);
void destroyGenerator(generator* g);
uint32_t runPrefill(generator* g, const uint32_t* tokens, int nTokens);
void generateTokens(generator* g, const uint32_t* prompt, int nPrompt, int maxNewTokens, void (*emit)(uint32_t token, void* ctx), void* ctx);
void generateScore(generator* g, const uint32_t* ids, size_t idCount, int prefillN, int decodeN, int chunks,
                   void (*progress)(void* ctx, int done, int total, int chunkTokens, int chunkTotal,
                                    double lossSum, long long count),
                   void* ctx, double* outLoss, long long* outCount);
void generatorRequestStop(generator* g);
int generatorFinishReason(const generator* g);
void generatorSetScoring(generator* g, int enabled);
void resetGenerator(generator* g);
void generatorKvUnstripe(generator* g, kvcache* kv, buffer src, const int* blockIndices, const int* slots, int count);
int generatorKvRestoreCold(generator* g, kvcache* kv, const kv_restore_block* blocks, int count);
void generatorKvRestripe(generator* g, kvcache* kv, const int* blockIndices, const int* slots, int count);
void generatorReadGdn(generator* g, kvcache* kv, void* out);
void generatorWriteGdn(generator* g, kvcache* kv, const void* in);
void generatorSetDumpDir(generator* g, const char* dir);
void generatorDumpPrefill(generator* g, int rows);
void generatorSetDumpLayers(generator* g, int layers);
void generatorDumpDecodeStep(generator* g, int step);
void generatorSetSampling(generator* g, const sample_params* p, uint32_t seed);
void generatorSetDumpHidden(generator* g, const char* dir, int reqIdx);
void generatorSetDumpTopP(generator* g, const char* path);
void generatorCloseDumpTopP(generator* g);
void generatorDumpSamplingDebug(generator* g, const uint32_t* generated, int nGen, int nPrompt);

#endif
