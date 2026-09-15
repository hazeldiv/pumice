#ifndef engine_h
#define engine_h

#include <stddef.h>
#include <stdint.h>
#include "model.h"
#include "state.h"

typedef struct {
    const char* weights;
    const char* quantConfig;
    int maxCtx;
    int prune;
    int expertsVram;
    int exportModel;
    const char* exportDir;
} engine_options;

typedef struct engine engine;

typedef void (*engine_emit)(void* ctx, uint32_t token, const char* delta, size_t deltaLen);
typedef void (*engine_progress)(void* ctx, int done, int total, double lossSum, long long count);

engine* engineOpen(const engine_options* opts, char* err, size_t errCap);
void engineClose(engine* e);

int engineTokenize(engine* e, const char* text, int addSpecial, uint32_t** out, size_t* outCount);
char* engineDecode(engine* e, const uint32_t* ids, size_t count);
void engineGenerate(engine* e, const uint32_t* prompt, size_t count, const sample_params* params,
                    uint32_t seed, int maxNew, engine_emit emit, void* ctx);
void engineScore(engine* e, const uint32_t* ids, int prefillN, int decodeN, int chunks,
                 engine_progress progress, void* ctx, double* outLoss, long long* outCount);

int engineVocab(const engine* e);
int engineEos(const engine* e);
int engineMaxCtx(const engine* e);
void engineRequestStop(engine* e);

#endif
