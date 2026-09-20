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
    int64_t kvRamBudget;
    int64_t kvDiskBudget;
    const char* kvStoreDir;
} engine_options;

typedef struct {
    int enabled;
    uint64_t blocks;
    uint64_t entries;
    uint64_t snapshots;
    uint64_t hits;
    uint64_t coldHits;
    uint64_t restores;
    uint64_t evictions;
    uint64_t coldDeletes;
    int64_t usedBytes;
    int64_t ramBudget;
    int64_t diskBudget;
    int64_t coldBytes;
    double restoreMs;
} engine_kv_stats;

typedef struct engine engine;

typedef void (*engine_emit)(void* ctx, uint32_t token, const char* delta, size_t deltaLen);
typedef void (*engine_progress)(void* ctx, int done, int total, int chunkTokens, int chunkTotal, double lossSum,
                                long long count);

engine* engineOpen(const engine_options* opts, char* err, size_t errCap);
void engineClose(engine* e);

int engineTokenize(engine* e, const char* text, int addSpecial, uint32_t** out, size_t* outCount);
char* engineDecode(engine* e, const uint32_t* ids, size_t count);
void engineGenerate(engine* e, const uint32_t* prompt, size_t count, const sample_params* params,
                    uint32_t seed, int maxNew, engine_emit emit, void* ctx);
void engineScore(engine* e, const uint32_t* ids, size_t count, int prefillN, int decodeN, int chunks,
                 engine_progress progress, void* ctx, double* outLoss, long long* outCount);

int engineVocab(const engine* e);
int engineEos(const engine* e);
int engineMaxCtx(const engine* e);
void engineRequestStop(engine* e);
int engineFinishReason(const engine* e);
int engineLastResume(const engine* e);
void engineMemoryStats(int64_t* hostVisible, int64_t* deviceLocal);
void engineKvStats(const engine* e, engine_kv_stats* out);

#endif
