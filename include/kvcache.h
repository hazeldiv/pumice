#ifndef kvcache_h
#define kvcache_h

#include <stdint.h>
#include <stdio.h>
#include "session.h"
#include "buffer.h"
#include "model.h"

#define KV_BLOCK_TOKENS 16
#define KV_SNAP_INTERVAL 1024
#define KV_SNAP_MAX 16
#define KV_STAGE_SLOTS 8

typedef struct {
    int layer;
    int quantized;
    int64_t kOff;
    int64_t vOff;
    int64_t kScaleOff;
    int64_t kZeroOff;
    int64_t vScaleOff;
    int64_t vZeroOff;
} kv_layer_layout;

typedef struct {
    uint64_t key;
    uint64_t child;
    int64_t coldOffset;
    int32_t slot;
    int32_t freq;
    int32_t pin;
    int32_t blockIndex;
    uint32_t lastUse;
    uint32_t crc;
    int32_t alive;
} kv_entry;

typedef struct {
    uint64_t key;
    int32_t slot;
    int32_t source;
} kv_restore_block;

typedef struct {
    int resume;
    int blockCount;
    int snapPos;
    const uint8_t* snapData;
    kv_restore_block* blocks;
} kv_plan;

typedef struct {
    uint64_t key;
    int32_t pos;
    int32_t used;
    uint32_t lastUse;
    uint8_t* data;
} kv_snapshot;

typedef struct kvcache {
    const model_dims* d;
    int fullCount;
    kv_layer_layout layout[MODEL_MAX_LAYERS];
    int64_t slotBytes;
    int64_t totalSlots;
    int64_t ramBudget;
    int64_t diskBudget;
    uint8_t* slotState;
    uint8_t* slotDirty;
    int32_t* freeSlots;
    int freeSlotCount;
    int nextSlot;
    int usedSlots;

    kv_entry* entries;
    int entryCount;
    int entryCap;
    int entryHigh;
    int32_t* table;
    int tableCap;
    int tableUsed;
    int32_t* freeEntries;
    int freeEntryCount;

    kv_snapshot snaps[KV_SNAP_MAX];
    int snapUsed;

    int deltaCount;
    int deltaLayers[MODEL_MAX_LAYERS];
    int64_t stateSBytes;
    int64_t convBytes;
    int64_t gdnBytes;

    buffer pool;
    buffer stage;
    int stageSlots;
    VkDevice device;

    FILE* cold;
    int64_t coldBytes;
    int64_t coldLive;
    int64_t coldDead;

    uint64_t seed0;
    char dir[512];
    char mirrorPath[600];
    char indexPath[600];
    char snapPath[600];
    char coldPath[600];

    uint32_t tick;
    int insertsSinceAge;

    uint64_t statBlocks;
    uint64_t statHits;
    uint64_t statColdHits;
    uint64_t statRestores;
    uint64_t statEvictions;
    uint64_t statColdDeletes;
    int64_t statRestoreBytes;
    double statRestoreMs;
} kvcache;

int kvOpen(kvcache* kv, session s, const model_config* spec, const char* root, int64_t ramBudget,
           int64_t diskBudget, int verbose);
void kvClose(kvcache* kv);
void kvChainHashes(const kvcache* kv, const uint32_t* tokens, int count, uint64_t* out);
int kvPlan(kvcache* kv, const uint64_t* hashes, int tokenCount, kv_plan* plan);
int kvColdLoad(kvcache* kv, uint64_t key, int stageSlot);
int kvAdmit(kvcache* kv, uint64_t key, int stageSlot);
void kvUnpin(kvcache* kv, kv_plan* plan);
void kvPlanFree(kv_plan* plan);
int kvHasBlock(const kvcache* kv, uint64_t key);
int kvBlockResident(const kvcache* kv, uint64_t key);
int kvAllocSlot(kvcache* kv);
void kvCommitBlock(kvcache* kv, uint64_t key, uint64_t parentKey, uint64_t childKey, int slot, int blockIndex);
int kvStoreSnapshot(kvcache* kv, const uint64_t* hashes, int pos, const void* data);
void kvPersist(kvcache* kv);
void kvAddRestoreTime(kvcache* kv, double ms);

#endif


