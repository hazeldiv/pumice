#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <direct.h>
#include "kvcache.h"
#include "data.h"
#include "xxhash.h"

#define KV_INDEX_MAGIC 0x4349564Bu
#define KV_INDEX_VERSION 3u
#define KV_COLD_DEAD 1u

typedef struct {
    uint32_t magic;
    uint32_t version;
    int64_t slotBytes;
    int64_t totalSlots;
    int64_t coldBytes;
    int64_t coldLive;
    int64_t coldDead;
    int32_t nextSlot;
    int32_t entryCount;
    int32_t snapPos;
    uint32_t tick;
    uint64_t snapKey;
} kv_index_header;

typedef struct {
    uint64_t key;
    uint64_t child;
    int64_t coldOffset;
    int32_t slot;
    int32_t freq;
    int32_t blockIndex;
    uint32_t lastUse;
    uint32_t crc;
} kv_index_record;

typedef struct {
    uint64_t key;
    int64_t bytes;
    uint32_t crc;
    uint32_t flags;
} kv_cold_record;

typedef struct {
    uint32_t magic;
    uint32_t count;
    int64_t gdnBytes;
} kv_snaps_header;

typedef struct {
    int32_t pos;
    uint32_t lastUse;
    uint64_t key;
} kv_snap_record;

#define KV_SNAPS_MAGIC 0x534E4150u

static uint32_t g_crcTable[256];
static int g_crcReady = 0;

static void crcBuild(void) {
    for (uint32_t i = 0; i < 256; i++) {
        uint32_t c = i;
        for (int k = 0; k < 8; k++) c = (c & 1u) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        g_crcTable[i] = c;
    }
    g_crcReady = 1;
}

static uint32_t crc32b(const void* data, size_t n) {
    if (!g_crcReady) crcBuild();
    const uint8_t* p = (const uint8_t*)data;
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < n; i++) c = g_crcTable[(c ^ p[i]) & 0xFFu] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

static uint32_t slotCrc(const kvcache* kv, int slot) {
    const uint8_t* p = (const uint8_t*)kv->pool.mappedMemory + (int64_t)slot * kv->slotBytes;
    return crc32b(p, (size_t)kv->slotBytes);
}

static int slotValid(const kvcache* kv, const kv_entry* e) {
    if (e->slot < 0 || e->slot >= kv->totalSlots) return 0;
    return slotCrc(kv, e->slot) == e->crc;
}

static int64_t hostHeapBytes(VkPhysicalDevice physicalDevice) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &mp);
    int64_t best = 0;
    for (uint32_t i = 0; i < mp.memoryHeapCount; i++) {
        if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) continue;
        int64_t size = (int64_t)mp.memoryHeaps[i].size;
        if (size > best) best = size;
    }
    return best;
}


static int ensureDir(const char* path) {
    if (_mkdir(path) == 0) return 1;
    struct _stat st;
    if (_stat(path, &st) == 0 && (st.st_mode & _S_IFDIR)) return 1;
    return 0;
}

static int tableFind(const kvcache* kv, uint64_t key) {
    if (kv->tableCap == 0) return -1;
    uint32_t mask = (uint32_t)kv->tableCap - 1u;
    uint32_t i = (uint32_t)(key * 0x9E3779B97F4A7C15ull) & mask;
    for (uint32_t n = 0; n < (uint32_t)kv->tableCap; n++) {
        int32_t v = kv->table[i];
        if (v == -1) return -1;
        if (v >= 0 && kv->entries[v].key == key) return v;
        i = (i + 1u) & mask;
    }
    return -1;
}

static void tableGrow(kvcache* kv) {
    int newCap = kv->tableCap ? kv->tableCap * 2 : 2048;
    int32_t* t = (int32_t*)malloc(sizeof(int32_t) * (size_t)newCap);
    for (int i = 0; i < newCap; i++) t[i] = -1;
    uint32_t mask = (uint32_t)newCap - 1u;
    for (int e = 0; e < kv->entryHigh; e++) {
        if (!kv->entries[e].alive) continue;
        uint32_t j = (uint32_t)(kv->entries[e].key * 0x9E3779B97F4A7C15ull) & mask;
        while (t[j] >= 0) j = (j + 1u) & mask;
        t[j] = e;
    }
    free(kv->table);
    kv->table = t;
    kv->tableCap = newCap;
    kv->tableUsed = kv->entryCount;
}

static void tableInsert(kvcache* kv, int idx) {
    if ((kv->tableUsed + 1) * 10 > kv->tableCap * 7) tableGrow(kv);
    uint64_t key = kv->entries[idx].key;
    uint32_t mask = (uint32_t)kv->tableCap - 1u;
    uint32_t i = (uint32_t)(key * 0x9E3779B97F4A7C15ull) & mask;
    while (kv->table[i] != -1 && kv->table[i] != -2) {
        if (kv->entries[kv->table[i]].key == key) {
            kv->table[i] = idx;
            return;
        }
        i = (i + 1u) & mask;
    }
    kv->table[i] = idx;
    kv->tableUsed++;
}

static void tableRemove(kvcache* kv, uint64_t key) {
    if (kv->tableCap == 0) return;
    uint32_t mask = (uint32_t)kv->tableCap - 1u;
    uint32_t i = (uint32_t)(key * 0x9E3779B97F4A7C15ull) & mask;
    for (uint32_t n = 0; n < (uint32_t)kv->tableCap; n++) {
        int32_t v = kv->table[i];
        if (v == -1) return;
        if (v >= 0 && kv->entries[v].key == key) {
            kv->table[i] = -2;
            kv->tableUsed--;
            return;
        }
        i = (i + 1u) & mask;
    }
}

static void growEntries(kvcache* kv, int newCap) {
    kv->entries = (kv_entry*)realloc(kv->entries, sizeof(kv_entry) * (size_t)newCap);
    kv->freeEntries = (int32_t*)realloc(kv->freeEntries, sizeof(int32_t) * (size_t)newCap);
    kv->entryCap = newCap;
}

static int entryAlloc(kvcache* kv, uint64_t key) {
    int idx;
    if (kv->freeEntryCount > 0) {
        idx = kv->freeEntries[--kv->freeEntryCount];
    } else {
        if (kv->entryHigh + 1 > kv->entryCap) growEntries(kv, kv->entryCap ? kv->entryCap * 2 : 1024);
        idx = kv->entryHigh++;
    }
    kv_entry* e = &kv->entries[idx];
    memset(e, 0, sizeof(*e));
    e->key = key;
    e->slot = -1;
    e->coldOffset = -1;
    e->alive = 1;
    kv->entryCount++;
    return idx;
}

static void entryRemove(kvcache* kv, int idx) {
    kv_entry* e = &kv->entries[idx];
    if (!e->alive) return;
    tableRemove(kv, e->key);
    e->alive = 0;
    kv->entryCount--;
    kv->freeEntries[kv->freeEntryCount++] = idx;
}

static void buildFingerprint(const model_config* spec, uint64_t* out) {
    const model_dims* d = &spec->dims;
    char sig[1024];
    int n = snprintf(sig, sizeof(sig), "%s|V%d|L%d|K%d|kv%d|hd%d|r%d|q", spec->name, d->vocab,
                     d->layerCount, d->K, d->kvHeads, d->headDim, d->kvRows);
    for (int L = 0; L < d->layerCount; L++) sig[n++] = (char)('0' + (int)spec->layers[L].attn.q);
    sig[n] = '\0';
    *out = XXH3_64bits(sig, (size_t)n);
}

static void buildLayout(kvcache* kv, const model_config* spec) {
    const model_dims* d = kv->d;
    int64_t cur = 0;
    kv->fullCount = 0;
    for (int L = 0; L < d->layerCount; L++) {
        if (spec->layers[L].attn.type != ATTENTION_FULL) continue;
        kv_layer_layout* ly = &kv->layout[kv->fullCount];
        memset(ly, 0, sizeof(*ly));
        ly->layer = L;
        ly->quantized = spec->layers[L].attn.q != QUANT_FP16;
        int64_t elem = ly->quantized ? 1 : 2;
        ly->kOff = cur;
        cur += (int64_t)d->kvRows * KV_BLOCK_TOKENS * elem;
        ly->vOff = cur;
        cur += (int64_t)KV_BLOCK_TOKENS * d->kvRows * elem;
        if (ly->quantized) {
            int64_t sb = (int64_t)d->kvHeads * KV_BLOCK_TOKENS * 4;
            ly->kScaleOff = cur; cur += sb;
            ly->kZeroOff = cur; cur += sb;
            ly->vScaleOff = cur; cur += sb;
            ly->vZeroOff = cur; cur += sb;
        }
        kv->fullCount++;
    }
    kv->slotBytes = cur;
    if (kv->slotBytes < 1) kv->slotBytes = 1;
}

static void buildGdn(kvcache* kv, const model_config* spec) {
    const model_dims* d = kv->d;
    kv->stateSBytes = (int64_t)d->nV * d->dim * d->dim * 2;
    kv->convBytes = (int64_t)d->convHist * d->zqkvN * 4;
    int64_t off = 0;
    kv->deltaCount = 0;
    for (int L = 0; L < d->layerCount; L++) {
        if (spec->layers[L].attn.type != ATTENTION_DELTA) continue;
        kv->deltaLayers[kv->deltaCount++] = L;
        off += kv->stateSBytes + kv->convBytes;
    }
    kv->gdnBytes = off;
}

static void freeSlot(kvcache* kv, int slot) {
    if (slot < 0 || slot >= kv->totalSlots) return;
    if (!kv->slotState[slot]) return;
    kv->slotState[slot] = 0;
    kv->slotDirty[slot] = 0;
    kv->freeSlots[kv->freeSlotCount++] = slot;
    kv->usedSlots--;
}

static int64_t coldAppend(kvcache* kv, uint64_t key, const uint8_t* data) {
    if (kv->cold == NULL) return -1;
    _fseeki64(kv->cold, 0, SEEK_END);
    int64_t pos = _ftelli64(kv->cold);
    kv_cold_record r;
    r.key = key;
    r.bytes = kv->slotBytes;
    r.crc = crc32b(data, (size_t)kv->slotBytes);
    r.flags = 0;
    if (fwrite(&r, sizeof(r), 1, kv->cold) != 1) return -1;
    if (fwrite(data, 1, (size_t)kv->slotBytes, kv->cold) != (size_t)kv->slotBytes) return -1;
    fflush(kv->cold);
    int64_t total = (int64_t)sizeof(r) + kv->slotBytes;
    kv->coldBytes += total;
    kv->coldLive += total;
    return pos;
}

static int coldRead(kvcache* kv, int64_t off, uint8_t* dst) {
    if (kv->cold == NULL || off < 0) return 0;
    if (_fseeki64(kv->cold, off, SEEK_SET) != 0) return 0;
    kv_cold_record r;
    if (fread(&r, sizeof(r), 1, kv->cold) != 1) return 0;
    if ((r.flags & KV_COLD_DEAD) || r.bytes != kv->slotBytes) return 0;
    if (fread(dst, 1, (size_t)kv->slotBytes, kv->cold) != (size_t)kv->slotBytes) return 0;
    if (crc32b(dst, (size_t)kv->slotBytes) != r.crc) return 0;
    return 1;
}

static void coldMarkDead(kvcache* kv, int64_t off) {
    if (kv->cold == NULL || off < 0) return;
    if (_fseeki64(kv->cold, off, SEEK_SET) != 0) return;
    kv_cold_record r;
    if (fread(&r, sizeof(r), 1, kv->cold) != 1) return;
    if (r.flags & KV_COLD_DEAD) return;
    r.flags |= KV_COLD_DEAD;
    _fseeki64(kv->cold, off, SEEK_SET);
    fwrite(&r, sizeof(r), 1, kv->cold);
    fflush(kv->cold);
    int64_t total = (int64_t)sizeof(r) + r.bytes;
    kv->coldLive -= total;
    kv->coldDead += total;
}

static int findVictim(const kvcache* kv, int cold) {
    int best = -1, bestLeaf = -1;
    int64_t bestScore = 0, bestLeafScore = 0;
    for (int i = 0; i < kv->entryHigh; i++) {
        const kv_entry* e = &kv->entries[i];
        if (!e->alive || e->pin > 0) continue;
        if (cold) {
            if (e->coldOffset < 0) continue;
        } else {
            if (e->slot < 0) continue;
        }
        int leaf = (e->child == 0) || (tableFind(kv, e->child) < 0);
        int64_t score = ((int64_t)e->freq << 32) | (int64_t)e->lastUse;
        if (leaf) {
            if (bestLeaf < 0 || score < bestLeafScore) { bestLeaf = i; bestLeafScore = score; }
        } else {
            if (best < 0 || score < bestScore) { best = i; bestScore = score; }
        }
    }
    return bestLeaf >= 0 ? bestLeaf : best;
}

static void demoteEntry(kvcache* kv, int idx) {
    kv_entry* e = &kv->entries[idx];
    if (e->slot < 0) return;
    int64_t off = coldAppend(kv, e->key, (const uint8_t*)kv->pool.mappedMemory + (int64_t)e->slot * kv->slotBytes);
    if (off >= 0) e->coldOffset = off;
    freeSlot(kv, e->slot);
    e->slot = -1;
    kv->statEvictions++;
}

static void evictCold(kvcache* kv) {
    int idx = findVictim(kv, 1);
    if (idx < 0) return;
    kv_entry* e = &kv->entries[idx];
    coldMarkDead(kv, e->coldOffset);
    e->coldOffset = -1;
    entryRemove(kv, idx);
    kv->statColdDeletes++;
}

static void enforceDiskBudget(kvcache* kv) {
    if (kv->diskBudget <= 0) return;
    while (kv->coldBytes > kv->diskBudget) {
        int before = (int)kv->statColdDeletes;
        evictCold(kv);
        if ((int)kv->statColdDeletes == before) break;
    }
}

static void compactCold(kvcache* kv) {
    if (kv->cold == NULL || kv->coldDead == 0) return;
    char tmp[640];
    snprintf(tmp, sizeof(tmp), "%s.tmp", kv->coldPath);
    FILE* out = fopen(tmp, "wb");
    if (out == NULL) return;
    uint8_t* buf = (uint8_t*)malloc((size_t)kv->slotBytes);
    if (buf == NULL) {
        fclose(out);
        return;
    }
    int64_t live = 0;
    for (int i = 0; i < kv->entryHigh; i++) {
        kv_entry* e = &kv->entries[i];
        if (!e->alive || e->coldOffset < 0) continue;
        if (!coldRead(kv, e->coldOffset, buf)) {
            e->coldOffset = -1;
            continue;
        }
        kv_cold_record r;
        r.key = e->key;
        r.bytes = kv->slotBytes;
        r.crc = crc32b(buf, (size_t)kv->slotBytes);
        r.flags = 0;
        int64_t pos = _ftelli64(out);
        fwrite(&r, sizeof(r), 1, out);
        fwrite(buf, 1, (size_t)kv->slotBytes, out);
        e->coldOffset = pos;
        live += (int64_t)sizeof(r) + kv->slotBytes;
    }
    free(buf);
    fclose(out);
    fclose(kv->cold);
    remove(kv->coldPath);
    rename(tmp, kv->coldPath);
    kv->cold = fopen(kv->coldPath, "r+b");
    if (kv->cold == NULL) kv->cold = fopen(kv->coldPath, "w+b");
    kv->coldBytes = live;
    kv->coldLive = live;
    kv->coldDead = 0;
}

int kvAllocSlot(kvcache* kv) {
    for (;;) {
        if (kv->freeSlotCount > 0) {
            int s = kv->freeSlots[--kv->freeSlotCount];
            kv->slotState[s] = 1;
            kv->usedSlots++;
            return s;
        }
        if (kv->nextSlot < kv->totalSlots) {
            int s = kv->nextSlot++;
            kv->slotState[s] = 1;
            kv->usedSlots++;
            return s;
        }
        int victim = findVictim(kv, 0);
        if (victim < 0) return -1;
        demoteEntry(kv, victim);
    }
}

static kv_snapshot* snapFind(kvcache* kv, int pos, uint64_t key) {
    for (int i = 0; i < KV_SNAP_MAX; i++) {
        kv_snapshot* s = &kv->snaps[i];
        if (s->used && s->pos == pos && s->key == key) return s;
    }
    return NULL;
}

static kv_snapshot* snapAlloc(kvcache* kv) {
    for (int i = 0; i < KV_SNAP_MAX; i++) {
        if (!kv->snaps[i].used) return &kv->snaps[i];
    }
    int best = 0;
    for (int i = 1; i < KV_SNAP_MAX; i++) {
        if (kv->snaps[i].lastUse < kv->snaps[best].lastUse) best = i;
    }
    return &kv->snaps[best];
}

static void loadSnapshots(kvcache* kv) {
    if (kv->gdnBytes <= 0) return;
    FILE* f = fopen(kv->snapPath, "rb");
    if (f == NULL) return;
    kv_snaps_header h;
    if (fread(&h, sizeof(h), 1, f) != 1 || h.magic != KV_SNAPS_MAGIC || h.gdnBytes != kv->gdnBytes) {
        fclose(f);
        return;
    }
    for (uint32_t i = 0; i < h.count && i < KV_SNAP_MAX; i++) {
        kv_snap_record r;
        if (fread(&r, sizeof(r), 1, f) != 1) break;
        kv_snapshot* s = &kv->snaps[i];
        s->data = (uint8_t*)malloc((size_t)kv->gdnBytes);
        if (s->data == NULL) break;
        if (fread(s->data, 1, (size_t)kv->gdnBytes, f) != (size_t)kv->gdnBytes) break;
        s->pos = r.pos;
        s->key = r.key;
        s->used = 1;
        s->lastUse = r.lastUse;
        kv->snapUsed++;
        if (r.lastUse > kv->tick) kv->tick = r.lastUse;
    }
    fclose(f);
}

static void loadIndex(kvcache* kv) {
    FILE* f = fopen(kv->indexPath, "rb");
    if (f == NULL) return;
    kv_index_header h;
    if (fread(&h, sizeof(h), 1, f) != 1 || h.magic != KV_INDEX_MAGIC || h.version != KV_INDEX_VERSION ||
        h.slotBytes != kv->slotBytes || h.totalSlots != kv->totalSlots) {
        fclose(f);
        return;
    }
    kv->nextSlot = h.nextSlot;
    kv->coldBytes = h.coldBytes;
    kv->coldLive = h.coldLive;
    kv->coldDead = h.coldDead;
    kv->tick = h.tick;
    for (int i = 0; i < h.entryCount; i++) {
        kv_index_record r;
        if (fread(&r, sizeof(r), 1, f) != 1) break;
        int idx = entryAlloc(kv, r.key);
        kv_entry* e = &kv->entries[idx];
        e->child = r.child;
        e->coldOffset = r.coldOffset;
        e->slot = r.slot;
        e->freq = r.freq;
        e->blockIndex = r.blockIndex;
        e->lastUse = r.lastUse;
        e->crc = r.crc;
        tableInsert(kv, idx);
        if (e->slot >= 0 && e->slot < kv->totalSlots) {
            kv->slotState[e->slot] = 1;
            if (e->slot + 1 > kv->nextSlot) kv->nextSlot = e->slot + 1;
        }
    }
    fclose(f);
    for (int i = 0; i < kv->nextSlot; i++) {
        if (kv->slotState[i]) kv->usedSlots++;
    }
    FILE* m = fopen(kv->mirrorPath, "rb");
    if (m != NULL) {
        int64_t want = (int64_t)kv->nextSlot * kv->slotBytes;
        if (kv->pool.mappedMemory != NULL && want > 0) {
            if (fread(kv->pool.mappedMemory, 1, (size_t)want, m) != (size_t)want) {
            }
        }
        fclose(m);
    }
}

int kvOpen(kvcache* kv, session s, const model_config* spec, const char* root, int64_t ramBudget,
           int64_t diskBudget, int verbose) {
    memset(kv, 0, sizeof(*kv));
    kv->d = &spec->dims;
    buildLayout(kv, spec);
    buildGdn(kv, spec);
    if (ramBudget < kv->slotBytes * 4) ramBudget = kv->slotBytes * 4;
    kv->ramBudget = ramBudget;
    kv->diskBudget = diskBudget;
    kv->totalSlots = ramBudget / kv->slotBytes;

    int64_t hostUsed = 0;
    int64_t deviceUsed = 0;
    bufferMemoryTotals(&hostUsed, &deviceUsed);
    int64_t hostHeap = hostHeapBytes(s.dev.physicalDevice);
    if (hostHeap > 0 && hostUsed + kv->ramBudget > hostHeap) {
        fprintf(stderr,
                "kvcache: warning: kvRamBudget %.0f MB + host pool in use %.0f MB exceeds host heap %.0f MB\n",
                (double)kv->ramBudget / (1024.0 * 1024.0), (double)hostUsed / (1024.0 * 1024.0),
                (double)hostHeap / (1024.0 * 1024.0));
    }

    kv->slotState = (uint8_t*)calloc((size_t)kv->totalSlots, 1);
    kv->slotDirty = (uint8_t*)calloc((size_t)kv->totalSlots, 1);
    kv->freeSlots = (int32_t*)malloc(sizeof(int32_t) * (size_t)kv->totalSlots);
    kv->nextSlot = 0;
    kv->usedSlots = 0;

    buildFingerprint(spec, &kv->seed0);
    const char* base = root ? root : "kvstore";
    snprintf(kv->dir, sizeof(kv->dir), "%s/%s-%016llx", base, spec->name, (unsigned long long)kv->seed0);
    ensureDir(base);
    ensureDir(kv->dir);
    snprintf(kv->mirrorPath, sizeof(kv->mirrorPath), "%s/mirror.bin", kv->dir);
    snprintf(kv->indexPath, sizeof(kv->indexPath), "%s/index.bin", kv->dir);
    snprintf(kv->snapPath, sizeof(kv->snapPath), "%s/snapshots.bin", kv->dir);
    snprintf(kv->coldPath, sizeof(kv->coldPath), "%s/cold.bin", kv->dir);

    uint8_t* zeros = (uint8_t*)calloc(1, (size_t)kv->ramBudget);
    if (zeros == NULL) return 0;
    kv->pool = createBufferNamed(s.dev.device, s.dev.physicalDevice, zeros, kv->ramBudget, MEMORY_RAM, "kvPool");
    kv->device = s.dev.device;
    free(zeros);
    if (kv->pool.buffer == VK_NULL_HANDLE) {
        free(kv->slotState);
        free(kv->slotDirty);
        free(kv->freeSlots);
        return 0;
    }

    kv->stageSlots = KV_STAGE_SLOTS;
    uint8_t* stageZeros = (uint8_t*)calloc(1, (size_t)kv->slotBytes * (size_t)kv->stageSlots);
    if (stageZeros == NULL) return 0;
    kv->stage = createBufferNamed(s.dev.device, s.dev.physicalDevice, stageZeros,
                                  kv->slotBytes * kv->stageSlots, MEMORY_RAM, "kvStage");
    free(stageZeros);
    if (kv->stage.buffer == VK_NULL_HANDLE) {
        destroyBuffer(s.dev.device, kv->pool);
        free(kv->slotState);
        free(kv->slotDirty);
        free(kv->freeSlots);
        return 0;
    }

    kv->cold = fopen(kv->coldPath, "r+b");
    if (kv->cold == NULL) kv->cold = fopen(kv->coldPath, "w+b");

    loadIndex(kv);
    loadSnapshots(kv);
    if (verbose) {
        fprintf(stderr, "kvcache: %s slots=%lld slotBytes=%lld gdn=%.1fMB snaps=%d ram=%.1fMB disk=%.1fMB resident=%d cold=%.1fMB\n",
                kv->dir, (long long)kv->totalSlots, (long long)kv->slotBytes,
                (double)kv->gdnBytes / (1024.0 * 1024.0), kv->snapUsed,
                (double)kv->ramBudget / (1024.0 * 1024.0),
                (double)kv->diskBudget / (1024.0 * 1024.0), kv->entryCount,
                (double)kv->coldBytes / (1024.0 * 1024.0));
    }
    return 1;
}

void kvClose(kvcache* kv) {
    if (kv->cold != NULL) fclose(kv->cold);
    if (kv->pool.buffer != VK_NULL_HANDLE) destroyBuffer(kv->device, kv->pool);
    if (kv->stage.buffer != VK_NULL_HANDLE) destroyBuffer(kv->device, kv->stage);
    free(kv->slotState);
    free(kv->slotDirty);
    free(kv->freeSlots);
    free(kv->entries);
    free(kv->table);
    free(kv->freeEntries);
    for (int i = 0; i < KV_SNAP_MAX; i++) free(kv->snaps[i].data);
}

void kvChainHashes(const kvcache* kv, const uint32_t* tokens, int count, uint64_t* out) {
    int nb = count / KV_BLOCK_TOKENS;
    uint64_t h = kv->seed0;
    for (int b = 0; b < nb; b++) {
        h = XXH3_64bits_withSeed(tokens + (size_t)b * KV_BLOCK_TOKENS,
                                 KV_BLOCK_TOKENS * sizeof(uint32_t), h);
        out[b] = h;
    }
}

int kvHasBlock(const kvcache* kv, uint64_t key) {
    return tableFind(kv, key) >= 0;
}

int kvBlockResident(const kvcache* kv, uint64_t key) {
    int idx = tableFind(kv, key);
    return idx >= 0 && kv->entries[idx].slot >= 0;
}

int kvPlan(kvcache* kv, const uint64_t* hashes, int tokenCount, kv_plan* plan) {
    memset(plan, 0, sizeof(*plan));
    plan->snapPos = -1;
    int nb = tokenCount / KV_BLOCK_TOKENS;
    int matched = 0;
    while (matched < nb) {
        int idx = tableFind(kv, hashes[matched]);
        if (idx < 0) break;
        kv_entry* e = &kv->entries[idx];
        if (e->slot >= 0 && !slotValid(kv, e)) {
            freeSlot(kv, e->slot);
            e->slot = -1;
            e->crc = 0;
            if (e->coldOffset < 0) break;
        }
        matched++;
    }
    int maxPos = matched * KV_BLOCK_TOKENS;
    if (maxPos > tokenCount - 1) maxPos = tokenCount - 1;
    for (int b = 0; b < matched; b++) {
        int idx = tableFind(kv, hashes[b]);
        kv_entry* e = &kv->entries[idx];
        e->freq++;
        e->lastUse = ++kv->tick;
    }
    kv->statHits += (uint64_t)matched;
    int best = -1;
    const uint8_t* snap = NULL;
    kv_snapshot* bestSnap = NULL;
    if (kv->gdnBytes <= 0) {
        best = maxPos - (maxPos % KV_BLOCK_TOKENS);
    } else {
        for (int i = 0; i < KV_SNAP_MAX; i++) {
            kv_snapshot* s = &kv->snaps[i];
            if (!s->used || s->data == NULL) continue;
            if (s->pos <= 0 || s->pos > maxPos || (s->pos % KV_BLOCK_TOKENS) != 0) continue;
            if (s->key != hashes[s->pos / KV_BLOCK_TOKENS - 1]) continue;
            if (s->pos > best) { best = s->pos; snap = s->data; bestSnap = s; }
        }
    }
    if (best < 0) return 0;
    if (bestSnap != NULL) bestSnap->lastUse = ++kv->tick;
    plan->resume = best;
    plan->snapPos = best;
    plan->snapData = snap;
    int bc = best / KV_BLOCK_TOKENS;
    plan->blockCount = bc;
    plan->blocks = (kv_restore_block*)calloc((size_t)(bc > 0 ? bc : 1), sizeof(kv_restore_block));
    for (int b = 0; b < bc; b++) {
        int idx = tableFind(kv, hashes[b]);
        kv_entry* e = &kv->entries[idx];
        e->pin++;
        plan->blocks[b].key = hashes[b];
        plan->blocks[b].slot = e->slot;
        plan->blocks[b].source = e->slot >= 0 ? 0 : 1;
    }
    return 1;
}

int kvColdLoad(kvcache* kv, uint64_t key, int stageSlot) {
    int idx = tableFind(kv, key);
    if (idx < 0) return 0;
    kv_entry* e = &kv->entries[idx];
    if (e->coldOffset < 0 || stageSlot < 0 || stageSlot >= kv->stageSlots) return 0;
    uint8_t* dst = (uint8_t*)kv->stage.mappedMemory + (int64_t)stageSlot * kv->slotBytes;
    if (!coldRead(kv, e->coldOffset, dst)) return 0;
    kv->statColdHits++;
    return 1;
}

int kvAdmit(kvcache* kv, uint64_t key, int stageSlot) {
    int idx = tableFind(kv, key);
    if (idx < 0) return 0;
    kv_entry* e = &kv->entries[idx];
    if (e->freq < 2 || e->slot >= 0) return 0;
    int slot = kvAllocSlot(kv);
    if (slot < 0) return 0;
    memcpy((uint8_t*)kv->pool.mappedMemory + (int64_t)slot * kv->slotBytes,
           (uint8_t*)kv->stage.mappedMemory + (int64_t)stageSlot * kv->slotBytes,
           (size_t)kv->slotBytes);
    if (e->coldOffset >= 0) {
        coldMarkDead(kv, e->coldOffset);
        e->coldOffset = -1;
    }
    e->slot = slot;
    e->crc = slotCrc(kv, slot);
    kv->slotDirty[slot] = 1;
    return 1;
}

void kvUnpin(kvcache* kv, kv_plan* plan) {
    for (int b = 0; b < plan->blockCount; b++) {
        int idx = tableFind(kv, plan->blocks[b].key);
        if (idx >= 0 && kv->entries[idx].pin > 0) kv->entries[idx].pin--;
    }
}

void kvPlanFree(kv_plan* plan) {
    free(plan->blocks);
    plan->blocks = NULL;
}

void kvCommitBlock(kvcache* kv, uint64_t key, uint64_t parentKey, uint64_t childKey, int slot, int blockIndex) {
    int idx = tableFind(kv, key);
    if (idx < 0) {
        idx = entryAlloc(kv, key);
        tableInsert(kv, idx);
    }
    kv_entry* e = &kv->entries[idx];
    if (e->coldOffset >= 0) {
        coldMarkDead(kv, e->coldOffset);
        e->coldOffset = -1;
    }
    e->slot = slot;
    e->blockIndex = blockIndex;
    e->lastUse = ++kv->tick;
    e->freq++;
    e->crc = slotCrc(kv, slot);
    if (childKey != 0) e->child = childKey;
    if (parentKey != 0) {
        int pi = tableFind(kv, parentKey);
        if (pi >= 0) kv->entries[pi].child = key;
    }
    kv->slotDirty[slot] = 1;
    kv->statBlocks++;
    if (++kv->insertsSinceAge >= 64) {
        for (int i = 0; i < kv->entryHigh; i++) {
            if (kv->entries[i].alive) kv->entries[i].freq >>= 1;
        }
        kv->insertsSinceAge = 0;
    }
    enforceDiskBudget(kv);
}

int kvStoreSnapshot(kvcache* kv, const uint64_t* hashes, int pos, const void* data) {
    if (kv->gdnBytes <= 0 || pos < KV_BLOCK_TOKENS || (pos % KV_BLOCK_TOKENS) != 0) return 0;
    uint64_t key = hashes[pos / KV_BLOCK_TOKENS - 1];
    kv_snapshot* s = snapFind(kv, pos, key);
    if (s == NULL) {
        s = snapAlloc(kv);
        if (s->data == NULL) {
            s->data = (uint8_t*)malloc((size_t)kv->gdnBytes);
            if (s->data == NULL) return 0;
        }
        if (!s->used) kv->snapUsed++;
    }
    memmove(s->data, data, (size_t)kv->gdnBytes);
    s->pos = pos;
    s->key = key;
    s->used = 1;
    s->lastUse = ++kv->tick;
    return 1;
}

void kvAddRestoreTime(kvcache* kv, double ms) {
    kv->statRestoreMs += ms;
}

static void persistMirror(kvcache* kv) {
    int dirty = 0;
    for (int i = 0; i < kv->nextSlot; i++) {
        if (kv->slotDirty[i]) { dirty = 1; break; }
    }
    if (!dirty) return;
    FILE* f = fopen(kv->mirrorPath, "r+b");
    if (f == NULL) f = fopen(kv->mirrorPath, "wb");
    if (f == NULL) return;
    for (int i = 0; i < kv->nextSlot; i++) {
        if (!kv->slotDirty[i]) continue;
        _fseeki64(f, (int64_t)i * kv->slotBytes, SEEK_SET);
        fwrite((const uint8_t*)kv->pool.mappedMemory + (int64_t)i * kv->slotBytes, 1,
               (size_t)kv->slotBytes, f);
        kv->slotDirty[i] = 0;
    }
    fclose(f);
}

static void persistIndex(kvcache* kv) {
    FILE* f = fopen(kv->indexPath, "wb");
    if (f == NULL) return;
    int latest = -1;
    uint32_t latestUse = 0;
    for (int i = 0; i < KV_SNAP_MAX; i++) {
        if (!kv->snaps[i].used) continue;
        if (latest < 0 || kv->snaps[i].lastUse >= latestUse) {
            latest = i;
            latestUse = kv->snaps[i].lastUse;
        }
    }
    kv_index_header h;
    memset(&h, 0, sizeof(h));
    h.magic = KV_INDEX_MAGIC;
    h.version = KV_INDEX_VERSION;
    h.slotBytes = kv->slotBytes;
    h.totalSlots = kv->totalSlots;
    h.coldBytes = kv->coldBytes;
    h.coldLive = kv->coldLive;
    h.coldDead = kv->coldDead;
    h.nextSlot = kv->nextSlot;
    h.entryCount = kv->entryCount;
    h.snapPos = latest >= 0 ? kv->snaps[latest].pos : 0;
    h.snapKey = latest >= 0 ? kv->snaps[latest].key : 0;
    h.tick = kv->tick;
    fwrite(&h, sizeof(h), 1, f);
    for (int i = 0; i < kv->entryHigh; i++) {
        kv_entry* e = &kv->entries[i];
        if (!e->alive) continue;
        kv_index_record r;
        r.key = e->key;
        r.child = e->child;
        r.coldOffset = e->coldOffset;
        r.slot = e->slot;
        r.freq = e->freq;
        r.blockIndex = e->blockIndex;
        r.lastUse = e->lastUse;
        r.crc = e->crc;
        fwrite(&r, sizeof(r), 1, f);
    }
    fclose(f);
}

static void persistSnapshots(kvcache* kv) {
    if (kv->gdnBytes <= 0) return;
    uint32_t count = 0;
    for (int i = 0; i < KV_SNAP_MAX; i++) {
        if (kv->snaps[i].used && kv->snaps[i].data != NULL) count++;
    }
    if (count == 0) {
        remove(kv->snapPath);
        return;
    }
    FILE* f = fopen(kv->snapPath, "wb");
    if (f == NULL) return;
    kv_snaps_header h;
    h.magic = KV_SNAPS_MAGIC;
    h.count = count;
    h.gdnBytes = kv->gdnBytes;
    fwrite(&h, sizeof(h), 1, f);
    for (int i = 0; i < KV_SNAP_MAX; i++) {
        kv_snapshot* s = &kv->snaps[i];
        if (!s->used || s->data == NULL) continue;
        kv_snap_record r;
        r.pos = s->pos;
        r.lastUse = s->lastUse;
        r.key = s->key;
        fwrite(&r, sizeof(r), 1, f);
        fwrite(s->data, 1, (size_t)kv->gdnBytes, f);
    }
    fclose(f);
}

void kvPersist(kvcache* kv) {
    persistMirror(kv);
    persistIndex(kv);
    persistSnapshots(kv);
    if (kv->coldDead > 0 && kv->coldDead >= kv->coldLive) compactCold(kv);
}

