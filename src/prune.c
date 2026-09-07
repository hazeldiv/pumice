#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <io.h>
#include <direct.h>
#include <windows.h>
#include "prune.h"
#include "safetensors.h"

#define EMBED_NAME "model.language_model.embed_tokens.weight"
#define HEAD_NAME "lm_head.weight"
#define GATHER_CHUNK 8192
#define TENSOR_FILE_MAGIC 0x54454E53

static void pfatal(const char* msg) {
    fprintf(stderr, "prune: %s\n", msg);
    exit(1);
}

static void perr(const char* msg, const char* arg) {
    fprintf(stderr, "prune: %s: %s\n", msg, arg);
    exit(1);
}

static int32_t* loadMapping(const char* path, int expected) {
    FILE* f = fopen(path, "rb");
    if (!f) perr("cannot open mapping", path);
    char magic[6];
    if (fread(magic, 1, 6, f) != 6 || memcmp(magic, "\x93NUMPY", 6) != 0) pfatal("bad npy magic");
    unsigned char ver[2];
    if (fread(ver, 1, 2, f) != 2) pfatal("bad npy version");
    if (ver[0] != 1) pfatal("unsupported npy version");
    unsigned char hlen[2];
    if (fread(hlen, 1, 2, f) != 2) pfatal("bad npy header");
    int hl = hlen[0] | (hlen[1] << 8);
    char* hdr = (char*)malloc(hl + 1);
    if (!hdr || fread(hdr, 1, hl, f) != (size_t)hl) pfatal("bad npy header");
    hdr[hl] = '\0';
    if (!strstr(hdr, "'<i4'")) pfatal("mapping npy must be int32");
    int count = 0;
    char* shp = strstr(hdr, "'shape':");
    if (!shp || sscanf(shp, "'shape': (%d,", &count) != 1) pfatal("bad npy shape");
    free(hdr);
    if (count != expected) pfatal("mapping length mismatch vs vocab size");
    int32_t* m = (int32_t*)malloc(sizeof(int32_t) * count);
    if (!m || fread(m, sizeof(int32_t), count, f) != (size_t)count) pfatal("mapping read error");
    fclose(f);
    return m;
}

static int copyFileTo(const char* src, const char* dstDir) {
    const char* base = strrchr(src, '/');
    const char* name = base ? base + 1 : src;
    char dst[512];
    snprintf(dst, sizeof(dst), "%s/%s", dstDir, name);
    if (fopen(dst, "rb") != NULL) return 0;
    BOOL ok = CopyFileA(src, dst, FALSE);
    if (!ok) perr("copy failed", src);
    return 1;
}

static void ensureTokenizerFiles(const char* vocabDir, const char* sourceDir) {
    _mkdir(vocabDir);
    const char* tokFiles[] = {"tokenizer.json", "tokenizer_config.json", "vocab.json", "mapping.npy"};
    char path[512];
    for (int i = 0; i < 4; i++) {
        snprintf(path, sizeof(path), "%s/%s", sourceDir, tokFiles[i]);
        if (fopen(path, "rb") != NULL) copyFileTo(path, vocabDir);
    }
}

static int cacheHeaderMatches(const char* path, int rows, int cols) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    int header[4];
    int ok = (fread(header, sizeof(int), 4, f) == 4 &&
              header[0] == TENSOR_FILE_MAGIC && header[1] == rows &&
              header[2] == cols && header[3] == (int)QUANT_FP16);
    fclose(f);
    return ok;
}

static int embedCacheValid(const char* modelName, int K, int V) {
    char path[512];
    snprintf(path, sizeof(path), "weights/%s/embed_%d_FP16.bin", modelName, V);
    return cacheHeaderMatches(path, K, V);
}

static int headCacheValid(const char* modelName, int K, int V) {
    char path[512];
    snprintf(path, sizeof(path), "weights/%s/lmHead_%d_FP16.bin", modelName, V);
    return cacheHeaderMatches(path, K, V);
}

static int vocabFileExists(const char* dir, const char* prefix) {
    char pattern[512];
    snprintf(pattern, sizeof(pattern), "%s/vocab/%s.*.safetensors", dir, prefix);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE) return 0;
    FindClose(h);
    return 1;
}

static int findShardPaths(const char* modelDir, char out[][512], int max) {
    char pattern[512];
    snprintf(pattern, sizeof(pattern), "%s/model*.safetensors", modelDir);
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern, &fd);
    if (h == INVALID_HANDLE_VALUE) return 0;
    int count = 0;
    do {
        if (count >= max) break;
        snprintf(out[count], 512, "%s/%s", modelDir, fd.cFileName);
        count++;
    } while (FindNextFileA(h, &fd));
    FindClose(h);
    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (strcmp(out[i], out[j]) > 0) {
                char tmp[512];
                strcpy(tmp, out[i]);
                strcpy(out[i], out[j]);
                strcpy(out[j], tmp);
            }
        }
    }
    return count;
}

static void writeSafetensorsHeader(FILE* f, const char* name, int rows, int cols, long dataBytes) {
    char entry[512];
    snprintf(entry, sizeof(entry),
             "{\"%s\":{\"dtype\":\"BF16\",\"shape\":[%d,%d],\"data_offsets\":[0,%ld]},\"__metadata__\":{\"format\":\"pt\"}}",
             name, rows, cols, dataBytes);
    int raw = (int)strlen(entry);
    int pad = (-raw) & 7;
    uint64_t total = (uint64_t)(raw + pad);
    fwrite(&total, sizeof(uint64_t), 1, f);
    fwrite(entry, 1, raw, f);
    for (int i = 0; i < pad; i++) fputc(' ', f);
}

static void gatherRows(FILE* src, FILE* dst, int64_t dataStart, const int32_t* mapping, int V, int K) {
    int rowBytes = K * 2;
    uint16_t* chunk = (uint16_t*)malloc((size_t)rowBytes * GATHER_CHUNK);
    if (!chunk) pfatal("out of memory");
    for (int start = 0; start < V; start += GATHER_CHUNK) {
        int stop = start + GATHER_CHUNK;
        if (stop > V) stop = V;
        for (int v = start; v < stop; v++) {
            _fseeki64(src, dataStart + (int64_t)mapping[v] * rowBytes, SEEK_SET);
            if (fread(chunk + (size_t)(v - start) * K, rowBytes, 1, src) != 1) pfatal("shard read error");
        }
        fwrite(chunk, rowBytes, stop - start, dst);
    }
    free(chunk);
}

static void gatherTensor(safetensors* sf, const char* tensorName, const char* outPrefix,
                         const char* vocabDir, const int32_t* mapping, int V, int K, int required) {
    const sa_tensor* t = safetensors_find(sf, tensorName);
    if (!t) {
        if (!required) return;
        perr("missing tensor in shard", tensorName);
    }
    if (t->dtype != SA_DTYPE_BF16 || t->ndim != 2 || t->shape[1] != K) pfatal("tensor shape/dtype mismatch vs config");
    int srcRows = (int)t->shape[0];
    for (int i = 0; i < V; i++) {
        if (mapping[i] < 0 || mapping[i] >= srcRows) pfatal("mapping id out of range");
    }
    FILE* src = sf->files[t->fileIndex];
    char dstPath[512];
    snprintf(dstPath, sizeof(dstPath), "%s/%s.%d.safetensors", vocabDir, outPrefix, V);
    FILE* dst = fopen(dstPath, "wb");
    if (!dst) perr("cannot create", dstPath);
    long dataBytes = (long)V * K * 2;
    writeSafetensorsHeader(dst, tensorName, V, K, dataBytes);
    gatherRows(src, dst, t->offset, mapping, V, K);
    fclose(dst);
    fprintf(stderr, "prune: wrote %s (%.1f MB, %d rows of %d)\n",
            dstPath, dataBytes / (1024.0 * 1024.0), V, srcRows);
}

int pruneVocab(const char* modelDir, const model_config* spec) {
    const model_dims* d = &spec->dims;
    char vocabDir[512];
    snprintf(vocabDir, sizeof(vocabDir), "%s/vocab", modelDir);

    if (embedCacheValid(spec->name, d->K, d->vocab) &&
        (d->tied || headCacheValid(spec->name, d->K, d->vocab))) {
        ensureTokenizerFiles(vocabDir, PRUNED_VOCAB_DIR);
        fprintf(stderr, "prune: weight cache present, skipping gather\n");
        return 0;
    }

    int needEmbed = !vocabFileExists(modelDir, "embed_tokens");
    int needHead = !d->tied && !vocabFileExists(modelDir, "lm_head");
    if (!needEmbed && !needHead) {
        ensureTokenizerFiles(vocabDir, PRUNED_VOCAB_DIR);
        return 0;
    }

    char path[512];
    snprintf(path, sizeof(path), "%s/mapping.npy", PRUNED_VOCAB_DIR);
    int32_t* mapping = loadMapping(path, d->vocab);

    char shardPaths[32][512];
    const char* shardPtrs[32];
    int shardCount = findShardPaths(modelDir, shardPaths, 32);
    if (shardCount == 0) pfatal("no model shards found");
    for (int i = 0; i < shardCount; i++) shardPtrs[i] = shardPaths[i];

    safetensors sf;
    if (safetensors_open(&sf, shardPtrs, shardCount) != 0) pfatal("cannot open shards");

    _mkdir(vocabDir);
    if (needEmbed) {
        gatherTensor(&sf, EMBED_NAME, "embed_tokens", vocabDir, mapping, d->vocab, d->K, 1);
    }
    if (needHead) {
        gatherTensor(&sf, HEAD_NAME, "lm_head", vocabDir, mapping, d->vocab, d->K, 1);
    }

    safetensors_close(&sf);
    free(mapping);
    ensureTokenizerFiles(vocabDir, PRUNED_VOCAB_DIR);
    return 1;
}
