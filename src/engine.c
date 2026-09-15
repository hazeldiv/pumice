#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "engine.h"
#include "generate.h"
#include "weights.h"
#include "prune.h"
#include "hqm.h"
#include "gguf.h"
#include "tokenizers_c.h"

struct engine {
    model_config spec;
    session s;
    generator* g;
    TokenizerHandle tok;
    uint32_t* ids;
    size_t idCount;
    size_t idCap;
    char* decoded;
    size_t decodedCap;
};

typedef struct {
    char* p;
    size_t len;
    size_t cap;
} strbuf;

static void sbReserve(strbuf* b, size_t extra) {
    if (b->len + extra + 1 <= b->cap) return;
    size_t cap = b->cap ? b->cap : 256;
    while (cap < b->len + extra + 1) cap *= 2;
    b->p = (char*)realloc(b->p, cap);
    b->cap = cap;
}

static void sbPutc(strbuf* b, char c) {
    sbReserve(b, 1);
    b->p[b->len++] = c;
    b->p[b->len] = '\0';
}

static void sbPuts(strbuf* b, const char* s) {
    size_t n = strlen(s);
    sbReserve(b, n);
    memcpy(b->p + b->len, s, n);
    b->len += n;
    b->p[b->len] = '\0';
}

static void sbPrintf(strbuf* b, const char* fmt, ...) {
    char tmp[64];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(tmp, sizeof(tmp), fmt, ap);
    va_end(ap);
    sbPuts(b, tmp);
}

static void sbEscape(strbuf* b, const char* s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
            case '"': sbPuts(b, "\\\""); break;
            case '\\': sbPuts(b, "\\\\"); break;
            case '\n': sbPuts(b, "\\n"); break;
            case '\r': sbPuts(b, "\\r"); break;
            case '\t': sbPuts(b, "\\t"); break;
            case '\b': sbPuts(b, "\\b"); break;
            case '\f': sbPuts(b, "\\f"); break;
            default:
                if (c < 0x20) sbPrintf(b, "\\u%04x", c);
                else sbPutc(b, (char)c);
        }
    }
}

static int readWhole(const char* path, char** out, size_t* outLen) {
    FILE* f = fopen(path, "rb");
    if (f == NULL) return 0;
    _fseeki64(f, 0, SEEK_END);
    int64_t n = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
    char* buf = (char*)malloc((size_t)n + 1);
    if (buf == NULL || (n > 0 && fread(buf, 1, (size_t)n, f) != (size_t)n)) {
        free(buf);
        fclose(f);
        return 0;
    }
    fclose(f);
    buf[n] = '\0';
    *out = buf;
    *outLen = (size_t)n;
    return 1;
}

static int parseStrArray(const void* raw, int64_t bytes, char*** out, int64_t* count) {
    if (bytes < 4) return 0;
    const uint8_t* p = (const uint8_t*)raw;
    uint32_t n;
    memcpy(&n, p, 4);
    size_t pos = 4;
    char** arr = (char**)calloc(n ? n : 1, sizeof(char*));
    for (uint32_t i = 0; i < n; i++) {
        if (pos + 4 > (size_t)bytes) { free(arr); return 0; }
        uint32_t len;
        memcpy(&len, p + pos, 4);
        pos += 4;
        if (pos + len > (size_t)bytes) { free(arr); return 0; }
        arr[i] = (char*)malloc((size_t)len + 1);
        memcpy(arr[i], p + pos, len);
        arr[i][len] = '\0';
        pos += len;
    }
    *out = arr;
    *count = (int64_t)n;
    return 1;
}

static void freeStrArray(char** arr, int64_t count) {
    if (arr == NULL) return;
    for (int64_t i = 0; i < count; i++) free(arr[i]);
    free(arr);
}

static char* bpeVocabJson(char** toks, int64_t n) {
    strbuf b = {0};
    sbPutc(&b, '{');
    for (int64_t i = 0; i < n; i++) {
        if (i > 0) sbPutc(&b, ',');
        sbPutc(&b, '"');
        sbEscape(&b, toks[i], strlen(toks[i]));
        sbPuts(&b, "\":");
        sbPrintf(&b, "%lld", (long long)i);
    }
    sbPutc(&b, '}');
    return b.p;
}

static char* bpeMergesText(char** merges, int64_t n) {
    strbuf b = {0};
    for (int64_t i = 0; i < n; i++) {
        if (i > 0) sbPutc(&b, '\n');
        sbPuts(&b, merges[i]);
    }
    return b.p;
}

static char* bpeAddedJson(char** toks, int64_t n, const int32_t* types, int64_t typeCount) {
    strbuf b = {0};
    sbPutc(&b, '{');
    int first = 1;
    for (int64_t i = 0; i < n && i < typeCount; i++) {
        if (types[i] != 2 && types[i] != 3 && types[i] != 4) continue;
        if (!first) sbPutc(&b, ',');
        first = 0;
        sbPutc(&b, '"');
        sbEscape(&b, toks[i], strlen(toks[i]));
        sbPuts(&b, "\":");
        sbPrintf(&b, "%lld", (long long)i);
    }
    sbPutc(&b, '}');
    return b.p;
}

static TokenizerHandle bpeFromParts(char** toks, int64_t nTok, char** merges, int64_t nMerge,
                                    const int32_t* types, int64_t nType) {
    if (toks == NULL || nTok <= 0) return NULL;
    char* vocab = bpeVocabJson(toks, nTok);
    char* mergesText = bpeMergesText(merges, nMerge);
    char* added = bpeAddedJson(toks, nTok, types, nType);
    TokenizerHandle h = byte_level_bpe_tokenizers_new_from_str(vocab, strlen(vocab), mergesText,
                                                               strlen(mergesText), added, strlen(added));
    free(vocab);
    free(mergesText);
    free(added);
    return h;
}

static TokenizerHandle tokenizerFromGguf(const char* path) {
    gguf g;
    if (gguf_open(&g, path) != 0) return NULL;
    char** toks = NULL;
    char** merges = NULL;
    int32_t* types = NULL;
    int64_t nTok = 0, nMerge = 0, nType = 0;
    gguf_meta_str_array(&g, "tokenizer.ggml.tokens", &toks, &nTok);
    gguf_meta_str_array(&g, "tokenizer.ggml.merges", &merges, &nMerge);
    gguf_meta_i32_array(&g, "tokenizer.ggml.token_type", &types, &nType);
    TokenizerHandle h = bpeFromParts(toks, nTok, merges, nMerge, types, nType);
    gguf_str_array_free(toks, nTok);
    gguf_str_array_free(merges, nMerge);
    free(types);
    gguf_close(&g);
    return h;
}

static TokenizerHandle tokenizerFromHqm(const char* path) {
    hqm h;
    if (hqm_open(&h, path) != 0) return NULL;
    TokenizerHandle handle = NULL;
    const hqm_tensor* tj = hqm_tensor_find(&h, "tokenizer.json");
    if (tj != NULL) {
        int64_t n = 0;
        void* raw = hqm_tensor_read(&h, tj, &n);
        if (raw != NULL) {
            handle = tokenizers_new_from_str((const char*)raw, (size_t)n);
            free(raw);
        }
    } else {
        const hqm_tensor* tt = hqm_tensor_find(&h, "tokenizer.tokens");
        if (tt != NULL) {
            int64_t n = 0;
            void* raw = hqm_tensor_read(&h, tt, &n);
            char** toks = NULL;
            int64_t nTok = 0;
            if (raw != NULL && parseStrArray(raw, n, &toks, &nTok)) {
                char** merges = NULL;
                int32_t* types = NULL;
                int64_t nMerge = 0, nType = 0;
                const hqm_tensor* mt = hqm_tensor_find(&h, "tokenizer.merges");
                if (mt != NULL) {
                    int64_t mn = 0;
                    void* mraw = hqm_tensor_read(&h, mt, &mn);
                    if (mraw != NULL) {
                        parseStrArray(mraw, mn, &merges, &nMerge);
                        free(mraw);
                    }
                }
                const hqm_tensor* yt = hqm_tensor_find(&h, "tokenizer.token_type");
                if (yt != NULL) {
                    int64_t yn = 0;
                    void* yraw = hqm_tensor_read(&h, yt, &yn);
                    if (yraw != NULL) {
                        types = (int32_t*)yraw;
                        nType = yn / 4;
                    }
                }
                handle = bpeFromParts(toks, nTok, merges, nMerge, types, nType);
                free(types);
                freeStrArray(merges, nMerge);
            }
            free(raw);
            freeStrArray(toks, nTok);
        }
    }
    hqm_close(&h);
    return handle;
}

static TokenizerHandle loadTokenizer(const char* weights, int pruned) {
    if (hqm_path_is_file(weights)) return tokenizerFromHqm(weights);
    if (gguf_path_is_file(weights)) return tokenizerFromGguf(weights);
    char path[512];
    if (pruned) snprintf(path, sizeof(path), "%s/vocab/tokenizer.json", weights);
    else snprintf(path, sizeof(path), "%s/tokenizer.json", weights);
    char* buf = NULL;
    size_t n = 0;
    if (!readWhole(path, &buf, &n)) return NULL;
    TokenizerHandle h = tokenizers_new_from_str(buf, n);
    free(buf);
    return h;
}

static void setError(char* err, size_t cap, const char* msg) {
    if (err != NULL && cap > 0) snprintf(err, cap, "%s", msg);
}

engine* engineOpen(const engine_options* opts, char* err, size_t errCap) {
    engine* e = (engine*)calloc(1, sizeof(engine));
    if (e == NULL) {
        setError(err, errCap, "out of memory");
        return NULL;
    }

    hqm_set_export_dir(opts->exportDir);
    loadModelConfig(&e->spec, opts->weights, opts->quantConfig, opts->maxCtx, opts->prune);

    char hqmPath[512];
    int hqmSource = hqm_resolve(&e->spec, opts->weights, hqmPath, sizeof(hqmPath));
    if (opts->prune && !hqmSource) pruneVocab(opts->weights, &e->spec);
    parseEos(&e->spec.dims, hqmSource ? hqmPath : opts->weights, opts->prune);

    weightsSetExport(opts->exportModel ? 1 : 0);
    if (opts->expertsVram > 0) e->spec.expertsVram = opts->expertsVram;

    e->s = createSession();
    e->g = createGenerator(e->s, &e->spec, opts->weights, 0);
    e->tok = loadTokenizer(opts->weights, opts->prune);
    if (e->tok == NULL) {
        destroyGenerator(e->g);
        destroySession(e->s);
        free(e);
        setError(err, errCap, "cannot load tokenizer");
        return NULL;
    }
    return e;
}

void engineClose(engine* e) {
    if (e == NULL) return;
    if (e->g != NULL) destroyGenerator(e->g);
    destroySession(e->s);
    if (e->tok != NULL) tokenizers_free(e->tok);
    free(e->ids);
    free(e->decoded);
    free(e);
}

int engineTokenize(engine* e, const char* text, int addSpecial, uint32_t** out, size_t* outCount) {
    if (e == NULL || e->tok == NULL) return -1;
    TokenizerEncodeResult r;
    tokenizers_encode(e->tok, text, strlen(text), addSpecial, &r);
    uint32_t* ids = (uint32_t*)malloc(sizeof(uint32_t) * (r.len ? r.len : 1));
    if (ids == NULL) return -1;
    for (size_t i = 0; i < r.len; i++) ids[i] = (uint32_t)r.token_ids[i];
    tokenizers_free_encode_results(&r, 1);
    *out = ids;
    *outCount = r.len;
    return 0;
}

char* engineDecode(engine* e, const uint32_t* ids, size_t count) {
    if (e == NULL || e->tok == NULL) return NULL;
    tokenizers_decode(e->tok, ids, count, 1);
    const char* data = NULL;
    size_t len = 0;
    tokenizers_get_decode_str(e->tok, &data, &len);
    char* out = (char*)malloc(len + 1);
    if (out == NULL) return NULL;
    memcpy(out, data, len);
    out[len] = '\0';
    return out;
}

static void appendId(engine* e, uint32_t token) {
    if (e->idCount + 1 > e->idCap) {
        e->idCap = e->idCap ? e->idCap * 2 : 256;
        e->ids = (uint32_t*)realloc(e->ids, e->idCap * sizeof(uint32_t));
    }
    e->ids[e->idCount++] = token;
}

static void storeDecoded(engine* e, const char* data, size_t len) {
    if (len + 1 > e->decodedCap) {
        e->decodedCap = len + 64;
        e->decoded = (char*)realloc(e->decoded, e->decodedCap);
    }
    memcpy(e->decoded, data, len);
    e->decoded[len] = '\0';
}

typedef struct {
    engine* e;
    engine_emit emit;
    void* ctx;
} emitContext;

static void onToken(uint32_t token, void* ctx) {
    emitContext* ec = (emitContext*)ctx;
    engine* e = ec->e;
    appendId(e, token);
    tokenizers_decode(e->tok, e->ids, e->idCount, 1);
    const char* data = NULL;
    size_t len = 0;
    tokenizers_get_decode_str(e->tok, &data, &len);
    size_t prev = e->decoded != NULL ? strlen(e->decoded) : 0;
    if (len > prev) ec->emit(ec->ctx, token, data + prev, len - prev);
    storeDecoded(e, data, len);
}

void engineGenerate(engine* e, const uint32_t* prompt, size_t count, const sample_params* params,
                    uint32_t seed, int maxNew, engine_emit emit, void* ctx) {
    if (e == NULL || e->g == NULL) return;
    e->idCount = 0;
    if (e->decoded != NULL) e->decoded[0] = '\0';
    generatorSetSampling(e->g, params, seed);
    resetGenerator(e->g);
    emitContext ec = {e, emit, ctx};
    generateTokens(e->g, prompt, (int)count, maxNew, onToken, &ec);
}

void engineScore(engine* e, const uint32_t* ids, int prefillN, int decodeN, int chunks,
                 engine_progress progress, void* ctx, double* outLoss, long long* outCount) {
    if (e == NULL || e->g == NULL) return;
    generatorSetScoring(e->g, 1);
    generateScore(e->g, ids, prefillN, decodeN, chunks, progress, ctx, outLoss, outCount);
}

int engineVocab(const engine* e) {
    return e->spec.dims.vocab;
}

int engineEos(const engine* e) {
    return e->spec.dims.eos;
}

int engineMaxCtx(const engine* e) {
    return e->spec.dims.maxCtx;
}

void engineRequestStop(engine* e) {
    if (e != NULL && e->g != NULL) generatorRequestStop(e->g);
}
