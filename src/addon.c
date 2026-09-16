#define NAPI_VERSION 8
#include <node_api.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "engine.h"

typedef struct {
    engine* e;
} engineRef;

typedef struct {
    napi_async_work work;
    napi_deferred deferred;
    engine_options opts;
    char* weights;
    char* quantConfig;
    char* exportDir;
    engine* result;
    char error[256];
} openJob;

typedef struct {
    uint32_t token;
    size_t len;
    char delta[1];
} tokenEvent;

typedef struct {
    napi_async_work work;
    napi_deferred deferred;
    napi_threadsafe_function tsfn;
    engine* e;
    uint32_t* prompt;
    size_t promptCount;
    sample_params params;
    uint32_t seed;
    int maxNew;
    int generated;
    char error[256];
} generateJob;

static int g_busy = 0;

static int getStringProp(napi_env env, napi_value obj, const char* name, char** out) {
    *out = NULL;
    napi_value v;
    if (napi_get_named_property(env, obj, name, &v) != napi_ok) return 0;
    napi_valuetype t;
    napi_typeof(env, v, &t);
    if (t != napi_string) return 0;
    size_t len = 0;
    napi_get_value_string_utf8(env, v, NULL, 0, &len);
    char* buf = (char*)malloc(len + 1);
    napi_get_value_string_utf8(env, v, buf, len + 1, &len);
    *out = buf;
    return 1;
}

static double getDoubleProp(napi_env env, napi_value obj, const char* name, double def) {
    napi_value v;
    if (napi_get_named_property(env, obj, name, &v) != napi_ok) return def;
    napi_valuetype t;
    napi_typeof(env, v, &t);
    if (t != napi_number) return def;
    double d = def;
    napi_get_value_double(env, v, &d);
    return d;
}

static int getBoolProp(napi_env env, napi_value obj, const char* name, int def) {
    napi_value v;
    if (napi_get_named_property(env, obj, name, &v) != napi_ok) return def;
    napi_valuetype t;
    napi_typeof(env, v, &t);
    if (t != napi_boolean) return def;
    bool b = def != 0;
    napi_get_value_bool(env, v, &b);
    return b ? 1 : 0;
}

static engine* unwrap(napi_env env, napi_value arg) {
    engineRef* ref = NULL;
    if (napi_get_value_external(env, arg, (void**)&ref) != napi_ok || ref == NULL) return NULL;
    return ref->e;
}

static void finalizeEngine(napi_env env, void* data, void* hint) {
    (void)env;
    (void)hint;
    engineRef* ref = (engineRef*)data;
    if (ref == NULL) return;
    if (ref->e != NULL) engineClose(ref->e);
    free(ref);
}

static napi_value throwError(napi_env env, const char* msg) {
    napi_throw_error(env, NULL, msg);
    return NULL;
}

static void openExecute(napi_env env, void* data) {
    (void)env;
    openJob* job = (openJob*)data;
    job->opts.weights = job->weights;
    job->opts.quantConfig = job->quantConfig;
    job->opts.exportDir = job->exportDir;
    job->result = engineOpen(&job->opts, job->error, sizeof(job->error));
}

static void openComplete(napi_env env, napi_status status, void* data) {
    (void)status;
    openJob* job = (openJob*)data;
    if (job->result != NULL) {
        engineRef* ref = (engineRef*)calloc(1, sizeof(engineRef));
        ref->e = job->result;
        napi_value ext;
        napi_create_external(env, ref, finalizeEngine, NULL, &ext);
        napi_resolve_deferred(env, job->deferred, ext);
    } else {
        napi_value msg;
        napi_create_string_utf8(env, job->error, NAPI_AUTO_LENGTH, &msg);
        napi_value err;
        napi_create_error(env, NULL, msg, &err);
        napi_reject_deferred(env, job->deferred, err);
    }
    napi_delete_async_work(env, job->work);
    free(job->weights);
    free(job->quantConfig);
    free(job->exportDir);
    free(job);
}

static napi_value createEngine(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1];
    napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
    if (argc < 1) return throwError(env, "createEngine requires options");

    openJob* job = (openJob*)calloc(1, sizeof(openJob));
    memset(&job->opts, 0, sizeof(job->opts));
    getStringProp(env, argv[0], "weights", &job->weights);
    getStringProp(env, argv[0], "quantConfig", &job->quantConfig);
    getStringProp(env, argv[0], "exportDir", &job->exportDir);
    job->opts.maxCtx = (int)getDoubleProp(env, argv[0], "maxCtx", 0);
    job->opts.prune = getBoolProp(env, argv[0], "prune", 0);
    job->opts.expertsVram = (int)getDoubleProp(env, argv[0], "expertsVram", 0);
    job->opts.exportModel = getBoolProp(env, argv[0], "exportModel", 0);

    if (job->weights == NULL) {
        free(job);
        return throwError(env, "weights is required");
    }

    napi_value promise;
    napi_create_promise(env, &job->deferred, &promise);
    napi_value name;
    napi_create_string_utf8(env, "vkCreateEngine", NAPI_AUTO_LENGTH, &name);
    napi_create_async_work(env, NULL, name, openExecute, openComplete, job, &job->work);
    napi_queue_async_work(env, job->work);
    return promise;
}

static napi_value destroyEngine(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1];
    napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
    if (argc < 1) return NULL;
    engineRef* ref = NULL;
    if (napi_get_value_external(env, argv[0], (void**)&ref) != napi_ok || ref == NULL) return NULL;
    if (ref->e != NULL) {
        engineClose(ref->e);
        ref->e = NULL;
    }
    return NULL;
}

static napi_value engineInfo(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1];
    napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
    engine* e = argc >= 1 ? unwrap(env, argv[0]) : NULL;
    if (e == NULL) return throwError(env, "engine is not loaded");
    napi_value obj;
    napi_create_object(env, &obj);
    napi_value v;
    napi_create_int32(env, engineVocab(e), &v);
    napi_set_named_property(env, obj, "vocab", v);
    napi_create_int32(env, engineEos(e), &v);
    napi_set_named_property(env, obj, "eos", v);
    napi_create_int32(env, engineMaxCtx(e), &v);
    napi_set_named_property(env, obj, "maxCtx", v);
    return obj;
}

static napi_value tokenize(napi_env env, napi_callback_info info) {
    size_t argc = 3;
    napi_value argv[3];
    napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
    if (argc < 2) return throwError(env, "tokenize requires engine and text");
    engine* e = unwrap(env, argv[0]);
    if (e == NULL) return throwError(env, "engine is not loaded");

    size_t textLen = 0;
    napi_get_value_string_utf8(env, argv[1], NULL, 0, &textLen);
    char* text = (char*)malloc(textLen + 1);
    napi_get_value_string_utf8(env, argv[1], text, textLen + 1, &textLen);

    int addSpecial = 0;
    if (argc >= 3) {
        napi_valuetype t;
        napi_typeof(env, argv[2], &t);
        if (t == napi_boolean) {
            bool b = false;
            napi_get_value_bool(env, argv[2], &b);
            addSpecial = b ? 1 : 0;
        }
    }

    uint32_t* ids = NULL;
    size_t count = 0;
    int rc = engineTokenize(e, text, addSpecial, &ids, &count);
    free(text);
    if (rc != 0) return throwError(env, "tokenize failed");

    void* data = NULL;
    napi_value arr;
    napi_create_arraybuffer(env, count * sizeof(uint32_t), &data, &arr);
    if (count > 0) memcpy(data, ids, count * sizeof(uint32_t));
    free(ids);
    napi_value result;
    napi_create_typedarray(env, napi_uint32_array, count, arr, 0, &result);
    return result;
}

static int readUint32Array(napi_env env, napi_value value, uint32_t** out, size_t* count) {
    bool isTyped = false;
    napi_is_typedarray(env, value, &isTyped);
    if (!isTyped) return 0;
    napi_typedarray_type type;
    size_t length = 0;
    void* data = NULL;
    napi_value buffer;
    size_t offset = 0;
    if (napi_get_typedarray_info(env, value, &type, &length, &data, &buffer, &offset) != napi_ok) return 0;
    if (type != napi_uint32_array) return 0;
    uint32_t* ids = (uint32_t*)malloc(sizeof(uint32_t) * (length ? length : 1));
    if (length > 0) memcpy(ids, data, length * sizeof(uint32_t));
    *out = ids;
    *count = length;
    return 1;
}

static napi_value decode(napi_env env, napi_callback_info info) {
    size_t argc = 2;
    napi_value argv[2];
    napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
    if (argc < 2) return throwError(env, "decode requires engine and ids");
    engine* e = unwrap(env, argv[0]);
    if (e == NULL) return throwError(env, "engine is not loaded");
    uint32_t* ids = NULL;
    size_t count = 0;
    if (!readUint32Array(env, argv[1], &ids, &count)) return throwError(env, "ids must be a Uint32Array");
    char* text = engineDecode(e, ids, count);
    free(ids);
    if (text == NULL) return throwError(env, "decode failed");
    napi_value result;
    napi_create_string_utf8(env, text, NAPI_AUTO_LENGTH, &result);
    free(text);
    return result;
}

static void emitToken(void* ctx, uint32_t token, const char* delta, size_t deltaLen) {
    generateJob* job = (generateJob*)ctx;
    job->generated++;
    tokenEvent* ev = (tokenEvent*)malloc(sizeof(tokenEvent) + deltaLen);
    ev->token = token;
    ev->len = deltaLen;
    memcpy(ev->delta, delta, deltaLen);
    ev->delta[deltaLen] = '\0';
    napi_call_threadsafe_function(job->tsfn, ev, napi_tsfn_blocking);
}

static void generateExecute(napi_env env, void* data) {
    (void)env;
    generateJob* job = (generateJob*)data;
    engineGenerate(job->e, job->prompt, job->promptCount, &job->params, job->seed, job->maxNew,
                   emitToken, job);
}

static void generateComplete(napi_env env, napi_status status, void* data) {
    (void)status;
    generateJob* job = (generateJob*)data;
    napi_release_threadsafe_function(job->tsfn, napi_tsfn_release);
    napi_value result;
    napi_create_int32(env, (int32_t)job->generated, &result);
    napi_resolve_deferred(env, job->deferred, result);
    napi_delete_async_work(env, job->work);
    free(job->prompt);
    free(job);
    g_busy = 0;
}

static void tsfnCall(napi_env env, napi_value jsCallback, void* context, void* data) {
    (void)context;
    if (env == NULL) {
        free(data);
        return;
    }
    tokenEvent* ev = (tokenEvent*)data;
    napi_value undefined;
    napi_get_undefined(env, &undefined);
    napi_value arg;
    napi_create_object(env, &arg);
    napi_value token;
    napi_create_uint32(env, ev->token, &token);
    napi_set_named_property(env, arg, "token", token);
    napi_value delta;
    napi_create_string_utf8(env, ev->delta, ev->len, &delta);
    napi_set_named_property(env, arg, "delta", delta);
    napi_value ignored;
    napi_call_function(env, undefined, jsCallback, 1, &arg, &ignored);
    free(ev);
}

static napi_value generate(napi_env env, napi_callback_info info) {
    size_t argc = 4;
    napi_value argv[4];
    napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
    if (argc < 4) return throwError(env, "generate requires engine, ids, params and callback");
    engine* e = unwrap(env, argv[0]);
    if (e == NULL) return throwError(env, "engine is not loaded");
    if (g_busy) return throwError(env, "generation already in progress");

    generateJob* job = (generateJob*)calloc(1, sizeof(generateJob));
    job->e = e;
    if (!readUint32Array(env, argv[1], &job->prompt, &job->promptCount)) {
        free(job);
        return throwError(env, "ids must be a Uint32Array");
    }
    job->params.temperature = (float)getDoubleProp(env, argv[2], "temperature", 0.0);
    job->params.repPenalty = (float)getDoubleProp(env, argv[2], "repPenalty", 1.0);
    job->params.penaltyLength = (uint32_t)getDoubleProp(env, argv[2], "penaltyLength", 0);
    job->params.topK = (uint32_t)getDoubleProp(env, argv[2], "topK", 0);
    job->params.topP = (float)getDoubleProp(env, argv[2], "topP", 1.0);
    job->params.minP = (float)getDoubleProp(env, argv[2], "minP", 0.0);
    job->params.presencePenalty = (float)getDoubleProp(env, argv[2], "presencePenalty", 0.0);
    job->seed = (uint32_t)getDoubleProp(env, argv[2], "seed", 1);
    job->maxNew = (int)getDoubleProp(env, argv[2], "maxNew", 128);
    if (job->seed == 0) job->seed = 1;

    napi_value promise;
    napi_create_promise(env, &job->deferred, &promise);
    napi_value name;
    napi_create_string_utf8(env, "vkGenerate", NAPI_AUTO_LENGTH, &name);
    napi_create_threadsafe_function(env, argv[3], NULL, name, 0, 1, NULL, NULL, NULL, tsfnCall,
                                    &job->tsfn);
    napi_create_async_work(env, NULL, name, generateExecute, generateComplete, job, &job->work);
    g_busy = 1;
    napi_queue_async_work(env, job->work);
    return promise;
}

typedef struct {
    int done;
    int total;
    int chunkTokens;
    int chunkTotal;
    double loss;
    long long count;
} scoreEvent;

typedef struct {
    napi_async_work work;
    napi_deferred deferred;
    napi_threadsafe_function tsfn;
    engine* e;
    uint32_t* ids;
    size_t idCount;
    int prefill;
    int decode;
    int chunks;
    double loss;
    long long count;
} scoreJob;

static void emitScore(void* ctx, int done, int total, int chunkTokens, int chunkTotal, double lossSum,
                      long long count) {
    scoreJob* job = (scoreJob*)ctx;
    scoreEvent* ev = (scoreEvent*)malloc(sizeof(scoreEvent));
    ev->done = done;
    ev->total = total;
    ev->chunkTokens = chunkTokens;
    ev->chunkTotal = chunkTotal;
    ev->loss = lossSum;
    ev->count = count;
    napi_call_threadsafe_function(job->tsfn, ev, napi_tsfn_blocking);
}

static void scoreExecute(napi_env env, void* data) {
    (void)env;
    scoreJob* job = (scoreJob*)data;
    engineScore(job->e, job->ids, job->idCount, job->prefill, job->decode, job->chunks, emitScore, job,
                &job->loss, &job->count);
}

static void scoreTsfnCall(napi_env env, napi_value jsCallback, void* context, void* data) {
    (void)context;
    if (env == NULL) {
        free(data);
        return;
    }
    scoreEvent* ev = (scoreEvent*)data;
    napi_value undefined;
    napi_get_undefined(env, &undefined);
    napi_value arg;
    napi_create_object(env, &arg);
    napi_value v;
    napi_create_int32(env, ev->done, &v);
    napi_set_named_property(env, arg, "done", v);
    napi_create_int32(env, ev->total, &v);
    napi_set_named_property(env, arg, "total", v);
    napi_create_int32(env, ev->chunkTokens, &v);
    napi_set_named_property(env, arg, "chunkTokens", v);
    napi_create_int32(env, ev->chunkTotal, &v);
    napi_set_named_property(env, arg, "chunkTotal", v);
    napi_create_double(env, ev->loss, &v);
    napi_set_named_property(env, arg, "loss", v);
    napi_create_double(env, (double)ev->count, &v);
    napi_set_named_property(env, arg, "count", v);
    double ppl = ev->count > 0 ? exp(ev->loss / (double)ev->count) : 0.0;
    napi_create_double(env, ppl, &v);
    napi_set_named_property(env, arg, "ppl", v);
    napi_value ignored;
    napi_call_function(env, undefined, jsCallback, 1, &arg, &ignored);
    free(ev);
}

static void scoreComplete(napi_env env, napi_status status, void* data) {
    (void)status;
    scoreJob* job = (scoreJob*)data;
    napi_release_threadsafe_function(job->tsfn, napi_tsfn_release);
    napi_value result;
    napi_create_object(env, &result);
    napi_value v;
    napi_create_double(env, job->loss, &v);
    napi_set_named_property(env, result, "loss", v);
    napi_create_double(env, (double)job->count, &v);
    napi_set_named_property(env, result, "count", v);
    double ppl = job->count > 0 ? exp(job->loss / (double)job->count) : 0.0;
    napi_create_double(env, ppl, &v);
    napi_set_named_property(env, result, "ppl", v);
    napi_resolve_deferred(env, job->deferred, result);
    napi_delete_async_work(env, job->work);
    free(job->ids);
    free(job);
    g_busy = 0;
}

static napi_value score(napi_env env, napi_callback_info info) {
    size_t argc = 4;
    napi_value argv[4];
    napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
    if (argc < 4) return throwError(env, "score requires engine, ids, params and callback");
    engine* e = unwrap(env, argv[0]);
    if (e == NULL) return throwError(env, "engine is not loaded");
    if (g_busy) return throwError(env, "generation already in progress");

    scoreJob* job = (scoreJob*)calloc(1, sizeof(scoreJob));
    job->e = e;
    if (!readUint32Array(env, argv[1], &job->ids, &job->idCount)) {
        free(job);
        return throwError(env, "ids must be a Uint32Array");
    }
    int vocab = engineVocab(e);
    for (size_t i = 0; i < job->idCount; i++) {
        if ((int)job->ids[i] >= vocab) {
            char msg[128];
            snprintf(msg, sizeof(msg), "token %u at index %zu is outside the vocab (%d)", job->ids[i],
                     i, vocab);
            free(job->ids);
            free(job);
            return throwError(env, msg);
        }
    }
    job->prefill = (int)getDoubleProp(env, argv[2], "prefill", 4096);
    job->decode = (int)getDoubleProp(env, argv[2], "decode", 4096);
    job->chunks = (int)getDoubleProp(env, argv[2], "chunks", 1);

    napi_value promise;
    napi_create_promise(env, &job->deferred, &promise);
    napi_value name;
    napi_create_string_utf8(env, "vkScore", NAPI_AUTO_LENGTH, &name);
    napi_create_threadsafe_function(env, argv[3], NULL, name, 0, 1, NULL, NULL, NULL, scoreTsfnCall,
                                    &job->tsfn);
    napi_create_async_work(env, NULL, name, scoreExecute, scoreComplete, job, &job->work);
    g_busy = 1;
    napi_queue_async_work(env, job->work);
    return promise;
}

static napi_value requestStop(napi_env env, napi_callback_info info) {
    size_t argc = 1;
    napi_value argv[1];
    napi_get_cb_info(env, info, &argc, argv, NULL, NULL);
    if (argc >= 1) {
        engine* e = unwrap(env, argv[0]);
        if (e != NULL) engineRequestStop(e);
    }
    return NULL;
}

NAPI_MODULE_INIT() {
    napi_value fn;
    napi_create_function(env, "createEngine", NAPI_AUTO_LENGTH, createEngine, NULL, &fn);
    napi_set_named_property(env, exports, "createEngine", fn);
    napi_create_function(env, "destroyEngine", NAPI_AUTO_LENGTH, destroyEngine, NULL, &fn);
    napi_set_named_property(env, exports, "destroyEngine", fn);
    napi_create_function(env, "engineInfo", NAPI_AUTO_LENGTH, engineInfo, NULL, &fn);
    napi_set_named_property(env, exports, "engineInfo", fn);
    napi_create_function(env, "tokenize", NAPI_AUTO_LENGTH, tokenize, NULL, &fn);
    napi_set_named_property(env, exports, "tokenize", fn);
    napi_create_function(env, "decode", NAPI_AUTO_LENGTH, decode, NULL, &fn);
    napi_set_named_property(env, exports, "decode", fn);
    napi_create_function(env, "generate", NAPI_AUTO_LENGTH, generate, NULL, &fn);
    napi_set_named_property(env, exports, "generate", fn);
    napi_create_function(env, "requestStop", NAPI_AUTO_LENGTH, requestStop, NULL, &fn);
    napi_set_named_property(env, exports, "requestStop", fn);
    napi_create_function(env, "score", NAPI_AUTO_LENGTH, score, NULL, &fn);
    napi_set_named_property(env, exports, "score", fn);
    return exports;
}
