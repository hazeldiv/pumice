#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "model.h"
#include "json.h"
#include "gguf.h"
#include "hqm.h"
#include "generated_vocab.h"

const char* model_shader(const char* base, QuantType q) {
    static char buf[160];
    const char* suffix = (q == QUANT_FP16) ? "FP16" : (q == QUANT_INT8) ? "INT8" : "INT4";
    snprintf(buf, sizeof(buf), "%s-%s.spv", base, suffix);
    return buf;
}

static QuantType parse_quant(const char* s, QuantType def) {
    if (s == NULL) return def;
    if (strcmp(s, "fp16") == 0) return QUANT_FP16;
    if (strcmp(s, "int8") == 0) return QUANT_INT8;
    if (strcmp(s, "int4") == 0) return QUANT_INT4;
    return def;
}

static void cfg_fatal(const char* msg) {
    fprintf(stderr, "model config: %s\n", msg);
    exit(1);
}

static int eosFromVocabJson(model_dims* d, const char* modelDir, const char* eosName) {
    char path[512];
    snprintf(path, sizeof(path), "%s/vocab/vocab.json", modelDir);
    json_value* vj = json_parse_file(path);
    if (vj == NULL || vj->type != JSON_OBJECT) cfg_fatal("cannot parse vocab/vocab.json");
    json_value* eosId = json_get(vj, eosName);
    if (eosId == NULL || eosId->type != JSON_NUMBER) return 0;
    d->eos = (int)eosId->number;
    json_free(vj);
    return 1;
}

static int eosFromTokenizerJson(model_dims* d, const char* modelDir, const char* eosName) {
    char path[512];
    snprintf(path, sizeof(path), "%s/tokenizer.json", modelDir);
    json_value* tj = json_parse_file(path);
    if (tj == NULL) cfg_fatal("cannot parse tokenizer.json");
    json_value* added = json_get(tj, "added_tokens");
    if (added == NULL || added->type != JSON_ARRAY) cfg_fatal("tokenizer.json missing added_tokens");
    int found = 0;
    for (int i = 0; i < added->count; i++) {
        json_value* tok = &added->items[i];
        const char* content = json_get_str(tok, "content", NULL);
        if (content != NULL && strcmp(content, eosName) == 0) {
            d->eos = json_get_int(tok, "id", -1);
            found = 1;
            break;
        }
    }
    json_free(tj);
    return found;
}

int parseEos(model_dims* d, const char* modelDir, int pruned) {
    if (hqm_path_is_file(modelDir)) {
        hqm h;
        if (hqm_open(&h, modelDir) != 0) cfg_fatal("cannot parse hqm");
        d->eos = (int)hqm_meta_int(&h, "hqm.eos", d->eos);
        hqm_close(&h);
        return 0;
    }
    if (!pruned && gguf_path_is_file(modelDir)) {
        gguf g;
        if (gguf_open(&g, modelDir) != 0) cfg_fatal("cannot open gguf");
        const gguf_kv* kv = gguf_kv_find(&g, "tokenizer.ggml.eos_token_id");
        if (kv == NULL) cfg_fatal("gguf missing eos token id");
        d->eos = (int)kv->ival;
        gguf_close(&g);
        return 0;
    }

    char dir[512];
    snprintf(dir, sizeof(dir), "%s", modelDir);
    if (gguf_path_is_file(modelDir)) gguf_dir_of(modelDir, dir, sizeof(dir));

    char path[512];
    const char* cfgName = pruned ? "vocab/tokenizer_config.json" : "tokenizer_config.json";
    snprintf(path, sizeof(path), "%s/%s", dir, cfgName);
    json_value* tc = json_parse_file(path);
    if (tc == NULL) {
        char msg[256];
        snprintf(msg, sizeof(msg), "cannot parse %s", cfgName);
        cfg_fatal(msg);
    }
    const char* eosSrc = json_get_str(tc, "eos_token", "<|im_end|>");
    json_value* eosTok = json_get(tc, "eos_token");
    if (eosTok != NULL && eosTok->type == JSON_OBJECT) {
        eosSrc = json_get_str(eosTok, "content", "<|im_end|>");
    }
    char eosName[128];
    snprintf(eosName, sizeof(eosName), "%s", eosSrc);
    json_free(tc);

    if (pruned) {
        if (!eosFromVocabJson(d, dir, eosName)) {
            char msg[256];
            snprintf(msg, sizeof(msg), "eos token %s not found in vocab/vocab.json", eosName);
            cfg_fatal(msg);
        }
        return 0;
    }

    char vocabPath[512];
    snprintf(vocabPath, sizeof(vocabPath), "%s/vocab.json", dir);
    json_value* vj = json_parse_file(vocabPath);
    if (vj != NULL && vj->type == JSON_OBJECT) {
        json_value* eosId = json_get(vj, eosName);
        if (eosId != NULL && eosId->type == JSON_NUMBER) {
            d->eos = (int)eosId->number;
            json_free(vj);
            return 0;
        }
        json_free(vj);
    }
    if (!eosFromTokenizerJson(d, dir, eosName)) {
        char msg[256];
        snprintf(msg, sizeof(msg), "eos token %s not found", eosName);
        cfg_fatal(msg);
    }
    return 0;
}

static void deriveDims(model_dims* d) {
    d->rotaryDim = (int)(d->headDim * d->partialRotary);
    d->rotaryHalf = d->rotaryDim / 2;
    d->qOff = d->heads * d->headDim;
    d->gOff = d->qOff;
    d->kOff = (d->heads + d->heads) * d->headDim;
    d->vOff = (d->heads + d->heads + d->kvHeads) * d->headDim;
    d->qkvN = d->vOff + d->kvHeads * d->headDim;
    d->kvRows = d->kvHeads * d->headDim;
    d->projKOff = d->nQk * d->dim;
    d->projVOff = d->projKOff + d->nQk * d->dim;
    d->projZOff = d->projVOff + d->nV * d->dim;
    d->projAOff = d->projZOff + d->nV * d->dim;
    d->projBOff = d->projAOff + d->nV;
    d->projN = d->projBOff + d->nV;
    d->zqkvN = 2 * d->nQk * d->dim + d->nV * d->dim;
}

static void validateDims(const model_dims* d) {
    if (d->K <= 0 || d->layerCount <= 0 || d->heads <= 0 || d->kvHeads <= 0 ||
        d->headDim <= 0 || d->ffnN <= 0 || d->nQk <= 0 || d->nV <= 0 || d->dim <= 0) {
        cfg_fatal("missing or invalid dimensions");
    }
    if (d->heads % d->kvHeads != 0) cfg_fatal("heads not divisible by kv_heads");
    if (d->nV % d->nQk != 0) cfg_fatal("n_v not divisible by n_qk");
    if (d->layerCount > MODEL_MAX_LAYERS) cfg_fatal("too many layers");
    if (d->convHist < 1) cfg_fatal("invalid linear_conv_kernel_dim");
    if (d->experts > 256) cfg_fatal("too many experts");
    if (d->experts > 0 && d->expertsPerTok != 8) cfg_fatal("only top-8 routing supported");
    if (d->experts > 0 && d->moeI <= 0) cfg_fatal("moe_intermediate_size missing");
}

static void loadQuantConfig(model_config* cfg, const char* dir, int maxCtxOverride, int maxPos) {
    model_dims* d = &cfg->dims;
    char path[512];
    snprintf(path, sizeof(path), "%s/quant_config.json", dir);
    json_value* qc = json_parse_file(path);
    if (qc == NULL) cfg_fatal("cannot parse quant_config.json");

    d->maxCtx = json_get_int(qc, "max_ctx", 0);
    if (d->maxCtx <= 0) d->maxCtx = json_get_int(qc, "max-ctx", 32768);
    if (maxCtxOverride > 0) d->maxCtx = maxCtxOverride;
    if (d->maxCtx < 1) cfg_fatal("invalid max_ctx");
    if (maxPos > 0 && d->maxCtx > maxPos) {
        fprintf(stderr, "max_ctx %d exceeds max_position_embeddings %d, clamping\n", d->maxCtx, maxPos);
        d->maxCtx = maxPos;
    }
    d->prefillChunk = json_get_int(qc, "prefill_chunk", 512);

    const char* combined = json_get_str(qc, "embed/lm_head", NULL);
    cfg->embedQ = parse_quant(combined ? combined : json_get_str(qc, "embed", "fp16"), QUANT_FP16);
    cfg->lmHeadQ = parse_quant(combined ? combined : json_get_str(qc, "lm_head", "fp16"), QUANT_FP16);

    json_value* layers = json_get(qc, "layers");
    if (layers == NULL || layers->type != JSON_ARRAY || layers->count != d->layerCount) {
        cfg_fatal("quant_config.json layers mismatch");
    }
    for (int i = 0; i < d->layerCount; i++) {
        json_value* ly = &layers->items[i];
        cfg->layers[i].attn.q = parse_quant(json_get_str(ly, "attn", "fp16"), QUANT_FP16);
        cfg->layers[i].ffn.q = parse_quant(json_get_str(ly, "ffn", "fp16"), QUANT_FP16);
        cfg->layers[i].ffn.type = d->experts > 0 ? FFN_MOE : FFN_SWIGLU;
    }

    cfg->expertsVram = json_get_int(qc, "experts_vram", d->experts);
    if (cfg->expertsVram < 1 || (d->experts > 0 && cfg->expertsVram > d->experts)) {
        cfg->expertsVram = d->experts;
    }

    snprintf(cfg->name, sizeof(cfg->name), "%s", json_get_str(qc, "name", "model"));
    snprintf(cfg->shaderDir, sizeof(cfg->shaderDir), "%s", json_get_str(qc, "shader_dir", ""));
    json_free(qc);
}

static void loadGgufConfig(model_config* cfg, const char* ggufPath, int maxCtxOverride, int pruned) {
    memset(cfg, 0, sizeof(model_config));

    gguf g;
    if (gguf_open(&g, ggufPath) != 0) cfg_fatal("cannot parse gguf");
    if (gguf_arch(&g)[0] == '\0') cfg_fatal("gguf missing general.architecture");

    model_dims* d = &cfg->dims;
    d->K = (int)gguf_meta_int_arch(&g, "embedding_length", 0);
    d->layerCount = (int)gguf_meta_int_arch(&g, "block_count", 0);
    d->heads = (int)gguf_meta_int_arch(&g, "attention.head_count", 0);
    d->kvHeads = (int)gguf_meta_int_arch(&g, "attention.head_count_kv", 0);
    d->headDim = (int)gguf_meta_int_arch(&g, "attention.key_length", 0);
    d->ffnN = (int)gguf_meta_int_arch(&g, "feed_forward_length", 0);
    d->ropeTheta = gguf_meta_num_arch(&g, "rope.freq_base", 1e7);
    d->convHist = (int)gguf_meta_int_arch(&g, "ssm.conv_kernel", 4) - 1;
    d->nQk = (int)gguf_meta_int_arch(&g, "ssm.group_count", 0);
    d->dim = (int)gguf_meta_int_arch(&g, "ssm.state_size", 0);
    d->experts = (int)gguf_meta_int_arch(&g, "expert_count", 0);
    d->expertsPerTok = (int)gguf_meta_int_arch(&g, "expert_used_count", 0);
    d->moeI = (int)gguf_meta_int_arch(&g, "expert_feed_forward_length", 0);
    if (d->ffnN <= 0 && d->moeI > 0) d->ffnN = d->moeI;

    int64_t inner = gguf_meta_int_arch(&g, "ssm.inner_size", 0);
    if (d->dim > 0 && inner > 0 && inner % d->dim == 0) d->nV = (int)(inner / d->dim);

    int64_t ropeDim = gguf_meta_int_arch(&g, "rope.dimension_count", 0);
    d->partialRotary = (d->headDim > 0 && ropeDim > 0) ? (double)ropeDim / d->headDim : 0.25;

    int64_t interval = gguf_meta_int_arch(&g, "full_attention_interval", 0);
    if (interval <= 0) cfg_fatal("gguf missing full_attention_interval");
    for (int i = 0; i < d->layerCount; i++) {
        cfg->layers[i].attn.type = ((i + 1) % (int)interval == 0) ? ATTENTION_FULL : ATTENTION_DELTA;
    }
    if (cfg->layers[0].attn.type != ATTENTION_DELTA) cfg_fatal("layer 0 must be linear_attention");

    d->tied = gguf_find(&g, "output.weight") == NULL;
    int64_t vocab = gguf_meta_arr_count(&g, "tokenizer.ggml.tokens", 0);
    if (vocab <= 0) {
        const gguf_tensor* te = gguf_find(&g, "token_embd.weight");
        if (te != NULL && te->nDims == 2) vocab = te->dims[1];
    }
    d->vocab = pruned ? MODEL_VOCAB : (int)vocab;
    if (d->vocab <= 0) cfg_fatal("invalid vocab size");
    cfg->pruned = pruned;

    int maxPos = (int)gguf_meta_int_arch(&g, "context_length", 0);
    validateDims(d);
    deriveDims(d);

    char dir[512];
    gguf_dir_of(ggufPath, dir, sizeof(dir));
    loadQuantConfig(cfg, dir, maxCtxOverride, maxPos);
    gguf_close(&g);
}

static void loadHqmConfig(model_config* cfg, const char* hqmPath, int maxCtxOverride) {
    memset(cfg, 0, sizeof(model_config));

    hqm h;
    if (hqm_open(&h, hqmPath) != 0) cfg_fatal("cannot parse hqm");

    model_dims* d = &cfg->dims;
    snprintf(cfg->name, sizeof(cfg->name), "%s", hqm_meta_str(&h, "general.name", "model"));
    snprintf(cfg->shaderDir, sizeof(cfg->shaderDir), "%s", hqm_meta_str(&h, "hqm.shader_dir", ""));
    d->K = (int)hqm_meta_int(&h, "hqm.K", 0);
    d->layerCount = (int)hqm_meta_int(&h, "hqm.layer_count", 0);
    d->ffnN = (int)hqm_meta_int(&h, "hqm.ffn", 0);
    d->heads = (int)hqm_meta_int(&h, "hqm.heads", 0);
    d->kvHeads = (int)hqm_meta_int(&h, "hqm.kv_heads", 0);
    d->headDim = (int)hqm_meta_int(&h, "hqm.head_dim", 0);
    int rotaryDim = (int)hqm_meta_int(&h, "hqm.rotary_dim", 0);
    d->ropeTheta = hqm_meta_num(&h, "hqm.rope_theta", 1e7);
    d->partialRotary = d->headDim > 0 ? (double)rotaryDim / d->headDim : 0.25;
    d->nQk = (int)hqm_meta_int(&h, "hqm.n_qk", 0);
    d->nV = (int)hqm_meta_int(&h, "hqm.n_v", 0);
    d->dim = (int)hqm_meta_int(&h, "hqm.dim", 0);
    d->convHist = (int)hqm_meta_int(&h, "hqm.conv_hist", 3);
    d->vocab = (int)hqm_meta_int(&h, "hqm.vocab", 0);
    d->eos = (int)hqm_meta_int(&h, "hqm.eos", 0);
    d->tied = (int)hqm_meta_int(&h, "hqm.tied", 0);
    d->maxCtx = (int)hqm_meta_int(&h, "hqm.max_ctx", 32768);
    d->prefillChunk = (int)hqm_meta_int(&h, "hqm.prefill_chunk", 512);
    d->experts = (int)hqm_meta_int(&h, "hqm.experts", 0);
    d->expertsPerTok = (int)hqm_meta_int(&h, "hqm.experts_per_tok", 0);
    d->moeI = (int)hqm_meta_int(&h, "hqm.moe_i", 0);
    if (d->ffnN <= 0 && d->moeI > 0) d->ffnN = d->moeI;
    cfg->embedQ = (QuantType)hqm_meta_int(&h, "hqm.embed_quant", QUANT_FP16);
    cfg->lmHeadQ = (QuantType)hqm_meta_int(&h, "hqm.lm_head_quant", QUANT_FP16);
    cfg->expertsVram = (int)hqm_meta_int(&h, "hqm.experts_vram", d->experts);
    cfg->pruned = (int)hqm_meta_int(&h, "hqm.pruned", 0);

    validateDims(d);
    deriveDims(d);

    const hqm_tensor* lt = hqm_tensor_find(&h, "config.layer_type");
    const hqm_tensor* la = hqm_tensor_find(&h, "config.layer_attn_quant");
    const hqm_tensor* lf = hqm_tensor_find(&h, "config.layer_ffn_quant");
    if (lt == NULL || la == NULL || lf == NULL) cfg_fatal("hqm missing layer config");
    int64_t n1 = 0, n2 = 0, n3 = 0;
    int32_t* types = (int32_t*)hqm_tensor_read(&h, lt, &n1);
    int32_t* aq = (int32_t*)hqm_tensor_read(&h, la, &n2);
    int32_t* fq = (int32_t*)hqm_tensor_read(&h, lf, &n3);
    if (types == NULL || aq == NULL || fq == NULL ||
        n1 / 4 != d->layerCount || n2 / 4 != d->layerCount || n3 / 4 != d->layerCount) {
        cfg_fatal("hqm layer config mismatch");
    }
    for (int i = 0; i < d->layerCount; i++) {
        cfg->layers[i].attn.type = (attention_type)types[i];
        cfg->layers[i].attn.q = (QuantType)aq[i];
        cfg->layers[i].ffn.type = d->experts > 0 ? FFN_MOE : FFN_SWIGLU;
        cfg->layers[i].ffn.q = (QuantType)fq[i];
    }
    free(types);
    free(aq);
    free(fq);

    if (maxCtxOverride > 0) d->maxCtx = maxCtxOverride;
    if (d->maxCtx < 1) cfg_fatal("invalid max_ctx");
    if (cfg->expertsVram < 1 || (d->experts > 0 && cfg->expertsVram > d->experts)) {
        cfg->expertsVram = d->experts;
    }
    hqm_close(&h);
}

int loadModelConfig(model_config* cfg, const char* modelDir, int maxCtxOverride, int pruned) {
    if (hqm_path_is_file(modelDir)) {
        loadHqmConfig(cfg, modelDir, maxCtxOverride);
        return 0;
    }
    if (gguf_path_is_file(modelDir)) {
        loadGgufConfig(cfg, modelDir, maxCtxOverride, pruned);
        return 0;
    }

    memset(cfg, 0, sizeof(model_config));

    char path[512];
    snprintf(path, sizeof(path), "%s/config.json", modelDir);
    json_value* hf = json_parse_file(path);
    if (hf == NULL) cfg_fatal("cannot parse config.json");

    json_value* txt = json_get(hf, "text_config");
    if (txt == NULL) cfg_fatal("config.json missing text_config");

    model_dims* d = &cfg->dims;
    d->K = json_get_int(txt, "hidden_size", 0);
    d->layerCount = json_get_int(txt, "num_hidden_layers", 0);
    d->heads = json_get_int(txt, "num_attention_heads", 0);
    d->kvHeads = json_get_int(txt, "num_key_value_heads", 0);
    d->headDim = json_get_int(txt, "head_dim", 0);
    d->ffnN = json_get_int(txt, "intermediate_size", 0);
    d->nQk = json_get_int(txt, "linear_num_key_heads", 0);
    d->nV = json_get_int(txt, "linear_num_value_heads", 0);
    d->dim = json_get_int(txt, "linear_value_head_dim", 0);
    int linKeyDim = json_get_int(txt, "linear_key_head_dim", 0);
    d->convHist = json_get_int(txt, "linear_conv_kernel_dim", 4) - 1;
    d->ropeTheta = json_get_num(txt, "rope_theta", 1e7);
    d->tied = json_get_bool(txt, "tie_word_embeddings", 0);
    d->partialRotary = json_get_num(txt, "partial_rotary_factor", 0.25);
    int hfVocab = json_get_int(txt, "vocab_size", MODEL_VOCAB);
    int maxPos = json_get_int(txt, "max_position_embeddings", 0);
    d->experts = json_get_int(txt, "num_experts", 0);
    d->expertsPerTok = json_get_int(txt, "num_experts_per_tok", 0);
    d->moeI = json_get_int(txt, "moe_intermediate_size", 0);
    if (d->ffnN <= 0 && d->moeI > 0) d->ffnN = d->moeI;
    json_value* rope = json_get(txt, "rope_parameters");
    if (rope != NULL) {
        d->ropeTheta = json_get_num(rope, "rope_theta", d->ropeTheta);
        d->partialRotary = json_get_num(rope, "partial_rotary_factor", d->partialRotary);
    }

    json_value* layerTypes = json_get(txt, "layer_types");
    if (layerTypes == NULL || layerTypes->type != JSON_ARRAY) cfg_fatal("config.json missing layer_types");

    validateDims(d);
    if (linKeyDim != d->dim) cfg_fatal("linear key/value head dims differ");
    if (layerTypes->count != d->layerCount) cfg_fatal("layer_types count mismatch");

    deriveDims(d);

    for (int i = 0; i < d->layerCount; i++) {
        json_value* lt = &layerTypes->items[i];
        if (lt->type != JSON_STRING) cfg_fatal("layer_types entry not a string");
        if (strcmp(lt->string, "full_attention") == 0) {
            cfg->layers[i].attn.type = ATTENTION_FULL;
        } else if (strcmp(lt->string, "linear_attention") == 0) {
            cfg->layers[i].attn.type = ATTENTION_DELTA;
        } else {
            cfg_fatal("unknown layer type");
        }
        if (i == 0 && cfg->layers[i].attn.type != ATTENTION_DELTA) {
            cfg_fatal("layer 0 must be linear_attention");
        }
    }

    json_free(hf);

    d->vocab = pruned ? MODEL_VOCAB : hfVocab;
    if (d->vocab <= 0) cfg_fatal("invalid vocab size");
    cfg->pruned = pruned;

    loadQuantConfig(cfg, modelDir, maxCtxOverride, maxPos);

    return 0;
}
