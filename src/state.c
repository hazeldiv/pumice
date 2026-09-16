#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "state.h"

static buffer createZeroed(session s, int64_t size, const char* name) {
    uint8_t* zeros = (uint8_t*)calloc(1, (size_t)size);
    buffer b = createBufferNamed(s.dev.device, s.dev.physicalDevice, zeros, size, MEMORY_VRAM, name);
    free(zeros);
    return b;
}

model_state createState(session s, const model_config* spec, int maxM, int vocab, int verbose) {
    model_state st = {0};
    const model_dims* d = &spec->dims;
    st.maxM = maxM;
    st.layerCount = d->layerCount;
    int maxCtx = d->maxCtx;
    int mn = maxM * d->K;

    st.layerBufs = (buffer*)calloc((size_t)d->layerCount * 8, sizeof(buffer));
    st.kCache = st.layerBufs + 0 * d->layerCount;
    st.vCache = st.layerBufs + 1 * d->layerCount;
    st.kScale = st.layerBufs + 2 * d->layerCount;
    st.kZero = st.layerBufs + 3 * d->layerCount;
    st.vScale = st.layerBufs + 4 * d->layerCount;
    st.vZero = st.layerBufs + 5 * d->layerCount;
    st.stateS = st.layerBufs + 6 * d->layerCount;
    st.convHist = st.layerBufs + 7 * d->layerCount;

    st.h = createZeroed(s, (int64_t)sizeof(float) * mn, "h");
    st.act = createZeroed(s, (int64_t)sizeof(float) * maxM * d->ffnN, "act");
    st.embOut = createZeroed(s, (int64_t)sizeof(float) * mn, "embOut");
    st.embStaged = createZeroed(s, (int64_t)sizeof(float) * mn, "embStaged");
    st.yGated = createZeroed(s, (int64_t)sizeof(float) * maxM * (d->nV * d->dim), "yGated");
    st.attnOut = createZeroed(s, (int64_t)sizeof(float) * maxM * d->qOff, "attnOut");
    st.gAttn = createZeroed(s, (int64_t)sizeof(float) * maxM * d->qOff, "gAttn");
    st.qOut = createZeroed(s, (int64_t)sizeof(float) * mn, "qOut");
    st.qProj = createZeroed(s, (int64_t)sizeof(float) * maxM * (d->nQk * d->dim), "qProj");
    st.kProj = createZeroed(s, (int64_t)sizeof(float) * maxM * (d->nQk * d->dim), "kProj");
    st.vProj = createZeroed(s, (int64_t)sizeof(float) * maxM * (d->nV * d->dim), "vProj");
    st.zProj = createZeroed(s, (int64_t)sizeof(float) * maxM * (d->nV * d->dim), "zProj");
    st.aProj = createZeroed(s, (int64_t)sizeof(float) * maxM * d->nV, "aProj");
    st.bProj = createZeroed(s, (int64_t)sizeof(float) * maxM * d->nV, "bProj");

    for (int L = 0; L < spec->dims.layerCount; L++) {
        const layer* ly = &spec->layers[L];
        if (ly->attn.type == ATTENTION_FULL) {
            int64_t cacheBytes = (int64_t)d->kvRows * maxCtx;
            char nk[64], nv[64], nks[64], nkz[64], nvs[64], nvz[64];
            snprintf(nk, sizeof(nk), "kCache_%d", L);
            snprintf(nv, sizeof(nv), "vCache_%d", L);
            snprintf(nks, sizeof(nks), "kScale_%d", L);
            snprintf(nkz, sizeof(nkz), "kZero_%d", L);
            snprintf(nvs, sizeof(nvs), "vScale_%d", L);
            snprintf(nvz, sizeof(nvz), "vZero_%d", L);
            if (ly->attn.q != QUANT_FP16) {
                st.kCache[L] = createZeroed(s, cacheBytes, nk);
                st.vCache[L] = createZeroed(s, cacheBytes, nv);
                st.kScale[L] = createZeroed(s, (int64_t)sizeof(float) * d->kvHeads * maxCtx, nks);
                st.kZero[L] = createZeroed(s, (int64_t)sizeof(float) * d->kvHeads * maxCtx, nkz);
                st.vScale[L] = createZeroed(s, (int64_t)sizeof(float) * d->kvHeads * maxCtx, nvs);
                st.vZero[L] = createZeroed(s, (int64_t)sizeof(float) * d->kvHeads * maxCtx, nvz);
                if (verbose) fprintf(stderr, "Allocating %lld MB for layer %d KV cache\n", (long long)cacheBytes * 2 / (1024 * 1024), L);
            } else {
                if (verbose) fprintf(stderr, "Allocating %lld MB for layer %d KV cache\n", (long long)cacheBytes * 4 / (1024 * 1024), L);
                st.kCache[L] = createZeroed(s, cacheBytes * 2, nk);
                st.vCache[L] = createZeroed(s, cacheBytes * 2, nv);
            }
        } else if (ly->attn.type == ATTENTION_DELTA) {
            char ns[64];
            snprintf(ns, sizeof(ns), "stateS_%d", L);
            st.stateS[L] = createZeroed(s, (int64_t)sizeof(float) * d->nV * d->dim * d->dim, ns);
            char nh[64];
            snprintf(nh, sizeof(nh), "convHist_%d", L);
            st.convHist[L] = createZeroed(s, (int64_t)sizeof(float) * d->convHist * d->zqkvN, nh);
        }
    }

    uint32_t zero = 0;
    st.position = createBufferNamed(s.dev.device, s.dev.physicalDevice, &zero, sizeof(uint32_t), MEMORY_RAM, "position");
    uint32_t* tokenInit = (uint32_t*)calloc(maxM, sizeof(uint32_t));
    st.tokenIds = createBufferNamed(s.dev.device, s.dev.physicalDevice, tokenInit, (int64_t)sizeof(uint32_t) * maxM, MEMORY_RAM, "tokenIds");
    free(tokenInit);
    float* lastInit = (float*)calloc(d->K, sizeof(float));
    st.lastRow = createBufferNamed(s.dev.device, s.dev.physicalDevice, lastInit, (int64_t)sizeof(float) * d->K, MEMORY_RAM, "lastRow");
    free(lastInit);
    int numGroups = (vocab + 255) / 256;
    st.maxValue = createZeroed(s, (int64_t)sizeof(float) * numGroups, "maxValue");
    st.maxIndex = createZeroed(s, (int64_t)sizeof(uint32_t) * numGroups, "maxIndex");
    st.result = createZeroed(s, sizeof(uint32_t) * 4, "result");
    st.logits = createZeroed(s, (int64_t)sizeof(float) * vocab, "logits");
    sample_params spInit = {0};
    st.sampleParams = createBufferNamed(s.dev.device, s.dev.physicalDevice, &spInit, sizeof(sample_params), MEMORY_RAM, "sampleParams");
    uint32_t seedInit = 0;
    st.sampleRng = createBufferNamed(s.dev.device, s.dev.physicalDevice, &seedInit, sizeof(uint32_t), MEMORY_RAM, "sampleRng");
    uint32_t* hist = (uint32_t*)malloc(sizeof(uint32_t) * MAX_PENALTY_LEN);
    for (int i = 0; i < MAX_PENALTY_LEN; i++) hist[i] = 0xFFFFFFFFu;
    st.sampleHistory = createBufferNamed(s.dev.device, s.dev.physicalDevice, hist, (int64_t)sizeof(uint32_t) * MAX_PENALTY_LEN, MEMORY_VRAM, "sampleHistory");
    free(hist);
    st.gemvPartial = createZeroed(s, (int64_t)sizeof(float) * 4 * d->K, "gemvPartial");
    st.qkvPartial = createZeroed(s, (int64_t)sizeof(float) * 4 * d->qkvN, "qkvPartial");
    st.ffnPartial = createZeroed(s, (int64_t)sizeof(float) * 8 * d->ffnN, "ffnPartial");
    st.linprojPartial = createZeroed(s, (int64_t)sizeof(float) * 4 * d->projN, "linprojPartial");
    st.attPartial = createZeroed(s, (int64_t)sizeof(float) * ATT_MAX_CHUNKS * d->heads * (2 + d->headDim), "attPartial");
    st.invRms = createZeroed(s, (int64_t)sizeof(float) * maxM, "invRms");
    st.attScores = createZeroed(s, (int64_t)maxM * maxCtx * d->kvHeads * 2, "attScores");
    st.smSum = createZeroed(s, (int64_t)sizeof(float) * maxM * d->kvHeads, "smSum");
    st.qkvRaw = createZeroed(s, (int64_t)sizeof(float) * maxM * d->qkvN, "qkvRaw");
    st.gAct = createZeroed(s, (int64_t)sizeof(float) * maxM * d->ffnN, "gAct");
    st.uAct = createZeroed(s, (int64_t)sizeof(float) * maxM * d->ffnN, "uAct");

    if (d->experts > 0) {
        int slots = d->expertsPerTok + 1;
        uint32_t* idsInit = (uint32_t*)calloc((size_t)maxM * slots, sizeof(uint32_t));
        for (int m = 0; m < maxM; m++) {
            idsInit[m * slots + slots - 1] = (uint32_t)d->experts;
        }
        st.moeIds = createBufferNamed(s.dev.device, s.dev.physicalDevice, idsInit, (int64_t)sizeof(uint32_t) * maxM * slots, MEMORY_VRAM, "moeIds");
        free(idsInit);
        st.moeWeights = createZeroed(s, (int64_t)sizeof(float) * maxM * slots, "moeWeights");
        st.moeSharedW = createZeroed(s, (int64_t)sizeof(float) * maxM, "moeSharedW");
        st.moeXn = createZeroed(s, (int64_t)sizeof(float) * maxM * d->K, "moeXn");
        st.moeH = createZeroed(s, (int64_t)sizeof(float) * maxM * slots * d->moeI, "moeH");
        st.moeP = createZeroed(s, (int64_t)sizeof(float) * maxM * slots * d->K, "moeP");
    }

    buffer* bufs[35 + 6 * MODEL_MAX_LAYERS] = {
        &st.h, &st.act, &st.embStaged, &st.embOut, &st.yGated, &st.attnOut, &st.gAttn, &st.qOut,
        &st.qProj, &st.kProj, &st.vProj, &st.zProj, &st.aProj, &st.bProj,
        &st.maxValue, &st.maxIndex, &st.result,
        &st.gAct, &st.uAct, &st.invRms, &st.qkvRaw, &st.gemvPartial, &st.qkvPartial,
        &st.ffnPartial, &st.linprojPartial, &st.attPartial, &st.smSum, &st.logits, &st.attScores,
        &st.moeIds, &st.moeWeights, &st.moeSharedW, &st.moeXn, &st.moeH, &st.moeP
    };
    int nb = 35;
    for (int L = 0; L < d->layerCount; L++) {
        if (spec->layers[L].attn.type == ATTENTION_FULL) {
            bufs[nb++] = &st.kCache[L];
            bufs[nb++] = &st.vCache[L];
            bufs[nb++] = &st.kScale[L];
            bufs[nb++] = &st.kZero[L];
            bufs[nb++] = &st.vScale[L];
            bufs[nb++] = &st.vZero[L];
        }
    }
    buffer transfer[35 + 6 * MODEL_MAX_LAYERS];
    for (int i = 0; i < nb; i++) transfer[i] = *bufs[i];
    createTransferAndCopy(s.dev.device, s.dev.queue, transfer, nb);
    for (int i = 0; i < nb; i++) releaseStaging(s.dev.device, bufs[i]);

    buffer* persist[2 * MODEL_MAX_LAYERS + 1];
    int np = 0;
    for (int L = 0; L < d->layerCount; L++) {
        if (spec->layers[L].attn.type == ATTENTION_DELTA) {
            persist[np++] = &st.stateS[L];
            persist[np++] = &st.convHist[L];
        }
    }
    persist[np++] = &st.sampleHistory;
    buffer persistTransfer[2 * MODEL_MAX_LAYERS + 1];
    for (int i = 0; i < np; i++) persistTransfer[i] = *persist[i];
    createTransferAndCopy(s.dev.device, s.dev.queue, persistTransfer, np);

    return st;
}

void destroyState(session s, model_state* st) {
    destroyBuffer(s.dev.device, st->h);
    destroyBuffer(s.dev.device, st->act);
    destroyBuffer(s.dev.device, st->embOut);
    destroyBuffer(s.dev.device, st->embStaged);
    destroyBuffer(s.dev.device, st->yGated);
    destroyBuffer(s.dev.device, st->attnOut);
    destroyBuffer(s.dev.device, st->gAttn);
    destroyBuffer(s.dev.device, st->qOut);
    destroyBuffer(s.dev.device, st->qProj);
    destroyBuffer(s.dev.device, st->kProj);
    destroyBuffer(s.dev.device, st->vProj);
    destroyBuffer(s.dev.device, st->zProj);
    destroyBuffer(s.dev.device, st->aProj);
    destroyBuffer(s.dev.device, st->bProj);
    for (int L = 0; L < st->layerCount; L++) {
        if (st->kCache[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, st->kCache[L]);
        if (st->vCache[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, st->vCache[L]);
        if (st->kScale[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, st->kScale[L]);
        if (st->kZero[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, st->kZero[L]);
        if (st->vScale[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, st->vScale[L]);
        if (st->vZero[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, st->vZero[L]);
        if (st->stateS[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, st->stateS[L]);
        if (st->convHist[L].buffer != VK_NULL_HANDLE) destroyBuffer(s.dev.device, st->convHist[L]);
    }
    free(st->layerBufs);
    destroyBuffer(s.dev.device, st->position);
    destroyBuffer(s.dev.device, st->tokenIds);
    destroyBuffer(s.dev.device, st->maxValue);
    destroyBuffer(s.dev.device, st->maxIndex);
    destroyBuffer(s.dev.device, st->result);
    destroyBuffer(s.dev.device, st->lastRow);
    destroyBuffer(s.dev.device, st->gemvPartial);
    destroyBuffer(s.dev.device, st->qkvPartial);
    destroyBuffer(s.dev.device, st->ffnPartial);
    destroyBuffer(s.dev.device, st->linprojPartial);
    destroyBuffer(s.dev.device, st->attPartial);
    destroyBuffer(s.dev.device, st->invRms);
    destroyBuffer(s.dev.device, st->attScores);
    destroyBuffer(s.dev.device, st->smSum);
    destroyBuffer(s.dev.device, st->qkvRaw);
    destroyBuffer(s.dev.device, st->gAct);
    destroyBuffer(s.dev.device, st->uAct);
    if (st->moeIds.buffer != VK_NULL_HANDLE) {
        destroyBuffer(s.dev.device, st->moeIds);
        destroyBuffer(s.dev.device, st->moeWeights);
        destroyBuffer(s.dev.device, st->moeSharedW);
        destroyBuffer(s.dev.device, st->moeXn);
        destroyBuffer(s.dev.device, st->moeH);
        destroyBuffer(s.dev.device, st->moeP);
    }
    destroyBuffer(s.dev.device, st->logits);
    destroyBuffer(s.dev.device, st->sampleParams);
    destroyBuffer(s.dev.device, st->sampleHistory);
    destroyBuffer(s.dev.device, st->sampleRng);
}

void stateSetPosition(session s, model_state* st, uint32_t pos) {
    (void)s;
    if (st->position.mappedMemory != NULL) {
        *(uint32_t*)st->position.mappedMemory = pos;
    }
}

uint32_t stateReadPosition(session s, model_state* st) {
    uint32_t pos = 0;
    readBuffer(s.dev.device, s.dev.physicalDevice, s.dev.queue, st->position, &pos);
    return pos;
}

int stateLayerCount(const model_state* st) {
    return st->layerCount;
}
