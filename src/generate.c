#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "generate.h"

#define ATT_SPLIT_THRESHOLD 256
#define FIRST_TOKEN_DELAY_MS 25.0
#define TOPP_KEEP 512

static double g_qpcFreqMs = 0.0;

static double nowMs(void) {
    if (g_qpcFreqMs == 0.0) {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        g_qpcFreqMs = (double)f.QuadPart / 1000.0;
    }
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / g_qpcFreqMs;
}

static void sleepMs(double ms) {
    if (ms <= 0.0) return;
    double target = nowMs() + ms;
    double coarse = ms - 1.0;
    if (coarse > 0.0) Sleep((DWORD)coarse);
    while (nowMs() < target) {
    }
}

static void addOp(operation* ops, int* n, const char* shader, int layer, buffer* bufs, int bc, const int* push, int pc, int dx, int dy) {
    operation* op = &ops[*n];
    memset(op, 0, sizeof(operation));
    snprintf(op->shader, sizeof(op->shader), "%s", shader);
    memcpy(op->buffers, bufs, sizeof(buffer) * bc);
    op->bufferCount = bc;
    memcpy(op->pushConstants, push, sizeof(int) * pc);
    op->pushConstantCount = pc;
    op->dispatchX = dx;
    op->dispatchY = dy;
    op->dispatchZ = 1;
    op->layer = layer;
    (*n)++;
}

static void addGemvSplit(generator* g, operation* ops, int* n, int L, const tensor* wt, buffer input, buffer output, buffer residual, QuantType q) {
    const model_dims* d = g->dims;
    buffer bufs[5];
    int b = 0;
    bufs[b++] = input;
    bufs[b++] = wt->data;
    bufs[b++] = g->st.gemvPartial;
    if (q != QUANT_FP16) {
        bufs[b++] = wt->scale;
        bufs[b++] = wt->zero;
    }
    int push[] = {1, d->K, wt->rows};
    addOp(ops, n, model_shader("GEMV-SplitK", q), L, bufs, b, push, 3, d->K / 256, 4);

    buffer rBufs[3];
    rBufs[0] = g->st.gemvPartial;
    rBufs[1] = residual;
    rBufs[2] = output;
    int pushR[] = {d->K, 4};
    addOp(ops, n, "Reduce-GEMV-ADD.spv", L, rBufs, 3, pushR, 2, d->K / 256, 1);
}

static void addReduceGemvAdd(generator* g, operation* ops, int* n, int L, buffer partial, buffer output, buffer residual) {
    const model_dims* d = g->dims;
    buffer rBufs[3];
    rBufs[0] = partial;
    rBufs[1] = residual;
    rBufs[2] = output;
    int pushR[] = {d->K, 4};
    addOp(ops, n, "Reduce-GEMV-ADD.spv", L, rBufs, 3, pushR, 2, d->K / 256, 1);
}

static void addGemmAdd(generator* g, operation* ops, int* n, int L, const tensor* wt, buffer input, buffer output, buffer residual, QuantType q, int m) {
    const model_dims* d = g->dims;
    if (m == 1) {
        addGemvSplit(g, ops, n, L, wt, input, output, residual, q);
        return;
    }

    buffer bufs[6];
    int b = 0;
    bufs[b++] = input;
    bufs[b++] = wt->data;
    bufs[b++] = output;
    if (q != QUANT_FP16) {
        bufs[b++] = wt->scale;
        bufs[b++] = wt->zero;
    }
    bufs[b++] = residual;
    int k = (input.buffer == g->st.act.buffer) ? d->ffnN : wt->rows;
    int tn = (q == QUANT_INT4) ? 64 : 32;
    int push[] = {m, d->K, k};
    addOp(ops, n, model_shader("GEMM-ADD2", q), L, bufs, b, push, 3, d->K / tn, (m + 15) / 16);
}

static void addLinearProj(generator* g, operation* ops, int* n, int L, int gemm, int m, buffer input) {
    model_state* st = &g->st;
    model_weights* w = &g->w;
    const model_dims* d = g->dims;
    QuantType q = g->spec->layers[L].attn.q;

    if (!gemm) {
        buffer bufs[6];
        int b = 0;
        bufs[b++] = input;
        bufs[b++] = w->gammaIn[L];
        bufs[b++] = w->proj[L].data;
        if (q != QUANT_FP16) {
            bufs[b++] = w->proj[L].scale;
            bufs[b++] = w->proj[L].zero;
        }
        bufs[b++] = st->linprojPartial;
        int push[] = {m, d->projN, d->K};
        addOp(ops, n, model_shader("RmsNorm-LinearProj-SplitK", q), L, bufs, b, push, 3,
              (d->projN + 255) / 256, 4);

        buffer rBufs[7];
        rBufs[0] = st->linprojPartial;
        rBufs[1] = st->qProj;
        rBufs[2] = st->kProj;
        rBufs[3] = st->vProj;
        rBufs[4] = st->zProj;
        rBufs[5] = st->aProj;
        rBufs[6] = st->bProj;
        int pushR[] = {d->projN, d->projKOff, d->projVOff, d->projZOff, d->projAOff, d->projBOff};
        addOp(ops, n, "Reduce-LinearProj.spv", L, rBufs, 7, pushR, 6,
              (d->projN + 255) / 256, 1);
        return;
    }

    buffer bufs[14];
    int b = 0;
    bufs[b++] = input;
    bufs[b++] = w->gammaIn[L];
    bufs[b++] = w->proj[L].data;
    bufs[b++] = st->qProj;
    bufs[b++] = st->kProj;
    bufs[b++] = st->vProj;
    bufs[b++] = st->zProj;
    bufs[b++] = st->aProj;
    bufs[b++] = st->bProj;
    if (q != QUANT_FP16) {
        bufs[b++] = w->proj[L].scale;
        bufs[b++] = w->proj[L].zero;
    }
    bufs[b++] = st->invRms;
    int push[] = {m, d->projN, d->K, d->projKOff, d->projVOff, d->projZOff, d->projAOff, d->projBOff};
    int pushP[] = {d->K};
    buffer proBufs[2];
    proBufs[0] = input;
    proBufs[1] = st->invRms;
    addOp(ops, n, "RmsNorm-Prologue.spv", L, proBufs, 2, pushP, 1, m, 1);
    addOp(ops, n, model_shader("RmsNorm-LinearProj-GEMM2", q), L, bufs, b, push, 8,
          d->projN / 32, (m + 15) / 16);
}

static void buildFfn(generator* g, operation* ops, int* n, int L, int gemm, int m) {
    model_state* st = &g->st;
    model_weights* w = &g->w;
    const model_dims* d = g->dims;
    QuantType f = g->spec->layers[L].ffn.q;

    if (!gemm) {
        buffer upBufs[9];
        int b = 0;
        upBufs[b++] = st->h;
        upBufs[b++] = w->gammaF[L];
        upBufs[b++] = w->gate[L].data;
        upBufs[b++] = w->up[L].data;
        upBufs[b++] = st->ffnPartial;
        if (f != QUANT_FP16) {
            upBufs[b++] = w->gate[L].scale;
            upBufs[b++] = w->gate[L].zero;
            upBufs[b++] = w->up[L].scale;
            upBufs[b++] = w->up[L].zero;
        }
        int push[] = {m, d->ffnN, d->K, d->ffnN};
        addOp(ops, n, model_shader("RmsNorm-up-ffn-SplitK", f), L, upBufs, b, push, 4,
              2 * d->ffnN / 256, 4);

        buffer dBufs[5];
        int bd = 0;
        dBufs[bd++] = st->ffnPartial;
        dBufs[bd++] = w->down[L].data;
        dBufs[bd++] = st->gemvPartial;
        if (f != QUANT_FP16) {
            dBufs[bd++] = w->down[L].scale;
            dBufs[bd++] = w->down[L].zero;
        }
        int pushD[] = {m, d->K, d->ffnN};
        addOp(ops, n, model_shader("FFN-Down-SplitK", f), L, dBufs, bd, pushD, 3,
              d->K / 256, 4);

        addReduceGemvAdd(g, ops, n, L, st->gemvPartial, st->h, st->h);
        return;
    }

    int pushP[] = {d->K};
    buffer proBufs[2];
    proBufs[0] = st->h;
    proBufs[1] = st->invRms;
    addOp(ops, n, "RmsNorm-Prologue.spv", L, proBufs, 2, pushP, 1, m, 1);

    if (f == QUANT_FP16) {
        buffer ffnBufs[6];
        int bf = 0;
        ffnBufs[bf++] = st->h;
        ffnBufs[bf++] = w->gammaF[L];
        ffnBufs[bf++] = w->gate[L].data;
        ffnBufs[bf++] = w->up[L].data;
        ffnBufs[bf++] = st->act;
        ffnBufs[bf++] = st->invRms;
        int push[] = {m, d->ffnN, d->K};
        addOp(ops, n, model_shader("RmsNorm-swiglu-ffn-GEMM2", f), L, ffnBufs, bf, push, 3,
              d->ffnN / 32, (m + 15) / 16);
    } else {
        buffer flatBufs[11];
        int bf2 = 0;
        flatBufs[bf2++] = st->h;
        flatBufs[bf2++] = w->gammaF[L];
        flatBufs[bf2++] = w->gate[L].data;
        flatBufs[bf2++] = w->up[L].data;
        flatBufs[bf2++] = st->gAct;
        flatBufs[bf2++] = st->uAct;
        flatBufs[bf2++] = w->gate[L].scale;
        flatBufs[bf2++] = w->gate[L].zero;
        flatBufs[bf2++] = w->up[L].scale;
        flatBufs[bf2++] = w->up[L].zero;
        flatBufs[bf2++] = st->invRms;
        int pushF[] = {m, d->ffnN, d->K, d->ffnN};
        addOp(ops, n, model_shader("RmsNorm-swiglu-flat-GEMM2", f), L, flatBufs, bf2, pushF, 4,
              (d->ffnN * 2) / 32, (m + 15) / 16);

        buffer cmbBufs[3];
        cmbBufs[0] = st->gAct;
        cmbBufs[1] = st->uAct;
        cmbBufs[2] = st->act;
        int pushC[] = {m * d->ffnN};
        addOp(ops, n, "Swiglu-combine.spv", L, cmbBufs, 3, pushC, 1, (m * d->ffnN + 255) / 256, 1);
    }

    addGemmAdd(g, ops, n, L, &w->down[L], st->act, st->h, st->h, f, m);
}

static void buildAttention(generator* g, operation* ops, int* n, int L, int gemm, int m, int splitAttn, int ctx, int offset) {
    model_state* st = &g->st;
    model_weights* w = &g->w;
    const model_dims* d = g->dims;
    QuantType q = g->spec->layers[L].attn.q;

    if (gemm) {
        int pushP[] = {d->K};
        buffer proBufs[2];
        proBufs[0] = st->h;
        proBufs[1] = st->invRms;
        addOp(ops, n, "RmsNorm-Prologue.spv", L, proBufs, 2, pushP, 1, m, 1);

        buffer qkvGBufs[7];
        int bq2 = 0;
        qkvGBufs[bq2++] = st->h;
        qkvGBufs[bq2++] = w->gammaIn[L];
        qkvGBufs[bq2++] = w->proj[L].data;
        if (q != QUANT_FP16) {
            qkvGBufs[bq2++] = w->proj[L].scale;
            qkvGBufs[bq2++] = w->proj[L].zero;
        }
        qkvGBufs[bq2++] = st->qkvRaw;
        qkvGBufs[bq2++] = st->invRms;
        int tnQkv = (q == QUANT_INT4) ? 64 : 32;
        int pushQ[] = {m, d->qkvN, d->K};
        addOp(ops, n, model_shader("RmsNorm-QKV-GEMM2", q), L, qkvGBufs, bq2, pushQ, 3,
              d->qkvN / tnQkv, (m + 15) / 16);

        buffer ropeGBufs[13];
        int brg = 0;
        ropeGBufs[brg++] = st->qkvRaw;
        ropeGBufs[brg++] = st->qOut;
        ropeGBufs[brg++] = st->kCache[L];
        ropeGBufs[brg++] = st->vCache[L];
        if (q != QUANT_FP16) {
            ropeGBufs[brg++] = st->kScale[L];
            ropeGBufs[brg++] = st->kZero[L];
            ropeGBufs[brg++] = st->vScale[L];
            ropeGBufs[brg++] = st->vZero[L];
        }
        ropeGBufs[brg++] = w->theta;
        ropeGBufs[brg++] = w->qNorm[L];
        ropeGBufs[brg++] = w->kNorm[L];
        ropeGBufs[brg++] = st->gAttn;
        int pushRG[] = {d->qkvN, d->gOff, d->kOff, d->vOff, offset, d->maxCtx, d->headDim, d->rotaryDim, d->heads, d->kvHeads};
        addOp(ops, n, model_shader("Rope-GEMM", q), L, ropeGBufs, brg, pushRG, 10,
              2 * d->heads + 2 * d->kvHeads, m);

        for (int hb = 0; hb < d->heads / d->kvHeads; hb++) {
            int pushA[] = {ctx, offset, m, hb * d->kvHeads, d->maxCtx, d->kvHeads, d->qOff, d->kvRows, d->heads / d->kvHeads};

            buffer qkBufs[5];
            int bq = 0;
            qkBufs[bq++] = st->qOut;
            qkBufs[bq++] = st->kCache[L];
            if (q != QUANT_FP16) {
                qkBufs[bq++] = st->kScale[L];
                qkBufs[bq++] = st->kZero[L];
            }
            qkBufs[bq++] = st->attScores;
            addOp(ops, n, model_shader("Att-QK2", q), L, qkBufs, bq, pushA, 9, (ctx + 63) / 64, ((m + 15) / 16) * d->kvHeads);

            buffer smBufs[2];
            smBufs[0] = st->attScores;
            smBufs[1] = st->smSum;
            addOp(ops, n, "Att-Softmax.spv", L, smBufs, 2, pushA, 9, m, d->kvHeads);

            buffer pvBufs[6];
            int bp = 0;
            pvBufs[bp++] = st->attScores;
            pvBufs[bp++] = st->vCache[L];
            if (q != QUANT_FP16) {
                pvBufs[bp++] = st->vScale[L];
                pvBufs[bp++] = st->vZero[L];
            }
            pvBufs[bp++] = st->attnOut;
            pvBufs[bp++] = st->smSum;
            addOp(ops, n, model_shader("Att-PV2", q), L, pvBufs, bp, pushA, 9, d->headDim / 64, ((m + 15) / 16) * d->kvHeads);
        }

        buffer gateBufs[2];
        gateBufs[0] = st->gAttn;
        gateBufs[1] = st->attnOut;
        int pushGate[] = {m * d->qOff};
        addOp(ops, n, "Gate-Sigmoid.spv", L, gateBufs, 2, pushGate, 1,
              (m * d->qOff + 255) / 256, 1);
    } else {
        buffer splitBufs[6];
        int bs = 0;
        splitBufs[bs++] = st->h;
        splitBufs[bs++] = w->gammaIn[L];
        splitBufs[bs++] = w->proj[L].data;
        if (q != QUANT_FP16) {
            splitBufs[bs++] = w->proj[L].scale;
            splitBufs[bs++] = w->proj[L].zero;
        }
        splitBufs[bs++] = st->qkvPartial;
        int push[] = {1, d->qkvN, d->K};
        addOp(ops, n, model_shader("RmsNorm-QKV-SplitK", q), L, splitBufs, bs, push, 3,
              d->qkvN / 256, 4);

        buffer ropeBufs[13];
        int br = 0;
        ropeBufs[br++] = st->qkvPartial;
        ropeBufs[br++] = st->qOut;
        ropeBufs[br++] = st->kCache[L];
        ropeBufs[br++] = st->vCache[L];
        if (q != QUANT_FP16) {
            ropeBufs[br++] = st->kScale[L];
            ropeBufs[br++] = st->kZero[L];
            ropeBufs[br++] = st->vScale[L];
            ropeBufs[br++] = st->vZero[L];
        }
        ropeBufs[br++] = w->theta;
        ropeBufs[br++] = st->position;
        ropeBufs[br++] = w->qNorm[L];
        ropeBufs[br++] = w->kNorm[L];
        ropeBufs[br++] = st->gAttn;
        int pushRope[] = {d->qkvN, d->gOff, d->kOff, d->vOff, d->maxCtx, d->headDim, d->rotaryDim, d->heads, d->kvHeads};
        addOp(ops, n, model_shader("Reduce-Rope", q), L, ropeBufs, br, pushRope, 9,
              2 * d->heads + 2 * d->kvHeads, 1);

        buffer attBufs[10];
        int ba = 0;
        attBufs[ba++] = st->kCache[L];
        attBufs[ba++] = st->vCache[L];
        attBufs[ba++] = st->qOut;
        if (q != QUANT_FP16) {
            attBufs[ba++] = st->kScale[L];
            attBufs[ba++] = st->kZero[L];
            attBufs[ba++] = st->vScale[L];
            attBufs[ba++] = st->vZero[L];
        }
        attBufs[ba++] = st->attPartial;
        attBufs[ba++] = st->position;
        int push0[6] = {d->maxCtx, d->kvHeads, d->kvRows, d->heads / d->kvHeads, d->headDim, d->heads};
        if (splitAttn) {
            int chunks = (d->maxCtx + ATT_CHUNK_TOKENS - 1) / ATT_CHUNK_TOKENS;
            if (chunks > ATT_MAX_CHUNKS) chunks = ATT_MAX_CHUNKS;
            addOp(ops, n, model_shader("Att-SplitK2", q), L, attBufs, ba, push0, 6, d->heads, chunks);

            buffer attRedBufs[3];
            attRedBufs[0] = st->attPartial;
            attRedBufs[1] = st->attnOut;
            attRedBufs[2] = st->position;
            addOp(ops, n, "Reduce-Att2.spv", L, attRedBufs, 3, push0, 6, d->heads * d->headDim / 256, 1);
        } else {
            buffer fullBufs[10];
            int bf = 0;
            fullBufs[bf++] = st->kCache[L];
            fullBufs[bf++] = st->vCache[L];
            fullBufs[bf++] = st->qOut;
            fullBufs[bf++] = st->attnOut;
            if (q != QUANT_FP16) {
                fullBufs[bf++] = st->kScale[L];
                fullBufs[bf++] = st->kZero[L];
                fullBufs[bf++] = st->vScale[L];
                fullBufs[bf++] = st->vZero[L];
            }
            fullBufs[bf++] = st->position;
            addOp(ops, n, model_shader("Att-full", q), L, fullBufs, bf, push0, 6, d->heads, 1);
        }

        buffer gateBufs[2];
        gateBufs[0] = st->gAttn;
        gateBufs[1] = st->attnOut;
        int pushGate[] = {d->qOff};
        addOp(ops, n, "Gate-Sigmoid.spv", L, gateBufs, 2, pushGate, 1,
              (d->qOff + 255) / 256, 1);
    }

    addGemmAdd(g, ops, n, L, &w->out[L], st->attnOut, st->h, st->h, q, m);
}

static void buildDelta(generator* g, operation* ops, int* n, int L, int gemm, int m, int realM) {
    model_state* st = &g->st;
    model_weights* w = &g->w;
    const model_dims* d = g->dims;
    QuantType q = g->spec->layers[L].attn.q;

    if (L != 0) {
        addLinearProj(g, ops, n, L, gemm, m, st->h);
    }

    buffer convBufs[] = {st->qProj, st->kProj, st->vProj, w->conv[L], st->convHist[L]};
    int pushConv[] = {m, d->projKOff, d->projVOff, d->zqkvN, realM};
    addOp(ops, n, "Conv-SiLU.spv", L, convBufs, 5, pushConv, 5, d->zqkvN / 256, 1);

    buffer dnBufs[] = {st->qProj, st->kProj, st->vProj, st->aProj, st->bProj, st->stateS[L], st->yGated, st->zProj, w->attnNorm[L], w->aLog[L], w->dtBias[L]};
    if (gemm) {
        int pushDn[] = {m, d->nV, d->nQk, realM};
        addOp(ops, n, "GatedDeltaNet-GEMM.spv", L, dnBufs, 11, pushDn, 4, d->nV, 1);
    } else {
        int pushDn[] = {d->nV, d->nQk, d->dim};
        addOp(ops, n, "GatedDeltaNet.spv", L, dnBufs, 11, pushDn, 3, d->nV, 1);
    }

    addGemmAdd(g, ops, n, L, &w->out[L], st->yGated, st->h, L == 0 ? st->embOut : st->h, q, m);
}

static void buildMoe(generator* g, operation* ops, int* n, int L, int m) {
    model_state* st = &g->st;
    model_weights* w = &g->w;
    const model_dims* d = g->dims;
    int slots = d->expertsPerTok + 1;

    buffer rtBufs[8];
    rtBufs[0] = st->h;
    rtBufs[1] = w->gammaF[L];
    rtBufs[2] = w->router[L];
    rtBufs[3] = w->sharedGate[L];
    rtBufs[4] = st->moeIds;
    rtBufs[5] = st->moeWeights;
    rtBufs[6] = st->moeSharedW;
    rtBufs[7] = st->moeXn;
    int pushR[] = {m, d->experts, d->K, slots};
    addOp(ops, n, "Router-TopK.spv", L, rtBufs, 8, pushR, 4, m, 1);

    buffer guBufs[9];
    guBufs[0] = st->moeXn;
    guBufs[1] = st->moeIds;
    guBufs[2] = w->guPool[L].vramData;
    guBufs[3] = w->guPool[L].vramScale;
    guBufs[4] = w->guPool[L].vramZero;
    guBufs[5] = st->moeH;
    guBufs[6] = w->guPool[L].ramData;
    guBufs[7] = w->guPool[L].ramScale;
    guBufs[8] = w->guPool[L].ramZero;
    int pushG[] = {d->K, d->moeI, slots, w->guPool[L].vramExperts, w->guPool[L].expertCount};
    addOp(ops, n, "Expert-Swiglu-INT4.spv", L, guBufs, 9, pushG, 5,
          d->moeI / 256, m * slots);

    buffer dnBufs[9];
    dnBufs[0] = st->moeH;
    dnBufs[1] = st->moeIds;
    dnBufs[2] = w->dnPool[L].vramData;
    dnBufs[3] = w->dnPool[L].vramScale;
    dnBufs[4] = w->dnPool[L].vramZero;
    dnBufs[5] = st->moeP;
    dnBufs[6] = w->dnPool[L].ramData;
    dnBufs[7] = w->dnPool[L].ramScale;
    dnBufs[8] = w->dnPool[L].ramZero;
    int pushD[] = {d->moeI, d->K, slots, w->dnPool[L].vramExperts, w->dnPool[L].expertCount};
    addOp(ops, n, "Expert-Down-INT4.spv", L, dnBufs, 9, pushD, 5,
          d->K / 256, m * slots);

    buffer cbBufs[5];
    cbBufs[0] = st->moeP;
    cbBufs[1] = st->moeWeights;
    cbBufs[2] = st->moeSharedW;
    cbBufs[3] = st->h;
    cbBufs[4] = st->h;
    int pushC[] = {m, d->K, slots};
    addOp(ops, n, "Moe-Combine.spv", L, cbBufs, 5, pushC, 3,
          (m * d->K + 255) / 256, 1);
}

static void buildLayer(generator* g, operation* ops, int* n, int L, int gemm, int m, int splitAttn, int ctx, int offset, int realM) {
    const layer* ly = &g->spec->layers[L];
    if (ly->attn.type == ATTENTION_FULL) {
        buildAttention(g, ops, n, L, gemm, m, splitAttn, ctx, offset);
    } else if (ly->attn.type == ATTENTION_DELTA) {
        buildDelta(g, ops, n, L, gemm, m, realM);
    }
    if (ly->ffn.type == FFN_SWIGLU) {
        buildFfn(g, ops, n, L, gemm, m);
    } else if (ly->ffn.type == FFN_MOE) {
        buildMoe(g, ops, n, L, m);
    }
}

static void buildLmHead(generator* g, operation* ops, int* n, buffer* input, int doIncrement, int passIdx) {
    model_state* st = &g->st;
    model_weights* w = &g->w;
    const model_dims* d = g->dims;
    if (g->sampling) {
        buffer lmBufs[] = {*input, w->lmHead, st->logits, w->gammaFinal};
        int pushL[] = {1, g->vocab, d->K};
        addOp(ops, n, "LMHead-GEMV-FP16.spv", -1, lmBufs, 4, pushL, 3, (g->vocab + 255) / 256, 1);
    } else {
        buffer lmBufs[] = {*input, w->lmHead, st->maxValue, st->maxIndex, w->gammaFinal};
        int pushL[] = {1, g->vocab, d->K};
        addOp(ops, n, "LMHead-GEMV-ArgMax-FP16.spv", -1, lmBufs, 5, pushL, 3, (g->vocab + 255) / 256, 1);
    }

    buffer redBufs[] = {st->maxValue, st->maxIndex, st->logits, st->sampleParams, st->sampleHistory, st->sampleRng, st->result, st->position, st->tokenIds};
    int pushR[] = {g->vocab, doIncrement, passIdx, g->sampling};
    addOp(ops, n, "ArgMax-Reduce.spv", -1, redBufs, 9, pushR, 4, 1, 1);
}

static void addDecodeEmbedLinearProj(generator* g, operation* ops, int* n) {
    model_state* st = &g->st;
    model_weights* w = &g->w;
    const model_dims* d = g->dims;
    QuantType q = g->spec->layers[0].attn.q;

    if (q == QUANT_FP16) {
        buffer embedBufs[] = {st->tokenIds, w->embed, w->gammaIn[0], w->proj[0].data, st->qProj, st->kProj, st->vProj, st->zProj, st->aProj, st->bProj, st->embOut};
        int pushE[] = {1, d->projN, d->K, d->projKOff, d->projVOff, d->projZOff, d->projAOff, d->projBOff, g->vocab};
        addOp(ops, n, "Embed-RmsNorm-LinearProj-FP16.spv", -1, embedBufs, 11, pushE, 9, (d->projN + 255) / 256, 1);
        return;
    }

    buffer gatherBufs[4];
    gatherBufs[0] = st->tokenIds;
    gatherBufs[1] = w->embed;
    gatherBufs[2] = st->embStaged;
    gatherBufs[3] = st->embOut;
    int pushG[] = {g->vocab, d->K};
    addOp(ops, n, "Embed-Gather.spv", -1, gatherBufs, 4, pushG, 2, 1, 1);

    buffer bufs[6];
    int b = 0;
    bufs[b++] = st->embStaged;
    bufs[b++] = w->gammaIn[0];
    bufs[b++] = w->proj[0].data;
    bufs[b++] = w->proj[0].scale;
    bufs[b++] = w->proj[0].zero;
    bufs[b++] = st->linprojPartial;
    int push[] = {1, d->projN, d->K};
    addOp(ops, n, model_shader("RmsNorm-LinearProj-SplitK", q), 0, bufs, b, push, 3,
          (d->projN + 255) / 256, 4);

    buffer rBufs[7];
    rBufs[0] = st->linprojPartial;
    rBufs[1] = st->qProj;
    rBufs[2] = st->kProj;
    rBufs[3] = st->vProj;
    rBufs[4] = st->zProj;
    rBufs[5] = st->aProj;
    rBufs[6] = st->bProj;
    int pushR[] = {d->projN, d->projKOff, d->projVOff, d->projZOff, d->projAOff, d->projBOff};
    addOp(ops, n, "Reduce-LinearProj.spv", 0, rBufs, 7, pushR, 6,
          (d->projN + 255) / 256, 1);
}

static int compileDecodeGroup(generator* g, operation* ops, int splitAttn) {
    model_state* st = &g->st;
    const model_dims* d = g->dims;
    int n = 0;

    for (int p = 0; p < DECODE_GROUP; p++) {
        addDecodeEmbedLinearProj(g, ops, &n);

        for (int L = 0; L < g->layerCount; L++) {
            buildLayer(g, ops, &n, L, 0, 1, splitAttn, 0, 0, 1);
        }

        buildLmHead(g, ops, &n, &st->h, 1, p);
    }
    return n;
}

static int compilePrefillHead(generator* g, operation* ops, int m) {
    model_state* st = &g->st;
    model_weights* w = &g->w;
    const model_dims* d = g->dims;
    int n = 0;

    buffer gatherBufs[4];
    gatherBufs[0] = st->tokenIds;
    gatherBufs[1] = w->embed;
    gatherBufs[2] = st->embStaged;
    gatherBufs[3] = st->embOut;
    int pushG[] = {g->vocab, d->K};
    addOp(ops, &n, "Embed-Gather.spv", -1, gatherBufs, 4, pushG, 2, m, 1);

    addLinearProj(g, ops, &n, 0, 1, m, st->embStaged);

    return n;
}

static int compilePrefill(generator* g, int m, int offset, int realM) {
    operation* ops = g->prefillOps;
    int n = compilePrefillHead(g, ops, m);
    for (int L = 0; L < g->layerCount; L++) {
        buildLayer(g, ops, &n, L, 1, m, 0, offset + m, offset, realM);
    }
    return n;
}

static int compileFinal(generator* g) {
    int n = 0;
    buildLmHead(g, g->finalOps, &n, &g->st.lastRow, 0, 0);
    return n;
}

generator* createGenerator(session s, const model_config* spec, const char* weightDir, int verboseWeights) {
    static generator g;
    memset(&g, 0, sizeof(generator));
    g.s = s;
    g.spec = spec;
    g.dims = &spec->dims;
    g.layerCount = spec->dims.layerCount;
    g.eos = spec->dims.eos;
    g.maxCtx = spec->dims.maxCtx;
    g.maxM = spec->dims.prefillChunk;
    setShaderRootDir(spec->shaderDir);
    pipelineSetSpecInt(0, spec->dims.K);
    g.w = createWeights(s, spec, weightDir, verboseWeights);
    g.vocab = g.w.vocab;
    g.st = createState(s, spec, g.maxM, g.vocab, verboseWeights);
    g.groupOpCount = compileDecodeGroup(&g, g.groupOps, 1);
    g.groupOpCountShort = compileDecodeGroup(&g, g.groupOpsShort, 0);
    g.prefillOpCount = compilePrefill(&g, g.maxM, 0, g.maxM);
    g.finalOpCount = compileFinal(&g);
    return &g;
}

void destroyGenerator(generator* g) {
    destroyState(g->s, &g->st);
    destroyWeights(g->s, &g->w);
}

static void dumpBuffer(generator* g, buffer buf, const char* name, int64_t floats) {
    if (g->dumpDir[0] == '\0') return;
    char path[320];
    snprintf(path, sizeof(path), "%s/%s.bin", g->dumpDir, name);
    float* host = (float*)malloc((size_t)buf.size);
    if (host == NULL) return;
    readBuffer(g->s.dev.device, g->s.dev.physicalDevice, g->s.dev.queue, buf, host);
    FILE* f = fopen(path, "wb");
    if (f != NULL) {
        int64_t avail = (int64_t)buf.size / (int64_t)sizeof(float);
        int64_t n = floats < avail ? floats : avail;
        fwrite(host, sizeof(float), (size_t)n, f);
        fclose(f);
    }
    free(host);
}

static void dumpBufferFp16(generator* g, buffer buf, const char* name, int64_t halves) {
    if (g->dumpDir[0] == '\0') return;
    char path[320];
    snprintf(path, sizeof(path), "%s/%s.bin", g->dumpDir, name);
    uint16_t* host = (uint16_t*)malloc((size_t)buf.size);
    if (host == NULL) return;
    readBuffer(g->s.dev.device, g->s.dev.physicalDevice, g->s.dev.queue, buf, host);
    FILE* f = fopen(path, "wb");
    if (f != NULL) {
        int64_t avail = (int64_t)buf.size / (int64_t)sizeof(uint16_t);
        int64_t n = halves < avail ? halves : avail;
        fwrite(host, sizeof(uint16_t), (size_t)n, f);
        fclose(f);
    }
    free(host);
}

static void dumpBufferU8(generator* g, buffer buf, const char* name, int64_t bytes) {
    if (g->dumpDir[0] == '\0') return;
    char path[320];
    snprintf(path, sizeof(path), "%s/%s.bin", g->dumpDir, name);
    uint8_t* host = (uint8_t*)malloc((size_t)buf.size);
    if (host == NULL) return;
    readBuffer(g->s.dev.device, g->s.dev.physicalDevice, g->s.dev.queue, buf, host);
    FILE* f = fopen(path, "wb");
    if (f != NULL) {
        int64_t avail = (int64_t)buf.size;
        int64_t n = bytes < avail ? bytes : avail;
        fwrite(host, 1, (size_t)n, f);
        fclose(f);
    }
    free(host);
}

typedef struct {
    float logit;
    uint32_t id;
} topp_entry;

static void heapPush(topp_entry* heap, int* n, float logit, uint32_t id) {
    int i = (*n)++;
    while (i > 0) {
        int p = (i - 1) / 2;
        if (heap[p].logit <= logit) break;
        heap[i] = heap[p];
        i = p;
    }
    heap[i].logit = logit;
    heap[i].id = id;
}

static void heapReplaceRoot(topp_entry* heap, int n, float logit, uint32_t id) {
    int i = 0;
    for (;;) {
        int l = 2 * i + 1;
        int r = l + 1;
        int m = i;
        float mv = logit;
        if (l < n && heap[l].logit < mv) { m = l; mv = heap[l].logit; }
        if (r < n && heap[r].logit < mv) { m = r; mv = heap[r].logit; }
        if (m == i) break;
        heap[i] = heap[m];
        i = m;
    }
    heap[i].logit = logit;
    heap[i].id = id;
}

static int toppCmp(const void* a, const void* b) {
    const topp_entry* x = (const topp_entry*)a;
    const topp_entry* y = (const topp_entry*)b;
    if (x->logit > y->logit) return -1;
    if (x->logit < y->logit) return 1;
    if (x->id < y->id) return -1;
    if (x->id > y->id) return 1;
    return 0;
}

static int toppFinite(float v) {
    return v > -1e29f && v < 1e29f;
}

static void dumpTopPStep(generator* g) {
    if (g->dumpTopPFile == NULL) return;
    model_state* st = &g->st;
    int vocab = g->vocab;
    float* logits = (float*)malloc(sizeof(float) * (size_t)vocab);
    if (logits == NULL) return;
    readBuffer(g->s.dev.device, g->s.dev.physicalDevice, g->s.dev.queue, st->logits, logits);
    double sum = 0.0;
    for (int i = 0; i < vocab; i++) {
        if (toppFinite(logits[i])) sum += (double)expf(logits[i]);
    }
    topp_entry* heap = (topp_entry*)malloc(sizeof(topp_entry) * TOPP_KEEP);
    if (heap == NULL) {
        free(logits);
        return;
    }
    int n = 0;
    for (uint32_t i = 0; i < (uint32_t)vocab; i++) {
        float lg = logits[i];
        if (!toppFinite(lg)) continue;
        if (n < TOPP_KEEP) {
            heapPush(heap, &n, lg, i);
        } else if (lg > heap[0].logit) {
            heapReplaceRoot(heap, TOPP_KEEP, lg, i);
        }
    }
    qsort(heap, (size_t)n, sizeof(topp_entry), toppCmp);
    uint32_t count = (uint32_t)n;
    double s = sum;
    fwrite(&g->dumpTopPStep, sizeof(uint32_t), 1, g->dumpTopPFile);
    fwrite(&count, sizeof(uint32_t), 1, g->dumpTopPFile);
    for (int i = 0; i < n; i++) {
        fwrite(&heap[i].id, sizeof(uint32_t), 1, g->dumpTopPFile);
        fwrite(&heap[i].logit, sizeof(float), 1, g->dumpTopPFile);
    }
    fwrite(&s, sizeof(double), 1, g->dumpTopPFile);
    fflush(g->dumpTopPFile);
    g->dumpTopPStep++;
    free(heap);
    free(logits);
}

static void executeChunked(session s, operation* ops, int opCount, const char* phase, int token) {
    int done = 0;
    while (done < opCount) {
        int cnt = opCount - done;
        if (cnt > 8) cnt = 8;
        executeRecord(&s, ops + done, cnt);
        executeSubmitNow(&s);
        executeWaitLast(&s);
        logLastFrame(&s, ops + done, cnt, phase, token);
        done += cnt;
    }
}

static void dumpHiddenStates(generator* g, int padded, int cur, int done) {
    model_state* st = &g->st;
    const model_dims* d = g->dims;
    char path[320];
    snprintf(path, sizeof(path), "%s/hidden_%d_%d_%d.f32", g->dumpHiddenDir, g->dumpHiddenReq, done, cur);
    FILE* f = fopen(path, "wb");
    if (f == NULL) return;
    float* host = (float*)malloc(sizeof(float) * (size_t)g->maxM * d->K);
    readBuffer(g->s.dev.device, g->s.dev.physicalDevice, g->s.dev.queue, st->h, host);
    fwrite(host, sizeof(float) * d->K, cur, f);
    free(host);
    fclose(f);
}

uint32_t runPrefill(generator* g, const uint32_t* tokens, int nTokens) {
    model_state* st = &g->st;
    const model_dims* d = g->dims;
    int m = g->maxM;

    int done = 0;
    int lastCur = 0;
    while (done < nTokens) {
        int cur = nTokens - done;
        if (cur > m) cur = m;
        int padded = (cur + 15) & ~15;
        memcpy(st->tokenIds.mappedMemory, tokens + done, sizeof(uint32_t) * cur);
        for (int i = cur; i < padded; i++) ((uint32_t*)st->tokenIds.mappedMemory)[i] = 0;

        if (g->dumpLayers > 0) {
            int headCount = compilePrefillHead(g, g->prefillOps, padded);
            executeChunked(g->s, g->prefillOps, headCount, "prefill", (int)(g->nextPos + done));
            dumpBuffer(g, st->embOut, "layer_00_embed", (int64_t)padded * d->K);
            for (int L = 0; L < g->layerCount && L < g->dumpLayers; L++) {
                const layer* ly = &g->spec->layers[L];
                int n = 0;
                if (ly->attn.type == ATTENTION_FULL) {
                    buildAttention(g, g->prefillOps, &n, L, 1, padded, 0, done + padded, done);
                } else if (ly->attn.type == ATTENTION_DELTA) {
                    buildDelta(g, g->prefillOps, &n, L, 1, padded, cur);
                }
                executeChunked(g->s, g->prefillOps, n, "prefill", (int)(g->nextPos + done));
                char name[64];
                snprintf(name, sizeof(name), "layer_%02d_hattn", L + 1);
                dumpBuffer(g, st->h, name, (int64_t)padded * d->K);

                int nf = 0;
                if (ly->ffn.type == FFN_SWIGLU) {
                    buildFfn(g, g->prefillOps, &nf, L, 1, padded);
                } else if (ly->ffn.type == FFN_MOE) {
                    buildMoe(g, g->prefillOps, &nf, L, padded);
                }
                executeChunked(g->s, g->prefillOps, nf, "prefill", (int)(g->nextPos + done));
                if (ly->ffn.type == FFN_MOE && st->moeIds.buffer != VK_NULL_HANDLE) {
                    int moeSlots = d->expertsPerTok + 1;
                    snprintf(name, sizeof(name), "layer_%02d_moeXn", L + 1);
                    dumpBuffer(g, st->moeXn, name, (int64_t)padded * d->K);
                    snprintf(name, sizeof(name), "layer_%02d_moeIds", L + 1);
                    dumpBuffer(g, st->moeIds, name, (int64_t)padded * moeSlots);
                    snprintf(name, sizeof(name), "layer_%02d_moeW", L + 1);
                    dumpBuffer(g, st->moeWeights, name, (int64_t)padded * moeSlots);
                    snprintf(name, sizeof(name), "layer_%02d_moeSw", L + 1);
                    dumpBuffer(g, st->moeSharedW, name, padded);
                    snprintf(name, sizeof(name), "layer_%02d_moeH", L + 1);
                    dumpBuffer(g, st->moeH, name, (int64_t)padded * moeSlots * d->moeI);
                    snprintf(name, sizeof(name), "layer_%02d_moeP", L + 1);
                    dumpBuffer(g, st->moeP, name, (int64_t)padded * moeSlots * d->K);
                }
                snprintf(name, sizeof(name), "layer_%02d_h", L + 1);
                dumpBuffer(g, st->h, name, (int64_t)padded * d->K);
                if (g->spec->layers[L].attn.type == ATTENTION_FULL) {
                    snprintf(name, sizeof(name), "layer_%02d_qkvRaw", L + 1);
                    dumpBuffer(g, st->qkvRaw, name, (int64_t)padded * d->qkvN);
                    snprintf(name, sizeof(name), "layer_%02d_qOut", L + 1);
                    dumpBuffer(g, st->qOut, name, (int64_t)padded * d->qOff);
                    snprintf(name, sizeof(name), "layer_%02d_attnOut", L + 1);
                    dumpBuffer(g, st->attnOut, name, (int64_t)padded * d->qOff);
                    snprintf(name, sizeof(name), "layer_%02d_gAttn", L + 1);
                    dumpBuffer(g, st->gAttn, name, (int64_t)padded * d->qOff);
                    snprintf(name, sizeof(name), "layer_%02d_scores", L + 1);
                    dumpBufferFp16(g, st->attScores, name, (int64_t)d->kvHeads * padded * d->maxCtx);
                    snprintf(name, sizeof(name), "layer_%02d_smSum", L + 1);
                    dumpBuffer(g, st->smSum, name, (int64_t)d->kvHeads * padded);
                    QuantType aq = g->spec->layers[L].attn.q;
                    if (aq == QUANT_FP16) {
                        snprintf(name, sizeof(name), "layer_%02d_kCache", L + 1);
                        dumpBufferFp16(g, st->kCache[L], name, (int64_t)(done + padded) * d->kvRows);
                        snprintf(name, sizeof(name), "layer_%02d_vCache", L + 1);
                        dumpBufferFp16(g, st->vCache[L], name, (int64_t)(done + padded) * d->kvRows);
                    } else {
                        int64_t nt = (int64_t)(done + padded);
                        snprintf(name, sizeof(name), "layer_%02d_kCacheU8", L + 1);
                        dumpBufferU8(g, st->kCache[L], name, nt * d->kvRows);
                        snprintf(name, sizeof(name), "layer_%02d_vCacheU8", L + 1);
                        dumpBufferU8(g, st->vCache[L], name, nt * d->kvRows);
                        snprintf(name, sizeof(name), "layer_%02d_kScale", L + 1);
                        dumpBuffer(g, st->kScale[L], name, (int64_t)d->kvHeads * d->maxCtx);
                        snprintf(name, sizeof(name), "layer_%02d_kZero", L + 1);
                        dumpBuffer(g, st->kZero[L], name, (int64_t)d->kvHeads * d->maxCtx);
                        snprintf(name, sizeof(name), "layer_%02d_vScale", L + 1);
                        dumpBuffer(g, st->vScale[L], name, (int64_t)d->kvHeads * d->maxCtx);
                        snprintf(name, sizeof(name), "layer_%02d_vZero", L + 1);
                        dumpBuffer(g, st->vZero[L], name, (int64_t)d->kvHeads * d->maxCtx);
                    }
                }
                snprintf(name, sizeof(name), "layer_%02d_act", L + 1);
                dumpBuffer(g, st->act, name, (int64_t)padded * d->ffnN);
                snprintf(name, sizeof(name), "layer_%02d_gAct", L + 1);
                dumpBuffer(g, st->gAct, name, (int64_t)padded * d->ffnN);
                snprintf(name, sizeof(name), "layer_%02d_uAct", L + 1);
                dumpBuffer(g, st->uAct, name, (int64_t)padded * d->ffnN);
                snprintf(name, sizeof(name), "layer_%02d_invRms", L + 1);
                dumpBuffer(g, st->invRms, name, (int64_t)padded);
                if (g->spec->layers[L].attn.type == ATTENTION_DELTA) {
                    snprintf(name, sizeof(name), "layer_%02d_yGated", L + 1);
                    dumpBuffer(g, st->yGated, name, (int64_t)padded * d->K);
                }
            }
            for (int L = g->dumpLayers; L < g->layerCount; L++) {
                int n = 0;
                buildLayer(g, g->prefillOps, &n, L, 1, padded, 0, done + padded, done, cur);
                executeChunked(g->s, g->prefillOps, n, "prefill", (int)(g->nextPos + done));
            }
        } else {
            g->prefillOpCount = compilePrefill(g, padded, done, cur);
            executeChunked(g->s, g->prefillOps, g->prefillOpCount, "prefill", (int)(g->nextPos + done));
        }
        if (g->dumpHiddenDir[0] != '\0') {
            dumpHiddenStates(g, padded, cur, done);
        }
        done += cur;
        lastCur = cur;
    }

    float* hCpu = (float*)malloc(sizeof(float) * m * d->K);
    readBuffer(g->s.dev.device, g->s.dev.physicalDevice, g->s.dev.queue, st->h, hCpu);
    memcpy(st->lastRow.mappedMemory, hCpu + (lastCur - 1) * d->K, sizeof(float) * d->K);
    free(hCpu);

    stateSetPosition(g->s, &g->st, g->nextPos + (uint32_t)nTokens);

    if (g->skipFinal) {
        g->nextPos += (uint32_t)nTokens;
        return 0;
    }

    if (g->dumpTopPFile != NULL && g->finalOpCount >= 2) {
        executeRecord(&g->s, g->finalOps, g->finalOpCount - 1);
        executeSubmitNow(&g->s);
        executeWaitLast(&g->s);
        dumpTopPStep(g);
        executeRecord(&g->s, g->finalOps + g->finalOpCount - 1, 1);
        executeSubmitNow(&g->s);
        executeWaitLast(&g->s);
    } else {
        executeLogged(g->s, g->finalOps, g->finalOpCount, "prefill", (int)g->nextPos);
    }

    g->nextPos += (uint32_t)nTokens;

    uint32_t results[4] = {0};
    readBuffer(g->s.dev.device, g->s.dev.physicalDevice, g->s.dev.queue, st->result, results);
    return results[0];
}

void generatorRequestStop(generator* g) {
    if (g != NULL) g->stop = 1;
}

void generateTokens(generator* g, const uint32_t* prompt, int nPrompt, int maxNewTokens, void (*emit)(uint32_t token, void* ctx), void* ctx) {
    g->stop = 0;
    g->skipFinal = 0;
    if (nPrompt >= g->maxCtx) {
        fprintf(stderr, "prompt length %d exceeds max ctx %d\n", nPrompt, g->maxCtx);
        return;
    }
    int limit = g->maxCtx - nPrompt;
    if (limit < 1) limit = 1;
    if (maxNewTokens > limit) maxNewTokens = limit;

    uint32_t token = runPrefill(g, prompt, nPrompt);
    int count = 1;
    emit(token, ctx);
    if (token == (uint32_t)g->eos || maxNewTokens <= 1) {
        return;
    }

    if (g->dumpTopPFile != NULL) {
        ((uint32_t*)g->st.tokenIds.mappedMemory)[0] = token;
        for (int i = 1; i < maxNewTokens; i++) {
            int split = (int)(g->nextPos >= ATT_SPLIT_THRESHOLD);
            operation* ops = split ? g->groupOps : g->groupOpsShort;
            int pt = (split ? g->groupOpCount : g->groupOpCountShort) / DECODE_GROUP;
            executeRecord(&g->s, ops, pt - 1);
            executeSubmitNow(&g->s);
            executeWaitLast(&g->s);
            dumpTopPStep(g);
            executeRecord(&g->s, ops + pt - 1, 1);
            executeSubmitNow(&g->s);
            executeWaitLast(&g->s);
            uint32_t picks[DECODE_GROUP];
            readBuffer(g->s.dev.device, g->s.dev.physicalDevice, g->s.dev.queue, g->st.result, picks);
            token = picks[0];
            emit(token, ctx);
            count++;
            g->nextPos++;
            if (token == (uint32_t)g->eos) break;
        }
        return;
    }

    int decodeTokens = maxNewTokens - 1;
    int fullGroups = decodeTokens / DECODE_GROUP;
    int rem = decodeTokens % DECODE_GROUP;

    int curSplit = (int)(g->nextPos >= ATT_SPLIT_THRESHOLD);
    operation* curOps = curSplit ? g->groupOps : g->groupOpsShort;
    int curCount = curSplit ? g->groupOpCount : g->groupOpCountShort;

    ((uint32_t*)g->st.tokenIds.mappedMemory)[0] = token;
    int dumpFirst = (g->dumpDir[0] != '\0');
    executeRecord(&g->s, curOps, dumpFirst ? curCount / DECODE_GROUP : curCount);
    executeSubmitNow(&g->s);
    if (dumpFirst) {
        executeWaitLast(&g->s);
        int passOps = curCount / DECODE_GROUP;
        int done = 0;
        int lastLayer = -2;
        while (done < passOps) {
            int seg = 1;
            while (done + seg < passOps && curOps[done + seg].layer == curOps[done].layer) seg++;
            executeRecord(&g->s, curOps + done, seg);
            executeSubmitNow(&g->s);
            executeWaitLast(&g->s);
            int layer = curOps[done].layer;
            if (layer != lastLayer) {
                char name[64];
                snprintf(name, sizeof(name), "decL%03d_h", layer + 1);
                dumpBuffer(g, g->st.h, name, g->dims->K);
                lastLayer = layer;
            }
            done += seg;
        }
        dumpBuffer(g, g->st.embOut, "decode_emb", g->dims->K);
        executeRecord(&g->s, curOps + passOps, curCount - passOps);
        executeSubmitNow(&g->s);
    }

    int eos = 0;
    int units = fullGroups + (rem > 0 ? 1 : 0);
    double prevDelay = FIRST_TOKEN_DELAY_MS;
    for (int u = 0; u < units && !eos && !g->stop; u++) {
        int cur = (u == fullGroups) ? rem : DECODE_GROUP;
        int next = ((rem > 0) && (u + 1 == fullGroups)) ? rem : DECODE_GROUP;

        int nextSplit = (int)((g->nextPos + cur) >= ATT_SPLIT_THRESHOLD);
        operation* nextOps = nextSplit ? g->groupOps : g->groupOpsShort;
        int nextCount = nextSplit ? g->groupOpCount : g->groupOpCountShort;

        uint32_t tokens[DECODE_GROUP];
        double t0 = nowMs();
        executeRecord(&g->s, nextOps, next * (nextCount / DECODE_GROUP));
        executeWaitLast(&g->s);
        logLastFrame(&g->s, curOps, cur * (curCount / DECODE_GROUP), "decode", (int)g->nextPos);
        readBuffer(g->s.dev.device, g->s.dev.physicalDevice, g->s.dev.queue, g->st.result, tokens);
        double t1 = nowMs();
        if (isTimingEnabled()) {
            fprintf(stderr, "[decode][tok=%5d] group=%d  %7.3f ms/tok\n", (int)g->nextPos, cur, (t1 - t0) / (double)cur);
        }
        ((uint32_t*)g->st.tokenIds.mappedMemory)[0] = tokens[cur - 1];
        executeSubmitNow(&g->s);
        g->nextPos += cur;

        for (int j = 0; j < cur && count < maxNewTokens; j++) {
            emit(tokens[j], ctx);
            count++;
            if (tokens[j] == (uint32_t)g->eos) {
                eos = 1;
                break;
            }
            if (g->stop) break;
            sleepMs(prevDelay);
        }

        prevDelay = (t1 - t0) / (double)cur;

        curSplit = nextSplit;
        curOps = nextOps;
        curCount = nextCount;
    }
    executeWaitLast(&g->s);
}

void generateScore(generator* g, const uint32_t* ids, size_t idCount, int prefillN, int decodeN, int chunks,
                   void (*progress)(void* ctx, int done, int total, int chunkTokens, int chunkTotal,
                                    double lossSum, long long count),
                   void* ctx, double* outLoss, long long* outCount) {
    g->stop = 0;
    g->skipFinal = 1;
    double lossSum = 0.0;
    long long count = 0;
    int scored = 0;
    int limit = g->maxCtx - prefillN;
    if (limit > decodeN) limit = decodeN;
    if (prefillN > 0 && limit > 0 && idCount > (size_t)(limit + 1)) {
        int maxChunks = (int)((idCount - (size_t)limit - 1) / (size_t)prefillN);
        if (chunks > maxChunks) chunks = maxChunks;
    } else {
        chunks = 0;
    }

    for (int k = 0; k < chunks && !g->stop && prefillN > 0 && limit > 0; k++) {
        int s = k * prefillN;
        int chunkTotal = limit + (k == 0 ? 1 : 0);
        if (progress != NULL) progress(ctx, scored, chunks, 0, chunkTotal, lossSum, count);
        resetGenerator(g);
        runPrefill(g, ids + s, prefillN);

        uint32_t* tokenIds = (uint32_t*)g->st.tokenIds.mappedMemory;
        tokenIds[1] = ids[s + prefillN];

        if (k == 0) {
            executeLogged(g->s, g->finalOps, g->finalOpCount, "score", 0);
            uint32_t bits[DECODE_GROUP] = {0};
            readBuffer(g->s.dev.device, g->s.dev.physicalDevice, g->s.dev.queue, g->st.result, bits);
            float loss = 0.0f;
            memcpy(&loss, &bits[0], sizeof(float));
            lossSum += (double)loss;
            count++;
            if (progress != NULL) progress(ctx, scored, chunks, 1, chunkTotal, lossSum, count);
        }
        tokenIds[0] = ids[s + prefillN];

        int j = 0;
        while (j < limit && !g->stop) {
            int cur = limit - j;
            if (cur > DECODE_GROUP) cur = DECODE_GROUP;
            for (int p = 0; p < cur; p++) tokenIds[1 + p] = ids[s + prefillN + j + 1 + p];

            int split = (int)(g->nextPos >= ATT_SPLIT_THRESHOLD);
            operation* ops = split ? g->groupOps : g->groupOpsShort;
            int opsPerPass = (split ? g->groupOpCount : g->groupOpCountShort) / DECODE_GROUP;
            executeRecord(&g->s, ops, cur * opsPerPass);
            executeSubmitNow(&g->s);
            executeWaitLast(&g->s);

            uint32_t bits[DECODE_GROUP] = {0};
            readBuffer(g->s.dev.device, g->s.dev.physicalDevice, g->s.dev.queue, g->st.result, bits);
            for (int p = 0; p < cur; p++) {
                float loss = 0.0f;
                memcpy(&loss, &bits[p], sizeof(float));
                lossSum += (double)loss;
                count++;
            }

            g->nextPos += (uint32_t)cur;
            j += cur;
            if (progress != NULL && (j & 31) == 0)
                progress(ctx, scored, chunks, j + (k == 0 ? 1 : 0), chunkTotal, lossSum, count);
        }

        scored++;
        if (progress != NULL) progress(ctx, scored, chunks, chunkTotal, chunkTotal, lossSum, count);
    }

    g->skipFinal = 0;
    if (outLoss != NULL) *outLoss = lossSum;
    if (outCount != NULL) *outCount = count;
}

void resetGenerator(generator* g) {
    int count = 0;
    buffer* states = (buffer*)malloc(sizeof(buffer) * (g->layerCount * 2 + 1));
    for (int L = 0; L < g->layerCount; L++) {
        if (g->spec->layers[L].attn.type == ATTENTION_DELTA) {
            states[count++] = g->st.stateS[L];
            states[count++] = g->st.convHist[L];
        }
    }
    states[count++] = g->st.sampleHistory;
    createTransferAndCopy(g->s.dev.device, g->s.dev.queue, states, count);
    free(states);
    g->nextPos = 0;
}

void generatorSetSampling(generator* g, const sample_params* p, uint32_t seed) {
    if (g->st.sampleParams.mappedMemory != NULL) {
        memcpy(g->st.sampleParams.mappedMemory, p, sizeof(sample_params));
    }
    if (g->st.sampleRng.mappedMemory != NULL) {
        *(uint32_t*)g->st.sampleRng.mappedMemory = seed;
    }
    int sampling = (p->temperature > 0.0f) ? 1 : 0;
    if (g->dumpTopPFile != NULL && !sampling) {
        sample_params sp = *p;
        sp.temperature = 1e-9f;
        memcpy(g->st.sampleParams.mappedMemory, &sp, sizeof(sample_params));
        sampling = 1;
    }
    if (g->sampling == sampling) return;
    g->sampling = sampling;
    g->groupOpCount = compileDecodeGroup(g, g->groupOps, 1);
    g->groupOpCountShort = compileDecodeGroup(g, g->groupOpsShort, 0);
    g->finalOpCount = compileFinal(g);
}

void generatorSetScoring(generator* g, int enabled) {
    int mode = enabled ? 2 : 0;
    if (g->sampling == mode) return;
    g->sampling = mode;
    g->groupOpCount = compileDecodeGroup(g, g->groupOps, 1);
    g->groupOpCountShort = compileDecodeGroup(g, g->groupOpsShort, 0);
    g->finalOpCount = compileFinal(g);
}

void generatorSetDumpDir(generator* g, const char* dir) {
    if (dir == NULL) {
        g->dumpDir[0] = '\0';
        return;
    }
    snprintf(g->dumpDir, sizeof(g->dumpDir), "%s", dir);
}

void generatorSetDumpHidden(generator* g, const char* dir, int reqIdx) {
    if (dir == NULL) {
        g->dumpHiddenDir[0] = '\0';
        return;
    }
    snprintf(g->dumpHiddenDir, sizeof(g->dumpHiddenDir), "%s", dir);
    g->dumpHiddenReq = reqIdx;
}

void generatorSetDumpTopP(generator* g, const char* path) {
    if (g->dumpTopPFile != NULL) {
        fclose(g->dumpTopPFile);
        g->dumpTopPFile = NULL;
    }
    if (path == NULL || path[0] == '\0') {
        g->dumpTopPPath[0] = '\0';
        return;
    }
    snprintf(g->dumpTopPPath, sizeof(g->dumpTopPPath), "%s", path);
    g->dumpTopPFile = fopen(g->dumpTopPPath, "ab");
    if (g->dumpTopPFile == NULL) {
        fprintf(stderr, "[dump-topp] failed to open %s\n", g->dumpTopPPath);
        return;
    }
    g->dumpTopPStep = 0;
}

void generatorCloseDumpTopP(generator* g) {
    if (g->dumpTopPFile != NULL) {
        fclose(g->dumpTopPFile);
        g->dumpTopPFile = NULL;
    }
}

void generatorDumpPrefill(generator* g, int rows) {
    if (g->dumpDir[0] == '\0') return;
    model_state* st = &g->st;
    dumpBuffer(g, st->embOut, "embOut", (int64_t)rows * g->dims->K);
    dumpBuffer(g, st->h, "h", (int64_t)rows * g->dims->K);
    dumpBuffer(g, st->lastRow, "lastRow", g->dims->K);
}

void generatorSetDumpLayers(generator* g, int layers) {
    g->dumpLayers = layers;
}

void generatorDumpDecodeStep(generator* g, int step) {
    if (g->dumpDir[0] == '\0') return;
    model_state* st = &g->st;
    char name[64];
    snprintf(name, sizeof(name), "decode_%02d_h", step);
    dumpBuffer(g, st->h, name, g->dims->K);
    snprintf(name, sizeof(name), "decode_%02d_q", step);
    dumpBuffer(g, st->qOut, name, g->dims->K);
}

void generatorDumpSamplingDebug(generator* g, const uint32_t* generated, int nGen, int nPrompt) {
    uint32_t* hist = (uint32_t*)malloc(sizeof(uint32_t) * (size_t)MAX_PENALTY_LEN);
    readBuffer(g->s.dev.device, g->s.dev.physicalDevice, g->s.dev.queue, g->st.sampleHistory, hist);
    uint32_t gpuPos = stateReadPosition(g->s, &g->st);

    fprintf(stderr, "[dbg-sampling] nPrompt=%d nGen=%d hostNextPos=%u gpuPosition=%u\n",
            nPrompt, nGen, (unsigned)g->nextPos, (unsigned)gpuPos);

    int sentinel = 0;
    for (int i = 0; i < MAX_PENALTY_LEN; i++) {
        if (hist[i] == 0xFFFFFFFFu) sentinel++;
    }
    fprintf(stderr, "[dbg-sampling] history %d/%d sentinel slots\n", sentinel, MAX_PENALTY_LEN);
    for (int i = 0; i < MAX_PENALTY_LEN; i++) {
        if (hist[i] != 0xFFFFFFFFu) {
            fprintf(stderr, "[dbg-sampling]   slot[%d]=%u\n", i, (unsigned)hist[i]);
        }
    }

    fprintf(stderr, "[dbg-sampling] generated tokens:\n");
    for (int i = 0; i < nGen; i++) {
        fprintf(stderr, "[dbg-sampling]   tok[%d]=%u\n", i, (unsigned)generated[i]);
    }
    free(hist);
}
