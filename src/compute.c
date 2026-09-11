#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <io.h>
#include <fcntl.h>
#include "session.h"
#include "model.h"
#include "generate.h"
#include "prune.h"
#include "gguf.h"

static const char* argval(int argc, char** argv, const char* name, const char* def) {
    for (int i = 1; i + 1 < argc; i++) {
        if (strcmp(argv[i], name) == 0) return argv[i + 1];
    }
    return def;
}

static int argflag(int argc, char** argv, const char* name) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], name) == 0) return 1;
    }
    return 0;
}

void memInfo(void) {
    session s = createSession();
    dumpMemoryInfo(s.dev.physicalDevice);
    destroySession(s);
}

static int dbgSampling = 0;
static uint32_t dbgGen[16384];
static int dbgCount = 0;

static void emitToken(uint32_t token, void* ctx) {
    (void)ctx;
    if (dbgSampling && dbgCount < 16384) dbgGen[dbgCount++] = token;
    fwrite(&token, sizeof(uint32_t), 1, stdout);
    fflush(stdout);
}

void serverMain(int argc, char** argv) {
    const char* weightDir = argval(argc, argv, "--weights", "../model/Qwen3.5-9B");
    int forceGguf = argflag(argc, argv, "--gguf");
    if (forceGguf && !gguf_path_is_file(weightDir)) {
        fprintf(stderr, "server: --gguf given but %s is not a gguf file\n", weightDir);
        exit(1);
    }
    int maxCtxOverride = atoi(argval(argc, argv, "--max-ctx", "0"));
    int maxNew = atoi(argval(argc, argv, "--max-new", "128"));
    const char* dumpDir = argval(argc, argv, "--dump", NULL);
    int dumpLayers = atoi(argval(argc, argv, "--dump-layers", "0"));
    int verboseWeights = argflag(argc, argv, "--verbose-weights");
    int timing = argflag(argc, argv, "--timing");
    const char* dumpHiddenDir = argval(argc, argv, "--dump-hidden", NULL);
    const char* dumpTopPPath = argval(argc, argv, "--dump-topp", NULL);
    int doPrune = argflag(argc, argv, "--prune");
    dbgSampling = argflag(argc, argv, "--debug-sampling");
    if (maxNew < 1) maxNew = 1;

    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);

    static model_config spec;
    loadModelConfig(&spec, weightDir, maxCtxOverride, doPrune);
    if (doPrune) pruneVocab(weightDir, &spec);
    parseEos(&spec.dims, weightDir, doPrune);
    int expertsVram = atoi(argval(argc, argv, "--experts-vram", "0"));
    if (expertsVram > 0) spec.expertsVram = expertsVram;

    if (timing) setTimingEnabled(1);
    session s = createSession();
    generator* g = createGenerator(s, &spec, weightDir, verboseWeights);
    if (dumpTopPPath != NULL) generatorSetDumpTopP(g, dumpTopPPath);
    if (dumpDir != NULL) {
        generatorSetDumpDir(g, dumpDir);
        generatorSetDumpLayers(g, dumpLayers);
    }
    int reqIdx = 0;

    uint32_t* prompt = NULL;
    int promptCap = 0;

    for (;;) {
        uint32_t n = 0;
        if (fread(&n, sizeof(uint32_t), 1, stdin) != 1) break;
        if (n == 0) break;

        sample_params sp = {0};
        uint32_t seed = 0;
        if (fread(&sp.temperature, sizeof(float), 1, stdin) != 1) break;
        if (fread(&sp.repPenalty, sizeof(float), 1, stdin) != 1) break;
        if (fread(&sp.penaltyLength, sizeof(uint32_t), 1, stdin) != 1) break;
        if (fread(&sp.topK, sizeof(uint32_t), 1, stdin) != 1) break;
        if (fread(&sp.topP, sizeof(float), 1, stdin) != 1) break;
        if (fread(&sp.minP, sizeof(float), 1, stdin) != 1) break;
        if (fread(&sp.presencePenalty, sizeof(float), 1, stdin) != 1) break;
        if (fread(&seed, sizeof(uint32_t), 1, stdin) != 1) break;
        if (sp.penaltyLength > MAX_PENALTY_LEN) sp.penaltyLength = MAX_PENALTY_LEN;
        if (seed == 0) seed = 1;

        if ((int)n > promptCap) {
            prompt = (uint32_t*)realloc(prompt, sizeof(uint32_t) * n);
            promptCap = (int)n;
        }
        if (fread(prompt, sizeof(uint32_t), n, stdin) != n) break;

        generatorSetSampling(g, &sp, seed);
        if (dumpHiddenDir != NULL) generatorSetDumpHidden(g, dumpHiddenDir, reqIdx);
        reqIdx++;
        resetGenerator(g);
        dbgCount = 0;
        generateTokens(g, prompt, (int)n, maxNew, emitToken, NULL);
        if (dbgSampling) generatorDumpSamplingDebug(g, dbgGen, dbgCount, (int)n);
        uint32_t sentinel = 0xFFFFFFFFu;
        fwrite(&sentinel, sizeof(uint32_t), 1, stdout);
        fflush(stdout);
    }

    free(prompt);
    generatorCloseDumpTopP(g);
    destroyGenerator(g);
    destroySession(s);
    if (timing) closeTimingLog();
}

