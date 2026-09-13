#include <vulkan/vulkan.h>
#include "dispatch.h"
#include "descriptor.h"
#include "pipeline.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PIPE_CACHE_MAX 4096
#define DESC_CACHE_MAX 1024
#define TIMING_AGG_MAX 128
#define QUERIES_PER_FRAME (TIMESTAMP_QUERY_COUNT / FRAME_COUNT)

typedef struct pipe_entry {
    char shader[128];
    int pushSize;
    int bufCount;
    VkPipeline pipeline;
    VkPipelineLayout layout;
    VkDescriptorSetLayout setLayout;
} pipe_entry;

typedef struct timing_agg {
    char shader[128];
    double totalMs;
    long calls;
} timing_agg;

static pipe_entry pipeCache[PIPE_CACHE_MAX];
static int pipeCacheCount = 0;

typedef struct desc_entry {
    VkDescriptorSetLayout layout;
    int bufferCount;
    VkBuffer handles[MAX_OP_BUFFERS];
    VkDescriptorSet set;
} desc_entry;

static VkDescriptorPool descPool = VK_NULL_HANDLE;
static desc_entry descCache[DESC_CACHE_MAX];
static int descCacheCount = 0;
static int timingEnabled = 0;

static FILE* timingFile = NULL;
static timing_agg aggTable[TIMING_AGG_MAX];
static int aggCount = 0;
static double grandTotalMs = 0.0;
static char shaderRoot[128] = "";

const char* shaderRootDir(void) {
    return shaderRoot;
}

void setShaderRootDir(const char* dir) {
    if (dir == NULL) {
        shaderRoot[0] = '\0';
        return;
    }
    snprintf(shaderRoot, sizeof(shaderRoot), "%s", dir);
}

void setTimingEnabled(int enabled) {
    timingEnabled = enabled;
}

int isTimingEnabled(void) {
    return timingEnabled;
}

void dispatchReset(VkDevice device) {
    for (int i = 0; i < pipeCacheCount; i++) {
        if (pipeCache[i].pipeline != VK_NULL_HANDLE) vkDestroyPipeline(device, pipeCache[i].pipeline, NULL);
        if (pipeCache[i].layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(device, pipeCache[i].layout, NULL);
        if (pipeCache[i].setLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(device, pipeCache[i].setLayout, NULL);
    }
    pipeCacheCount = 0;
    if (descPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(device, descPool, NULL);
    descPool = VK_NULL_HANDLE;
    descCacheCount = 0;
    pipelineClearSpec();
}

static VkDescriptorSet getDescriptorSet(session s, VkDescriptorSetLayout layout, const operation* op) {
    for (int i = 0; i < descCacheCount; i++) {
        desc_entry* e = &descCache[i];
        if (e->layout != layout || e->bufferCount != op->bufferCount) continue;
        int match = 1;
        for (int b = 0; b < op->bufferCount; b++) {
            if (e->handles[b] != op->buffers[b].buffer) { match = 0; break; }
        }
        if (match) return e->set;
    }

    if (descPool == VK_NULL_HANDLE) {
        VkDescriptorPoolSize ps = {0};
        ps.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        ps.descriptorCount = DESC_CACHE_MAX * MAX_OP_BUFFERS;
        VkDescriptorPoolCreateInfo ci = {0};
        ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        ci.poolSizeCount = 1;
        ci.pPoolSizes = &ps;
        ci.maxSets = DESC_CACHE_MAX;
        if (vkCreateDescriptorPool(s.dev.device, &ci, NULL, &descPool) != VK_SUCCESS) {
            fprintf(stderr, "Error: Failed to create descriptor pool\n");
            exit(EXIT_FAILURE);
        }
    }

    VkDescriptorSetAllocateInfo ai = {0};
    ai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool = descPool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &layout;
    VkDescriptorSet set;
    if (vkAllocateDescriptorSets(s.dev.device, &ai, &set) != VK_SUCCESS) {
        fprintf(stderr, "Error: Failed to allocate descriptor set\n");
        exit(EXIT_FAILURE);
    }

    VkDescriptorBufferInfo bi[MAX_OP_BUFFERS];
    VkWriteDescriptorSet wr[MAX_OP_BUFFERS];
    for (int i = 0; i < op->bufferCount; i++) {
        bi[i].buffer = op->buffers[i].buffer;
        bi[i].offset = 0;
        bi[i].range = VK_WHOLE_SIZE;
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        wr[i].pNext = NULL;
        wr[i].dstSet = set;
        wr[i].dstBinding = i;
        wr[i].dstArrayElement = 0;
        wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].descriptorCount = 1;
        wr[i].pBufferInfo = &bi[i];
        wr[i].pImageInfo = NULL;
        wr[i].pTexelBufferView = NULL;
    }
    vkUpdateDescriptorSets(s.dev.device, op->bufferCount, wr, 0, NULL);

    if (descCacheCount < DESC_CACHE_MAX) {
        desc_entry* e = &descCache[descCacheCount++];
        e->layout = layout;
        e->bufferCount = op->bufferCount;
        for (int i = 0; i < op->bufferCount; i++) e->handles[i] = op->buffers[i].buffer;
        e->set = set;
    }
    return set;
}

static pipe_entry* getPipeEntry(session s, const operation* op) {
    int pushSize = sizeof(int) * op->pushConstantCount;
    for (int i = 0; i < pipeCacheCount; i++) {
        if (pipeCache[i].pushSize == pushSize &&
            pipeCache[i].bufCount == op->bufferCount &&
            strcmp(pipeCache[i].shader, op->shader) == 0) {
            return &pipeCache[i];
        }
    }
    if (pipeCacheCount >= PIPE_CACHE_MAX) {
        fprintf(stderr, "Error: pipeline cache overflow\n");
        exit(EXIT_FAILURE);
    }

    pipe_entry* e = &pipeCache[pipeCacheCount++];
    snprintf(e->shader, sizeof(e->shader), "%s", op->shader);
    e->pushSize = pushSize;
    e->bufCount = op->bufferCount;
    e->setLayout = createDescriptorSetLayout(s.dev.device, op->bufferCount);

    char shaderPath[288];
    const char* root = shaderRootDir();
    if (root[0] != '\0') {
        snprintf(shaderPath, sizeof(shaderPath), "%s/%s", root, op->shader);
    } else {
        snprintf(shaderPath, sizeof(shaderPath), "shader/%s", op->shader);
    }
    pipeline p = createPipeline(s.dev.device, e->setLayout, shaderPath, (uint32_t)pushSize);
    e->pipeline = p.pipeline;
    e->layout = p.layout;
    return e;
}

static void ensureTimingLog(void) {
    if (timingFile != NULL) return;
    timingFile = fopen("timing_log.txt", "w");
    if (timingFile == NULL) {
        fprintf(stderr, "Error: Failed to open timing_log.txt\n");
        exit(EXIT_FAILURE);
    }
}

static timing_agg* findAgg(const char* shader) {
    for (int i = 0; i < aggCount; i++) {
        if (strcmp(aggTable[i].shader, shader) == 0) return &aggTable[i];
    }
    if (aggCount >= TIMING_AGG_MAX) return NULL;
    timing_agg* a = &aggTable[aggCount++];
    snprintf(a->shader, sizeof(a->shader), "%s", shader);
    a->totalMs = 0.0;
    a->calls = 0;
    return a;
}

static int compareAgg(const void* pa, const void* pb) {
    const timing_agg* a = (const timing_agg*)pa;
    const timing_agg* b = (const timing_agg*)pb;
    if (a->totalMs < b->totalMs) return 1;
    if (a->totalMs > b->totalMs) return -1;
    return 0;
}

static double getTimestampPeriod(VkPhysicalDevice physicalDevice) {
    static double cached = 0.0;
    static VkPhysicalDevice cachedDevice = VK_NULL_HANDLE;
    if (cachedDevice != physicalDevice) {
        VkPhysicalDeviceProperties properties;
        vkGetPhysicalDeviceProperties(physicalDevice, &properties);
        cached = (double)properties.limits.timestampPeriod;
        cachedDevice = physicalDevice;
    }
    return cached;
}

static void logFrame(session s, uint32_t slot, operation ops[], int opCount, const char* phase, int token) {
    uint32_t qBase = slot * QUERIES_PER_FRAME;
    uint64_t stamps[2 + 2 * opCount];
    int queryCount = 2 + 2 * opCount;
    vkGetQueryPoolResults(s.dev.device, s.qpool, qBase, queryCount,
                          queryCount * sizeof(uint64_t), stamps,
                          sizeof(uint64_t), VK_QUERY_RESULT_WAIT_BIT);

    double period = getTimestampPeriod(s.dev.physicalDevice);
    ensureTimingLog();

    for (int i = 0; i < opCount; i++) {
        uint32_t ticks = (uint32_t)stamps[3 + 2 * i] - (uint32_t)stamps[2 + 2 * i];
        double ms = (double)ticks * period / 1000000.0;
        char layerStr[8];
        if (ops[i].layer >= 0) {
            snprintf(layerStr, sizeof(layerStr), "%3d", ops[i].layer);
        } else {
            snprintf(layerStr, sizeof(layerStr), " --");
        }
        fprintf(timingFile, "[%-7s][tok=%5d][L=%3s] %-38s %10.3f ms\n",
                phase, token, layerStr, ops[i].shader, ms);

        timing_agg* a = findAgg(ops[i].shader);
        if (a != NULL) {
            a->totalMs += ms;
            a->calls++;
        }
        grandTotalMs += ms;
    }
    fflush(timingFile);
}

void closeTimingLog(void) {
    if (timingFile == NULL) return;
    qsort(aggTable, aggCount, sizeof(timing_agg), compareAgg);
    fprintf(timingFile, "\n==== timing summary ====\n");
    for (int i = 0; i < aggCount; i++) {
        double avg = aggTable[i].calls > 0 ? aggTable[i].totalMs / aggTable[i].calls : 0.0;
        fprintf(timingFile, "%-42s calls=%5ld total=%12.3f ms avg=%9.3f ms\n",
                aggTable[i].shader, aggTable[i].calls, aggTable[i].totalMs, avg);
    }
    fprintf(timingFile, "grand total: %.3f ms\n", grandTotalMs);
    fclose(timingFile);
    timingFile = NULL;
    aggCount = 0;
    grandTotalMs = 0.0;
}

static void recordFrame(session* s, operation ops[], int opCount) {
    uint32_t slot = s->frame;
    uint32_t qBase = slot * QUERIES_PER_FRAME;
    VkCommandBuffer cb = s->buffer[slot];

    vkWaitForFences(s->dev.device, 1, &s->fence[slot], VK_TRUE, UINT64_MAX);
    vkResetFences(s->dev.device, 1, &s->fence[slot]);
    vkResetCommandBuffer(cb, 0);

    VkCommandBufferBeginInfo beginInfo = {0};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &beginInfo);

    vkCmdResetQueryPool(cb, s->qpool, qBase, QUERIES_PER_FRAME);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, s->qpool, qBase);

    for (int i = 0; i < opCount; i++) {
        operation* op = &ops[i];
        pipe_entry* entry = getPipeEntry(*s, op);
        VkDescriptorSet set = getDescriptorSet(*s, entry->setLayout, op);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, entry->pipeline);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, entry->layout, 0, 1, &set, 0, NULL);

        if (op->pushConstantCount > 0) {
            vkCmdPushConstants(cb, entry->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                               sizeof(int) * op->pushConstantCount, op->pushConstants);
        }
        {
            VkMemoryBarrier barrier = {
                .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_SHADER_READ_BIT
            };
            vkCmdPipelineBarrier(cb,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &barrier, 0, NULL, 0, NULL);
        }
        if (timingEnabled) {
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, s->qpool, qBase + 2 + 2 * i);
        }
        vkCmdDispatch(cb, (uint32_t)op->dispatchX, (uint32_t)op->dispatchY, (uint32_t)op->dispatchZ);
        if (timingEnabled) {
            vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, s->qpool, qBase + 3 + 2 * i);
        }
    }

    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, s->qpool, qBase + 1);
    vkEndCommandBuffer(cb);
}

static void submitFrame(session* s) {
    uint32_t slot = s->frame;
    VkSubmitInfo submitInfo = {0};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &s->buffer[slot];
    vkQueueSubmit(s->dev.queue, 1, &submitInfo, s->fence[slot]);
    s->lastSubmitted = (int)slot;
    s->frame = (slot + 1) % FRAME_COUNT;
}

static void waitLast(session* s) {
    if (s->lastSubmitted < 0) return;
    vkWaitForFences(s->dev.device, 1, &s->fence[s->lastSubmitted], VK_TRUE, UINT64_MAX);
}

void executeRecord(session* s, operation ops[], int opCount) {
    recordFrame(s, ops, opCount);
}

void executeSubmitNow(session* s) {
    submitFrame(s);
}

void executeWaitLast(session* s) {
    waitLast(s);
}

void logLastFrame(session* s, operation ops[], int opCount, const char* phase, int token) {
    if (!timingEnabled || s->lastSubmitted < 0) return;
    logFrame(*s, (uint32_t)s->lastSubmitted, ops, opCount, phase, token);
}

void execute(session s, operation ops[], int opCount) {
    s.frame = 0;
    recordFrame(&s, ops, opCount);
    submitFrame(&s);
    waitLast(&s);
}

void executeLogged(session s, operation ops[], int opCount, const char* phase, int token) {
    execute(s, ops, opCount);
    if (timingEnabled && phase != NULL) {
        logFrame(s, 0, ops, opCount, phase, token);
    }
}

double getExecutionTime(session s) {
    uint64_t timestamps[2];
    vkGetQueryPoolResults(s.dev.device, s.qpool, 0, 2, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_WAIT_BIT);

    double timestampPeriod = getTimestampPeriod(s.dev.physicalDevice);
    return (double)((unsigned int)timestamps[1] - (unsigned int)timestamps[0]) * timestampPeriod / 1000000.0;
}
