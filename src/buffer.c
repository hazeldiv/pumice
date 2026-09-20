#include <vulkan/vulkan.h>
#include "device.h"
#include "buffer.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

static uint32_t findMemoryType(VkPhysicalDevice physicalDevice, VkMemoryRequirements memReqs, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((memReqs.memoryTypeBits & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return UINT32_MAX;
}

static int64_t g_deviceLocalBytes = 0;
static int64_t g_hostVisibleBytes = 0;

static int g_allocFailed = 0;
static char g_allocError[256] = {0};

void bufferAllocClear(void) {
    g_allocFailed = 0;
    g_allocError[0] = '\0';
}

int bufferAllocFailed(void) {
    return g_allocFailed;
}

const char* bufferAllocError(void) {
    return g_allocError;
}

void bufferAllocFail(const char* message) {
    if (g_allocFailed) return;
    g_allocFailed = 1;
    snprintf(g_allocError, sizeof(g_allocError), "%s", message);
}

void bufferMemoryTotals(int64_t* deviceLocal, int64_t* hostVisible) {
    if (deviceLocal != NULL) *deviceLocal = g_deviceLocalBytes;
    if (hostVisible != NULL) *hostVisible = g_hostVisibleBytes;
}

static int allocateBufferMemory(VkDevice device, VkPhysicalDevice physicalDevice, VkBuffer buffer, VkMemoryRequirements memReqs, VkMemoryPropertyFlags properties, VkDeviceMemory* memory, const char* name) {
    int isDev = (properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
    uint32_t memoryTypeIndex = findMemoryType(physicalDevice, memReqs, properties);
    if (memoryTypeIndex == UINT32_MAX) {
        char msg[256];
        snprintf(msg, sizeof(msg), "out of %s memory: no suitable memory type for '%s' (%.2f MB)",
                 isDev ? "GPU" : "host", name ? name : "(unnamed)",
                 (double)memReqs.size / (1024.0 * 1024.0));
        bufferAllocFail(msg);
        return -1;
    }
    VkMemoryAllocateInfo allocInfo = {0};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memReqs.size;
    allocInfo.memoryTypeIndex = memoryTypeIndex;
    if (vkAllocateMemory(device, &allocInfo, NULL, memory) != VK_SUCCESS) {
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "out of %s memory: failed to allocate '%s' (%.2f MB) | device_local=%.2f MB host_visible=%.2f MB",
                 isDev ? "GPU" : "host", name ? name : "(unnamed)",
                 (double)memReqs.size / (1024.0 * 1024.0),
                 (double)g_deviceLocalBytes / (1024.0 * 1024.0),
                 (double)g_hostVisibleBytes / (1024.0 * 1024.0));
        bufferAllocFail(msg);
        return -1;
    }
    vkBindBufferMemory(device, buffer, *memory, 0);
    if (properties & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
        g_deviceLocalBytes += memReqs.size;
    } else if (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        g_hostVisibleBytes += memReqs.size;
    }
    return 0;
}

buffer createBuffer(VkDevice device, VkPhysicalDevice physicalDevice, void* data, int64_t size, int memoryType) {
    return createBufferNamed(device, physicalDevice, data, size, memoryType, NULL);
}

buffer createBufferNamed(VkDevice device, VkPhysicalDevice physicalDevice, void* data, int64_t size, int memoryType, const char* name) {
    buffer buf = {0};
    buf.memoryType = memoryType;
    buf.size = size;
    if (name) {
        snprintf(buf.name, sizeof(buf.name), "%s", name);
    }
    if (g_allocFailed) return buf;
    if (data == NULL && size > 0) {
        char msg[256];
        snprintf(msg, sizeof(msg), "out of host memory: could not stage '%s' (%.2f MB)",
                 name ? name : "(unnamed)", (double)size / (1024.0 * 1024.0));
        bufferAllocFail(msg);
        return buf;
    }

    VkBufferCreateInfo bufferInfo = {0};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bufferInfo, NULL, &buf.buffer) != VK_SUCCESS) {
        bufferAllocFail("out of GPU memory: vkCreateBuffer failed");
        return buf;
    }
    vkGetBufferMemoryRequirements(device, buf.buffer, &buf.memReqs);

    if (memoryType == MEMORY_RAM) {
        if (allocateBufferMemory(device, physicalDevice, buf.buffer, buf.memReqs, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &buf.memory, name) != 0) {
            vkDestroyBuffer(device, buf.buffer, NULL);
            buf.buffer = VK_NULL_HANDLE;
            return buf;
        }
        void* mappedMemory = NULL;
        vkMapMemory(device, buf.memory, 0, size, 0, &mappedMemory);
        memcpy(mappedMemory, data, size);
        buf.mappedMemory = mappedMemory;
    } else {
        VkBufferCreateInfo stagingInfo = {0};
        stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        stagingInfo.size = size;
        stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        stagingInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device, &stagingInfo, NULL, &buf.stagingBuffer) != VK_SUCCESS) {
            vkDestroyBuffer(device, buf.buffer, NULL);
            buf.buffer = VK_NULL_HANDLE;
            bufferAllocFail("out of GPU memory: vkCreateBuffer failed");
            return buf;
        }
        VkMemoryRequirements stagingMemReqs;
        vkGetBufferMemoryRequirements(device, buf.stagingBuffer, &stagingMemReqs);
        if (allocateBufferMemory(device, physicalDevice, buf.stagingBuffer, stagingMemReqs, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &buf.stagingMemory, name) != 0) {
            vkDestroyBuffer(device, buf.stagingBuffer, NULL);
            vkDestroyBuffer(device, buf.buffer, NULL);
            buf.stagingBuffer = VK_NULL_HANDLE;
            buf.buffer = VK_NULL_HANDLE;
            return buf;
        }
        void* mappedMemory = NULL;
        vkMapMemory(device, buf.stagingMemory, 0, size, 0, &mappedMemory);
        memcpy(mappedMemory, data, size);
        vkUnmapMemory(device, buf.stagingMemory);

        if (allocateBufferMemory(device, physicalDevice, buf.buffer, buf.memReqs, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &buf.memory, name) != 0) {
            vkDestroyBuffer(device, buf.stagingBuffer, NULL);
            vkFreeMemory(device, buf.stagingMemory, NULL);
            vkDestroyBuffer(device, buf.buffer, NULL);
            buf.stagingBuffer = VK_NULL_HANDLE;
            buf.stagingMemory = VK_NULL_HANDLE;
            buf.buffer = VK_NULL_HANDLE;
            return buf;
        }
    }

    return buf;
}

void createTransferAndCopy(VkDevice device, VkQueue queue, buffer* buffers, int bufferCount) {
    VkCommandPoolCreateInfo poolInfo = {0};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = 0;
    VkCommandPool transferPool;
    vkCreateCommandPool(device, &poolInfo, NULL, &transferPool);

    VkCommandBufferAllocateInfo allocInfo = {0};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = transferPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer transferCmd;
    vkAllocateCommandBuffers(device, &allocInfo, &transferCmd);

    VkCommandBufferBeginInfo beginInfo = {0};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(transferCmd, &beginInfo);

    for (int i = 0; i < bufferCount; i++) {
        if (buffers[i].memoryType == MEMORY_VRAM && buffers[i].stagingBuffer != VK_NULL_HANDLE) {
            VkBufferCopy copyRegion = {0};
            copyRegion.size = buffers[i].size;
            vkCmdCopyBuffer(transferCmd, buffers[i].stagingBuffer, buffers[i].buffer, 1, &copyRegion);
        }
    }

    vkEndCommandBuffer(transferCmd);

    VkSubmitInfo submitInfo = {0};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &transferCmd;
    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    vkDestroyCommandPool(device, transferPool, NULL);
}

void destroyBuffer(VkDevice device, buffer buf) {
    if (buf.buffer == VK_NULL_HANDLE) return;
    if (buf.stagingBuffer != VK_NULL_HANDLE) {
        VkMemoryRequirements reqs;
        vkGetBufferMemoryRequirements(device, buf.stagingBuffer, &reqs);
        g_hostVisibleBytes -= (int64_t)reqs.size;
        vkDestroyBuffer(device, buf.stagingBuffer, NULL);
        vkFreeMemory(device, buf.stagingMemory, NULL);
    }
    if (buf.mappedMemory != NULL) {
        vkUnmapMemory(device, buf.memory);
    }
    VkMemoryRequirements reqs;
    vkGetBufferMemoryRequirements(device, buf.buffer, &reqs);
    if (buf.memoryType == MEMORY_RAM) g_hostVisibleBytes -= (int64_t)reqs.size;
    else g_deviceLocalBytes -= (int64_t)reqs.size;
    vkDestroyBuffer(device, buf.buffer, NULL);
    vkFreeMemory(device, buf.memory, NULL);
}

void writeStaging(VkDevice device, buffer* buf, const void* data, int64_t size) {
    if (buf->stagingMemory == VK_NULL_HANDLE) return;
    void* mapped;
    vkMapMemory(device, buf->stagingMemory, 0, size, 0, &mapped);
    memcpy(mapped, data, (size_t)size);
    vkUnmapMemory(device, buf->stagingMemory);
}

void clearStaging(VkDevice device, buffer* buf) {
    if (buf->stagingMemory == VK_NULL_HANDLE) return;
    void* mapped;
    vkMapMemory(device, buf->stagingMemory, 0, buf->size, 0, &mapped);
    memset(mapped, 0, (size_t)buf->size);
    vkUnmapMemory(device, buf->stagingMemory);
}

void releaseStaging(VkDevice device, buffer* buf) {
    if (buf->stagingBuffer != VK_NULL_HANDLE) {
        VkMemoryRequirements reqs;
        vkGetBufferMemoryRequirements(device, buf->stagingBuffer, &reqs);
        g_hostVisibleBytes -= (int64_t)reqs.size;
        vkDestroyBuffer(device, buf->stagingBuffer, NULL);
        vkFreeMemory(device, buf->stagingMemory, NULL);
        buf->stagingBuffer = VK_NULL_HANDLE;
        buf->stagingMemory = VK_NULL_HANDLE;
    }
}

void readBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkQueue queue, buffer buf, void* output) {
    if (buf.mappedMemory != NULL) {
        memcpy(output, buf.mappedMemory, buf.size);
    } else {
        VkBufferCreateInfo stagingInfo = {0};
        stagingInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        stagingInfo.size = buf.size;
        stagingInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        VkBuffer staging;
        vkCreateBuffer(device, &stagingInfo, NULL, &staging);
        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(device, staging, &memReqs);
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        uint32_t memType;
        for (memType = 0; memType < memProperties.memoryTypeCount; memType++) {
            if ((memReqs.memoryTypeBits & (1 << memType)) && (memProperties.memoryTypes[memType].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))) break;
        }
        VkMemoryAllocateInfo allocInfo = {0};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memReqs.size;
        allocInfo.memoryTypeIndex = memType;
        VkDeviceMemory stagingMem;
        vkAllocateMemory(device, &allocInfo, NULL, &stagingMem);
        vkBindBufferMemory(device, staging, stagingMem, 0);
        VkCommandBufferBeginInfo beginInfo = {0};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VkCommandPool pool;
        VkCommandPoolCreateInfo poolInfo = {0};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.queueFamilyIndex = 0;
        vkCreateCommandPool(device, &poolInfo, NULL, &pool);
        VkCommandBuffer cmd;
        VkCommandBufferAllocateInfo allocInfoCmd = {0};
        allocInfoCmd.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfoCmd.commandPool = pool;
        allocInfoCmd.commandBufferCount = 1;
        vkAllocateCommandBuffers(device, &allocInfoCmd, &cmd);
        vkBeginCommandBuffer(cmd, &beginInfo);
        VkBufferCopy copyRegion = {0};
        copyRegion.size = buf.size;
        vkCmdCopyBuffer(cmd, buf.buffer, staging, 1, &copyRegion);
        vkEndCommandBuffer(cmd);
        VkSubmitInfo submitInfo = {0};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;
        vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);
        vkDestroyCommandPool(device, pool, NULL);
        void* mapped;
        vkMapMemory(device, stagingMem, 0, buf.size, 0, &mapped);
        memcpy(output, mapped, buf.size);
        vkUnmapMemory(device, stagingMem);
        vkDestroyBuffer(device, staging, NULL);
        vkFreeMemory(device, stagingMem, NULL);
    }
}
