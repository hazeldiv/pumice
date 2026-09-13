#ifndef dispatch_h
#define dispatch_h

#include <vulkan/vulkan.h>
#include "buffer.h"
#include "device.h"
#include "descriptor.h"
#include "pipeline.h"
#include "session.h"

#define MAX_OP_BUFFERS 18
#define MAX_PUSH_CONSTANTS 16

typedef struct operation {
    char shader[128];
    buffer buffers[MAX_OP_BUFFERS];
    int bufferCount;
    int pushConstants[MAX_PUSH_CONSTANTS];
    int pushConstantCount;
    int dispatchX;
    int dispatchY;
    int dispatchZ;
    int layer;
} operation;

void execute(session s, operation ops[], int opCount);
void dispatchReset(VkDevice device);
void executeLogged(session s, operation ops[], int opCount, const char* phase, int token);
void executeRecord(session* s, operation ops[], int opCount);
void executeSubmitNow(session* s);
void executeWaitLast(session* s);
void logLastFrame(session* s, operation ops[], int opCount, const char* phase, int token);
void setTimingEnabled(int enabled);
int isTimingEnabled(void);
void closeTimingLog(void);
double getExecutionTime(session s);
const char* shaderRootDir(void);
void setShaderRootDir(const char* dir);

#endif