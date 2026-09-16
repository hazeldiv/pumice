CC           := gcc
VULKAN_INC   := -I"$(VULKAN_SDK)/Include"
TOKENIZERS   := tokenizers
NODE_EXE     := $(shell node -p "process.execPath")
NODE_CACHE   := $(subst \,/,$(LOCALAPPDATA))/node-gyp/Cache
NODE_INC     ?= $(firstword $(wildcard $(NODE_CACHE)/*/include/node))

CFLAGS       := -O2 -Wall -Wextra -Iinclude $(VULKAN_INC) -I"$(TOKENIZERS)/include"
ifneq ($(NODE_INC),)
CFLAGS       += -I"$(NODE_INC)"
endif

LDFLAGS      := -L"$(VULKAN_SDK)/Lib" -lvulkan-1 -luser32 -lgdi32
NODE_LDFLAGS := -shared -L"$(VULKAN_SDK)/Lib" -lvulkan-1 -luser32 -lgdi32 \
                -lws2_32 -luserenv -lbcrypt -lntdll -ladvapi32 -lole32 -loleaut32 \
                -lpsapi -lshell32 -lshlwapi -lcrypt32

TARGET       := main
NODE_MODULE  := vk_compute.node
SRC_DIR      := src
BUILD_DIR    := build
BIN_DIR      := bin
SHADER_DIR   := shader
NODE_LIB     := $(BUILD_DIR)/libnode.a

SRCS         := $(wildcard $(SRC_DIR)/*.c)
CORE_SRCS    := $(filter-out $(SRC_DIR)/main.c $(SRC_DIR)/addon.c $(SRC_DIR)/engine.c,$(SRCS))
CORE_OBJS    := $(CORE_SRCS:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
ENGINE_OBJ   := $(BUILD_DIR)/engine.o
MAIN_OBJS    := $(BUILD_DIR)/main.o $(CORE_OBJS)
ADDON_OBJS   := $(BUILD_DIR)/addon.o $(ENGINE_OBJ) $(CORE_OBJS)
DEPS         := $(CORE_OBJS:.o=.d) $(ENGINE_OBJ:.o=.d) $(BUILD_DIR)/main.d $(BUILD_DIR)/addon.d
rwildcard    = $(foreach d,$(wildcard $(1)*),$(call rwildcard,$(d)/,$(2)) $(filter $(subst *,%,$(2)),$(d)))
SHADERS      := $(filter-out $(SHADER_DIR)/Prototype/%,$(call rwildcard,$(SHADER_DIR)/,*.comp))
SHADER_OUT   := $(BIN_DIR)/shader
SHADER_OUT_W := $(subst /,\,$(SHADER_OUT))
SHADERS_OBJS := $(addprefix $(SHADER_OUT)/,$(notdir $(SHADERS:.comp=.spv)))

all: ${SHADERS_OBJS} $(BIN_DIR)/$(TARGET) $(BIN_DIR)/$(NODE_MODULE)

$(BIN_DIR)/$(TARGET): $(MAIN_OBJS)
	@if not exist $(BIN_DIR) mkdir $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "Built successfully: $@.exe"

$(BUILD_DIR)/libnode.a:
	@if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)
	powershell -NoProfile -ExecutionPolicy Bypass -File gen_node_lib.ps1 -NodeExe "$(NODE_EXE)" -OutLib "$(NODE_LIB)"

$(BIN_DIR)/$(NODE_MODULE): $(ADDON_OBJS) $(NODE_LIB) $(TOKENIZERS)/lib/libtokenizers_c.a
	@if not exist $(BIN_DIR) mkdir $(BIN_DIR)
	$(CC) $(CFLAGS) $^ -o $@ $(NODE_LDFLAGS)
	@echo "Built successfully: $@"

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@if not exist $(BUILD_DIR) mkdir $(BUILD_DIR)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

define COMPILE_SHADER
$(SHADER_OUT)/$(notdir $(basename $(1))).spv: $(1)
	@if not exist $(SHADER_OUT_W) mkdir $(SHADER_OUT_W)
	"$(VULKAN_SDK)/Bin/glslangValidator" -V --target-env vulkan1.1 "$(1)" -o $$@
endef
$(foreach f,$(SHADERS),$(eval $(call COMPILE_SHADER,$(f))))

-include $(DEPS)

run: all
	@cd bin && main.exe

clean:
	@if exist $(BIN_DIR) rmdir /S /Q $(BIN_DIR)
	@if exist $(BUILD_DIR) rmdir /S /Q $(BUILD_DIR)
