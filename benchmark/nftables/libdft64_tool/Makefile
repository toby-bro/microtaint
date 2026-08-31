# Standalone Pin/libdft64 tool Makefile for the nft_byteorder canary_taint tool.
#
# Reuses Pin's Config makefiles (like libdft64/tools/Makefile) but hardcodes
# absolute paths so the tool can live outside the libdft tree.

PIN_ROOT   ?= /home/jns/Documents/Telecom/PRIM/benchmark/external/pin-3.20-98437-gf02b61307-gcc-linux
LIBDFT_SRC ?= /home/jns/Documents/Telecom/PRIM/benchmark/external/libdft64/src
LIBDFT_OBJ ?= $(LIBDFT_SRC)/obj-intel64

CONFIG_ROOT := $(PIN_ROOT)/source/tools/Config
include $(CONFIG_ROOT)/makefile.config

# Pin 3.20 headers trip -Werror=non-c-typedef-for-linkage on modern GCC.
TOOL_CXXFLAGS += -Wno-deprecated-declarations -Wno-error=non-c-typedef-for-linkage \
                 -Wno-error -DNO_PINTOOL_LOG -I$(LIBDFT_SRC)
TOOL_LIBS     += -L$(LIBDFT_OBJ) -ldft

TOOL_ROOTS := canary_taint

include $(TOOLS_ROOT)/Config/makefile.default.rules

.PHONY: tool
tool: $(OBJDIR)canary_taint$(PINTOOL_SUFFIX)
