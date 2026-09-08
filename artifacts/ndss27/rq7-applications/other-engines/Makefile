# Standalone Pin/libdft64 tools for the RQ7 multitool applications eval.
# Absolute paths, built out-of-tree.

PIN_ROOT   ?= /home/jns/Documents/Telecom/PRIM/benchmark/external/pin-3.20-98437-gf02b61307-gcc-linux
LIBDFT_SRC ?= /home/jns/Documents/Telecom/PRIM/benchmark/external/libdft64/src
LIBDFT_OBJ ?= $(LIBDFT_SRC)/obj-intel64

CONFIG_ROOT := $(PIN_ROOT)/source/tools/Config
include $(CONFIG_ROOT)/makefile.config

TOOL_CXXFLAGS += -Wno-deprecated-declarations -Wno-error=non-c-typedef-for-linkage \
                 -Wno-error -DNO_PINTOOL_LOG -I$(LIBDFT_SRC)
TOOL_LIBS     += -L$(LIBDFT_OBJ) -ldft

TOOL_ROOTS := cf_leak dns_taint

include $(TOOLS_ROOT)/Config/makefile.default.rules

.PHONY: tools
tools: $(OBJDIR)cf_leak$(PINTOOL_SUFFIX) $(OBJDIR)dns_taint$(PINTOOL_SUFFIX)
