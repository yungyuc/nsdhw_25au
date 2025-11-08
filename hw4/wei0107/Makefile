# Robust Makefile for HW4 (_matrix) — works on CI without pybind11 pip pkg
PYTHON3 ?= python3

# 1) Resolve extension suffix
EXT_SUFFIX := $(shell $(PYTHON3) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '')")
ifeq ($(strip $(EXT_SUFFIX)),)
EXT_SUFFIX := .so
endif

# 2) Try to get Python include dir via sysconfig (toolcache-friendly)
PY_INC_DIR := $(shell $(PYTHON3) -c "import sysconfig; print(sysconfig.get_paths().get('include',''))")

# 3) Try python-config includes (fallback)
PY_CFG_INC := $(shell sh -c 'command -v $(PYTHON3)-config >/dev/null 2>&1 && $(PYTHON3)-config --includes || (command -v python3-config >/dev/null 2>&1 && python3-config --includes)')

# 4) Try pybind11 includes from module; if not, use system headers path
PYB11_INC := $(shell $(PYTHON3) -m pybind11 --includes 2>/dev/null)
ifeq ($(strip $(PYB11_INC)),)
ifneq ("$(wildcard /usr/include/pybind11)","")
  PYB11_INC := -I/usr/include/pybind11
endif
endif

INCLUDES := $(PYB11_INC) $(if $(PY_INC_DIR),-I$(PY_INC_DIR),) $(PY_CFG_INC)

CXX ?= g++
CXXFLAGS ?= -O3 -fPIC -Wall -Wextra -std=c++17 -march=native
LDFLAGS ?= -shared
LDLIBS ?= -ldl

TARGET := _matrix$(EXT_SUFFIX)
SOURCES := matrix_ops.cpp

all: $(TARGET)

$(TARGET): $(SOURCES)
	@echo "==> Using INCLUDES: $(INCLUDES)"
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCES) -o $(TARGET) $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGET) *.o
