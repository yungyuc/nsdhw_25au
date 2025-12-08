PYTHON3 ?= python3

EXT_SUFFIX := $(shell $(PYTHON3) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '')")
ifeq ($(strip $(EXT_SUFFIX)),)
EXT_SUFFIX := .so
endif

PY_INC_DIR := $(shell $(PYTHON3) -c "import sysconfig; print(sysconfig.get_paths().get('include',''))")
PY_CFG_INC := $(shell sh -c 'command -v $(PYTHON3)-config >/dev/null 2>&1 && $(PYTHON3)-config --includes || (command -v python3-config >/dev/null 2>&1 && python3-config --includes)')
PYB11_INC := $(shell $(PYTHON3) -m pybind11 --includes 2>/dev/null)
NUMPY_INC := $(shell $(PYTHON3) -c "import numpy; print(numpy.get_include())")

INCLUDES := $(PYB11_INC) -I$(PY_INC_DIR) $(PY_CFG_INC) -I$(NUMPY_INC) -I. -I..

CXX ?= g++
CXXFLAGS ?= -O3 -fPIC -Wall -Wextra -std=c++17 -march=native
LDFLAGS ?= -shared
LDLIBS ?= -ldl

TARGET  := _matrix$(EXT_SUFFIX)
SOURCES := mod.cpp matrix.cpp

all: $(TARGET)

$(TARGET): $(SOURCES) matrix.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCES) -o $(TARGET) $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGET) *.o
