# Build Python extension _matrix for HW4
# Usage:
#   make
#   make clean

PYTHON3 ?= python3
EXT_SUFFIX := $(shell $(PYTHON3) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
PY_INCLUDES := $(shell $(PYTHON3) -m pybind11 --includes)

CXX ?= g++
CXXFLAGS ?= -O3 -fPIC -Wall -Wextra -std=c++17 -march=native
LDFLAGS ?= -shared
LDLIBS ?= -ldl

TARGET := _matrix$(EXT_SUFFIX)
SOURCES := matrix_ops.cpp

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(PY_INCLUDES) $(SOURCES) -o $(TARGET) $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGET) *.o
