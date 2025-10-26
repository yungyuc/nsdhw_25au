# nsdhw_25au/hw3/wei0107/Makefile (CI-friendly)

PYTHON      ?= python3
CXX         ?= g++

# 擷取 Python 的 .so 後綴
EXT_SUFFIX  := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# Python.h 的 include 路徑
PY_INCLUDE  := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('INCLUDEPY'))")

# 系統安裝的 pybind11
PYBIND11_INCLUDE := /usr/include/pybind11

CXXFLAGS    ?= -O3 -march=native -fPIC -std=c++17 -DNDEBUG
CXXFLAGS    += -I$(PY_INCLUDE) -I$(PYBIND11_INCLUDE)

LDFLAGS     ?= -shared

# BLAS 連結
BLAS_LDFLAGS := $(shell pkg-config --libs openblas 2>/dev/null)
ifeq ($(strip $(BLAS_LDFLAGS)),)
BLAS_LDFLAGS := -lblas
endif

TARGET = _matrix$(EXT_SUFFIX)
SRC    = matrix_ops.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -c $(SRC) -o matrix_ops.o
	$(CXX) $(LDFLAGS) matrix_ops.o -o $(TARGET) $(BLAS_LDFLAGS)

clean:
	rm -f *.o _matrix*.so performance.txt

test: all
	$(PYTHON) -m pytest -q

perf: all
	$(PYTHON) bench.py

.PHONY: all clean test perf
