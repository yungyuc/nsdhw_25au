PYTHON      ?= python3
PYINCLUDES  := $(shell $(PYTHON) -m pybind11 --includes)
EXT_SUFFIX  := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
CXX         ?= g++
CXXFLAGS    ?= -O3 -march=native -fPIC -std=c++17 -DNDEBUG
LDFLAGS     ?= -shared

BLAS_LDFLAGS := $(shell pkg-config --libs openblas 2>/dev/null)
ifeq ($(strip $(BLAS_LDFLAGS)),)
BLAS_LDFLAGS := -lblas
endif

TARGET = _matrix$(EXT_SUFFIX)
SRC    = matrix_ops.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(PYINCLUDES) -I. -c $(SRC) -o matrix_ops.o
	$(CXX) $(LDFLAGS) matrix_ops.o -o $(TARGET) $(BLAS_LDFLAGS)

clean:
	rm -f *.o $(TARGET) performance.txt

test: all
	$(PYTHON) -m pytest -q

perf: all
	$(PYTHON) bench.py

.PHONY: all clean test perf
