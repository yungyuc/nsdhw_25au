CXX := g++
CXXFLAGS := -O2 -std=c++17 -Wall -Wextra -fPIC
NUMPY_INCLUDE := $(shell python3 -c "import numpy,sys; sys.stdout.write(numpy.get_include())" 2>/dev/null)
PYINCLUDES := $(shell python3 -m pybind11 --includes 2>/dev/null || python3-config --includes) \
              -I/usr/include/pybind11 \
              $(if $(NUMPY_INCLUDE),-I$(NUMPY_INCLUDE))
PYEXT := $(shell python3-config --extension-suffix)
PYLDFLAGS := $(shell python3-config --ldflags)
BLAS_LIBS ?= -lblas

TARGET := _matrix$(PYEXT)
SOURCES := matrix.cpp ../mod.cpp

.PHONY: all test clean perf

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -shared $(PYINCLUDES) -I.. -I. $(SOURCES) -o $@ $(PYLDFLAGS) $(BLAS_LIBS)

test: all
	PYTHONPATH=".:$$PYTHONPATH" python3 -m pytest -v

perf: all
	PYTHONPATH=".:$$PYTHONPATH" python3 benchmark.py > performance.txt
	@echo "performance written to performance.txt"

clean:
	rm -f $(TARGET)
	rm -rf ../__pycache__ ../.pytest_cache


