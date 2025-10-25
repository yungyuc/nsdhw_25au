CXX := g++
CXXFLAGS := -O2 -std=c++17 -Wall -Wextra -fPIC
# Prefer pybind11 --includes; fall back to python3-config when pybind11 module is missing (CI)
PYINCLUDES := $(shell python3 -m pybind11 --includes 2>/dev/null || python3-config --includes) -I/usr/include/pybind11
PYEXT := $(shell python3-config --extension-suffix)
PYLDFLAGS := $(shell python3-config --ldflags)
# Link OpenBLAS if available (apt install libopenblas-dev). Override externally if needed.
BLAS_LIBS ?= -lopenblas

TARGET := _matrix$(PYEXT)
SOURCES := matrix.cpp binding.cpp

.PHONY: all test clean perf

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) -shared $(PYINCLUDES) $(SOURCES) -o $@ $(PYLDFLAGS) $(BLAS_LIBS)

test: all
	PYTHONPATH=".:$$PYTHONPATH" python3 -m pytest -v

perf: all
	PYTHONPATH=".:$$PYTHONPATH" python3 benchmark.py > performance.txt
	@echo "performance written to performance.txt"

clean:
	rm -f $(TARGET)
	rm -rf __pycache__ .pytest_cache


