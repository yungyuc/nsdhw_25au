CXX := g++
CXXFLAGS := -std=c++17 -O3 -fPIC -Wall -fopenmp -march=native

PYBIND11_INCLUDES := $(shell python3 -m pybind11 --includes)
PYTHON_INCLUDES   := $(shell python3-config --includes)
PYTHON_LIBS       := $(shell python3-config --ldflags)

ifneq ($(MKLROOT),)
CBLAS_INCLUDES := -I$(MKLROOT)/include
CBLAS_LIBS := -L$(MKLROOT)/lib/intel64 -lmkl_rt
else
CBLAS_INCLUDES :=
CBLAS_LIBS ?= -lopenblas
endif

ALL_INCLUDES := -I. $(PYBIND11_INCLUDES) $(PYTHON_INCLUDES) $(CBLAS_INCLUDES)
ALL_LIBS     := $(CBLAS_LIBS) $(PYTHON_LIBS)

SOURCES := matrix.cpp ../mod.cpp
TARGET := _matrix$(shell python3-config --extension-suffix)


.PHONY: all clean test benchmark
all: $(TARGET)

$(TARGET): $(SOURCES) matrix.hpp
	$(CXX) $(CXXFLAGS) $(ALL_INCLUDES) -shared -o $@ $(SOURCES) $(ALL_LIBS)

test: $(TARGET)
	python3 -m pytest -v

benchmark: $(TARGET)
	python3 benchmark.py > performance.txt

clean:
	rm -f $(TARGET) *.o performance.txt