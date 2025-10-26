CXX := g++
CXXFLAGS := -std=c++17 -O3 -fPIC -Wall -fopenmp -march=native


PYBIND11_INCLUDES := $(shell python3 -m pybind11 --includes)
PYTHON_INCLUDES   := $(shell python3-config --includes)
PYTHON_LIBS       := $(shell python3-config --ldflags)

MKL_INC           := -I$(MKLROOT)/include

MKL_LIBS          := -L$(MKLROOT)/lib/intel64 -lmkl_rt

ALL_INCLUDES := $(PYBIND11_INCLUDES) $(PYTHON_INCLUDES) $(MKL_INC)
ALL_LIBS     := $(MKL_LIBS) $(PYTHON_LIBS)

SOURCES := matrix.cpp
TARGET := _matrix$(shell python3-config --extension-suffix)


.PHONY: all clean test benchmark
all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(ALL_INCLUDES) -shared -o $@ $(SOURCES) $(ALL_LIBS)

test: $(TARGET)
	python3 -m pytest -v

benchmark: $(TARGET)
	python3 benchmark.py --n 1024 --block 64 > performance.txt

clean:
	rm -f $(TARGET) *.o performance.txt