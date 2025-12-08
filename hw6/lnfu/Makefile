CXX := g++
CXXFLAGS = -std=c++17 -O3 -march=native -fPIC -fopenmp -Wall -Wextra

PYTHON = python3

IOMPROOT := /opt/intel/oneapi/compiler/latest
MKLROOT := /opt/intel/oneapi/mkl/latest

NUMPY_INCLUDES := $(shell $(PYTHON) -c "import numpy; print(numpy.get_include())")
INCLUDES := -I. -I$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))") $(shell $(PYTHON) -m pybind11 --includes 2>/dev/null || echo "-I/usr/include/pybind11") -I$(NUMPY_INCLUDES) -I$(MKLROOT)/include
# INCLUDES := -I$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))") $(shell $(PYTHON) -m pybind11 --includes 2>/dev/null || echo "-I/usr/include/pybind11") -I$(MKLROOT)/include
LDFLAGS := -shared -L$(MKLROOT)/lib -lmkl_rt -Wl,-rpath,$(MKLROOT)/lib
TARGET := _matrix$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

all: $(TARGET)

$(TARGET): mod.cpp matrix.cpp matrix.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) mod.cpp matrix.cpp -o $(TARGET) $(LDFLAGS)

test: $(TARGET)
	LD_LIBRARY_PATH=$(IOMPROOT)/lib:$(MKLROOT)/lib:$$LD_LIBRARY_PATH $(PYTHON) -m pytest -q -vv

clean:
	rm -f $(TARGET) performance.txt

.PHONY: all test clean
