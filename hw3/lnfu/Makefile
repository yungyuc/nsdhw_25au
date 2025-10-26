CXX := g++
CXXFLAGS = -std=c++17 -O3 -march=native -fPIC -fopenmp -Wall -Wextra

PYTHON = python3

IOMPROOT := /opt/intel/oneapi/compiler/latest
MKLROOT := /opt/intel/oneapi/mkl/latest

INCLUDES := -I$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))") $(shell $(PYTHON) -m pybind11 --includes 2>/dev/null || echo "-I/usr/include/pybind11") -I$(MKLROOT)/include
LDFLAGS := -shared -L$(MKLROOT)/lib -lmkl_rt -Wl,-rpath,$(MKLROOT)/lib
TARGET := _matrix$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

all: $(TARGET)

$(TARGET): matrix.cpp matrix.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) matrix.cpp -o $(TARGET) $(LDFLAGS)

test: $(TARGET)
	LD_LIBRARY_PATH=$(IOMPROOT)/lib:$(MKLROOT)/lib:$$LD_LIBRARY_PATH $(PYTHON) -m pytest test_matrix.py -v

benchmark: $(TARGET)
	LD_LIBRARY_PATH=$(IOMPROOT)/lib:$(MKLROOT)/lib:$$LD_LIBRARY_PATH $(PYTHON) benchmark.py > performance.txt

clean:
	rm -f $(TARGET) performance.txt

.PHONY: all test benchmark clean
