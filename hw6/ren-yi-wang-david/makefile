CXX := g++
CXXFLAGS := -std=c++17 -O3 -fPIC -Wall -Wextra

PYTHON := python3

# Python include path
PY_INCLUDE := -I$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")

# System pybind11
PYBIND11_INCLUDE := -I/usr/include/pybind11

# System BLAS
BLAS_LIB := -lblas -lpthread -lm -ldl

# Output extension (.so)
EXT := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
TARGET := _matrix$(EXT)

INCLUDES := -I. $(PY_INCLUDE) $(PYBIND11_INCLUDE)

all: $(TARGET)

$(TARGET): mod.cpp matrix.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) mod.cpp -shared -o $(TARGET) $(BLAS_LIB)

clean:
	rm -f _matrix*.so

.PHONY: all clean
