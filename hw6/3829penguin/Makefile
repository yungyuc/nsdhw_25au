CXX := g++
CXXFLAGS = -std=c++17 -O3 -march=native -fPIC -fopenmp -Wall -Wextra

PYTHON = python3

# Python include path
PY_INCLUDE := -I$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")

# System pybind11 (CI has this)
PYBIND11_INCLUDE := -I/usr/include/pybind11

# Do not use numpy includes (not needed for HW6)
NUMPY_INCLUDES :=

# System BLAS (CI always has this)
BLAS_LIB := -lblas -lpthread -lm -ldl

# Output file name
TARGET := _matrix$(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# Collect includes
INCLUDES := -I. $(PY_INCLUDE) $(PYBIND11_INCLUDE)

all: $(TARGET)

$(TARGET): mod.cpp matrix.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) mod.cpp -o $(TARGET) -shared $(BLAS_LIB)

test: $(TARGET)
	$(PYTHON) -m pytest -q -vv

clean:
	rm -f *.o *.so *.pyd

.PHONY: all test clean
