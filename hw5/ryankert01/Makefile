CXX := g++
CXXFLAGS := -std=c++17 -O3 -fPIC -Wall -Wextra -march=native

# Python + pybind11 includes and libs
PYINC := $(shell python3 -m pybind11 --includes)
PYTHON_INCLUDES := $(shell python3-config --includes)
PYTHON_LIBS := $(shell python3-config --ldflags)
EXT := $(shell python3-config --extension-suffix)

# BLAS (OpenBLAS or fallback)
BLAS_LIBS := -lopenblas

# Module name and C++ sources
MOD := _matrix
SOURCES := matrix.cpp ../mod.cpp 

all: $(MOD)$(EXT)

# Build rule: depend on matrix.hpp, but DO NOT compile it
$(MOD)$(EXT): $(SOURCES) matrix.hpp
	$(CXX) $(CXXFLAGS) \
	    -I. $(PYTHON_INCLUDES) $(PYINC) \
	    -shared $(SOURCES) -o $@ \
	    $(PYTHON_LIBS) $(BLAS_LIBS)

clean:
	rm -f *.o *.so *.pyd
