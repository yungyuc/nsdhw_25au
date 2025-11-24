CXX := g++
CXXFLAGS := -O3 -march=native -fPIC -std=c++17 -Wall -Wextra

PYINC := $(shell python3 -m pybind11 --includes)
PYTHON_INCLUDES = $(shell python3-config --includes)
PYTHON_LIB = $(shell python3-config --ldflags)
EXT   := $(shell python3-config --extension-suffix)
# NUMPY_INCLUDES := $(shell python3 -c "import numpy; print('-I' + numpy.get_include())")

BLAS_LIBS := $(shell pkg-config --libs openblas 2>/dev/null)
ifeq ($(strip $(BLAS_LIBS)),)
BLAS_LIBS := -lblas
endif

MOD := _matrix
SRC := ../mod.cpp matrix.hpp

all: $(MOD)$(EXT)

$(MOD)$(EXT): $(SRC)
	$(CXX) $(CXXFLAGS) -I. $(PYTHON_INCLUDES) $(PYINC) \
	$(NUMPY_INCLUDES) \
	-shared $^ -o $@  $(PYTHON_LIB) $(BLAS_LIBS)

clean:
	rm -f *.o *.so *.pyd
