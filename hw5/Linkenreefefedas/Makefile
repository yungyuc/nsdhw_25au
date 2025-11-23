# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -shared -fPIC

# Python and pybind11
PYTHON = python3
PYBIND11_INCLUDES = $(shell $(PYTHON) -m pybind11 --includes 2>/dev/null || $(PYTHON) -c "import pybind11; print('-I' + pybind11.get_include() + ' -I' + pybind11.get_include(user=True))")
NUMPY_INCLUDES = $(shell $(PYTHON) -c "import numpy; print('-I' + numpy.get_include())")

# MKL configuration - detect from Python environment
MKL_INCLUDES := $(shell $(PYTHON) -c "import os; home = os.path.expanduser('~'); paths = [os.path.join(home, '.local/include'), '/usr/include/mkl', '/usr/include']; print(' '.join(['-I' + p for p in paths if os.path.exists(os.path.join(p, 'mkl.h'))]))" 2>/dev/null)
MKL_LIBS := $(shell $(PYTHON) -c "import os; home = os.path.expanduser('~'); lib_path = os.path.join(home, '.local/lib'); print(f'-L{lib_path} -Wl,-rpath,{lib_path} -lmkl_rt') if os.path.exists(lib_path) and any('libmkl_rt' in f for f in os.listdir(lib_path) if f.startswith('libmkl_rt')) else print('-lmkl_rt -lpthread -lm -ldl')" 2>/dev/null)

# Fallback if Python detection fails
ifeq ($(MKL_INCLUDES),)
    MKL_INCLUDES = -I$(HOME)/.local/include
endif
ifeq ($(MKL_LIBS),)
    MKL_LIBS = -L$(HOME)/.local/lib -Wl,-rpath,$(HOME)/.local/lib -lmkl_rt
endif

# Python extension suffix
PYTHON_EXT_SUFFIX = $(shell $(PYTHON)-config --extension-suffix 2>/dev/null || echo ".so")

# Target
TARGET = _matrix$(PYTHON_EXT_SUFFIX)

# Source files
SOURCES = ../mod.cpp
HEADERS = matrix.hpp

# Build rules
.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -I. $(PYBIND11_INCLUDES) $(NUMPY_INCLUDES) $(MKL_INCLUDES) \
		-o $@ $(SOURCES) $(MKL_LIBS)

clean:
	rm -f $(TARGET) _matrix*.so *.o

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
