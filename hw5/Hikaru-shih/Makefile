CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -shared -fPIC

PYTHON = python3
PYTHON_INCLUDES = $(shell $(PYTHON)-config --includes 2>/dev/null || $(PYTHON) -c "import sysconfig; print('-I' + sysconfig.get_path('include'))")
PYBIND11_INCLUDES = $(shell $(PYTHON) -m pybind11 --includes 2>/dev/null || $(PYTHON) -c "import pybind11; print('-I' + pybind11.get_include() + ' -I' + pybind11.get_include(user=True))" 2>/dev/null || echo "-I/usr/include/pybind11")
NUMPY_INCLUDES = $(shell $(PYTHON) -c "import numpy; print('-I' + numpy.get_include())" 2>/dev/null || echo "")

MKL_INCLUDES := $(shell $(PYTHON) -c "import os; home = os.path.expanduser('~'); paths = [os.path.join(home, '.local/include'), '/usr/include/mkl', '/usr/include']; print(' '.join(['-I' + p for p in paths if os.path.exists(os.path.join(p, 'mkl.h'))]))" 2>/dev/null)
MKL_LIBS := $(shell $(PYTHON) -c "import os; home = os.path.expanduser('~'); lib_path = os.path.join(home, '.local/lib'); print(f'-L{lib_path} -Wl,-rpath,{lib_path} -lmkl_rt') if os.path.exists(lib_path) and any('libmkl_rt' in f for f in os.listdir(lib_path) if f.startswith('libmkl_rt')) else print('-lmkl_rt -lpthread -lm -ldl')" 2>/dev/null)

ifeq ($(MKL_INCLUDES),)
    MKL_INCLUDES = -I$(HOME)/.local/include
endif
ifeq ($(MKL_LIBS),)
    MKL_LIBS = -L$(HOME)/.local/lib -Wl,-rpath,$(HOME)/.local/lib -lmkl_rt
endif

PYTHON_EXT_SUFFIX = $(shell $(PYTHON)-config --extension-suffix 2>/dev/null || echo ".so")

TARGET = _matrix$(PYTHON_EXT_SUFFIX)

SOURCES = ../mod.cpp
HEADERS = matrix.hpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -I. $(PYTHON_INCLUDES) $(PYBIND11_INCLUDES) $(NUMPY_INCLUDES) $(MKL_INCLUDES) \
		-o $@ $(SOURCES) $(MKL_LIBS)

clean:
	rm -f $(TARGET) _matrix*.so *.o

