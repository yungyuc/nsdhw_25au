# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -fPIC -march=native

# Detect OS
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    PYTHON = python
else
    DETECTED_OS := $(shell uname -s)
    PYTHON = python3
endif

# Paths
ifeq ($(DETECTED_OS),Windows)
    MKLROOT ?= C:/Users/user/anaconda3/Library
else
    # For Linux, use pip-installed MKL
    USER_BASE = $(shell $(PYTHON) -c "import site; print(site.USER_BASE)")
    MKLROOT = $(USER_BASE)
endif

PYTHON_INCLUDE = $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND11_INCLUDE = $(shell $(PYTHON) -m pybind11 --includes)
EXTENSION_SUFFIX = $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# Include directories
INCLUDES = -I$(PYTHON_INCLUDE) $(PYBIND11_INCLUDE)

# MKL linking flags
ifeq ($(DETECTED_OS),Windows)
    # Windows MKL linking
    MKL_LIBS = -L$(MKLROOT)/lib -lmkl_rt
    MKL_INCLUDES = -I$(MKLROOT)/include
else
    # Linux MKL linking (pip-installed)
    MKL_LIBS = -L$(MKLROOT)/lib -Wl,--no-as-needed -Wl,-rpath,$(MKLROOT)/lib $(MKLROOT)/lib/libmkl_rt.so -lpthread -lm -ldl
    MKL_INCLUDES = -I$(MKLROOT)/include
endif

# Target
TARGET = _matrix$(EXTENSION_SUFFIX)

# Source files
SOURCES = matrix_multiplication.cpp
HEADERS = Matrix.h

# Default target
all: $(TARGET)

# Build the Python module
$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(MKL_INCLUDES) -shared $(SOURCES) -o $(TARGET) $(MKL_LIBS)

# Run tests
test: $(TARGET)
	$(PYTHON) -m pytest test_matrix.py -v -s

# Generate performance report
performance: $(TARGET)
	$(PYTHON) benchmark.py

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o *.so *.pyd performance.txt
	rm -rf __pycache__ .pytest_cache

.PHONY: all test performance clean
