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
    # For Linux, try to detect MKL location
    # First try pip-installed MKL
    USER_BASE = $(shell $(PYTHON) -c "import site; print(site.USER_BASE)" 2>/dev/null || echo "")
    ifneq ($(wildcard $(USER_BASE)/lib/libmkl_rt.so*),)
        MKLROOT = $(USER_BASE)
        MKL_FROM_PIP = 1
    else
        # Try system-installed MKL (intel-mkl-full on Ubuntu)
        ifneq ($(wildcard /usr/lib/x86_64-linux-gnu/libmkl_rt.so*),)
            MKLROOT = /usr
            MKL_FROM_SYSTEM = 1
        else
            # Fallback to environment variable
            MKLROOT = $(shell echo $$MKLROOT)
        endif
    endif
endif

PYTHON_INCLUDE = $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND11_INCLUDE = $(shell $(PYTHON) -m pybind11 --includes 2>/dev/null || echo "-I/usr/include/pybind11")
EXTENSION_SUFFIX = $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# Include directories
INCLUDES = -I$(PYTHON_INCLUDE) $(PYBIND11_INCLUDE)

# MKL linking flags
ifeq ($(DETECTED_OS),Windows)
    # Windows MKL linking
    MKL_LIBS = -L$(MKLROOT)/lib -lmkl_rt
    MKL_INCLUDES = -I$(MKLROOT)/include
else ifdef MKL_FROM_PIP
    # Linux MKL linking (pip-installed)
    MKL_LIBS = -L$(MKLROOT)/lib -Wl,--no-as-needed -Wl,-rpath,$(MKLROOT)/lib $(MKLROOT)/lib/libmkl_rt.so -lpthread -lm -ldl
    MKL_INCLUDES = -I$(MKLROOT)/include
else ifdef MKL_FROM_SYSTEM
    # Linux MKL linking (system-installed, e.g., intel-mkl-full)
    MKL_LIBS = -L/usr/lib/x86_64-linux-gnu -Wl,--no-as-needed -lmkl_rt -lpthread -lm -ldl
    MKL_INCLUDES = -I/usr/include/mkl
else
    # Fallback for custom MKLROOT
    MKL_LIBS = -L$(MKLROOT)/lib -Wl,--no-as-needed -lmkl_rt -lpthread -lm -ldl
    MKL_INCLUDES = -I$(MKLROOT)/include
endif

# Target
TARGET = _matrix$(EXTENSION_SUFFIX)

# Source files
SOURCES = matrix_multiplication.cpp
HEADERS = Matrix.h CustomAllocator.h

# Default target
all: $(TARGET)

# Build the Python module
$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(MKL_INCLUDES) -shared $(SOURCES) -o $(TARGET) $(MKL_LIBS)

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o *.so *.pyd
	rm -rf __pycache__ .pytest_cache

.PHONY: all clean
