CXX ?= c++
CXXFLAGS ?= -std=c++17 -Wall -Wextra -O2 -fPIC
CXXFLAGS += -I .
PYTHON ?= python3
PYBIND11_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes 2>/dev/null)
NUMPY_INCLUDES := $(shell $(PYTHON) -c "import numpy, sys; print('-I' + numpy.get_include())")
PYTHON_INCLUDES := $(shell python3-config --includes)
PYTHON_LIBS := $(shell python3-config --ldflags)
MKL_INCLUDES := -I$(MKLROOT)/include
MKL_LIBS := -L$(MKLROOT)/lib/intel64 -lmkl_rt

TARGET := _matrix$(shell python3-config --extension-suffix)
SRCS := mod.cpp matrix.cpp

UNAME_S := $(shell uname -s)
LDFLAGS := 
LDLIBS :=

ifeq ($(UNAME_S), Darwin)
	CXXFLAGS += -DACCELERATE_NEW_LAPACK
	LDFLAGS += -undefined dynamic_lookup
	LDLIBS += -framework Accelerate
	MKL_INCLUDES :=
	MKL_LIBS :=
endif

.PHONY: all build test clean

all: build

build: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) $(PYTHON_INCLUDES) $(NUMPY_INCLUDES) \
	$(MKL_INCLUDES) -shared -o $@ $(SRCS) $(PYTHON_LIBS) \
	$(LDFLAGS) $(LDLIBS) $(MKL_LIBS) 

test: build
	$(PYTHON) -m pytest -q -vv

clean:
	@rm -f $(TARGET) _matrix*.so
	@rm -rf __pycache__ .pytest_cache