# Makefile — pybind11 + Python dev; auto-detect MKL (or fallback)
PYTHON   := python3
CXX      := g++
CXXFLAGS := -O3 -march=native -fPIC -std=c++17 -Wall -Wextra -Wno-unused-parameter
LDFLAGS  := -shared

# Python & pybind11 includes
PYCFG_I  := $(shell $(PYTHON)-config --includes)
PYCFG_L  := $(shell $(PYTHON)-config --ldflags)
PYBIND_I := $(shell $(PYTHON) -m pybind11 --includes 2>/dev/null || echo -I/usr/include/pybind11)
EXT      := $(shell $(PYTHON)-config --extension-suffix)

SRC      := _matrix.cpp
SO       := _matrix$(EXT)

# ---------- MKL auto-detect ----------
HAVE_MKL := 0
MKL_INC  :=
MKL_LIBS :=

ifneq ($(strip $(MKLROOT)),)
  HAVE_MKL := 1
  MKL_INC  := -I$(MKLROOT)/include
  MKL_LIBS := -Wl,--no-as-needed -L$(MKLROOT)/lib/intel64 -lmkl_rt -lpthread -lm -ldl
else ifneq ($(wildcard /usr/include/mkl/mkl.h),)
  HAVE_MKL := 1
  MKL_INC  := -I/usr/include/mkl
  MKL_LIBS := -Wl,--no-as-needed -L/usr/lib/x86_64-linux-gnu \
              -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl
endif

# Expose to C++ so _matrix.cpp can choose DGEMM vs fallback
CXXFLAGS += -DHAVE_MKL=$(HAVE_MKL)

.PHONY: all clean test benchmark

all: $(SO)

$(SO): $(SRC)
	$(CXX) $(CXXFLAGS) $(PYCFG_I) $(PYBIND_I) $(MKL_INC) -o $@ $< $(LDFLAGS) $(PYCFG_L) $(MKL_LIBS)

clean:
	rm -f $(SO) performance.txt

test: all
	$(PYTHON) -m pytest -q

# Optional: large-size benchmark (>= 1000^2) writing performance.txt (naive/tile/mkl)
benchmark: all
	MATRIX_SIZE=1024 TILE_SIZE=64 python3 benchmark.py

