# ===== Python / target naming =====
PYTHON        ?= python3
EXT_SUFFIX    := $(shell $(PYTHON) -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")
TARGET        := _matrix$(EXT_SUFFIX)

# ===== Python includes & libdir =====
PY_INC        := $(shell $(PYTHON) -c "import sysconfig;print(sysconfig.get_paths().get('include',''))")
PY_PLATINC    := $(shell $(PYTHON) -c "import sysconfig;print(sysconfig.get_paths().get('platinclude',''))")
PY_LIBDIR     := $(shell $(PYTHON) -c "import sysconfig;print(sysconfig.get_config_var('LIBDIR') or '')")

# ===== Compiler / Flags =====
CXX           := g++
CXXFLAGS      := -O3 -march=native -fPIC -std=c++17 -Wall -Wextra -Wno-unused-parameter
LDFLAGS       :=
LDLIBS        := -ldl -lm

HAVE_MKL     ?= 0
MKL_INC      ?= /usr/include/mkl

# ===== Includes =====
INCLUDES      := \
  $(if $(PY_INC),-I$(PY_INC),) \
  $(if $(PY_PLATINC),-I$(PY_PLATINC),) \
  -I/usr/include/pybind11 \
  -I/usr/lib/python3/dist-packages/pybind11/include \
  -I.  

ifeq ($(HAVE_MKL),1)
  CXXFLAGS  += -DHAVE_MKL=1
  INCLUDES  += -I$(MKL_INC)
  LDLIBS    += -Wl,--no-as-needed -L/usr/lib/x86_64-linux-gnu \
               -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lpthread -lm -ldl
endif

SRC          := matrix.cpp ../mod.cpp
OBJ          := matrix.o mod.o

# ===== Rules =====
all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) -shared -o $@ $(OBJ) -L$(PY_LIBDIR) $(LDFLAGS) $(LDLIBS)

matrix.o: matrix.cpp matrix.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c matrix.cpp -o $@

mod.o: ../mod.cpp matrix.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c ../mod.cpp -o $@

clean:
	rm -f _matrix*.so *.o performance.txt

test: clean $(TARGET)
	$(PYTHON) -m pytest -q
