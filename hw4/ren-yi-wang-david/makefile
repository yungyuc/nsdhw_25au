# === Makefile (safe version, full fix, no main error) ===
.RECIPEPREFIX = >

CXX := g++
CXXFLAGS := -O3 -march=native -std=c++11 -fPIC
PYTHON := python3

# === Python Config ===
PYTHON_VERSION := $(shell $(PYTHON) -c "import sys; print('python{}.{}'.format(sys.version_info.major, sys.version_info.minor))")
PYTHON_INCLUDE := $(shell $(PYTHON) -c "from sysconfig import get_path; print(get_path('include'))")

# --- 加上 --embed，確保包含 -lpython3.x ---
PYTHON_LDFLAGS := $(shell $(PYTHON)-config --ldflags --embed 2>/dev/null || $(PYTHON)-config --ldflags)
PYTHON_LIB := $(shell $(PYTHON)-config --libs 2>/dev/null || echo "-lpython$(shell $(PYTHON) -c 'import sys; print(sys.version_info.major)')")

# === Pybind11 Include Path ===
PYBIND11_INCLUDE := $(shell $(PYTHON) -m pybind11 --includes 2>/dev/null | sed 's/-I//')
ifeq ($(PYBIND11_INCLUDE),)
	PYBIND11_INCLUDE := /usr/include/pybind11
endif

# === NumPy Include (安全判斷) ===
NUMPY_INCLUDE := $(shell $(PYTHON) -c "import importlib.util; spec=importlib.util.find_spec('numpy'); print(__import__('numpy').get_include() if spec else '')" 2>/dev/null)

# === MKL / OpenBLAS 設定 ===
MKL_INC := /usr/include/mkl
MKL_LIB := -lmkl_rt -lpthread -lm -ldl
OPENBLAS_LIB := -lopenblas -lpthread -lm -ldl

ifeq ($(wildcard $(MKL_INC)/mkl.h),)
	LIBS := $(OPENBLAS_LIB)
	CXXFLAGS += -DUSE_OPENBLAS
	$(info >>> Using OpenBLAS backend)
else
	LIBS := $(MKL_LIB)
	CXXFLAGS += -DUSE_MKL
	$(info >>> Using Intel MKL backend)
endif

# === Include & Link Flags ===
INCLUDES := -I$(PYBIND11_INCLUDE) -I$(PYTHON_INCLUDE)
ifneq ($(strip $(NUMPY_INCLUDE)),)
	INCLUDES += -I$(NUMPY_INCLUDE)
endif
ifneq ($(wildcard $(MKL_INC)/mkl.h),)
	INCLUDES += -I$(MKL_INC)
endif

# --- LDFLAGS: 加上 PYTHON_LIB 確保連到 libpython3.x.so ---
LDFLAGS := $(PYTHON_LDFLAGS) $(PYTHON_LIB) $(LIBS)

# === Output .so name for import _matrix ===
SO_NAME := _matrix$(shell $(PYTHON)-config --extension-suffix)

# === Build Targets ===
all: $(SO_NAME)

$(SO_NAME): matrix.cpp
> $(CXX) $(CXXFLAGS) -DBUILD_PYBIND -shared -o $@ $< $(INCLUDES) $(LDFLAGS)

# === pytest will automatically load _matrix*.so ===
test: $(SO_NAME)
> $(PYTHON) -m pytest --maxfail=1 --disable-warnings -q

clean:
> rm -f _matrix*.so
