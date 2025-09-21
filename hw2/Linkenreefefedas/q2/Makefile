PYTHON := python3
CXX := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -fPIC

# 取得 Python/pybind11 相關 flags
PY_INC := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_paths()['include'])")
PY_LIBDIR := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or '')")
PY_EXT := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
PY_CFLAGS := -I$(PY_INC)
# pybind11 includes
PYBIND11_INC := $(shell $(PYTHON) -m pybind11 --includes)

MODULE := vecgeom$(PY_EXT)
OBJS := pybind_module.o

.PHONY: all test clean

all: $(MODULE)

pybind_module.o: pybind_module.cpp angle.hpp
	$(CXX) $(CXXFLAGS) $(PY_CFLAGS) $(PYBIND11_INC) -c $< -o $@

$(MODULE): $(OBJS)
	$(CXX) -shared -o $@ $^ -L$(PY_LIBDIR)

test: all
	$(PYTHON) -m pytest -q

clean:
	rm -f *.o $(MODULE)
