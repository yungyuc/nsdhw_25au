CXX      := g++
CXXFLAGS := -O3 -march=native -fPIC -std=c++17 -Wall -Wextra

# pybind11 / python
PYINC := $(shell python3 -m pybind11 --includes)
PYTHON_INCLUDES = $(shell python3-config --includes)
PYTHON_LIB = $(shell python3-config --ldflags)
EXT   := $(shell python3-config --extension-suffix)

BLAS_LIBS := $(shell pkg-config --libs openblas 2>/dev/null)
ifeq ($(strip $(BLAS_LIBS)),)
BLAS_LIBS := -lblas
endif
  
MOD := _matrix

# ====== targets ======
.PHONY: all clean test run perf

all: $(MOD)$(EXT) perf

$(MOD)$(EXT): pybind.o
	$(CXX) -shared -fPIC $^ -o $@ $(BLAS_LIBS) $(PYTHON_LIB)

pybind.o: pybind.cpp mul_set.hpp
	$(CXX) $(CXXFLAGS) $(PYTHON_INCLUDES) $(PYINC) -c $< -o $@

perf: perf.o
	$(CXX) $^ -o $@ $(BLAS_LIBS)

perf.o: perf.cpp mul_set.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

test: $(MOD)$(EXT)
	python3 -m pytest -q

run: all
	./perf

clean:
	rm -f *.o *.so *.pyd $(MOD)$(EXT) perf performance.txt
	@find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache .mypy_cache .cache 2>/dev/null || true