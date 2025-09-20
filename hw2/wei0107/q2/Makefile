# Robust Makefile for Q2 (_vector)

PYTHON   ?= python3
CXX      := g++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -fPIC

EXT_SUFFIX = $(shell $(PYTHON) -c "import sysconfig,sys;print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')" 2>/dev/null || echo .so)

TARGET := _vector$(EXT_SUFFIX)
SRC    := pybind_module.cpp
OBJ    := $(SRC:.cpp=.o)

.PHONY: all test clean

all: $(TARGET)

# 只在真正編譯時決定 include 旗標，避免 make test 時沒裝 pybind11 也爆
define GET_PY_INCLUDES
`{ \
  $(PYTHON) -m pybind11 --includes 2>/dev/null \
  || { \
       INC_PY=$$($(PYTHON)-config --includes 2>/dev/null || python3-config --includes 2>/dev/null); \
       if [ -d /usr/include/pybind11 ]; then \
         echo "-I/usr/include/pybind11 $$INC_PY"; \
       else \
         echo "$$INC_PY"; \
       fi; \
     }; \
}`
endef

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -shared $(OBJ) -o $@ $$($(PYTHON)-config --ldflags 2>/dev/null || python3-config --ldflags 2>/dev/null)

%.o: %.cpp angle.hpp
	@echo "Compiling $<"
	$(CXX) $(CXXFLAGS) $(GET_PY_INCLUDES) -c $< -o $@

test: $(TARGET)
	PYTHONPATH=. $(PYTHON) -m pytest -q

clean:
	rm -f $(OBJ) $(TARGET)
	find . -name "__pycache__" -type d -exec rm -rf {} +
