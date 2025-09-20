# Build pybind11 extension module named "_vector" and run pytest

PYTHON     ?= python3
CXX        := g++
CXXFLAGS   := -std=c++17 -O3 -Wall -Wextra -fPIC

EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
PY_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes) $(shell $(PYTHON)-config --includes 2>/dev/null || python3-config --includes)
PY_LDFLAGS  := $(shell $(PYTHON)-config --ldflags 2>/dev/null || python3-config --ldflags)

TARGET := _vector$(EXT_SUFFIX)
SRC    := pybind_module.cpp
OBJ    := $(SRC:.cpp=.o)

.PHONY: all test clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -shared $(OBJ) -o $@ $(PY_LDFLAGS)

%.o: %.cpp angle.hpp
	$(CXX) $(CXXFLAGS) $(PY_INCLUDES) -c $< -o $@

test: $(TARGET)
	PYTHONPATH=. pytest -q

clean:
	rm -f $(OBJ) $(TARGET)
	find . -name "__pycache__" -type d -exec rm -rf {} +
