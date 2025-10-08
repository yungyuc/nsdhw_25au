PYTHON     := python3
CXX        := g++
CXXFLAGS   := -std=c++17 -O3 -Wall -Wextra -fPIC

EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

PY_INCLUDES := $(shell $(PYTHON) -m pybind11 --includes 2>/dev/null || \
                 $(PYTHON)-config --includes 2>/dev/null || \
                 python3-config --includes 2>/dev/null)

PY_LDFLAGS  := $(shell $(PYTHON)-config --ldflags 2>/dev/null || \
                 python3-config --ldflags 2>/dev/null)

TARGET := _vector$(EXT_SUFFIX)
SRC    := angle.cpp bind.cpp
OBJ    := $(SRC:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -shared $(OBJ) -o $@ $(PY_LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(PY_INCLUDES) -c $< -o $@

test: $(TARGET)
	PYTHONPATH=. $(PYTHON) -m pytest -q

clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all test clean
