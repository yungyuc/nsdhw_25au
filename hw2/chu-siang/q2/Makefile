# Q2 Makefile: build a pybind11 extension named _vector
PYTHON   ?= python3
CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2 -Wall -Wextra -fPIC
INCLUDES := $(shell $(PYTHON) -m pybind11 --includes)
EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX') or '')")

TARGET   := _vector$(EXT_SUFFIX)
SRC      := vector.cpp
OBJ      := $(SRC:.cpp=.o)

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

test: all
	env PYTHONPATH=".:$$PYTHONPATH" $(PYTHON) -m pytest -v

clean:
	$(RM) -f $(OBJ) $(TARGET)

