CXX = g++
CXXFLAGS = -O2 -std=c++17 -fPIC -Wall

PYTHON = python3
INC = $(shell $(PYTHON) -m pybind11 --includes)
EXT = $(shell $(PYTHON)-config --extension-suffix)

SRC = matrix.cpp
HDR = matrix.hpp

TARGET = _matrix$(EXT)

all: $(TARGET)

$(TARGET): $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) $(INC) -I. $(SRC) -shared -o $@

clean:
	rm -f _matrix*.so
