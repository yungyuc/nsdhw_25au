PYTHON      := python3
PYCONFIG    := python3-config

CXX         := g++
CXXFLAGS    := -O3 -std=c++17 -fPIC $(shell $(PYTHON) -m pybind11 --includes)
LDFLAGS     := -shared $(shell $(PYCONFIG) --ldflags)

EXT_SUFFIX  := $(shell $(PYCONFIG) --extension-suffix)
TARGET      := _matrix$(EXT_SUFFIX)

SOURCES     := mod.cpp matrix.cpp
OBJECTS     := $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(LDFLAGS) -o $@ $^

%.o: %.cpp matrix.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) _matrix*.so