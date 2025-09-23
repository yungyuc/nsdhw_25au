CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -shared -fPIC
PYINCLUDES := $(shell python3 -m pybind11 --includes 2>/dev/null || python3-config --includes)
PYLDFLAGS := $(shell python3-config --ldflags)
TARGET := _vector$(shell python3-config --extension-suffix)
SRC := angle.cpp bind.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(PYINCLUDES) $(SRC) -o $@ $(PYLDFLAGS)

test: $(TARGET)
	python3 -m pytest -v test_angle.py

clean:
	rm -f $(TARGET) *.o

.PHONY: all test clean
