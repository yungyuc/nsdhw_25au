CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -shared -fPIC
PYBIND11_FLAGS = $(shell python3 -m pybind11 --includes)
PYTHON_CFLAGS = $(shell python3-config --includes)

SRC = angle2d.cpp
TARGET = _vector.so

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(PYBIND11_FLAGS) $(PYTHON_CFLAGS) $^ -o $@

test: $(TARGET)
	python3 -m pytest -v test_angle.py

clean:
	rm -f $(TARGET) *.o

.PHONY: test clean