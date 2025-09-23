CXX = g++
CXXFLAGS = -std=c++14 -Wall -Wextra -O2 -shared -fPIC
PYBIND11_FLAGS = $(shell python3 -m pybind11 --includes)

SRC = angle.cpp bind.cpp
TARGET = _vector.so

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(PYBIND11_FLAGS) $^ -o $@

test: $(TARGET)
	python3 -m pytest -v test_angle.py

clean:
	rm -f $(TARGET) *.o

.PHONY: test clean
