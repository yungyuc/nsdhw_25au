PYBIND11_INCLUDES := $(shell python3 -m pybind11 --includes)
PYTHON_INCLUDES := $(shell python3-config --includes)
PYTHON_LIB := $(shell python3-config --ldflags --embed)

CXX := g++
CXXFLAGS := -O3 -Wall -shared -std=c++17 -fPIC

TARGET := _vector$(shell python3-config --extension-suffix)
SRC := vector_angle.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) $(PYTHON_INCLUDES) $(SRC) -o $(TARGET) $(PYTHON_LIB)

test: all
	env PYTHONPATH=. python3 -m pytest -q test_vector.py

clean:
	rm -f $(TARGET)
