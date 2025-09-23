PYBIND11_INCLUDES := $(shell python3 -m pybind11 --includes)
CXX := g++
CXXFLAGS := -O3 -Wall -shared -std=c++17 -fPIC

TARGET := _vector$(shell python3-config --extension-suffix)
SRC := vector_angle.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(PYBIND11_INCLUDES) $(SRC) -o $(TARGET)

test: all
	python3 -m pytest -q test_vector.py

clean:
	rm -f $(TARGET)
