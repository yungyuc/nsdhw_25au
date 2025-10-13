CXX = g++
CXXFLAGS = -O3 -Wall -shared -std=c++17 -fPIC
PYBIND11_FLAGS = $(shell python3 -m pybind11 --includes)
PYTHON_INCLUDES = $(shell python3-config --includes)
PYTHON_LIB = $(shell python3-config --ldflags)
PYTHON_EXT = $(shell python3-config --extension-suffix)

TARGET = _vector$(PYTHON_EXT)
SRC = vector_angle.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(PYBIND11_FLAGS) $(PYTHON_INCLUDES) $(SRC) -o $@ $(PYTHON_LIB)

test: $(TARGET) 
	python3 -m pytest -q

clean:
	rm -f $(TARGET) *.o