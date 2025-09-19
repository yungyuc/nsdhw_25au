CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -shared -fPIC 
PYBIND11_FLAGS = $(shell python3 -m pybind11 --includes)
PYTHON_INCLUDES = $(shell python3-config --includes)
PYTHON_LIB = $(shell python3-config --ldflags --embed)

TARGET = _vector$(shell python3-config --extension-suffix)
SRC = angle.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(PYBIND11_FLAGS) $(PYTHON_INCLUDES) $(SRC) -o $@ $(PYTHON_LIB)

test: $(TARGET)
	env PYTHONPATH=. python3 -m pytest -v

clean:
	rm -f $(TARGET) *.o