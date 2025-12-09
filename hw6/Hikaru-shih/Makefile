CXX = g++
CXXFLAGS = -O3 -std=c++17 -fPIC -Wall -Wextra

PY_INC = -I/usr/include/python3.10
PYBIND11_INC = -I/usr/include/pybind11

SRC = matrix.cpp
HDR = matrix.hpp

TARGET = _matrix.cpython-310-x86_64-linux-gnu.so

all: $(TARGET)

$(TARGET): $(SRC) $(HDR)
	$(CXX) $(CXXFLAGS) $(PY_INC) $(PYBIND11_INC) -shared $(SRC) -o $@

clean:
	rm -f _matrix*.so
