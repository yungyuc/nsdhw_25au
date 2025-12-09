CXX = g++
CXXFLAGS = -O3 -std=c++17 -fPIC -DUSE_MKL

PY_INCLUDES = $(shell python3 -m pybind11 --includes)
PY_LDFLAGS = $(shell python3-config --ldflags)

TARGET = _matrix$(shell python3-config --extension-suffix)

SRC = mod.cpp

LDFLAGS = $(PY_LDFLAGS) \
          -Wl,--start-group \
          -lmkl_intel_lp64 -lmkl_core -lmkl_sequential \
          -Wl,--end-group \
          -lpthread -lm -ldl


all: $(TARGET)

$(TARGET): $(SRC) matrix.hpp
	$(CXX) $(CXXFLAGS) $(PY_INCLUDES) -shared $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)
