PYTHON      := python3
PY_INCLUDES := $(shell $(PYTHON)-config --includes)
EXT_SUFFIX  := $(shell $(PYTHON)-config --extension-suffix)

TARGET      := _matrix$(EXT_SUFFIX)
SRC         := matrix.cpp

CXX         := g++
CXXFLAGS    := -O3 -Wall -shared -std=c++17 -fPIC

PYBIND11_INC := $(shell $(PYTHON) -m pybind11 --includes)
MKL_INC  := /usr/include/mkl
MKL_LIB  := /usr/lib/x86_64-linux-gnu/mkl
MKL_LINK := -L$(MKL_LIB) -lmkl_rt -lblas


$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(PY_INCLUDES) $(PYBIND11_INC) -I$(MKL_INC) \
	$(SRC) -o $(TARGET) $(MKL_LINK)


rebuild: clean $(TARGET)


test: $(TARGET)
	$(PYTHON) -m pytest -v


clean:
	rm -rf __pycache__ *.so *.pyd *.egg-info *.o build
