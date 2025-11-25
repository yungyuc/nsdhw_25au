CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -fPIC

TARGET = matrix
OBJ = matrix.o
LIB = matrix.so

PYTHON_INCLUDE ?= $(shell python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])")
PYTHON_LIB ?= $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")

BLAS = -lblas -llapack

all: $(LIB)

$(OBJ): matrix.cpp matrix.hpp
	$(CXX) $(CXXFLAGS) -c matrix.cpp -I$(PYTHON_INCLUDE)

$(LIB): $(OBJ)
	$(CXX) $(CXXFLAGS) -shared -o $@ $^ -L$(PYTHON_LIB) $(BLAS)

clean:
	rm -f *.o *.so *.pyd