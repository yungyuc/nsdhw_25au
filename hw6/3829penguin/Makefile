CXX = g++
CXXFLAGS = -O3 -std=c++17 -fPIC -DUSE_MKL

PYTHON_INCLUDE = $(shell python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])")
PYTHON_LIBDIR  = $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
PYSUFFIX       = $(shell python3-config --extension-suffix)

PYBIND11 = -I/usr/include/pybind11

TARGET = _matrix$(PYSUFFIX)

SRC = mod.cpp

LDFLAGS = -shared \
          -L$(PYTHON_LIBDIR) \
          -lmkl_rt -lpthread -lm -ldl

all: $(TARGET)

$(TARGET): $(SRC) matrix.hpp
	$(CXX) $(CXXFLAGS) -I$(PYTHON_INCLUDE) $(PYBIND11) \
		-shared $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)
