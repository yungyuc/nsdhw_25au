PYTHON = python
CXX = g++
CXXFLAGS = -O3 -Wall -shared -std=c++17 -fPIC
PYBIND11_CFLAGS = $(shell $(PYTHON) -m pybind11 --includes)
PYTHON_EXT = $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

TARGET = _vector$(PYTHON_EXT)
SRC = vector_angle.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(PYBIND11_CFLAGS) $(SRC) -o $(TARGET)

run: all
	$(PYTHON) -c "import vector_angle; print('angle:', vector_angle.vector_angle(1,0,0,1))"

test: all
	pytest -q

clean:
	rm -f vector_angle*.pyd vector_angle*.so result.txt
	rm -rf __pycache__ .pytest_cache build