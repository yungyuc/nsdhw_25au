CXX = g++
CXXFLAGS = -O3 -Wall -shared -std=c++17 -fPIC $(shell python3 -m pybind11 --includes)
LDFLAGS = $(shell python3-config --ldflags)
TARGET = vector_angle$(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
SRCS = angle.cpp bind.cpp

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

test: $(TARGET)
	pytest -q

clean:
	rm -f $(TARGET) *.o
