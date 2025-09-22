TARGET := _vector
SRC := angle.cpp

# Get the necessary Python include flags (e.g., -I/usr/include/python3.10)
PYTHON_CFLAGS = $(shell python3-config --cflags)

# Manually specify the pybind11 include path based on the install script
# This tells the compiler to look for headers in /usr/include
PYBIND11_CFLAGS = -I/usr/include

# Combine all compiler flags for clarity
CXXFLAGS = -O3 -Wall -shared -std=c++11 -fPIC $(PYTHON_CFLAGS) $(PYBIND11_CFLAGS)

all:
	c++ $(CXXFLAGS) $(SRC) -o $(TARGET).so

test:
	python3 -m pytest -v

clean:
	rm -f $(TARGET).so
	rm -rf __pycache__ .pytest_cache