# Makefile for Question 2 - pybind11 module

# Python and compiler settings
PYTHON := python3
CXX := g++
CXXFLAGS := -std=c++17 -O3 -Wall -shared -fPIC

# Source files
SOURCES := matrix,cpp
MODULE_NAME := _matrix

# Default target
all: build

# Build the pybind11 module using setup.py
build:
	$(PYTHON) -m pip install pybind11 pytest
	$(PYTHON) setup.py build_ext --inplace

# Run tests using pytest
test: build
	$(PYTHON) -m pytest test_matrix.py

# Clean all generated files
clean:
	rm -rf build/
	rm -f *.so
	rm -f *.o
	rm -rf __pycache__/
	rm -rf *.egg-info/


.PHONY: all build test clean help