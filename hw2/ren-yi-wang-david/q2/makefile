# 系統 Python
PYTHON := python3

# 模組名稱 
MODULE_NAME := vector

# 系統類型
UNAME_S := $(shell uname -s)

# 編譯器
CXX := g++
CXXFLAGS := -O3 -Wall -std=c++17 -fPIC

# pybind11 header
PYBIND_INCLUDES := -I/usr/include/pybind11
PYTHON_INCLUDES := $(shell $(PYTHON) -c "import sysconfig; print('-I' + sysconfig.get_config_var('INCLUDEPY'))")

# linker flag
ifeq ($(UNAME_S),Darwin)
    LDFLAGS := -shared -undefined dynamic_lookup
else
    LDFLAGS := -shared
endif

# Python 擴展名
EXT_SUFFIX := $(shell $(PYTHON) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

TARGET := _$(MODULE_NAME)$(EXT_SUFFIX)
SOURCES := vector.cpp binding.cpp

.PHONY: all test clean

all: $(TARGET)

$(TARGET): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(PYTHON_INCLUDES) $(PYBIND_INCLUDES) $(SOURCES) -o $@ $(LDFLAGS)

test: $(TARGET)
	PYTHONPATH=".:$(PYTHONPATH)" $(PYTHON) -m pytest -v test_rad.py

clean:
 rm -rf $(TARGET) __pycache__ .pytest_cache *.o *.so *.pyd

