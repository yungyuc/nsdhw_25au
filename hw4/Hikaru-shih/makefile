PYTHON3 ?= python3

EXT_SUFFIX := $(shell $(PYTHON3) -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')")

PY_INC_DIR := $(shell $(PYTHON3) -c "import sysconfig; print(sysconfig.get_paths().get('include', ''))")

PY_CFG_INC := $(shell sh -c 'command -v $(PYTHON3)-config >/dev/null 2>&1 && $(PYTHON3)-config --includes || (command -v python3-config >/dev/null 2>&1 && python3-config --includes)')

PYB11_INC := $(shell $(PYTHON3) -m pybind11 --includes 2>/dev/null)
ifeq ($(strip $(PYB11_INC)),)
ifneq ("$(wildcard /usr/include/pybind11)","")
	PYB11_INC := -I/usr/include/pybind11
endif
endif

INCLUDES := $(PYB11_INC) $(if $(PY_INC_DIR),-I$(PY_INC_DIR),) $(PY_CFG_INC)

CXX      ?= g++
CXXFLAGS ?= -O3 -fPIC -Wall -Wextra -std=c++17 -march=native
LDFLAGS  ?= -shared
LDLIBS   ?= -ldl

TARGET   := _matrix$(EXT_SUFFIX)
SOURCES  := matrix.cpp

all: $(TARGET)

$(TARGET): $(SOURCES)
	@echo "==> Building: $(TARGET)"
	@echo "==> Using includes: $(INCLUDES)"
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SOURCES) -o $(TARGET) $(LDFLAGS) $(LDLIBS)

clean:
	rm -f $(TARGET) *.o
	rm -rf __pycache__