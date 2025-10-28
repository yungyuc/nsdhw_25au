# ============================================
#   Matrix Multiplication Build (System MKL)
#   Author: ren-yi-wang-david
#   Compatible with: WSL / Ubuntu + Python 3.x
# ============================================


CXX := g++
PYTHON := python3

# === 編譯選項 ===
CXXFLAGS := -O3 -Wall -std=c++11 -fPIC -march=native -DHASMKL

# === 自動偵測 Python include 路徑 ===
PYTHON_INCLUDE := $(shell $(PYTHON) -c "from sysconfig import get_path; print(get_path('include'))")

# === 固定的 pybind11 與 MKL 系統路徑 ===
PYBIND11_INCLUDE := /usr/include/pybind11
MKL_INC := /usr/include/mkl
MKL_LIB := -L/usr/lib/x86_64-linux-gnu -lmkl_rt -lpthread -lm -ldl

# === 合併 include / link flag ===
INCLUDES := -I$(PYTHON_INCLUDE) -I$(PYBIND11_INCLUDE) -I$(MKL_INC)
LDFLAGS := $(MKL_LIB)

# === Python 模組名稱 (_matrix.so) ===
SO_NAME := _matrix$(shell $(PYTHON)-config --extension-suffix)

# === Python 測試腳本 ===
TEST_SCRIPT := test_multiply.py  

# === 主要目標 ===
all: $(SO_NAME)

$(SO_NAME): matrix.cpp StopWatch.hpp
	@echo ">>> Building $@ using system MKL + pybind11..."
	$(CXX) $(CXXFLAGS) -shared -DBUILD_PYBIND -o $@ $< $(INCLUDES) $(LDFLAGS)
	@echo ">>> Build complete: $@"

# === 清理檔案（CI 會執行 make clean） ===
clean:
	rm -f $(SO_NAME) *.o *.so 
	@echo ">>> Clean complete"

# === 測試模組可否成功匯入 ===
test: $(SO_NAME)
	@echo ">>> Running Python test script: $(TEST_SCRIPT)"
	$(PYTHON) $(TEST_SCRIPT)

.PHONY: all clean test
