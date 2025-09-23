CXX=g++
CXXFLAGS = -std=c++11
#確保一定執行
.PHONY: all check clean run
all:main

#默認目標
default:main

main:main.o line.o
#$^ 依賴的檔案，$@目標檔案
	$(CXX) $^ -o $@
#僅生成.o檔
main.o: main.cpp line.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
line.o: line.cpp line.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
#清理	
clean:
	rm -rf result.txt main.o line.o main
#執行
run:
	./main
#檢查
check:
	./main>>result.txt
