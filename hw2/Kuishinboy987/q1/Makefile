CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra

TARGET = main

all: $(TARGET)

$(TARGET): Line.o main.o
	$(CXX) $(CXXFLAGS) -o $(TARGET) Line.o main.o

Line.o: Line.cpp Line.hpp
	$(CXX) $(CXXFLAGS) -I. -c Line.cpp

main.o: main.cpp Line.hpp
	$(CXX) $(CXXFLAGS) -I. -c main.cpp

run: $(TARGET)
	./$(TARGET)

check: $(TARGET)
	./$(TARGET) > result.txt

clean:
	rm -f *.o $(TARGET) result.txt