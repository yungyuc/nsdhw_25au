CXX := g++
CXXFLAGS := -O2 -std=c++17 -Wall -Wextra

TARGET := line_demo
SOURCES := main.cpp line.cpp
OBJECTS := $(SOURCES:.cpp=.o)

.PHONY: all run check clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJECTS)

main.o: main.cpp line.hpp
	$(CXX) $(CXXFLAGS) -c main.cpp -o $@

line.o: line.cpp line.hpp
	$(CXX) $(CXXFLAGS) -c line.cpp -o $@

run: $(TARGET)
	./$(TARGET)

check: $(TARGET)
	./$(TARGET) > result.txt

clean:
	rm -f $(TARGET) $(OBJECTS) result.txt


