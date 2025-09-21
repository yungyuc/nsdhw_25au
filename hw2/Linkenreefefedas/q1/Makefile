CXX := g++
CXXFLAGS := -std=c++17

TARGET := line_demo
SRC := main.cpp Line.cpp
OBJ := $(SRC:.cpp=.o)

.PHONY: all run check clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp Line.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

check: $(TARGET)
	./$(TARGET) > result.txt

clean:
	rm -f $(TARGET) $(OBJ) result.txt
