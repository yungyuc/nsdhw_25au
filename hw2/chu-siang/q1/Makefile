# Q1 Makefile (single-file build)
CXX := g++
CXXFLAGS := -std=gnu++17 -O2 -Wall -Wextra -Wpedantic
LDFLAGS :=

TARGET := app
SRC := main.cpp
OBJ := $(SRC:.cpp=.o)

.PHONY: all run check clean

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

check: $(TARGET)
	./$(TARGET) > result.txt

clean:
	$(RM) -f $(OBJ) $(TARGET) result.txt

