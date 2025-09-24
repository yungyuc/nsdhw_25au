# Makefile for C++ project

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O2
TARGET = main
SOURCES = main.cpp
OBJECTS = $(SOURCES:.cpp=.o)

# Default target
all: $(TARGET)

# Build the main executable
$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $(TARGET)

# Build object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Run the program
run: $(TARGET)
	./$(TARGET)

# Run and redirect output to result.txt
check: $(TARGET)
	./$(TARGET) > result.txt

# Clean build artifacts
clean:
	rm -f $(TARGET) $(OBJECTS) result.txt

# Debug build
debug: CXXFLAGS += -g -DDEBUG
debug: $(TARGET)

# Rebuild everything
rebuild: clean all

.PHONY: all clean run check debug rebuild