CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -pedantic

SRC := main.cpp
OBJ := $(SRC:.cpp=.o)
BIN := line_app

.PHONY: all run check clean

all: $(BIN)

$(BIN): $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) -o $@

%.o: %.cpp line.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(BIN)
	./$(BIN)

check: $(BIN)
	./$(BIN) > result.txt

clean:
	rm -f $(OBJ) $(BIN) result.txt