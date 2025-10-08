CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

TARGET = line
OBJS = main.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

main.o: main.cpp Line.h

run: $(TARGET)
	./$(TARGET)

check: $(TARGET)
	./$(TARGET) > result.txt

clean:
	rm -f $(OBJS) $(TARGET) result.txt
