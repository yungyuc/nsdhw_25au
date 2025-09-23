CXX := g++
CXXFLAGS := -Wall -O2
TARGET := main
SRC := main.cpp


all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)


run: $(TARGET)
	./$(TARGET)


check: $(TARGET)
	./$(TARGET) > result.txt

clean:
	rm -f $(TARGET) result.txt
