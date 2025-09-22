TARGET := main
SRC := main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	g++ $(SRC) -o $(TARGET)

run: $(TARGET)
	./main

check: $(TARGET)
	./main >> result.txt

clean:
	rm ./main ./result.txt