.PHONY: clean run check

CXX = g++
CXXFLAGS = -std=c++11

constructor: constructor.o
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: constructor
	./constructor

check: constructor
	./constructor > result.txt

clean:
	rm -f *.o constructor result.txt
