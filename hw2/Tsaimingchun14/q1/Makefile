line: line.cpp
	g++ -std=c++17 -Wall -Wextra -O2 $< -o $@

run: line
	./line

check: line
	./line > result.txt

clean:
	rm -f line result.txt

.PHONY: run check clean
