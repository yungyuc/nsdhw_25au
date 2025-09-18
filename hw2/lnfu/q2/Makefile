.PHONY: clean test all

all: vector.cpp
	python3 -m pip install pybind11
	python3 setup.py build_ext --inplace

test:
	python3 -m pytest test_vector.py -v

clean:
	python3 setup.py clean --all || true
	rm -f _vector*.so
	rm -rf build __pycache__ *.egg-info
