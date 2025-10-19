import numpy as np
import timeit
import _matrix

def benchmark():
    """
    Benchmarks the performance of different matrix multiplication implementations.
    """
    matrix_size = 1024
    tile_size = 64
    number_of_runs = 3

    print(f"--- Matrix Multiplication Benchmark ---")
    print(f"Matrix Size: {matrix_size}x{matrix_size}")
    print(f"Number of runs for timing: {number_of_runs}\n")

    try:
        mat_a = _matrix.Matrix(matrix_size, matrix_size)
        mat_b = _matrix.Matrix(matrix_size, matrix_size)

        np_a = np.random.rand(matrix_size, matrix_size)
        np_b = np.random.rand(matrix_size, matrix_size)

        for i in range(matrix_size):
            for j in range(matrix_size):
                mat_a[i, j] = np_a[i, j]
                mat_b[i, j] = np_b[i, j]

    except AttributeError:
        print("Error: Could not find the 'Matrix' class in the '_matrix' module.")
        print("Please ensure your C++ module is compiled and in the same directory.")
        return
    except Exception as e:
        print(f"An error occurred during data preparation: {e}")
        return

    time_naive = timeit.timeit(
        lambda: _matrix.multiply_naive(mat_a, mat_b),
        number=number_of_runs,
        globals=globals()
    )

    time_tile = timeit.timeit(
        lambda: _matrix.multiply_tile(mat_a, mat_b),
        number=number_of_runs,
        globals=globals()
    )

    time_mkl = timeit.timeit(
        lambda: _matrix.multiply_mkl(mat_a, mat_b),
        number=number_of_runs,
        globals=globals()
    )

    avg_naive = time_naive / number_of_runs
    avg_tile = time_tile / number_of_runs
    avg_mkl = time_mkl / number_of_runs

    print("--- Average Runtimes ---")
    print(f"Naive implementation: {avg_naive:.6f} s")
    print(f"Tiled implementation: {avg_tile:.6f} s")
    print(f"MKL (DGEMM) implementation: {avg_mkl:.6f} s")
    print("------------------------\n")

    if avg_tile > 0:
        speedup_tile = avg_naive / avg_tile
        print(f"Tiling speedup over naive: {speedup_tile:.2f}x faster")
    
    if avg_mkl > 0:
        speedup_mkl = avg_naive / avg_mkl
        print(f"MKL speedup over naive:    {speedup_mkl:.2f}x faster")
        
    if avg_mkl > 0 and avg_tile > 0:
        speedup_mkl_tile = avg_tile / avg_mkl
        print(f"MKL speedup over tiling:   {speedup_mkl_tile:.2f}x faster")


if __name__ == "__main__":
    benchmark()