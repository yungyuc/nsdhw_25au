import numpy as np
import timeit
import _matrix

def benchmark():
    matrix_size = 500
    tile_size = 16
    number_of_runs = 1

    mat_a = _matrix.Matrix(matrix_size, matrix_size)
    mat_b = _matrix.Matrix(matrix_size, matrix_size)

    np_a = np.random.rand(matrix_size, matrix_size)
    np_b = np.random.rand(matrix_size, matrix_size)

    for i in range(matrix_size):
        for j in range(matrix_size):
            mat_a[i, j] = np_a[i, j]
            mat_b[i, j] = np_b[i, j]

    time_naive = timeit.timeit(
        lambda: _matrix.multiply_naive(mat_a, mat_b),
        number=number_of_runs,
        globals=globals()
    )

    time_tile = timeit.timeit(
        lambda: _matrix.multiply_tile(mat_a, mat_b, tile_size),
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

    with open("performance.txt", "w") as f:
        f.write(f"Average time for Naive method: {avg_naive:.6f} seconds\n")
        f.write(f"Average time for Tiling method (tile size={tile_size}): {avg_tile:.6f} seconds\n")
        f.write(f"Average time for MKL method: {avg_mkl:.6f} seconds\n")

if __name__ == "__main__":
    benchmark()