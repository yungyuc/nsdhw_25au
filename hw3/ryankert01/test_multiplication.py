import pytest
import numpy as np
from _matrix import Matrix, multiply_naive, multiply_tile, multiply_mkl

def fill_matrix(mat):
    for i in range(mat.nrow):
        for j in range(mat.ncol):
            mat[i, j] = i * mat.ncol + j

def to_numpy(mat):
    np_mat = np.empty((mat.nrow, mat.ncol))
    for i in range(mat.nrow):
        for j in range(mat.ncol):
            np_mat[i, j] = mat[i, j]
    return np_mat

def check_multiplication_correctness(nrow_a, ncol_a, nrow_b, ncol_b):
    mat_a = Matrix(nrow_a, ncol_a)
    mat_b = Matrix(nrow_b, ncol_b)

    fill_matrix(mat_a)
    fill_matrix(mat_b)

    np_a = to_numpy(mat_a)
    np_b = to_numpy(mat_b)
    expected = np.dot(np_a, np_b)

    res_naive = multiply_naive(mat_a, mat_b)
    res_tile = multiply_tile(mat_a, mat_b, 64)
    res_mkl = multiply_mkl(mat_a, mat_b)

    np_naive = to_numpy(res_naive)
    np_tile = to_numpy(res_tile)
    np_mkl = to_numpy(res_mkl)

    # use np as golden ref
    assert np.allclose(expected, np_naive)
    assert np.allclose(expected, np_tile)
    assert np.allclose(expected, np_mkl)

def test_multiplication_correctness():
    check_multiplication_correctness(128, 256, 256, 128)

def test_multiplication_correctness_weird_shape():
    # weird shape
    check_multiplication_correctness(123, 231, 231, 123)
    check_multiplication_correctness(123, 1, 1, 123)
    check_multiplication_correctness(1, 34, 34, 1)

def test_dimension_mismatch():
    mat_a = Matrix(2, 3)
    mat_b = Matrix(4, 2)
    with pytest.raises(Exception):
        multiply_naive(mat_a, mat_b)