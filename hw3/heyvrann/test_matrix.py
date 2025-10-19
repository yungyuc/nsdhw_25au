import pytest
from _matrix import Matrix, multiply_naive, multiply_tile, multiply_mkl
import math

def make_matrix(r, c, func):
    m = Matrix(r, c)
    for i in range(r):
        for j in range(c):
            m[i, j] = func(i, j)
    return m

def assert_matrix_equal(mat1: Matrix, mat2: Matrix, eps: float = 1e-9):
    assert mat1.nrow == mat2.nrow
    assert mat1.ncol == mat1.ncol
    for i in range(mat1.nrow):
        for j in range(mat2.ncol):
            assert abs(mat1[i, j]-mat2[i, j]) < eps

def test_matrix_basic():
    m = make_matrix(2, 3, lambda i, j: i*10+j)
    assert m.nrow == 2
    assert m.ncol == 3
    assert m[1, 2] == 12

def test_transpose():
    m = make_matrix(2, 3, lambda i, j: i*10+j)
    t = m.transpose()
    assert t.nrow == 3
    assert t.ncol == 2
    assert t[2, 1] == m[1, 2]

def test_multiply_match():
    mat1 = make_matrix(3, 4, lambda i, j: i+j+1)
    mat2 = make_matrix(4, 2, lambda i, j: (i+1)*(j+1))

    mat_naive = multiply_naive(mat1, mat2)
    mat_tile = multiply_tile(mat1, mat2)
    mat_mkl = multiply_mkl(mat1, mat2)

    assert_matrix_equal(mat_naive, mat_tile)
    assert_matrix_equal(mat_tile, mat_mkl)
    assert_matrix_equal(mat_naive, mat_mkl)

    assert mat_naive.nflo == mat1.nrow * mat1.ncol * mat2.ncol
    assert mat_tile.nflo == mat_naive.nflo
    assert mat_mkl.nflo == mat_naive.nflo

def test_matrix_mismatch():
    mat1 = Matrix(2, 3)
    mat2 = Matrix(4, 5)
    with pytest.raises(Exception):
        multiply_naive(mat1, mat2)