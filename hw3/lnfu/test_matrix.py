#!/usr/bin/env python3

import pytest
import numpy as np

import _matrix


def test_matrix_creation():
    """Test matrix creation"""
    m = _matrix.Matrix(3, 4)
    assert m.nrow == 3
    assert m.ncol == 4
    assert m.size == 12


def test_matrix_with_data():
    """Test matrix creation with data"""
    data = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]
    m = _matrix.Matrix(data)
    assert m.nrow == 2
    assert m.ncol == 3
    assert m.size == 6
    assert m[0, 0] == 1.0
    assert m[0, 1] == 2.0
    assert m[0, 2] == 3.0
    assert m[1, 0] == 4.0
    assert m[1, 1] == 5.0
    assert m[1, 2] == 6.0


def test_matrix_indexing():
    """Test matrix element access"""
    m = _matrix.Matrix(2, 2)
    m[0, 0] = 1.0
    m[0, 1] = 2.0
    m[1, 0] = 3.0
    m[1, 1] = 4.0

    assert m[0, 0] == 1.0
    assert m[0, 1] == 2.0
    assert m[1, 0] == 3.0
    assert m[1, 1] == 4.0


def test_multiply_naive_small():
    """Test naive multiplication with small matrices"""

    A = _matrix.Matrix(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    B = _matrix.Matrix(
        [
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )

    # C = [[19, 22], [43, 50]]
    C = _matrix.multiply_naive(A, B)

    assert A[0, 0] == 1.0
    assert A[0, 1] == 2.0
    assert A[1, 0] == 3.0
    assert A[1, 1] == 4.0

    assert B[0, 0] == 5.0
    assert B[0, 1] == 6.0
    assert B[1, 0] == 7.0
    assert B[1, 1] == 8.0

    assert C[0, 0] == 19.0
    assert C[0, 1] == 22.0
    assert C[1, 0] == 43.0
    assert C[1, 1] == 50.0


def test_multiply_tile_small():
    """Test tiled multiplication with small matrices"""
    A = _matrix.Matrix(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    B = _matrix.Matrix(
        [
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )

    # C = [[19, 22], [43, 50]]
    C = _matrix.multiply_tile(A, B)

    assert A[0, 0] == 1.0
    assert A[0, 1] == 2.0
    assert A[1, 0] == 3.0
    assert A[1, 1] == 4.0

    assert B[0, 0] == 5.0
    assert B[0, 1] == 6.0
    assert B[1, 0] == 7.0
    assert B[1, 1] == 8.0

    assert C[0, 0] == 19.0
    assert C[0, 1] == 22.0
    assert C[1, 0] == 43.0
    assert C[1, 1] == 50.0


def test_multiply_tile_large():
    """Test tiled multiplication with larger matrices"""
    n = 128
    np.random.seed(0)
    A_data = np.random.randn(n, n).tolist()
    B_data = np.random.randn(n, n).tolist()

    A = _matrix.Matrix(A_data)
    B = _matrix.Matrix(B_data)

    C_tile = _matrix.multiply_tile(A, B)

    # Verify against numpy result
    C_np = np.dot(np.array(A_data), np.array(B_data))

    for i in range(n):
        for j in range(n):
            assert abs(C_tile[i, j] - C_np[i, j]) < 1e-6

    C_tile = _matrix.multiply_tile(A, B, tile_size=32)

    # Verify against numpy result
    C_np = np.dot(np.array(A_data), np.array(B_data))

    for i in range(n):
        for j in range(n):
            assert abs(C_tile[i, j] - C_np[i, j]) < 1e-6


def test_multiply_mkl_small():
    """Test MKL multiplication with small matrices"""
    A = _matrix.Matrix(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )
    B = _matrix.Matrix(
        [
            [5.0, 6.0],
            [7.0, 8.0],
        ]
    )

    # C = [[19, 22], [43, 50]]
    C = _matrix.multiply_mkl(A, B)

    assert A[0, 0] == 1.0
    assert A[0, 1] == 2.0
    assert A[1, 0] == 3.0
    assert A[1, 1] == 4.0

    assert B[0, 0] == 5.0
    assert B[0, 1] == 6.0
    assert B[1, 0] == 7.0
    assert B[1, 1] == 8.0

    assert C[0, 0] == 19.0
    assert C[0, 1] == 22.0
    assert C[1, 0] == 43.0
    assert C[1, 1] == 50.0


def test_multiply_consistency():
    """Test that all three methods give the same result"""
    np.random.seed(42)
    n = 100

    A = _matrix.Matrix([np.random.randn(n).tolist() for _ in range(n)])
    B = _matrix.Matrix([np.random.randn(n).tolist() for _ in range(n)])

    C_naive = _matrix.multiply_naive(A, B)
    C_tile = _matrix.multiply_tile(A, B)
    C_mkl = _matrix.multiply_mkl(A, B)

    # Check consistency
    assert C_naive == C_tile
    assert C_naive == C_mkl


def test_multiply_large():
    """Test multiplication with large matrices"""
    np.random.seed(123)
    n = 500

    A = _matrix.Matrix([np.random.randn(n).tolist() for _ in range(n)])
    B = _matrix.Matrix([np.random.randn(n).tolist() for _ in range(n)])

    C_naive = _matrix.multiply_naive(A, B)
    C_tile = _matrix.multiply_tile(A, B)
    C_mkl = _matrix.multiply_mkl(A, B)

    # Check consistency
    assert C_naive == C_tile
    assert C_naive == C_mkl


def test_invalid_dimensions():
    """Test that incompatible dimensions raise an error"""
    A = _matrix.Matrix(2, 3)
    B = _matrix.Matrix(4, 2)

    with pytest.raises(Exception):
        _matrix.multiply_naive(A, B)

    with pytest.raises(Exception):
        _matrix.multiply_tile(A, B)

    with pytest.raises(Exception):
        _matrix.multiply_mkl(A, B)


def test_rectangular_matrices():
    """Test multiplication with non-square matrices"""
    # A: 3x2, B: 2x4, C: 3x4
    A = _matrix.Matrix([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    B = _matrix.Matrix([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

    C_naive = _matrix.multiply_naive(A, B)
    C_tile = _matrix.multiply_tile(A, B)
    C_mkl = _matrix.multiply_mkl(A, B)

    assert C_naive.nrow == 3
    assert C_naive.ncol == 4

    # Verify consistency
    assert C_naive == C_tile
    assert C_naive == C_mkl


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
