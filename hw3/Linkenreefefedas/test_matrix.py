import pytest
import _matrix
import time
import sys

def test_matrix_creation():
    """Test matrix creation and basic properties"""
    m = _matrix.Matrix(3, 4)
    assert m.nrow == 3
    assert m.ncol == 4
    assert m.rows() == 3
    assert m.cols() == 4
    
    m2 = _matrix.Matrix(2, 2, 5.0)
    assert m2[0, 0] == 5.0
    assert m2[1, 1] == 5.0
    assert m2(0, 0) == 5.0
    assert m2(1, 1) == 5.0

def test_matrix_access():
    """Test matrix element access and modification"""
    m = _matrix.Matrix(3, 3)
    m[0, 0] = 1.0
    m[1, 1] = 2.0
    m[2, 2] = 3.0
    
    assert m[0, 0] == 1.0
    assert m[1, 1] == 2.0
    assert m[2, 2] == 3.0
    assert m(0, 0) == 1.0
    assert m(1, 1) == 2.0
    assert m(2, 2) == 3.0

def test_matrix_fill():
    """Test matrix fill operation"""
    m = _matrix.Matrix(3, 3)
    m.fill(7.5)
    
    for i in range(3):
        for j in range(3):
            assert m[i, j] == 7.5

def create_test_matrices(size):
    """Helper function to create test matrices"""
    A = _matrix.Matrix(size, size)
    B = _matrix.Matrix(size, size)
    
    # Initialize with simple values
    for i in range(size):
        for j in range(size):
            A[i, j] = float(i + j)
            B[i, j] = float(i - j + 1)
    
    return A, B

def test_multiply_naive_small():
    """Test naive multiplication with small matrices"""
    A = _matrix.Matrix(2, 3)
    B = _matrix.Matrix(3, 2)
    
    # Set A = [[1, 2, 3], [4, 5, 6]]
    A[0, 0] = 1.0
    A[0, 1] = 2.0
    A[0, 2] = 3.0
    A[1, 0] = 4.0
    A[1, 1] = 5.0
    A[1, 2] = 6.0
    
    # Set B = [[1, 2], [3, 4], [5, 6]]
    B[0, 0] = 1.0
    B[0, 1] = 2.0
    B[1, 0] = 3.0
    B[1, 1] = 4.0
    B[2, 0] = 5.0
    B[2, 1] = 6.0
    
    C = _matrix.multiply_naive(A, B)
    
    # Expected C = [[22, 28], [49, 64]]
    assert abs(C[0, 0] - 22.0) < 1e-10
    assert abs(C[0, 1] - 28.0) < 1e-10
    assert abs(C[1, 0] - 49.0) < 1e-10
    assert abs(C[1, 1] - 64.0) < 1e-10

def test_multiply_tile_small():
    """Test tiled multiplication with small matrices"""
    A = _matrix.Matrix(2, 3)
    B = _matrix.Matrix(3, 2)
    
    # Set A = [[1, 2, 3], [4, 5, 6]]
    A[0, 0] = 1.0
    A[0, 1] = 2.0
    A[0, 2] = 3.0
    A[1, 0] = 4.0
    A[1, 1] = 5.0
    A[1, 2] = 6.0
    
    # Set B = [[1, 2], [3, 4], [5, 6]]
    B[0, 0] = 1.0
    B[0, 1] = 2.0
    B[1, 0] = 3.0
    B[1, 1] = 4.0
    B[2, 0] = 5.0
    B[2, 1] = 6.0
    
    C = _matrix.multiply_tile(A, B, 2)
    
    # Expected C = [[22, 28], [49, 64]]
    assert abs(C[0, 0] - 22.0) < 1e-10
    assert abs(C[0, 1] - 28.0) < 1e-10
    assert abs(C[1, 0] - 49.0) < 1e-10
    assert abs(C[1, 1] - 64.0) < 1e-10

def test_multiply_mkl_small():
    """Test MKL multiplication with small matrices"""
    A = _matrix.Matrix(2, 3)
    B = _matrix.Matrix(3, 2)
    
    # Set A = [[1, 2, 3], [4, 5, 6]]
    A[0, 0] = 1.0
    A[0, 1] = 2.0
    A[0, 2] = 3.0
    A[1, 0] = 4.0
    A[1, 1] = 5.0
    A[1, 2] = 6.0
    
    # Set B = [[1, 2], [3, 4], [5, 6]]
    B[0, 0] = 1.0
    B[0, 1] = 2.0
    B[1, 0] = 3.0
    B[1, 1] = 4.0
    B[2, 0] = 5.0
    B[2, 1] = 6.0
    
    C = _matrix.multiply_mkl(A, B)
    
    # Expected C = [[22, 28], [49, 64]]
    assert abs(C[0, 0] - 22.0) < 1e-10
    assert abs(C[0, 1] - 28.0) < 1e-10
    assert abs(C[1, 0] - 49.0) < 1e-10
    assert abs(C[1, 1] - 64.0) < 1e-10

def test_all_methods_match():
    """Test that all three multiplication methods produce the same result"""
    size = 100
    A, B = create_test_matrices(size)
    
    C_naive = _matrix.multiply_naive(A, B)
    C_tile = _matrix.multiply_tile(A, B, 32)
    C_mkl = _matrix.multiply_mkl(A, B)
    
    # Check if results match
    assert C_naive.equals(C_tile, 1e-9), "Naive and tile results don't match"
    assert C_naive.equals(C_mkl, 1e-9), "Naive and MKL results don't match"
    assert C_tile.equals(C_mkl, 1e-9), "Tile and MKL results don't match"

def test_large_matrices():
    """Test with larger matrices (1000x1000)"""
    size = 1000
    A, B = create_test_matrices(size)
    
    print(f"\nTesting with {size}x{size} matrices...")
    
    # Test naive (might be slow)
    print("Running naive multiplication...")
    start = time.time()
    C_naive = _matrix.multiply_naive(A, B)
    time_naive = time.time() - start
    print(f"Naive time: {time_naive:.4f} seconds")
    
    # Test tiled
    print("Running tiled multiplication...")
    start = time.time()
    C_tile = _matrix.multiply_tile(A, B, 64)
    time_tile = time.time() - start
    print(f"Tiled time: {time_tile:.4f} seconds")
    
    # Test MKL
    print("Running MKL multiplication...")
    start = time.time()
    C_mkl = _matrix.multiply_mkl(A, B)
    time_mkl = time.time() - start
    print(f"MKL time: {time_mkl:.4f} seconds")
    
    # Verify results match
    assert C_naive.equals(C_tile, 1e-8), "Naive and tile results don't match"
    assert C_naive.equals(C_mkl, 1e-8), "Naive and MKL results don't match"
    
    # Check performance requirement: tiled should be at least 20% faster than naive
    speedup = time_naive / time_tile
    print(f"Speedup (tile vs naive): {speedup:.2f}x")
    print(f"Tile is {(1 - time_tile/time_naive)*100:.1f}% faster than naive")
    
    assert time_tile < time_naive * 0.8, \
        f"Tiled version must be at least 20% faster than naive (current: {time_tile/time_naive*100:.1f}%)"

def test_dimension_mismatch():
    """Test that multiplication with incompatible dimensions raises an error"""
    A = _matrix.Matrix(2, 3)
    B = _matrix.Matrix(4, 2)
    
    with pytest.raises(Exception):
        _matrix.multiply_naive(A, B)
    
    with pytest.raises(Exception):
        _matrix.multiply_tile(A, B)
    
    with pytest.raises(Exception):
        _matrix.multiply_mkl(A, B)

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
