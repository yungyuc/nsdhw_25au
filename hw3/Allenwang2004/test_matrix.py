#!/usr/bin/env python3
"""
Pytest test suite for matrix multiplication methods.
This script tests three methods: naive, tiling, and MKL-based implementations
to ensure they produce consistent results and meet performance requirements.
"""

import time
import pytest
import _matrix
import random
import sys
from typing import Tuple

class TestMatrixMultiplication:
    """Test class for matrix multiplication methods."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup for each test method."""
        self.tolerance = 1e-10  # Tolerance for floating point comparison
        
    def create_random_matrix(self, rows, cols, seed=None):
        """Create a matrix with random values."""
        if seed is not None:
            random.seed(seed)
        
        matrix = _matrix.Matrix(rows, cols)
        for i in range(rows):
            for j in range(cols):
                matrix[i, j] = random.uniform(-10.0, 10.0)
        return matrix
    
    def matrices_are_equal(self, mat1, mat2, tolerance=None):
        """Check if two matrices are approximately equal."""
        if tolerance is None:
            tolerance = self.tolerance
            
        if mat1.rows() != mat2.rows() or mat1.cols() != mat2.cols():
            return False
            
        for i in range(mat1.rows()):
            for j in range(mat1.cols()):
                diff = abs(mat1[i, j] - mat2[i, j])
                if diff > tolerance:
                    return False, f"Difference at ({i}, {j}): {diff}"
        return True, "Matrices are equal"
    
    def print_matrix_info(self, matrix, name):
        """Print basic information about a matrix."""
        print(f"{name}: {matrix.rows()}x{matrix.cols()}")
        if matrix.rows() <= 5 and matrix.cols() <= 5:
            print("Matrix values:")
            for i in range(matrix.rows()):
                row = [f"{matrix[i, j]:.3f}" for j in range(matrix.cols())]
                print("  [" + ", ".join(row) + "]")

    def test_large_matrices_correctness_and_performance(self, size=1000):
        """Test with large matrices for performance and correctness."""
        # Create large test matrices
        A = self.create_random_matrix(size, size, seed=12345)
        B = self.create_random_matrix(size, size, seed=54321)
        
        results = {}
        times = {}
        
        # Test naive method
        start_time = time.time()
        C_naive = _matrix.multiply_naive(A, B)
        times['naive'] = time.time() - start_time
        results['naive'] = C_naive
        
        # Test tiling method with different tile sizes to find optimal
        best_time = float('inf')
        best_tile_size = 16
        test_tile_sizes = [16]
        
        for tile_size in test_tile_sizes:
            start_time = time.time()
            C_tiling_test = _matrix.multiply_tile(A, B, tile_size)
            test_time = time.time() - start_time
            if test_time < best_time:
                best_time = test_time
                best_tile_size = tile_size
        
        tile_size = best_tile_size
        start_time = time.time()
        C_tiling = _matrix.multiply_tile(A, B, tile_size)
        times['tiling'] = time.time() - start_time
        results['tiling'] = C_tiling
        
        # Test MKL method
        start_time = time.time()
        C_mkl = _matrix.multiply_mkl(A, B)
        times['mkl'] = time.time() - start_time
        results['mkl'] = C_mkl
        
        # Check correctness (sample a few elements due to large size)
        sample_size = min(100, size)
        tolerance = 1e-8  # Slightly more lenient for large matrices
        
        for _ in range(sample_size):
            i = random.randint(0, size - 1)
            j = random.randint(0, size - 1)
            
            naive_val = results['naive'][i, j]
            tiling_val = results['tiling'][i, j]
            mkl_val = results['mkl'][i, j]
            
            assert abs(naive_val - tiling_val) <= tolerance, \
                f"Naive vs Tiling mismatch at ({i}, {j}): {naive_val} vs {tiling_val}"
                
            assert abs(naive_val - mkl_val) <= tolerance, \
                f"Naive vs MKL mismatch at ({i}, {j}): {naive_val} vs {mkl_val}"
        
        # Performance analysis - Check if tiling is at least 20% faster than naive
        speedup = times['naive'] / times['tiling']
        required_speedup = 1.2  # 20% faster
        
        print(f"\n=== Performance Results for {size}x{size} ===")
        print(f"Naive method:  {times['naive']:.2f} seconds")
        print(f"Tiling method: {times['tiling']:.2f} seconds (tile_size={tile_size})")
        print(f"MKL method:    {times['mkl']:.2f} seconds")
        print(f"Tiling speedup over naive: {speedup:.2f}x")
        
        assert speedup >= required_speedup, \
            f"Tiling is only {((speedup - 1) * 100):.1f}% faster than naive (requirement: ≥20%)"
    
    def test_matrix_dimensions_validation(self):
        """Test that matrix multiplication validates dimensions correctly."""
        A = self.create_random_matrix(3, 4, seed=42)
        B = self.create_random_matrix(5, 3, seed=43)  # Incompatible dimensions
        
        with pytest.raises(Exception):
            _matrix.multiply_naive(A, B)
            
        with pytest.raises(Exception):
            _matrix.multiply_tile(A, B, 2)
            
        with pytest.raises(Exception):
            _matrix.multiply_mkl(A, B)
    
    def test_different_tile_sizes(self):
        """Test tiling method with different tile sizes."""
        A = self.create_random_matrix(100, 100, seed=42)
        B = self.create_random_matrix(100, 100, seed=43)
        
        C_naive = _matrix.multiply_naive(A, B)
        
        # Test different tile sizes
        for tile_size in [16, 32, 64, 128]:
            C_tiling = _matrix.multiply_tile(A, B, tile_size)
            equal, msg = self.matrices_are_equal(C_naive, C_tiling, tolerance=1e-10)
            assert equal, f"Tiling result with tile_size={tile_size} doesn't match naive: {msg}"