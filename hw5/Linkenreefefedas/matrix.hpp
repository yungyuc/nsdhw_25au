#pragma once

#include <vector>
#include <stdexcept>
#include <algorithm>
#include <mkl.h>

class Matrix {
public:
    Matrix(size_t nrow, size_t ncol) 
        : m_nrow(nrow), m_ncol(ncol), m_data(nrow * ncol, 0.0) {}

    Matrix(Matrix const & other)
        : m_nrow(other.m_nrow), m_ncol(other.m_ncol), m_data(other.m_data) {}

    Matrix(Matrix && other) noexcept
        : m_nrow(other.m_nrow), m_ncol(other.m_ncol), m_data(std::move(other.m_data)) {}

    Matrix & operator=(Matrix const & other) {
        if (this != &other) {
            m_nrow = other.m_nrow;
            m_ncol = other.m_ncol;
            m_data = other.m_data;
        }
        return *this;
    }

    Matrix & operator=(Matrix && other) noexcept {
        if (this != &other) {
            m_nrow = other.m_nrow;
            m_ncol = other.m_ncol;
            m_data = std::move(other.m_data);
        }
        return *this;
    }

    ~Matrix() = default;

    bool operator==(Matrix const & other) const {
        if (m_nrow != other.m_nrow || m_ncol != other.m_ncol) {
            return false;
        }
        return m_data == other.m_data;
    }

    bool operator!=(Matrix const & other) const {
        return !(*this == other);
    }

    double operator()(size_t i, size_t j) const {
        return m_data[i * m_ncol + j];
    }

    double & operator()(size_t i, size_t j) {
        return m_data[i * m_ncol + j];
    }

    size_t nrow() const { return m_nrow; }
    size_t ncol() const { return m_ncol; }

private:
    size_t m_nrow;
    size_t m_ncol;
    std::vector<double> m_data;
};

// Naive matrix multiplication - O(n^3)
Matrix multiply_naive(Matrix const & mat1, Matrix const & mat2) {
    if (mat1.ncol() != mat2.nrow()) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }

    Matrix result(mat1.nrow(), mat2.ncol());
    
    for (size_t i = 0; i < mat1.nrow(); ++i) {
        for (size_t j = 0; j < mat2.ncol(); ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < mat1.ncol(); ++k) {
                sum += mat1(i, k) * mat2(k, j);
            }
            result(i, j) = sum;
        }
    }
    
    return result;
}

// Tiled matrix multiplication for better cache locality
Matrix multiply_tile(Matrix const & mat1, Matrix const & mat2, size_t tile_size) {
    if (mat1.ncol() != mat2.nrow()) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }

    Matrix result(mat1.nrow(), mat2.ncol());
    
    const size_t M = mat1.nrow();
    const size_t N = mat2.ncol();
    const size_t K = mat1.ncol();
    
    // Process matrix in tiles for better cache performance
    for (size_t i0 = 0; i0 < M; i0 += tile_size) {
        const size_t i_end = std::min(i0 + tile_size, M);
        
        for (size_t j0 = 0; j0 < N; j0 += tile_size) {
            const size_t j_end = std::min(j0 + tile_size, N);
            
            for (size_t k0 = 0; k0 < K; k0 += tile_size) {
                const size_t k_end = std::min(k0 + tile_size, K);
                
                // Compute the tile
                for (size_t i = i0; i < i_end; ++i) {
                    for (size_t j = j0; j < j_end; ++j) {
                        double sum = result(i, j);
                        for (size_t k = k0; k < k_end; ++k) {
                            sum += mat1(i, k) * mat2(k, j);
                        }
                        result(i, j) = sum;
                    }
                }
            }
        }
    }
    
    return result;
}

// MKL-based matrix multiplication using DGEMM
Matrix multiply_mkl(Matrix const & mat1, Matrix const & mat2) {
    if (mat1.ncol() != mat2.nrow()) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication");
    }

    Matrix result(mat1.nrow(), mat2.ncol());
    
    const MKL_INT m = static_cast<MKL_INT>(mat1.nrow());
    const MKL_INT n = static_cast<MKL_INT>(mat2.ncol());
    const MKL_INT k = static_cast<MKL_INT>(mat1.ncol());
    
    const double alpha = 1.0;
    const double beta = 0.0;
    
    // Create contiguous copies for MKL (row-major to column-major)
    std::vector<double> a_copy(m * k);
    std::vector<double> b_copy(k * n);
    
    // Transpose mat1 to column-major
    for (size_t i = 0; i < mat1.nrow(); ++i) {
        for (size_t j = 0; j < mat1.ncol(); ++j) {
            a_copy[j * m + i] = mat1(i, j);
        }
    }
    
    // Transpose mat2 to column-major
    for (size_t i = 0; i < mat2.nrow(); ++i) {
        for (size_t j = 0; j < mat2.ncol(); ++j) {
            b_copy[j * k + i] = mat2(i, j);
        }
    }
    
    std::vector<double> c_result(m * n);
    
    // Call MKL DGEMM: C = alpha * A * B + beta * C
    cblas_dgemm(
        CblasColMajor,     // Column-major order
        CblasNoTrans,      // Don't transpose A
        CblasNoTrans,      // Don't transpose B
        m,                 // Rows of A and C
        n,                 // Columns of B and C
        k,                 // Columns of A, rows of B
        alpha,             // Scalar alpha
        a_copy.data(),     // Matrix A
        m,                 // Leading dimension of A
        b_copy.data(),     // Matrix B
        k,                 // Leading dimension of B
        beta,              // Scalar beta
        c_result.data(),   // Matrix C
        m                  // Leading dimension of C
    );
    
    // Convert result back to row-major
    for (size_t i = 0; i < result.nrow(); ++i) {
        for (size_t j = 0; j < result.ncol(); ++j) {
            result(i, j) = c_result[j * m + i];
        }
    }
    
    return result;
}

// vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
