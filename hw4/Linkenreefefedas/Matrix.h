#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include "CustomAllocator.h"

class Matrix {
public:
    // Constructor
    Matrix(size_t rows, size_t cols) 
        : m_rows(rows), m_cols(cols), m_data(rows * cols) {}
    
    // Constructor with initial value
    Matrix(size_t rows, size_t cols, double val) 
        : m_rows(rows), m_cols(cols), m_data(rows * cols, val) {}
    
    // Copy constructor
    Matrix(const Matrix& other)
        : m_rows(other.m_rows), m_cols(other.m_cols), m_data(other.m_data) {}
    
    // Move constructor
    Matrix(Matrix&& other) noexcept
        : m_rows(other.m_rows), m_cols(other.m_cols), m_data(std::move(other.m_data)) {
        other.m_rows = 0;
        other.m_cols = 0;
    }
    
    // Assignment operator
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            m_rows = other.m_rows;
            m_cols = other.m_cols;
            m_data = other.m_data;
        }
        return *this;
    }
    
    // Move assignment operator
    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            m_rows = other.m_rows;
            m_cols = other.m_cols;
            m_data = std::move(other.m_data);
            other.m_rows = 0;
            other.m_cols = 0;
        }
        return *this;
    }
    
    // Get dimensions
    size_t rows() const { return m_rows; }
    size_t cols() const { return m_cols; }
    
    // Element access
    double operator()(size_t i, size_t j) const {
        return m_data[i * m_cols + j];
    }
    
    double& operator()(size_t i, size_t j) {
        return m_data[i * m_cols + j];
    }
    
    // Get raw data pointer
    double* data() { return m_data.data(); }
    const double* data() const { return m_data.data(); }
    
    // Fill with value
    void fill(double val) {
        std::fill(m_data.begin(), m_data.end(), val);
    }
    
    // Check if matrices are equal (with tolerance)
    bool equals(const Matrix& other, double tol = 1e-10) const {
        if (m_rows != other.m_rows || m_cols != other.m_cols) {
            return false;
        }
        for (size_t i = 0; i < m_data.size(); ++i) {
            if (std::abs(m_data[i] - other.m_data[i]) > tol) {
                return false;
            }
        }
        return true;
    }

private:
    size_t m_rows;
    size_t m_cols;
    std::vector<double, CustomAllocator<double>> m_data;
};

// Forward declarations of multiplication functions
Matrix multiply_naive(const Matrix& A, const Matrix& B);
Matrix multiply_tile(const Matrix& A, const Matrix& B, size_t tile_size = 64);
Matrix multiply_mkl(const Matrix& A, const Matrix& B);

#endif // MATRIX_H
