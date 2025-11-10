#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>
#include "allocator.hpp"

class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols)
      : m_rows(rows), m_cols(cols), m_data(rows * cols, 0.0) {}

    std::size_t rows() const { return m_rows; }
    std::size_t cols() const { return m_cols; }

    double const & at(std::size_t r, std::size_t c) const {
        if (r >= m_rows || c >= m_cols) throw std::out_of_range("index out of range");
        return m_data[r * m_cols + c];
    }

    double & at(std::size_t r, std::size_t c) {
        if (r >= m_rows || c >= m_cols) throw std::out_of_range("index out of range");
        return m_data[r * m_cols + c];
    }

    // Raw data accessors (row-major)
    double const * data() const { return m_data.data(); }
    double * data() { return m_data.data(); }

    double get(std::size_t r, std::size_t c) const { return at(r, c); }
    void set(std::size_t r, std::size_t c, double v) { at(r, c) = v; }

private:
    std::size_t m_rows{0};
    std::size_t m_cols{0};
    std::vector<double, CountingAllocator<double>> m_data;
};

Matrix multiply_naive(Matrix const &A, Matrix const &B);
Matrix multiple_tile(Matrix const &A, Matrix const &B, std::size_t tile);
Matrix multiply_mkl(Matrix const &A, Matrix const &B);
bool blas_available();

// Expose memory statistics query in C++
inline std::size_t bytes() { return memstats::bytes(); }
inline std::size_t allocated() { return memstats::allocated(); }
inline std::size_t deallocated() { return memstats::deallocated(); }

