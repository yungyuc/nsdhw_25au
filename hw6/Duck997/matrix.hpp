#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

class Matrix {
public:
    Matrix() = default;
    Matrix(std::size_t rows, std::size_t cols)
      : m_rows(rows), m_cols(cols), m_data(rows * cols, 0.0) {}

    std::size_t rows() const { return m_rows; }
    std::size_t cols() const { return m_cols; }
    std::size_t nrow() const { return m_rows; }
    std::size_t ncol() const { return m_cols; }

    double const & at(std::size_t r, std::size_t c) const {
        if (r >= m_rows || c >= m_cols) throw std::out_of_range("index out of range");
        return m_data[r * m_cols + c];
    }

    double & at(std::size_t r, std::size_t c) {
        if (r >= m_rows || c >= m_cols) throw std::out_of_range("index out of range");
        return m_data[r * m_cols + c];
    }

    double operator()(std::size_t r, std::size_t c) const {
        return at(r, c);
    }
    double & operator()(std::size_t r, std::size_t c) {
        return at(r, c);
    }

    // Raw data accessors (row-major)
    double const * data() const { return m_data.data(); }
    double * data() { return m_data.data(); }

    double get(std::size_t r, std::size_t c) const { return at(r, c); }
    void set(std::size_t r, std::size_t c, double v) { at(r, c) = v; }

    bool operator==(Matrix const& other) const {
        if (m_rows != other.m_rows || m_cols != other.m_cols) return false;
        for (std::size_t i = 0; i < m_rows; ++i) {
            for (std::size_t j = 0; j < m_cols; ++j) {
                if (at(i, j) != other.at(i, j)) return false;
            }
        }
        return true;
    }

private:
    std::size_t m_rows{0};
    std::size_t m_cols{0};
    std::vector<double> m_data;
};

Matrix multiply_naive(Matrix const &A, Matrix const &B);
Matrix multiply_tile(Matrix const &A, Matrix const &B, std::size_t tile);
Matrix multiple_tile(Matrix const &A, Matrix const &B, std::size_t tile);
Matrix multiply_mkl(Matrix const &A, Matrix const &B);
bool blas_available();


