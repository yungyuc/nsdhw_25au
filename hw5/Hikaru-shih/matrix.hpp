#pragma once

#include <cstddef>
#include <vector>
#include <stdexcept>

class Matrix
{
public:
    using size_type = std::size_t;

    Matrix(size_type nrow, size_type ncol);

    size_type nrow() const noexcept { return m_nrow; }
    size_type ncol() const noexcept { return m_ncol; }

    // element access
    double & operator()(size_type i, size_type j);
    double const & operator()(size_type i, size_type j) const;

    // equality for testing
    bool operator==(Matrix const & other) const noexcept;

private:
    size_type m_nrow{0};
    size_type m_ncol{0};
    std::vector<double> m_data;   // row-major, size = nrow * ncol

    size_type index(size_type i, size_type j) const noexcept
    {
        return i * m_ncol + j;
    }
};

// ---- free functions for multiplication ----

// naive i-j-k triple loop
Matrix multiply_naive(Matrix const & A, Matrix const & B);

// tiled (blocked) matmul
Matrix multiply_tile(Matrix const & A, Matrix const & B);

// placeholder: here we simply reuse naive; you can later change to MKL if available.
Matrix multiply_mkl(Matrix const & A, Matrix const & B);