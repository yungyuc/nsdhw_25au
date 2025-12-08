#pragma once

#include <cstddef>
#include <vector>

class Matrix
{
public:
    using size_type = std::size_t;

    Matrix() = default;

    Matrix(size_type nrow, size_type ncol)
      : m_nrow(nrow)
      , m_ncol(ncol)
      , m_data(nrow * ncol, 0.0)
    {}

    // 行列數（兩套名稱都提供，和之前作業、wrapper 對齊）
    size_type rows() const noexcept { return m_nrow; }
    size_type cols() const noexcept { return m_ncol; }
    size_type nrow() const noexcept { return m_nrow; }
    size_type ncol() const noexcept { return m_ncol; }

    // 直接拿到底層連續 buffer
    double* data() noexcept { return m_data.data(); }
    double const* data() const noexcept { return m_data.data(); }

    // (i, j) 索引，row-major：i * ncol + j
    double& operator()(size_type i, size_type j)
    {
        return m_data[i * m_ncol + j];
    }

    double operator()(size_type i, size_type j) const
    {
        return m_data[i * m_ncol + j];
    }

    // 相等比較：形狀一樣且每個元素一樣
    bool operator==(Matrix const& other) const noexcept
    {
        return m_nrow == other.m_nrow &&
               m_ncol == other.m_ncol &&
               m_data == other.m_data;
    }

private:
    size_type m_nrow = 0;
    size_type m_ncol = 0;
    std::vector<double> m_data;
};

// 乘法函式宣告，實作在 matrix.cpp
Matrix multiply_naive(Matrix const& A, Matrix const& B);
Matrix multiply_tile(Matrix const& A, Matrix const& B, std::size_t tile);
Matrix multiply_mkl(Matrix const& A, Matrix const& B);
