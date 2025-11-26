#include "matrix.hpp"

#include <algorithm>
#include <stdexcept>

#if defined(USE_MKL)
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif

Matrix::Matrix(std::size_t rows, std::size_t cols)
    : m_rows(rows), m_cols(cols), m_data(rows * cols, 0.0)
{
}

std::size_t Matrix::nrow() const { return m_rows; }
std::size_t Matrix::ncol() const { return m_cols; }

double &Matrix::operator()(std::size_t r, std::size_t c)
{
    return m_data[r * m_cols + c];
}

double const &Matrix::operator()(std::size_t r, std::size_t c) const
{
    return m_data[r * m_cols + c];
}

bool Matrix::operator==(Matrix const &other) const
{
    return m_rows == other.m_rows && m_cols == other.m_cols && m_data == other.m_data;
}

bool Matrix::operator!=(Matrix const &other) const
{
    return !(*this == other);
}

Matrix Matrix::transpose() const
{
    Matrix result(m_cols, m_rows);
    for (std::size_t i = 0; i < m_rows; ++i)
    {
        for (std::size_t j = 0; j < m_cols; ++j)
        {
            result(j, i) = (*this)(i, j);
        }
    }
    return result;
}

std::vector<double> &Matrix::data() { return m_data; }
std::vector<double> const &Matrix::data() const { return m_data; }

Matrix multiply_naive(Matrix const &a, Matrix const &b)
{
    if (a.ncol() != b.nrow())
    {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication.");
    }

    Matrix result(a.nrow(), b.ncol());
    for (std::size_t i = 0; i < a.nrow(); ++i)
    {
        for (std::size_t j = 0; j < b.ncol(); ++j)
        {
            double sum = 0.0;
            for (std::size_t k = 0; k < a.ncol(); ++k)
            {
                sum += a(i, k) * b(k, j);
            }
            result(i, j) = sum;
        }
    }

    return result;
}

Matrix multiply_tile(Matrix const &a, Matrix const &b, std::size_t tile_size)
{
    if (tile_size == 0)
    {
        throw std::invalid_argument("tile_size must be positive");
    }
    if (a.ncol() != b.nrow())
    {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication.");
    }

    Matrix bt = b.transpose(); // improves locality for the inner loop
    Matrix c(a.nrow(), b.ncol());

    const std::size_t M = a.nrow();
    const std::size_t N = b.ncol();
    const std::size_t K = a.ncol();

    auto const &A = a.data();
    auto const &B = bt.data();
    auto &C = c.data();

    for (std::size_t i0 = 0; i0 < M; i0 += tile_size)
    {
        const std::size_t i_max = std::min(i0 + tile_size, M);
        for (std::size_t j0 = 0; j0 < N; j0 += tile_size)
        {
            const std::size_t j_max = std::min(j0 + tile_size, N);
            for (std::size_t k0 = 0; k0 < K; k0 += tile_size)
            {
                const std::size_t k_max = std::min(k0 + tile_size, K);

                for (std::size_t i = i0; i < i_max; ++i)
                {
                    for (std::size_t j = j0; j < j_max; ++j)
                    {
                        double sum = C[i * N + j];
                        for (std::size_t k = k0; k < k_max; ++k)
                        {
                            sum += A[i * K + k] * B[j * K + k];
                        }
                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }

    return c;
}

Matrix multiply_mkl(Matrix const &a, Matrix const &b)
{
    if (a.ncol() != b.nrow())
    {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication.");
    }

    Matrix result(a.nrow(), b.ncol());
    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        static_cast<int>(a.nrow()),
        static_cast<int>(b.ncol()),
        static_cast<int>(a.ncol()),
        1.0,
        a.data().data(),
        static_cast<int>(a.ncol()),
        b.data().data(),
        static_cast<int>(b.ncol()),
        0.0,
        result.data().data(),
        static_cast<int>(b.ncol()));

    return result;
}