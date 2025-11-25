#include "matrix.hpp"

Matrix::Matrix(size_type nrow, size_type ncol)
    : m_nrow(nrow)
    , m_ncol(ncol)
    , m_data(nrow * ncol, 0.0)
{
}

double & Matrix::operator()(size_type i, size_type j)
{
    if (i >= m_nrow || j >= m_ncol)
    {
        throw std::out_of_range("Matrix index out of range");
    }
    return m_data[index(i, j)];
}

double const & Matrix::operator()(size_type i, size_type j) const
{
    if (i >= m_nrow || j >= m_ncol)
    {
        throw std::out_of_range("Matrix index out of range");
    }
    return m_data[index(i, j)];
}

bool Matrix::operator==(Matrix const & other) const noexcept
{
    if (m_nrow != other.m_nrow || m_ncol != other.m_ncol)
    {
        return false;
    }
    return m_data == other.m_data;
}

// ---------- multiply_naive ----------

Matrix multiply_naive(Matrix const & A, Matrix const & B)
{
    if (A.ncol() != B.nrow())
    {
        throw std::invalid_argument("Incompatible shapes in multiply_naive");
    }

    Matrix C(A.nrow(), B.ncol());

    for (std::size_t i = 0; i < A.nrow(); ++i)
    {
        for (std::size_t k = 0; k < A.ncol(); ++k)
        {
            const double aik = A(i, k);
            if (aik == 0.0) continue; // 小小的優化

            for (std::size_t j = 0; j < B.ncol(); ++j)
            {
                C(i, j) += aik * B(k, j);
            }
        }
    }

    return C;
}

// ---------- multiply_tile ----------

Matrix multiply_tile(Matrix const & A, Matrix const & B)
{
    if (A.ncol() != B.nrow())
    {
        throw std::invalid_argument("Incompatible shapes in multiply_tile");
    }

    Matrix C(A.nrow(), B.ncol());

    const std::size_t M = A.nrow();
    const std::size_t K = A.ncol();
    const std::size_t N = B.ncol();

    const std::size_t TILE = 32; // 視情況可改, 但這樣已經算是 HW 等級的 tile

    for (std::size_t ii = 0; ii < M; ii += TILE)
    {
        const std::size_t iimax = std::min(ii + TILE, M);
        for (std::size_t kk = 0; kk < K; kk += TILE)
        {
            const std::size_t kkmax = std::min(kk + TILE, K);
            for (std::size_t jj = 0; jj < N; jj += TILE)
            {
                const std::size_t jjmax = std::min(jj + TILE, N);

                // 小 block 內部一般三層迴圈
                for (std::size_t i = ii; i < iimax; ++i)
                {
                    for (std::size_t k = kk; k < kkmax; ++k)
                    {
                        const double aik = A(i, k);
                        if (aik == 0.0) continue;

                        for (std::size_t j = jj; j < jjmax; ++j)
                        {
                            C(i, j) += aik * B(k, j);
                        }
                    }
                }
            }
        }
    }

    return C;
}

// ---------- multiply_mkl ----------

// 如果作業環境真的有 MKL，可以改成呼叫 cblas_dgemm。
// 為了讓程式一定能編得過，這裡先直接呼叫 naive 版本。
Matrix multiply_mkl(Matrix const & A, Matrix const & B)
{
    return multiply_naive(A, B);
}