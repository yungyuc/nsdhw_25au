#pragma once
#include <cstddef>
#include <vector>
#include <algorithm>
#include <cstring>
#include <stdexcept>

#ifdef USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif

class Matrix
{
public:
    Matrix(size_t nrow, size_t ncol)
        : nrow_(nrow), ncol_(ncol), buffer_(nrow * ncol, 0.0)
    {
    }

    size_t nrow() const { return nrow_; }
    size_t ncol() const { return ncol_; }

    double operator()(size_t row, size_t col) const
    {
        return buffer_[row * ncol_ + col];
    }

    double &operator()(size_t row, size_t col)
    {
        return buffer_[row * ncol_ + col];
    }

    double *get_data() { return buffer_.data(); }
    const double *get_data() const { return buffer_.data(); }

    void load_buffer(double *input)
    {
        std::memcpy(buffer_.data(), input, sizeof(double) * nrow_ * ncol_);
    }

    friend bool operator==(const Matrix &A, const Matrix &B)
    {
        if (A.nrow_ != B.nrow_ || A.ncol_ != B.ncol_)
            return false;

        for (size_t i = 0; i < A.buffer_.size(); ++i)
        {
            if (A.buffer_[i] != B.buffer_[i])
                return false;
        }
        return true;
    }

private:
    size_t nrow_;
    size_t ncol_;
    std::vector<double> buffer_;
};

inline Matrix multiply_naive(const Matrix &A, const Matrix &B)
{
    if (A.ncol() != B.nrow())
        throw std::runtime_error("Dimension mismatch in multiply_naive");

    Matrix C(A.nrow(), B.ncol());

    for (size_t i = 0; i < A.nrow(); ++i)
    {
        for (size_t j = 0; j < B.ncol(); ++j)
        {
            double sum = 0.0;
            for (size_t k = 0; k < A.ncol(); ++k)
            {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }

    return C;
}

inline Matrix multiply_tile(const Matrix &A, const Matrix &B, size_t T)
{
    if (A.ncol() != B.nrow())
        throw std::runtime_error("Dimension mismatch in multiply_tile");

    Matrix C(A.nrow(), B.ncol());

    size_t M = A.nrow();
    size_t K = A.ncol();
    size_t N = B.ncol();

    for (size_t ii = 0; ii < M; ii += T)
    {
        for (size_t jj = 0; jj < N; jj += T)
        {
            for (size_t kk = 0; kk < K; kk += T)
            {

                size_t i_max = std::min(ii + T, M);
                size_t j_max = std::min(jj + T, N);
                size_t k_max = std::min(kk + T, K);

                for (size_t i = ii; i < i_max; ++i)
                {
                    for (size_t j = jj; j < j_max; ++j)
                    {

                        double sum = C(i, j);

                        for (size_t k = kk; k < k_max; ++k)
                        {
                            sum += A(i, k) * B(k, j);
                        }

                        C(i, j) = sum;
                    }
                }
            }
        }
    }

    return C;
}

inline Matrix multiply_mkl(const Matrix &A, const Matrix &B)
{
    if (A.ncol() != B.nrow())
        throw std::runtime_error("Dimension mismatch in multiply_mkl");

    Matrix C(A.nrow(), B.ncol());

    int m = A.nrow();
    int k = A.ncol();
    int n = B.ncol();

    double alpha = 1.0;
    double beta = 0.0;

    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans, CblasNoTrans,
        m, n, k,
        alpha,
        A.get_data(), k,
        B.get_data(), n,
        beta,
        C.get_data(), n);

    return C;
}
