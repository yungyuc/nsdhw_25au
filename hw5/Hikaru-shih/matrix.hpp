#include <iostream>
#include <algorithm>
#include <vector>
#include <cblas.h>

class Matrix {
public:
    Matrix(size_t r, size_t c)
        : r_(r), c_(c), buf_(r * c)
    {}

    double operator()(size_t r, size_t c) const {
        return buf_[r * c_ + c];
    }

    double& operator()(size_t r, size_t c) {
        return buf_[r * c_ + c];
    }

    size_t nrow() const { return r_; }
    size_t ncol() const { return c_; }

    const double* data() const { return buf_.data(); }
    double* data() { return buf_.data(); }

    bool operator==(Matrix const& o) const {
        return (r_ == o.r_ && c_ == o.c_ && buf_ == o.buf_);
    }

private:
    size_t r_;
    size_t c_;
    std::vector<double> buf_;
};

void populate(Matrix& m) {
    for (size_t i = 0; i < m.nrow(); ++i) {
        for (size_t j = 0; j < m.ncol(); ++j) {
            m(i, j) = 1;
        }
    }
}

Matrix multiply_naive(const Matrix& a, const Matrix& b) {
    Matrix out(a.nrow(), b.ncol());

    for (size_t i = 0; i < out.nrow(); ++i) {
        for (size_t k = 0; k < out.ncol(); ++k) {
            double acc = 0;
            for (size_t j = 0; j < a.ncol(); ++j) {
                acc += a(i, j) * b(j, k);
            }
            out(i, k) = acc;
        }
    }

    return out;
}

Matrix multiply_tile(const Matrix& a, const Matrix& b, size_t T) {
    size_t R = a.nrow();
    size_t M = a.ncol();
    size_t C = b.ncol();

    Matrix out(R, C);

    for (size_t i = 0; i < R; i += T) {
        for (size_t j = 0; j < C; j += T) {
            for (size_t k = 0; k < M; k += T) {
                size_t i2 = std::min(i + T, R);
                size_t j2 = std::min(j + T, C);
                size_t k2 = std::min(k + T, M);

                for (size_t ii = i; ii < i2; ++ii) {
                    for (size_t jj = j; jj < j2; ++jj) {
                        double acc = out(ii, jj);
                        for (size_t kk = k; kk < k2; ++kk) {
                            acc += a(ii, kk) * b(kk, jj);
                        }
                        out(ii, jj) = acc;
                    }
                }
            }
        }
    }

    return out;
}

Matrix multiply_mkl(const Matrix& a, const Matrix& b) {
    int R = a.nrow();
    int M = a.ncol();
    int C = b.ncol();

    Matrix out(a.nrow(), b.ncol());

    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        R, C, M,
        1.0,
        a.data(), M,
        b.data(), C,
        0.0,
        out.data(), C
    );

    return out;
}