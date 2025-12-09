#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

class Mat {
public:
    Mat(std::size_t r, std::size_t c)
        : r_(r), c_(c), buf_(r * c) {}

    Mat(const Mat& o)
        : r_(o.r_), c_(o.c_), buf_(o.buf_) {}

    Mat(Mat&& o) noexcept
        : r_(o.r_), c_(o.c_), buf_(std::move(o.buf_)) {}

    Mat& operator=(const Mat& o) {
        if (this != &o) {
            r_ = o.r_;
            c_ = o.c_;
            buf_ = o.buf_;
        }
        return *this;
    }

    Mat& operator=(Mat&& o) noexcept {
        if (this != &o) {
            r_ = o.r_;
            c_ = o.c_;
            buf_ = std::move(o.buf_);
        }
        return *this;
    }

    bool operator==(const Mat& o) const {
        return r_ == o.r_ && c_ == o.c_ && buf_ == o.buf_;
    }

    double operator()(std::size_t i, std::size_t j) const {
        return buf_[i * c_ + j];
    }

    double& operator()(std::size_t i, std::size_t j) {
        return buf_[i * c_ + j];
    }

    std::size_t nrow() const { return r_; }
    std::size_t ncol() const { return c_; }

    pybind11::array_t<double> array() {
        return pybind11::array_t<double>(
            {r_, c_},
            {sizeof(double) * c_, sizeof(double)},
            buf_.data(),
            pybind11::cast(this)
        );
    }

    std::size_t r_, c_;
    std::vector<double> buf_;
};

inline Mat mul_naive(const Mat& A, const Mat& B) {
    Mat M(A.nrow(), B.ncol());
    auto R = A.nrow();
    auto C = B.ncol();
    auto K = A.ncol();

    for (std::size_t i = 0; i < R; ++i) {
        for (std::size_t j = 0; j < C; ++j) {
            double x = 0;
            const double* pa = &A.buf_[i * K];
            const double* pb = &B.buf_[j];
            for (std::size_t k = 0; k < K; ++k) {
                x += pa[k] * pb[k * C];
            }
            M(i, j) = x;
        }
    }
    return M;
}

inline Mat mul_mkl(const Mat& A, const Mat& B) {
    return mul_naive(A, B);
}
