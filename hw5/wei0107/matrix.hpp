// matrix.hpp
#pragma once

#include <cstddef>
#include <vector>
#include <stdexcept>
#include <algorithm>

class Matrix {
public:
    Matrix() : r_(0), c_(0), data_() {}
    Matrix(std::size_t r, std::size_t c)
        : r_(r), c_(c), data_(r * c, 0.0)
    {}

    // 這兩個是原本的命名
    std::size_t rows() const { return r_; }
    std::size_t cols() const { return c_; }

    // 這兩個是 mod.cpp 期待的接口名稱
    std::size_t nrow() const { return r_; }
    std::size_t ncol() const { return c_; }

    double* data() { return data_.data(); }
    const double* data() const { return data_.data(); }

    double& operator()(std::size_t i, std::size_t j) {
        return data_[i * c_ + j];
    }
    double operator()(std::size_t i, std::size_t j) const {
        return data_[i * c_ + j];
    }

    void fill(double v) { std::fill(data_.begin(), data_.end(), v); }

    bool operator==(Matrix const& other) const {
        return r_ == other.r_ && c_ == other.c_ && data_ == other.data_;
    }

private:
    std::size_t r_, c_;
    std::vector<double> data_;
};

// 這三個函式會被 mod.cpp 綁到 Python：
Matrix multiply_naive(Matrix const& A, Matrix const& B);
Matrix multiply_tile(Matrix const& A, Matrix const& B, std::size_t tile);
Matrix multiply_mkl  (Matrix const& A, Matrix const& B);
