#ifndef MATRIX_HPP_
#define MATRIX_HPP_

#include <cstddef>
#include <vector>
#include <stdexcept>

class Matrix {
public:
    Matrix(std::size_t r, std::size_t c)
        : nrow_(r), ncol_(c), buf_(r * c, 0.0)
    {}

    std::size_t nrow() const noexcept { return nrow_; }
    std::size_t ncol() const noexcept { return ncol_; }

    // element access (with bound check)
    double& operator()(std::size_t i, std::size_t j) {
        bounds_check(i, j);
        return buf_[i * ncol_ + j];
    }

    const double& operator()(std::size_t i, std::size_t j) const {
        bounds_check(i, j);
        return buf_[i * ncol_ + j];
    }

    // pointer access for BLAS / MKL
    double* data() noexcept { return buf_.data(); }
    const double* data() const noexcept { return buf_.data(); }

    // equality: same shape + all elements equal
    bool operator==(const Matrix& other) const noexcept {
        if (nrow_ != other.nrow_ || ncol_ != other.ncol_) return false;
        return buf_ == other.buf_;
    }

private:
    std::size_t nrow_;
    std::size_t ncol_;
    std::vector<double> buf_;

    void bounds_check(std::size_t i, std::size_t j) const {
        if (i >= nrow_ || j >= ncol_) {
            throw std::out_of_range("Matrix index out of range");
        }
    }

    friend Matrix multiply_naive(const Matrix&, const Matrix&);
    friend Matrix multiply_tile(const Matrix&, const Matrix&, int);
    friend Matrix multiply_mkl(const Matrix&, const Matrix&);
};

// ==== function declarations ====
Matrix multiply_naive(const Matrix& A, const Matrix& B);
Matrix multiply_tile(const Matrix& A, const Matrix& B, int tsize);
Matrix multiply_mkl(const Matrix& A, const Matrix& B);

#endif // MATRIX_HPP_
