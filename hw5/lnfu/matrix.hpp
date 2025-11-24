#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

const double EPSILON = 1e-9;

class Matrix {
 public:
  Matrix(size_t n_row, size_t n_col)
      : rows_(n_row), cols_(n_col), data_(n_row * n_col) {
    if (n_row == 0 || n_col == 0) {
      throw std::invalid_argument(
          "Matrix dimensions must be greater than zero");
    }
  }

  Matrix(const std::vector<std::vector<double>>& data) {
    rows_ = data.size();
    if (rows_ == 0) {
      throw std::invalid_argument("Matrix must have at least one row");
    }
    cols_ = data[0].size();
    if (cols_ == 0) {
      throw std::invalid_argument("Matrix must have at least one column");
    }

    data_.resize(rows_ * cols_);

    for (size_t i = 0; i < rows_; ++i) {
      if (data[i].size() != cols_) {
        throw std::invalid_argument(
            "All rows must have the same number of columns");
      }
      for (size_t j = 0; j < cols_; ++j) {
        (*this)(i, j) = data[i][j];
      }
    }
  }

  ~Matrix() = default;
  Matrix(const Matrix& other) = default;
  Matrix(Matrix&& other) noexcept = default;
  Matrix& operator=(const Matrix& other) = default;
  Matrix& operator=(Matrix&& other) noexcept = default;

  // Const access operator
  double operator()(size_t i, size_t j) const { return data_[i * cols_ + j]; }

  // Non-const access operator
  double& operator()(size_t i, size_t j) { return data_[i * cols_ + j]; }

  bool operator==(const Matrix& other) const {
    if (rows_ != other.rows_ || cols_ != other.cols_) {
      return false;
    }
    for (size_t i = 0; i < rows_ * cols_; ++i) {
      if (std::abs(data_[i] - other.data_[i]) > EPSILON) {
        return false;
      }
    }
    return true;
  }

  std::string to_string() const {
    std::string result;
    for (size_t i = 0; i < rows_; ++i) {
      for (size_t j = 0; j < cols_; ++j) {
        result += std::to_string((*this)(i, j)) + " ";
      }
      result += "\n";
    }
    return result;
  }

  double* data() { return data_.data(); }
  const double* data() const { return data_.data(); }

  size_t nrow() const { return rows_; }
  size_t ncol() const { return cols_; }

 private:
  size_t rows_;
  size_t cols_;
  std::vector<double> data_;
};

Matrix multiply_naive(const Matrix& A, const Matrix& B);
Matrix multiply_tile(const Matrix& A, const Matrix& B, size_t tile_size);
Matrix multiply_mkl(const Matrix& A, const Matrix& B);

#endif  // MATRIX_HPP