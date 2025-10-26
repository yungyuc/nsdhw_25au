#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <cstddef>
#include <cmath>

const double EPSILON = 1e-9;

class Matrix
{

public:
    Matrix(size_t n_row, size_t n_col)
        : rows_(n_row), cols_(n_col)
    {
        if (n_row == 0 || n_col == 0)
        {
            throw std::invalid_argument("Matrix dimensions must be greater than zero");
        }
        data_ = std::make_unique<double[]>(n_row * n_col);
    }

    Matrix(const std::vector<std::vector<double>> &data)
    {
        rows_ = data.size();
        if (rows_ == 0)
        {
            throw std::invalid_argument("Matrix must have at least one row");
        }

        cols_ = data[0].size();
        if (cols_ == 0)
        {
            throw std::invalid_argument("Matrix must have at least one column");
        }

        data_ = std::make_unique<double[]>(rows_ * cols_);
        for (size_t i = 0; i < rows_; ++i)
        {
            if (data[i].size() != cols_)
            {
                throw std::invalid_argument("All rows must have the same number of columns");
            }
            for (size_t j = 0; j < cols_; ++j)
            {
                (*this)(i, j) = data[i][j];
            }
        }
    }

    ~Matrix() = default;

    // Copy constructor
    Matrix(const Matrix &other)
        : rows_(other.rows_), cols_(other.cols_), data_(std::make_unique<double[]>(other.rows_ * other.cols_))
    {
        std::copy(other.data_.get(), other.data_.get() + (other.rows_ * other.cols_), data_.get());
    }

    // Move constructor
    Matrix(Matrix &&other) noexcept
        : rows_(other.rows_), cols_(other.cols_), data_(std::move(other.data_))
    {
        other.rows_ = 0;
        other.cols_ = 0;
    }

    // Copy assignment
    Matrix &operator=(const Matrix &other)

    {
        if (this != &other)
        {
            rows_ = other.rows_;
            cols_ = other.cols_;
            data_ = std::make_unique<double[]>(other.rows_ * other.cols_);
            std::copy(other.data_.get(), other.data_.get() + (other.rows_ * other.cols_), data_.get());
        }
        return *this;
    }

    // Move assignment
    Matrix &operator=(Matrix &&other) noexcept
    {
        if (this != &other)
        {
            rows_ = other.rows_;
            cols_ = other.cols_;
            data_ = std::move(other.data_);
            other.rows_ = 0;
            other.cols_ = 0;
        }
        return *this;
    }

    // Const access operator
    double operator()(size_t i, size_t j) const
    {
        return data_[i * cols_ + j];
    }

    // Non-const access operator
    double &operator()(size_t i, size_t j)
    {
        return data_[i * cols_ + j];
    }

    bool operator==(const Matrix &other) const
    {
        if (rows_ != other.rows_ || cols_ != other.cols_)
        {
            return false;
        }

        for (size_t i = 0; i < rows_ * cols_; ++i)
        {
            if (std::abs(data_[i] - other.data_[i]) > EPSILON)
            {
                return false;
            }
        }
        return true;
    }

    std::string to_string() const
    {
        std::string result;
        for (size_t i = 0; i < rows_; ++i)
        {
            for (size_t j = 0; j < cols_; ++j)
            {
                result += std::to_string((*this)(i, j)) + " ";
            }
            result += "\n";
        }
        return result;
    }

    double *data() { return data_.get(); }

    const double *data() const { return data_.get(); }

    size_t get_rows() const { return rows_; }
    size_t get_cols() const { return cols_; }

private:
    size_t rows_;
    size_t cols_;
    std::unique_ptr<double[]> data_;
};

Matrix multiply_naive(const Matrix &A, const Matrix &B);
Matrix multiply_tile(const Matrix &A, const Matrix &B, size_t tile_size);
Matrix multiply_mkl(const Matrix &A, const Matrix &B);

#endif // MATRIX_HPP