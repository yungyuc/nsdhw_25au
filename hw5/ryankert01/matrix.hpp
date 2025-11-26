#pragma once

#include <cstddef>
#include <vector>

class Matrix
{
public:
    Matrix(std::size_t rows, std::size_t cols);

    std::size_t nrow() const;
    std::size_t ncol() const;

    double &operator()(std::size_t r, std::size_t c);
    double const &operator()(std::size_t r, std::size_t c) const;

    bool operator==(Matrix const &other) const;
    bool operator!=(Matrix const &other) const;

    Matrix transpose() const;

    std::vector<double> &data();
    std::vector<double> const &data() const;

private:
    std::size_t m_rows;
    std::size_t m_cols;
    std::vector<double> m_data;
};

Matrix multiply_naive(Matrix const &a, Matrix const &b);
Matrix multiply_tile(Matrix const &a, Matrix const &b, std::size_t tile_size);
Matrix multiply_mkl(Matrix const &a, Matrix const &b);
