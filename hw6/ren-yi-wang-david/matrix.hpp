#pragma once
#include <cstddef>
#include <vector>

class Matrix
{
public:
    size_t nrow, ncol;
    std::vector<double> buf;

    Matrix(size_t r, size_t c)
        : nrow(r), ncol(c), buf(r * c, 0.0) {}

    inline double &operator()(size_t r, size_t c)
    {
        return buf[r * ncol + c];
    }

    inline double operator()(size_t r, size_t c) const
    {
        return buf[r * ncol + c];
    }

    bool operator==(const Matrix &other) const
    {
        return nrow == other.nrow &&
               ncol == other.ncol &&
               buf == other.buf;
    }
};
