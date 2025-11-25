#include<iostream>
#include<algorithm>
#include<vector>
#include <cblas.h>

class Matrix {
public:

    Matrix(size_t nrow, size_t ncol)
    : m_nrow(nrow), m_ncol(ncol), m_buffer(nrow * ncol)
    {}

    double operator()(size_t row, size_t col) const
    {
        return m_buffer[row*m_ncol + col];
    }
    double & operator()(size_t row, size_t col)
    {
        return m_buffer[row*m_ncol + col];
    }

    size_t nrow() const {return m_nrow;}
    size_t ncol() const {return m_ncol;}

    const double* data() const { return m_buffer.data(); }
    double* data() { return m_buffer.data(); }
    
    bool operator==(Matrix const & other) const {
        if (m_nrow != other.m_nrow || m_ncol != other.m_ncol) return false;
        return m_buffer == other.m_buffer;
    }
private:

    size_t m_nrow;
    size_t m_ncol;
    std::vector<double> m_buffer;
};

/**
 * Populate the matrix object.
 */
void populate(Matrix & matrix)
{
    for (size_t i=0; i<matrix.nrow(); ++i) // the i-th row
    {
        for (size_t j=0; j<matrix.ncol(); ++j) // the j-th column
        {
            matrix(i, j) = 1;
        }
    }
}

Matrix multiply_naive(const Matrix & mat1, const Matrix & mat2) {
    Matrix mat_result(mat1.nrow(), mat2.ncol());

    for (size_t i = 0; i < mat_result.nrow(); ++i)
    {
        for (size_t k = 0; k < mat_result.ncol(); ++k)
        {
            double v = 0;
            for (size_t j = 0; j < mat1.ncol(); ++j)
            {
                v += mat1(i, j) * mat2(j, k);
            }
            mat_result(i, k) = v;
        }
    }

    return mat_result;
};

Matrix multiply_tile(const Matrix & mat1, const Matrix & mat2, size_t T) {
    size_t row = mat1.nrow();
    size_t mid = mat1.ncol();
    size_t col = mat2.ncol();
    Matrix mat_result(row, col);

    for (size_t i = 0; i < row; i += T)
    {
        for (size_t j = 0; j < col; j+= T)
        {
            for (size_t k = 0; k < mid; k += T)
            {
                size_t i_max = std::min(i + T, row);
                size_t j_max = std::min(j + T, col);
                size_t k_max = std::min(k + T, mid);
                for (size_t ii = i; ii < i_max; ++ii)
                {
                    for (size_t jj = j; jj < j_max; ++jj)
                    {
                        double sum = mat_result(ii, jj);
                        for (size_t kk = k; kk < k_max; ++kk)
                        {
                            sum += mat1(ii, kk) * mat2(kk, jj);
                        }
                        mat_result(ii, jj) = sum;
                    }
                }
            }
        }
    }

    return mat_result;
};


Matrix multiply_mkl(const Matrix & mat1, const Matrix & mat2) {
    const int row = mat1.nrow();
    const int mid = mat1.ncol();
    const int col = mat2.ncol();

    Matrix mat_result(mat1.nrow(), mat2.ncol());

    cblas_dgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        row, col, mid,
        1.0,
        mat1.data(), mid,
        mat2.data(), col,
        0.0,
        mat_result.data(), col
    );
    
    return mat_result;
};