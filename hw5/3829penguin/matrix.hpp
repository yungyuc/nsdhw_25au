#include <cstddef>
#include <vector>
#include <stdexcept>
#include <algorithm>
#ifdef USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
#include <iostream>
#include <string>
#include <sstream>
#include <cstring>

class Matrix {

public:
    size_t nrow_;
    size_t ncol_;
    Matrix(size_t n_row, size_t n_col)
    {
        nrow_ = n_row;
        ncol_ = n_col;
        size_t nelement = nrow_ * ncol_;
        m_buffer = new double[nelement];
        memset(m_buffer, 0, nrow_ * ncol_ * sizeof(double));
    }
    size_t nrow() const {return nrow_;}
    size_t ncol() const {return ncol_;}
    double   operator() (size_t row, size_t col) const
    {
        return m_buffer[row* ncol() + col];
    }
    double & operator() (size_t row, size_t col)
    {
        return m_buffer[row* ncol() + col];
    }
	void load_buffer(double* input) {
		memcpy(m_buffer, input, nrow() * ncol() * sizeof(double));
	}

    friend bool operator == (const Matrix& A, const Matrix& B){
        if ((A.nrow()!=B.nrow()) || (A.ncol()!=B.ncol())) return false;
        for (size_t i=0;i<A.nrow()*A.ncol();i++){
            if (A.m_buffer[i] != B.m_buffer[i]) return false;
        }
        return true;
    }

    double* get_data() const {
        return m_buffer;
    }

private:
    double * m_buffer;

};

Matrix multiply_naive(Matrix& matrix_a, Matrix& matrix_b){
    
    Matrix result(matrix_a.nrow(), matrix_b.ncol());
    for (size_t row = 0; row < matrix_a.nrow(); row++){
        for (size_t col = 0; col < matrix_b.ncol(); col++){
            for (size_t i = 0; i < matrix_a.ncol(); i++){
                result(row,col) += matrix_a(row,i) * matrix_b(i,col);
            }
        }
    }
    return result;
}

Matrix multiply_tile(Matrix& matrix_a, Matrix& matrix_b, size_t tsize){

    Matrix result(matrix_a.nrow(), matrix_b.ncol());
    for (size_t tile_row_start=0; tile_row_start<matrix_a.nrow(); tile_row_start+=tsize){
        size_t tile_row_end = tile_row_start+tsize;
        tile_row_end = (tile_row_end>matrix_a.nrow())?matrix_a.nrow():tile_row_end;

        for (size_t tile_col_start=0; tile_col_start<matrix_b.ncol(); tile_col_start+=tsize){
            size_t tile_col_end = tile_col_start+tsize;
            tile_col_end = (tile_col_end>matrix_b.ncol())?matrix_b.ncol():tile_col_end;

            for (size_t t=0; t<matrix_a.ncol(); t++){
                for (size_t i=tile_row_start; i<tile_row_end;i++){
                    for (size_t j=tile_col_start; j<tile_col_end;j++){
                        result(i,j) += matrix_a(i,t)*matrix_b(t,j);
                    }
                }
            }
        }
    }
    return result;
}

Matrix multiply_mkl(Matrix& matrix_a, Matrix& matrix_b) {

	double* C = new double[matrix_a.nrow() * matrix_b.ncol()];

	int m = matrix_a.nrow();
	int k = matrix_a.ncol();
	int n = matrix_b.ncol();
	double alpha = 1.0, beta = 0.0;

	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
				m, n, k, alpha, matrix_a.get_data(), k, matrix_b.get_data(), n, beta, C, n);

	Matrix res(matrix_a.nrow(), matrix_b.ncol());
    res.load_buffer(C);

	delete[] C;

	return res;
}