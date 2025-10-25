// nsdhw_25au/hw3/wei0107/matrix_ops.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <string>
#include <immintrin.h>

namespace py = pybind11;

// ---- cblas dgemm (row-major) ----
extern "C" {
    void cblas_dgemm(const int Order, const int TransA, const int TransB,
                     const int M, const int N, const int K,
                     const double alpha, const double *A, const int lda,
                     const double *B, const int ldb,
                     const double beta, double *C, const int ldc);
}
static constexpr int CblasRowMajor = 101;
static constexpr int CblasNoTrans  = 111;

// ---------------- Matrix ----------------
class Matrix {
public:
    Matrix() : r_(0), c_(0) {}
    Matrix(size_t r, size_t c) : r_(r), c_(c), data_(r*c, 0.0) {}

    size_t rows() const { return r_; }
    size_t cols() const { return c_; }

    double* data() { return data_.data(); }
    const double* data() const { return data_.data(); }

    double& operator()(size_t i, size_t j) { return data_[i*c_ + j]; }
    double  operator()(size_t i, size_t j) const { return data_[i*c_ + j]; }

    void fill(double v) { std::fill(data_.begin(), data_.end(), v); }

    py::buffer_info buffer_info() {
        return py::buffer_info(
            data_.data(),
            (py::ssize_t)sizeof(double),
            std::string(py::format_descriptor<double>::format()),
            (py::ssize_t)2,
            std::vector<py::ssize_t>{(py::ssize_t)r_, (py::ssize_t)c_},
            std::vector<py::ssize_t>{(py::ssize_t)(c_*sizeof(double)), (py::ssize_t)sizeof(double)}
        );
    }

    py::array_t<double> to_numpy() const {
        return py::array_t<double>(
            {(py::ssize_t)r_, (py::ssize_t)c_},
            {(py::ssize_t)(c_*sizeof(double)), (py::ssize_t)sizeof(double)},
            data_.data()
        );
    }

    static Matrix from_numpy(const py::array_t<double, py::array::c_style | py::array::forcecast>& arr) {
        if (arr.ndim() != 2) throw std::runtime_error("expected 2D numpy array");
        size_t r = (size_t)arr.shape(0), c = (size_t)arr.shape(1);
        Matrix M(r, c);
        std::memcpy(M.data(), arr.data(), sizeof(double)*r*c);
        return M;
    }

private:
    size_t r_, c_;
    std::vector<double> data_;
};

// --------- naive ----------
Matrix multiply_naive(const Matrix& A, const Matrix& B) {
    if (A.cols() != B.rows()) throw std::runtime_error("dimension mismatch");
    const size_t M=A.rows(), K=A.cols(), N=B.cols();
    Matrix C(M,N);
    const double* a=A.data(); const double* b=B.data(); double* c=C.data();
    const size_t ldb=N, ldc=N;
    for (size_t i=0;i<M;++i){
        for(size_t k=0;k<K;++k){
            const double aik=a[i*K+k];
            const double* b_row=b+k*ldb;
            double* c_row=c+i*ldc;
            for(size_t j=0;j<N;++j){
                c_row[j]+=aik*b_row[j];
            }
        }
    }
    return C;
}

// --------- tiled ----------
Matrix multiply_tile(const Matrix& A, const Matrix& B, int tile=128){
    if (A.cols()!=B.rows()) throw std::runtime_error("dimension mismatch");
    if (tile<=0) tile=64;

    const size_t M=A.rows(), K=A.cols(), N=B.cols();
    Matrix C(M,N); 

    const double* a=A.data();
    const double* b=B.data();
    double* c=C.data();

    const size_t lda=K, ldb=N, ldc=N;
    const size_t vec_size = 4; // AVX double 數量

    // K 軸塊迴圈 (最外層)
    for(size_t k0=0;k0<K;k0+=tile){
        const size_t k_max=std::min(k0+(size_t)tile,K);

        // M 軸塊迴圈
        for(size_t i0=0;i0<M;i0+=tile){
            const size_t i_max=std::min(i0+(size_t)tile,M);
            
            // N 軸塊迴圈
            for(size_t j0=0;j0<N;j0+=tile){ 
                const size_t j_max=std::min(j0+(size_t)tile,N);
                
                // 註冊分塊 i 迴圈 (i+=2)
                for(size_t i=i0;i<i_max;i+=2){ 
                    const size_t i1 = std::min(i + 1, i_max - 1);

                    // 註冊分塊 j 迴圈 (j+=4)
                    const size_t j_vec_max = j0 + ((j_max - j0) & ~(vec_size - 1));
                    for(size_t j=j0;j<j_vec_max;j+=vec_size){
                        
                        // 1. 載入 C 的原有值到累積器 (這是關鍵修正點!)
                        //    C[i,j] = C[i,j] + A[i,k] * B[k,j]
                        __m256d c_acc0 = _mm256_loadu_pd(c + i*ldc + j); 
                        __m256d c_acc1 = _mm256_loadu_pd(c + i1*ldc + j);

                        // K 軸內迴圈 (k0 到 k_max)
                        for(size_t k=k0;k<k_max;++k){
                            
                            const double aik = a[i*lda + k];
                            const double a1k = a[i1*lda + k];
                            
                            // 載入 B 的元素
                            const double* __restrict b_row = b + k*ldb + j; 
                            __m256d b_vec = _mm256_loadu_pd(b_row); 
                            
                            // 執行 FMA 累積
                            c_acc0 = _mm256_fmadd_pd(_mm256_set1_pd(aik), b_vec, c_acc0); 
                            c_acc1 = _mm256_fmadd_pd(_mm256_set1_pd(a1k), b_vec, c_acc1);
                        }

                        // 寫回 C 矩陣
                        _mm256_storeu_pd(c + i*ldc + j, c_acc0);
                        _mm256_storeu_pd(c + i1*ldc + j, c_acc1);
                    }

                    // 處理 j 迴圈剩餘部分 (非向量化)
                    for(size_t j=j_vec_max; j<j_max; ++j){
                        
                        double sum0 = c[i*ldc + j]; // 載入 C 原有值
                        double sum1 = c[i1*ldc + j];

                        for(size_t k=k0;k<k_max;++k){
                            sum0 += a[i*lda + k] * b[k*ldb + j];
                            sum1 += a[i1*lda + k] * b[k*ldb + j];
                        }

                        c[i*ldc + j] = sum0; // 寫回 C
                        c[i1*ldc + j] = sum1;
                    }
                } // 結束 i 迴圈
            } // 結束 j0 迴圈
        } // 結束 i0 迴圈
    } // 結束 k0 迴圈 (最外層)

    return C;
}

// --------- mkl (dgemm) ----------
Matrix multiply_mkl(const Matrix& A, const Matrix& B, double alpha=1.0, double beta=0.0){
    if (A.cols()!=B.rows()) throw std::runtime_error("dimension mismatch");
    const int M=(int)A.rows(), K=(int)A.cols(), N=(int)B.cols();
    Matrix C(M,N);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M,N,K, alpha,
                A.data(), K,
                B.data(), N,
                beta, C.data(), N);
    return C;
}

PYBIND11_MODULE(_matrix, m){
    m.doc()="HW3 matrix multiply: naive / tiled / DGEMM";

    py::class_<Matrix>(m,"Matrix",py::buffer_protocol())
        .def(py::init<>())
        .def(py::init<size_t,size_t>())
        // validate.py 需要的屬性名稱
        .def_property_readonly("rows",&Matrix::rows)
        .def_property_readonly("cols",&Matrix::cols)
        .def_property_readonly("nrow",&Matrix::rows)
        .def_property_readonly("ncol",&Matrix::cols)
        .def("fill",&Matrix::fill)
        .def("to_numpy",&Matrix::to_numpy)
        .def_static("from_numpy",&Matrix::from_numpy)
        .def_buffer(&Matrix::buffer_info)
        // Python 索引：A[i, j]
        .def("__getitem__", [](const Matrix& M, std::pair<size_t,size_t> ij){
            return M(ij.first, ij.second);
        })
        .def("__setitem__", [](Matrix& M, std::pair<size_t,size_t> ij, double v){
            M(ij.first, ij.second) = v;
        })
        .def("__eq__", [](const Matrix& A, const Matrix& B){
            if (A.rows()!=B.rows() || A.cols()!=B.cols()) return false;
            const size_t n = A.rows()*A.cols();
            const double* ad = A.data();
            const double* bd = B.data();
            for (size_t i=0;i<n;++i){
                if (ad[i] != bd[i]) return false;
            }
            return true;
        });

    m.def("multiply_naive",&multiply_naive);
    m.def("multiply_tile",&multiply_tile, py::arg("A"), py::arg("B"), py::arg("tile")=128);
    m.def("multiple_tile",&multiply_tile, py::arg("A"), py::arg("B"), py::arg("tile")=128); // 兼容命名
    m.def("multiply_mkl",&multiply_mkl, py::arg("A"), py::arg("B"), py::arg("alpha")=1.0, py::arg("beta")=0.0);
}
