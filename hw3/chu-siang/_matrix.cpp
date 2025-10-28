// _matrix.cpp — pybind11 Matrix + GEMM (naive/tile/MKL-or-fallback)
#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cstddef>

namespace py = pybind11;

// ---------- Optional MKL/CBLAS detection (compile-time) ----------
#if defined(HAVE_MKL) && (HAVE_MKL+0)
  #if __has_include(<mkl.h>)
    #include <mkl.h>
    #define MATRIX_HAVE_MKL 1
  #elif __has_include(<cblas.h>)
    #include <cblas.h>
    #define MATRIX_HAVE_MKL 1
  #else
    #define MATRIX_HAVE_MKL 0
  #endif
#else
  #define MATRIX_HAVE_MKL 0
#endif

class Matrix {
public:
    Matrix(std::size_t r, std::size_t c)
        : nrow_(r), ncol_(c), data_(r*c, 0.0) {
        if (r == 0 || c == 0) throw std::invalid_argument("nrow/ncol must be positive");
    }
    std::size_t nrow() const { return nrow_; }
    std::size_t ncol() const { return ncol_; }

    double get(std::size_t i, std::size_t j) const {
        if (i >= nrow_ || j >= ncol_) throw std::out_of_range("index out of range");
        return data_[i*ncol_ + j];
    }
    void set(std::size_t i, std::size_t j, double v) {
        if (i >= nrow_ || j >= ncol_) throw std::out_of_range("index out of range");
        data_[i*ncol_ + j] = v;
    }

    bool operator==(const Matrix& o) const {
        return nrow_ == o.nrow_ && ncol_ == o.ncol_ && data_ == o.data_;
    }

    const double* raw() const { return data_.data(); }
    double*       raw()       { return data_.data(); }

private:
    std::size_t nrow_, ncol_;
    std::vector<double> data_;
};

// ---------------- naive (intentionally cache-unfriendly i-j-k) ----------------
Matrix multiply_naive(const Matrix& A, const Matrix& B) {
    if (A.ncol() != B.nrow()) throw std::invalid_argument("incompatible dimensions");
    const std::size_t M = A.nrow(), K = A.ncol(), N = B.ncol();
    Matrix C(M, N);
    const double* ad = A.raw();
    const double* bd = B.raw();
    double* cd = C.raw();

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < K; ++k) {
                sum += ad[i*K + k] * bd[k*N + j]; // stride-N through B's column
            }
            cd[i*N + j] = sum;
        }
    }
    return C;
}

// ---------------- tiled (blocked i0-k0-j0; contiguous within tiles) ----------------
Matrix multiply_tile(const Matrix& A, const Matrix& B, int tsize) {
    if (A.ncol() != B.nrow()) throw std::invalid_argument("incompatible dimensions");
    if (tsize <= 0) throw std::invalid_argument("tile size must be positive");
    const std::size_t M = A.nrow(), K = A.ncol(), N = B.ncol();
    Matrix C(M, N);

    const double* ad = A.raw();
    const double* bd = B.raw();
    double* cd = C.raw();

    const std::size_t TM = (std::size_t)tsize;
    const std::size_t TK = (std::size_t)tsize;
    const std::size_t TN = (std::size_t)tsize;

    for (std::size_t i0 = 0; i0 < M; i0 += TM) {
        const std::size_t i_max = std::min(i0 + TM, M);
        for (std::size_t k0 = 0; k0 < K; k0 += TK) {
            const std::size_t k_max = std::min(k0 + TK, K);
            for (std::size_t j0 = 0; j0 < N; j0 += TN) {
                const std::size_t j_max = std::min(j0 + TN, N);

                for (std::size_t i = i0; i < i_max; ++i) {
                    double* ci = &cd[i*N + j0];
                    for (std::size_t k = k0; k < k_max; ++k) {
                        const double aik = ad[i*K + k];
                        const double* bk = &bd[k*N + j0];
                        for (std::size_t j = j0; j < j_max; ++j) {
                            ci[j - j0] += aik * bk[j - j0]; // contiguous on tile
                        }
                    }
                }
            }
        }
    }
    return C;
}

// ---------------- MKL (or fallback) ----------------
Matrix multiply_mkl(const Matrix& A, const Matrix& B) {
    if (A.ncol() != B.nrow()) throw std::invalid_argument("incompatible dimensions");
#if MATRIX_HAVE_MKL
    const std::size_t M = A.nrow(), K = A.ncol(), N = B.ncol();
    Matrix C(M, N);
    const double alpha = 1.0, beta = 0.0;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                (int)M, (int)N, (int)K,
                alpha,
                A.raw(), (int)A.ncol(),
                B.raw(), (int)B.ncol(),
                beta,
                C.raw(), (int)C.ncol());
    return C;
#else
    // No MKL/CBLAS headers: fallback for correctness & portability
    return multiply_naive(A, B);
#endif
}

// ---------------- pybind11 module (signed indices for safe bound checks) ----------------
PYBIND11_MODULE(_matrix, m) {
    m.doc() = "Matrix class + GEMM (naive/tile/MKL)";

    py::class_<Matrix>(m, "Matrix")
        .def(py::init<std::size_t, std::size_t>(), py::arg("nrow"), py::arg("ncol"))
        .def_property_readonly("nrow", &Matrix::nrow)
        .def_property_readonly("ncol", &Matrix::ncol)
        .def("__getitem__", [](const Matrix& self, py::tuple ij) {
            if (ij.size() != 2) throw std::out_of_range("index must be (i,j)");
            py::ssize_t is = ij[0].cast<py::ssize_t>();
            py::ssize_t js = ij[1].cast<py::ssize_t>();
            if (is < 0 || js < 0) throw std::out_of_range("index out of range");
            std::size_t i = static_cast<std::size_t>(is);
            std::size_t j = static_cast<std::size_t>(js);
            return self.get(i, j);
        })
        .def("__setitem__", [](Matrix& self, py::tuple ij, double v) {
            if (ij.size() != 2) throw std::out_of_range("index must be (i,j)");
            py::ssize_t is = ij[0].cast<py::ssize_t>();
            py::ssize_t js = ij[1].cast<py::ssize_t>();
            if (is < 0 || js < 0) throw std::out_of_range("index out of range");
            std::size_t i = static_cast<std::size_t>(is);
            std::size_t j = static_cast<std::size_t>(js);
            self.set(i, j, v);
        })
        .def(py::self == py::self)
        .def("__repr__", [](const Matrix& mtx) {
            return "Matrix(" + std::to_string(mtx.nrow()) + ", " + std::to_string(mtx.ncol()) + ")";
        });

    m.def("multiply_naive", &multiply_naive, "Naive matrix-matrix multiply (i-j-k)");
    m.def("multiply_tile",  &multiply_tile,  py::arg("A"), py::arg("B"), py::arg("tsize"),
          "Tiled matrix-matrix multiply (tsize > 0)");
    m.def("multiply_mkl",   &multiply_mkl,   "DGEMM via MKL/CBLAS when available; fallback otherwise");
}

