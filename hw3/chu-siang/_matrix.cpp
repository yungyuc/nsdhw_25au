#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <vector>
#include <stdexcept>
#include <cstddef>
#include <cassert>

namespace py = pybind11;

#ifdef HAVE_MKL
  #include <mkl.h>
  #include <mkl_cblas.h>
#endif

struct Matrix {
    std::size_t nrow;
    std::size_t ncol;
    std::vector<double> buf; // row-major

    Matrix(std::size_t r, std::size_t c)
        : nrow(r), ncol(c), buf(r * c, 0.0) {}

    const double* data() const { return buf.data(); }
    double*       data()       { return buf.data(); }

    inline void bounds_check(std::ptrdiff_t i, std::ptrdiff_t j) const {
        if (i < 0 || j < 0 || static_cast<std::size_t>(i) >= nrow || static_cast<std::size_t>(j) >= ncol) {
            throw py::index_error("index out of range");
        }
    }

    double get(std::ptrdiff_t i, std::ptrdiff_t j) const {
        bounds_check(i, j);
        return buf[static_cast<std::size_t>(i) * ncol + static_cast<std::size_t>(j)];
    }
    void set(std::ptrdiff_t i, std::ptrdiff_t j, double v) {
        bounds_check(i, j);
        buf[static_cast<std::size_t>(i) * ncol + static_cast<std::size_t>(j)] = v;
    }

    bool equals(const Matrix& other) const {
        if (nrow != other.nrow || ncol != other.ncol) return false;
        for (std::size_t i = 0; i < buf.size(); ++i) {
            if (buf[i] != other.buf[i]) return false;
        }
        return true;
    }
};

static inline void check_mul_dims(std::size_t a_m, std::size_t a_k,
                                  std::size_t b_m, std::size_t b_n) {
    if (a_k != b_m) {
        throw py::value_error("shape mismatch: A(M,K) x B(K,N) required");
    }
    (void)a_m; (void)b_n;
}

static Matrix multiply_naive(const Matrix& A, const Matrix& B) {
    const std::size_t M = A.nrow, K = A.ncol;
    const std::size_t K2 = B.nrow, N = B.ncol;
    check_mul_dims(M, K, K2, N);

    Matrix C(M, N);
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            const double aik = A.buf[i * A.ncol + k];
            if (aik == 0.0) continue;
            for (std::size_t j = 0; j < N; ++j) {
                C.buf[i * N + j] += aik * B.buf[k * N + j];
            }
        }
    }
    return C;
}

static Matrix multiply_tile(const Matrix& A, const Matrix& B, int tsize) {
    if (tsize <= 0) {
        throw py::value_error("tsize must be positive");
    }
    const std::size_t T = static_cast<std::size_t>(tsize);

    const std::size_t M = A.nrow, K = A.ncol;
    const std::size_t K2 = B.nrow, N = B.ncol;
    check_mul_dims(M, K, K2, N);

    Matrix C(M, N);

    for (std::size_t ii = 0; ii < M; ii += T) {
        for (std::size_t kk = 0; kk < K; kk += T) {
            for (std::size_t jj = 0; jj < N; jj += T) {
                const std::size_t iimax = std::min(ii + T, M);
                const std::size_t kkmax = std::min(kk + T, K);
                const std::size_t jjmax = std::min(jj + T, N);
                for (std::size_t i = ii; i < iimax; ++i) {
                    for (std::size_t k = kk; k < kkmax; ++k) {
                        const double aik = A.buf[i * A.ncol + k];
                        if (aik == 0.0) continue;
                        double* __restrict cptr = &C.buf[i * N + jj];
                        const double* __restrict bptr = &B.buf[k * N + jj];
                        for (std::size_t j = jj; j < jjmax; ++j) {
                            *cptr++ += aik * (*bptr++);
                        }
                    }
                }
            }
        }
    }
    return C;
}

static Matrix multiply_mkl(const Matrix& A, const Matrix& B) {
#ifdef HAVE_MKL
    const std::size_t M = A.nrow, K = A.ncol;
    const std::size_t K2 = B.nrow, N = B.ncol;
    check_mul_dims(M, K, K2, N);

    Matrix C(M, N);

    // Row-major：lda=K, ldb=N, ldc=N
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                1.0,
                A.data(), static_cast<int>(K),
                B.data(), static_cast<int>(N),
                0.0,
                C.data(), static_cast<int>(N));
    return C;
#else
    return multiply_naive(A, B);
#endif
}

PYBIND11_MODULE(_matrix, m) {
    m.doc() = "Matrix + GEMM implementations (naive/tiled/MKL)";

    py::class_<Matrix>(m, "Matrix")
        .def(py::init<std::size_t, std::size_t>(),
             py::arg("nrow"), py::arg("ncol"))
        .def_property_readonly("nrow", [](const Matrix& self){ return self.nrow; })
        .def_property_readonly("ncol", [](const Matrix& self){ return self.ncol; })
        .def("__getitem__", [](const Matrix& self, py::tuple ij) {
            if (ij.size() != 2) throw py::index_error("need 2 indices (i, j)");
            auto i = ij[0].cast<std::ptrdiff_t>();
            auto j = ij[1].cast<std::ptrdiff_t>();
            return self.get(i, j);
        })
        .def("__setitem__", [](Matrix& self, py::tuple ij, double v) {
            if (ij.size() != 2) throw py::index_error("need 2 indices (i, j)");
            auto i = ij[0].cast<std::ptrdiff_t>();
            auto j = ij[1].cast<std::ptrdiff_t>();
            self.set(i, j, v);
        })
        .def("__eq__", [](const Matrix& a, const Matrix& b){ return a.equals(b); })
        .def("__ne__", [](const Matrix& a, const Matrix& b){ return !a.equals(b); })
        ;

    m.def("multiply_naive", &multiply_naive, "C = A * B (naive triple loop)",
          py::arg("A"), py::arg("B"));
    m.def("multiply_tile",  &multiply_tile,  "C = A * B (tiled)",
          py::arg("A"), py::arg("B"), py::arg("tsize"));
    m.def("multiply_mkl",   &multiply_mkl,   "C = A * B (MKL dgemm if available; fallback to naive otherwise)",
          py::arg("A"), py::arg("B"));
}

