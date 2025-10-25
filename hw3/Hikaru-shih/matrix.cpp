#include <vector>
#include <cstddef>
#include <cassert>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <limits>
#include <cblas.h>
#include <cmath>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace py = pybind11;

class Matrix {
public:
    Matrix() : rows_(0), cols_(0) {}

    Matrix(std::size_t r, std::size_t c, double init = 0.0)
        : rows_(r), cols_(c), data_(r * c, init) {}

    static Matrix Random(std::size_t r, std::size_t c,
                         unsigned seed = 0xC0FFEEu,
                         double lo = -1.0, double hi = 1.0) {
        Matrix m(r, c);
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> dist(lo, hi);
        for (double &x : m.data_) x = dist(rng);
        return m;
    }

    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    std::size_t size() const noexcept { return data_.size(); }

    double* data() noexcept { return data_.data(); }
    const double* data() const noexcept { return data_.data(); }

    void fill(double v) { std::fill(data_.begin(), data_.end(), v); }

    // Element access (row-major)
    double& operator()(std::size_t i, std::size_t j) noexcept {
        return data_[i * cols_ + j];
    }
    const double& operator()(std::size_t i, std::size_t j) const noexcept {
        return data_[i * cols_ + j];
    }

    // Helpful for unit tests
    bool approx_equal(const Matrix& other, double tol = 1e-9) const {
        if (rows_ != other.rows_ || cols_ != other.cols_) return false;
        for (std::size_t i = 0; i < size(); ++i) {
            double a = data_[i], b = other.data_[i];
            double diff = std::abs(a - b);
            double scale = std::max({1.0, std::abs(a), std::abs(b)});
            if (diff > tol * scale) return false;
        }
        return true;
    }

private:
    std::size_t rows_, cols_;
    std::vector<double> data_;
};

static inline void check_mul_dims(const Matrix& A, const Matrix& B) {
    if (A.cols() != B.rows()) {
        throw std::invalid_argument("Dimension mismatch: A.cols()!=B.rows().");
    }
}

Matrix multiply_naive(const Matrix& A, const Matrix& B) {
    check_mul_dims(A, B);
    Matrix C(A.rows(), B.cols(), 0.0);

    const std::size_t M = A.rows();
    const std::size_t K = A.cols();
    const std::size_t N = B.cols();

    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t k = 0; k < K; ++k) {
            const double aik = A(i, k);
            const std::size_t bk = k * N;
            double* ci = C.data() + i * N;
            const double* bptr = B.data() + bk;
            for (std::size_t j = 0; j < N; ++j) {
                ci[j] += aik * bptr[j];
            }
        }
    }
    return C;
}

Matrix multiply_tile(const Matrix& A, const Matrix& B, int block) {
    check_mul_dims(A, B);
    if (block <= 0) block = 64;

    const size_t M = A.rows();
    const size_t K = A.cols();
    const size_t N = B.cols();

    Matrix C(M, N, 0.0);

    std::vector<double> BT((size_t)N * K);
    {
        const double* Bd = B.data();
        for (size_t k = 0; k < K; ++k) {
            const double* bk = Bd + k * N;
            for (size_t j = 0; j < N; ++j) {
                BT[j * K + k] = bk[j];
            }
        }
    }

    const double* Ad = A.data();
    double*       Cd = C.data();

    for (size_t ii = 0; ii < M; ii += (size_t)block) {
        const size_t i_max = std::min(ii + (size_t)block, M);
        for (size_t jj = 0; jj < N; jj += (size_t)block) {
            const size_t j_max = std::min(jj + (size_t)block, N);
            for (size_t kk = 0; kk < K; kk += (size_t)block) {
                const size_t k_max = std::min(kk + (size_t)block, K);
                const size_t len   = k_max - kk;

                for (size_t i = ii; i < i_max; ++i) {
                    double* ci = Cd + i * N;
                    const double* ai = Ad + i * K;
                    const double* ak = ai + kk;

                    for (size_t j = jj; j < j_max; ++j) {
                        const double* btj = &BT[j * K + kk];
                        double acc = ci[j];
                        for (size_t t = 0; t < len; ++t) {
                            acc += ak[t] * btj[t];
                        }
                        ci[j] = acc;
                    }
                }
            }
        }
    }

    return C;
}

Matrix multiply_mkl(const Matrix& A, const Matrix& B) {
    check_mul_dims(A, B);
    Matrix C(A.rows(), B.cols(), 0.0);

    const int M = static_cast<int>(A.rows());
    const int K = static_cast<int>(A.cols());
    const int N = static_cast<int>(B.cols());

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0,
                A.data(), K,
                B.data(), N,
                0.0,
                C.data(), N);
    return C;
}


PYBIND11_MODULE(_matrix, m) {
    py::class_<Matrix>(m, "Matrix")
        .def(py::init<std::size_t, std::size_t>(), py::arg("nrow"), py::arg("ncol"))
        .def_property_readonly("nrow", &Matrix::rows)
        .def_property_readonly("ncol", &Matrix::cols)
        .def("__getitem__", [](const Matrix& m, std::pair<std::size_t,std::size_t> ij){
            return m(ij.first, ij.second);
        })
        .def("__setitem__", [](Matrix& m, std::pair<std::size_t,std::size_t> ij, double v){
            m(ij.first, ij.second) = v;
        })
        .def("__eq__", [](const Matrix& a, const Matrix& b){ return a.approx_equal(b, 1e-9); })
        .def("__ne__", [](const Matrix& a, const Matrix& b){ return !a.approx_equal(b, 1e-9); })
        .def("approx_equal", &Matrix::approx_equal, py::arg("other"), py::arg("tol") = 1e-9);

    m.def("multiply_naive", &multiply_naive);
    m.def("multiply_tile",
          [](const Matrix& A, const Matrix& B, int block){ return multiply_tile(A, B, block); },
          py::arg("A"), py::arg("B"), py::arg("block") = 64);
    m.def("multiply_mkl", &multiply_mkl);
}