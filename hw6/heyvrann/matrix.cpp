#include "matrix.hpp"
#include <iostream>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <limits>

#if defined(USE_MKL) || defined(HASMKL)
#include <mkl_cblas.h>
#include <mkl_service.h>
#elif defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

ByteCounter::ByteCounter(): m_impl(new ByteCounterImpl) { incref(); }

ByteCounter::ByteCounter(const ByteCounter& other): m_impl(other.m_impl) { incref(); }

ByteCounter& ByteCounter::operator=(const ByteCounter& other) {
    if (&other != this) {
        decref();
        m_impl = other.m_impl;
        incref();
    }
    return *this;
}

ByteCounter::ByteCounter(ByteCounter&& other): m_impl(other.m_impl) { incref(); }


ByteCounter& ByteCounter::operator=(ByteCounter&& other) {
    if (&other != this) {
        decref();
        m_impl = other.m_impl;
        incref();
    }
    return *this;
}

ByteCounter::~ByteCounter() { decref(); }

void ByteCounter::swap(ByteCounter& other) {
    std::swap(m_impl, other.m_impl);
}

void ByteCounter::increase(std::size_t amount) {
    m_impl->allocated += amount;
}

void ByteCounter::decrease(std::size_t amount) {
        m_impl->deallocated += amount;
}

std::size_t ByteCounter::bytes() const { return m_impl->allocated - m_impl->deallocated; }

std::size_t ByteCounter::allocated() const { return m_impl->allocated; }

std::size_t ByteCounter::deallocated() const { return m_impl->deallocated; }

std::size_t ByteCounter::refcount() const { return m_impl->refcount; }

bool ByteCounter::operator==(const ByteCounter& other) { return m_impl == other.m_impl; }

bool ByteCounter::operator!=(const ByteCounter& other) { return !((*this) == other); }

void ByteCounter::incref() { ++m_impl->refcount; }

void ByteCounter::decref() {
    if (m_impl == nullptr) {

    }
    else if (m_impl->refcount == 1) {
        delete m_impl;
        m_impl = nullptr;
    }
    else {
        --m_impl->refcount;
    }
}

template<class T>
template <class U> constexpr
CustomAllocator<T>::CustomAllocator(const CustomAllocator<U>& other) noexcept {
    counter = other.counter;
}

template<class T>
T* CustomAllocator<T>::allocate(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::bad_alloc();
    }

    const std::size_t bytes = n * sizeof(T);
    T* p = static_cast<T*>(std::malloc(bytes));
    if (p) {
        counter.increase(bytes);
        return p;
    }
    else {
        throw std::bad_alloc();
    }
}

template<class T>
void CustomAllocator<T>::deallocate(T* p, std::size_t n) {
    std::free(p);

    const std::size_t bytes = n * sizeof(T);
    counter.decrease(bytes);
}

template <class T, class U>
bool operator==(const CustomAllocator<T>& a, const CustomAllocator<U>& b)
{
    return a.counter == b.counter;
}

template <class T, class U>
bool operator!=(const CustomAllocator<T>& a, const CustomAllocator<U>& b)
{
    return !(a == b);
}

template<class T>
ByteCounter CustomAllocator<T>::counter{};

Matrix::Matrix(size_t nrow, size_t ncol): m_nrow(nrow), m_ncol(ncol) {
    reset_buffer(nrow, ncol);
}

Matrix::Matrix(size_t nrow, size_t ncol, const std::vector<double>& vec) {
        reset_buffer(nrow, ncol);
        *this = vec;
}

Matrix& Matrix::operator=(const std::vector<double>& vec) {
    if (size() != vec.size()) 
        throw std::out_of_range("number of elements mismatch");

    size_t k = 0;
    for (size_t i = 0 ; i < m_nrow ; i++)
        for (size_t j = 0 ; j < m_ncol ; j++)
            (*this)(i, j) = vec[k++];

    return *this;
}

Matrix::Matrix(const Matrix& other)
: m_nflo(other.m_nflo) {
    reset_buffer(other.m_nrow, other.m_ncol);
    for (size_t i = 0 ; i < m_nrow ; i++)
        for (size_t j = 0 ; j < m_ncol ; j++)
            (*this)(i, j) = other(i, j);
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this == &other) return *this;

    if (m_nrow != other.m_nrow || m_ncol != other.m_ncol)
        reset_buffer(other.m_nrow, other.m_ncol);
    for (size_t i = 0 ; i < m_nrow ; i++)
        for (size_t j = 0 ; j < m_ncol ; j++)
            (*this)(i, j) = other(i, j);
    m_nflo = other.m_nflo;

    return *this;
}

Matrix::Matrix(Matrix && other)
: m_nflo(other.m_nflo) {
    std::swap(m_nrow, other.m_nrow);
    std::swap(m_ncol, other.m_ncol);
    std::swap(m_buffer, other.m_buffer);
}

Matrix& Matrix::operator=(Matrix && other) {
    if (this == &other) return *this;

    std::swap(m_nrow, other.m_nrow);
    std::swap(m_ncol, other.m_ncol);
    std::swap(m_buffer, other.m_buffer);
    std::swap(m_nflo, other.m_nflo);

    return *this;
}

Matrix::~Matrix() { reset_buffer(0, 0); }

double Matrix::operator()(size_t row, size_t col) const { 
    return m_buffer[index(row, col)]; 
}
double& Matrix::operator()(size_t row, size_t col) {
    return m_buffer[index(row, col)];
}

double Matrix::operator[](size_t idx) const { return m_buffer[idx]; }
double& Matrix::operator[](size_t idx) { return m_buffer[idx]; }

size_t Matrix::nrow() const { return m_nrow; }
size_t Matrix::ncol() const { return m_ncol; }
size_t Matrix::size() const { return m_nrow * m_ncol; }

size_t Matrix::nflo() const { return m_nflo; }
size_t& Matrix::nflo() { return m_nflo; }

double* Matrix::data() { return m_buffer.data(); }

const double* Matrix::data() const { return m_buffer.data(); }

size_t Matrix::index(size_t row, size_t col) const { return row * m_ncol + col; }

void Matrix::reset_buffer(size_t nrow, size_t ncol) {
    const size_t nelement = nrow * ncol;
    m_buffer.assign(nelement, 0.0);
    m_nrow = nrow; m_ncol = ncol;
}

Matrix Matrix::transpose() const {
    Matrix ret(ncol(), nrow());
    for (size_t i = 0 ; i < ret.nrow() ; i++) 
        for (size_t j = 0 ; j < ret.ncol() ; j++)
            ret(i, j) = (*this)(j, i);
    return ret;
}

bool operator==(const Matrix& mat1, const Matrix& mat2) {
    if (mat1.nrow() != mat2.nrow() || mat1.ncol() != mat2.ncol())
        return false;

    for (size_t i = 0 ; i < mat1.nrow() ; i++)
        for (size_t j = 0 ; j < mat1.ncol() ; j++) 
            if (mat1(i, j) != mat2(i, j))
                return false;


    return true;
}

bool operator!=(const Matrix& mat1, const Matrix& mat2) {
    return !(mat1 == mat2);
}

void validate_multiplication(const Matrix& mat1, const Matrix& mat2) {
    if (mat1.ncol() != mat2.nrow()) {
        throw std::out_of_range("the number of first matrix column differs from that of second matrix row");
    }
}

size_t calculate_nflo(const Matrix& mat1, const Matrix& mat2) {
    return mat1.nrow() * mat1.ncol() * mat2.ncol();
}

Matrix multiply_naive(const Matrix& mat1, const Matrix& mat2) {
    validate_multiplication(mat1, mat2);

    Matrix ret(mat1.nrow(), mat2.ncol());

    const size_t nrow = ret.nrow();
    const size_t ncol = ret.ncol();
    const size_t nrow1 = mat1.nrow();
    const size_t ncol1 = mat1.ncol();
    const size_t ncol2 = mat2.ncol();

    for (size_t i = 0 ; i < nrow ; i++)
        for (size_t j = 0 ; j < ncol ; j++)
            ret(i, j) = 0;

    for (size_t i = 0 ; i < nrow1 ; i++) 
        for (size_t j = 0 ; j < ncol1 ; j++) 
            for (size_t k = 0 ; k < ncol2 ; k++)
                ret(i, k) += mat1(i, j) * mat2(j, k);

    ret.nflo() = calculate_nflo(mat1, mat2);

    return ret;
}

Matrix multiply_tile(const Matrix& mat1, const Matrix& mat2, const size_t B) {
    validate_multiplication(mat1, mat2);

    Matrix ret(mat1.nrow(), mat2.ncol()), mat2_trans = mat2.transpose();

    const size_t nrow = ret.nrow();
    const size_t ncol = ret.ncol();
    const size_t nrow1 = mat1.nrow();
    const size_t ncol1 = mat1.ncol();
    const size_t ncol2 = mat2.ncol();

    for (size_t i = 0 ; i < nrow ; i++)
        for (size_t j = 0 ; j < ncol ; j++)
            ret[i*ncol+j] = 0;

    for (size_t it = 0 ; it < nrow1 ; it += B) {
        const size_t n_i = std::min(it+B, nrow1); 
        for (size_t jt = 0 ; jt < ncol2 ; jt += B) {
            const size_t n_j = std::min(jt+B, ncol2);
            for (size_t kt = 0  ; kt < ncol1 ; kt += B) {
                const size_t n_k = std::min(kt+B, ncol1);
                for (size_t i = it ; i < n_i ; i++) {
                    for (size_t j = jt ; j < n_j ; j++) {
                        for (size_t k = kt ; k < n_k ; k++) {
                            ret.m_buffer[i*ncol+j] += mat1.m_buffer[i*ncol1+k] * mat2_trans.m_buffer[j*ncol1+k];
                        }
                    }
                }
            }
        }
    }

    ret.nflo() = calculate_nflo(mat1, mat2);

    return ret;
}

Matrix multiply_mkl(const Matrix& mat1, const Matrix& mat2) {
#if !defined(HASMKL) || defined(NOMKL)
    // run with VECLIB_MAXIMUM_THREAD=1
#else // HASMKL NOMKL
    mkl_set_num_threads(1);
#endif //HASMKL NOMKL

    validate_multiplication(mat1, mat2);

    Matrix ret(mat1.nrow(), mat2.ncol());

    const size_t ncol = ret.ncol();
    const size_t nrow1 = mat1.nrow();
    const size_t ncol1 = mat1.ncol();
    const size_t ncol2 = mat2.ncol();

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                nrow1, ncol2, ncol1, 1.0, 
                mat1.m_buffer.data(), ncol1,
                mat2.m_buffer.data(), ncol2,
                0.0, ret.m_buffer.data(), ncol);

    ret.nflo() = calculate_nflo(mat1, mat2);

    return ret;
}