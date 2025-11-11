#include<iostream>
#include<algorithm>
#include<vector>
#include <cblas.h>
#include <atomic>
#include <limits>
#include <cstdlib>

struct ByteCounterImpl {
    std::atomic_size_t allocated = 0;
    std::atomic_size_t deallocated = 0;
    std::atomic_size_t refcount = 0;
};  

class ByteCounter
{
public:
    ByteCounter()
      : m_impl(new ByteCounterImpl)
    { incref(); }

    ByteCounter(ByteCounter const & other)
      : m_impl(other.m_impl)
    { incref(); }

    ByteCounter & operator=(ByteCounter const & other)
    {
        if (&other != this)
        {
            decref();
            m_impl = other.m_impl;
            incref();
        }

        return *this;
    }

    ByteCounter(ByteCounter && other)
      : m_impl(other.m_impl)
    { incref(); }

    ByteCounter & operator=(ByteCounter && other)
    {
        if (&other != this)
        {
            decref();
            m_impl = other.m_impl;
            incref();
        }

        return *this;
    }

    ~ByteCounter() { decref(); }

    void swap(ByteCounter & other)
    {
        std::swap(m_impl, other.m_impl);
    }

    void increase(std::size_t amount)
    {
        m_impl->allocated += amount;
    }

    void decrease(std::size_t amount)
    {
        m_impl->deallocated += amount;
    }

    std::size_t bytes() const { return m_impl->allocated - m_impl->deallocated; }
    std::size_t allocated() const { return m_impl->allocated; }
    std::size_t deallocated() const { return m_impl->deallocated; }
    /* This is for debugging. */
    std::size_t refcount() const { return m_impl->refcount; }

private:

    void incref() { ++m_impl->refcount; }

    void decref()
    {
        if (nullptr == m_impl)
        {
            // Do nothing.
        }
        else if (1 == m_impl->refcount)
        {
            delete m_impl;
            m_impl = nullptr;
        }
        else
        {
            --m_impl->refcount;
        }
    }

    ByteCounterImpl * m_impl = nullptr;

};

template <class T>
struct myAllocator
{
    using value_type = T;
    myAllocator() = default;
    template <class U>
    myAllocator(const myAllocator<U>& other) noexcept { counter = other.counter; }

    T * allocate(std::size_t n)
    {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
        {
            throw std::bad_alloc();
        }
        const std::size_t bytes = n*sizeof(T);
        T * p = static_cast<T *>(std::malloc(bytes));
        if (p)
        {
            counter.increase(bytes);
            return p;
        }
        else
        {
            throw std::bad_alloc();
        }
    }

    void deallocate(T* p, std::size_t n) noexcept
    {
        std::free(p);

        const std::size_t bytes = n*sizeof(T);
        counter.decrease(bytes);
    }

    template <class U> struct rebind { using other = myAllocator<U>; };

    template <class U> bool operator==(const myAllocator<U>&) const noexcept { return true; }
    template <class U> bool operator!=(const myAllocator<U>&) const noexcept { return false; }

    inline static ByteCounter counter{};
};

template <class T, class U>
bool operator==(const myAllocator<T> & a, const myAllocator<U> & b)
{
    return a.counter == b.counter;
}

template <class T, class U>
bool operator!=(const myAllocator<T> & a, const myAllocator<U> & b)
{
    return !(a == b);
}

class Matrix {
public:

    using Buffer = std::vector<double, myAllocator<double>>; 

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

    bool operator==(const Matrix& rhs) const {
        return m_nrow == rhs.m_nrow && m_ncol == rhs.m_ncol &&
               std::equal(m_buffer.begin(), m_buffer.end(), rhs.m_buffer.begin(), rhs.m_buffer.end());
    }
    
private:

    size_t m_nrow;
    size_t m_ncol;
    Buffer m_buffer;
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