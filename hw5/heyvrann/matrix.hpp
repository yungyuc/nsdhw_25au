#include <atomic>
#include <vector>
#include <cstddef>

#if defined(USE_MKL) || defined(HASMKL)
#include <mkl_cblas.h>
#include <mkl_service.h>
#elif defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

struct ByteCounterImpl {
    std::atomic_size_t allocated = 0;
    std::atomic_size_t deallocated = 0;
    std::atomic_size_t refcount = 0;
};  

class ByteCounter {
public:
    ByteCounter();

    ByteCounter(const ByteCounter& other);

    ByteCounter& operator=(const ByteCounter& other);

    ByteCounter(ByteCounter&& other);

    ByteCounter& operator=(ByteCounter&& other);

    ~ByteCounter();

    void swap(ByteCounter& other);

    void increase(std::size_t amount);

    void decrease(std::size_t amount);

    std::size_t bytes() const;

    std::size_t allocated() const;

    std::size_t deallocated() const;

    std::size_t refcount() const;

    bool operator==(const ByteCounter& other);

    bool operator!=(const ByteCounter& other);

private:
    ByteCounterImpl* m_impl;
    void incref();
    void decref();
};

template <class T>
struct CustomAllocator {
    using value_type = T;

    static ByteCounter counter;

    CustomAllocator() = default;

    template <class U> constexpr
    CustomAllocator(const CustomAllocator<U>& other) noexcept;

    T* allocate(std::size_t n);

    void deallocate(T* p, std::size_t n);
};

template <class T, class U>
bool operator==(const CustomAllocator<T>& a, const CustomAllocator<U>& b);

template <class T, class U>
bool operator!=(const CustomAllocator<T>& a, const CustomAllocator<U>& b);

// template<class T>
// ByteCounter CustomAllocator<T>::counter{};

struct Matrix {

    std::vector<double, CustomAllocator<double>> m_buffer;

    Matrix(size_t nrow, size_t ncol);

    Matrix(size_t nrow, size_t ncol, const std::vector<double>& vec);

    Matrix& operator=(const std::vector<double>& vec);

    Matrix(const Matrix& other);

    Matrix& operator=(const Matrix& other);

    Matrix(Matrix && other);

    Matrix& operator=(Matrix && other);

    ~Matrix();

    double operator()(size_t row, size_t col) const;
    double& operator()(size_t row, size_t col);

    double operator[](size_t idx) const;
    double& operator[](size_t idx);

    size_t nrow() const;
    size_t ncol() const;
    size_t size() const;

    size_t nflo() const;
    size_t& nflo();

    Matrix transpose() const;

private:

    size_t m_nrow, m_ncol;
    size_t m_nflo = 0;

    size_t index(size_t row, size_t col) const;

    void reset_buffer(size_t nrow, size_t ncol);
};

bool operator==(const Matrix& mat1, const Matrix& mat2);

bool operator!=(const Matrix& mat1, const Matrix& mat2);

void validate_multiplication(const Matrix& mat1, const Matrix& mat2);

size_t calculate_nflo(const Matrix& mat1, const Matrix& mat2);

Matrix multiply_naive(const Matrix& mat1, const Matrix& mat2);

Matrix multiply_tile(const Matrix& mat1, const Matrix& mat2, const size_t B);

Matrix multiply_mkl(const Matrix& mat1, const Matrix& mat2);