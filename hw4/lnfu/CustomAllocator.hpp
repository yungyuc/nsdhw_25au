#ifndef CUSTOM_ALLOCATOR_HPP
#define CUSTOM_ALLOCATOR_HPP

#include <cstddef>
#include <cstdlib>

namespace MemoryTracker
{
    inline std::size_t total_bytes = 0;
    inline std::size_t total_allocated = 0;
    inline std::size_t total_deallocated = 0;
}

template <typename T>
class CustomAllocator
{
public:
    using value_type = T;
    using pointer = T *;
    using const_pointer = const T *;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    // Rebind 結構（必須）
    template <typename U>
    struct rebind
    {
        using other = CustomAllocator<U>;
    };

    CustomAllocator() noexcept = default;

    template <typename U>
    CustomAllocator(const CustomAllocator<U> &) noexcept {}

    // 分配記憶體
    T *allocate(std::size_t n)
    {
        if (n == 0)
            return nullptr;

        std::size_t bytes = n * sizeof(T);
        T *ptr = static_cast<T *>(::operator new(bytes));

        // 更新追蹤資訊
        MemoryTracker::total_bytes += bytes;
        MemoryTracker::total_allocated += bytes;

        return ptr;
    }

    // 釋放記憶體
    void deallocate(T *p, std::size_t n) noexcept
    {
        if (p == nullptr || n == 0)
            return;

        std::size_t bytes = n * sizeof(T);

        // 更新追蹤資訊
        MemoryTracker::total_bytes -= bytes;
        MemoryTracker::total_deallocated += bytes;

        ::operator delete(p);
    }
};

// Allocator 比較運算子（必須）
template <typename T, typename U>
bool operator==(const CustomAllocator<T> &, const CustomAllocator<U> &) noexcept
{
    return true;
}

template <typename T, typename U>
bool operator!=(const CustomAllocator<T> &, const CustomAllocator<U> &) noexcept
{
    return false;
}

#endif // CUSTOM_ALLOCATOR_HPP