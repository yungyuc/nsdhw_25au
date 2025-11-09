#pragma once

#include <cstddef>
#include <new>
#include <atomic>

namespace memstats {
inline std::atomic<std::size_t>& bytes_in_use() {
    static std::atomic<std::size_t> v{0};
    return v;
}
inline std::atomic<std::size_t>& total_allocated() {
    static std::atomic<std::size_t> v{0};
    return v;
}
inline std::atomic<std::size_t>& total_deallocated() {
    static std::atomic<std::size_t> v{0};
    return v;
}

inline std::size_t bytes() { return bytes_in_use().load(); }
inline std::size_t allocated() { return total_allocated().load(); }
inline std::size_t deallocated() { return total_deallocated().load(); }
} 

template <class T>
struct CountingAllocator {
    using value_type = T;

    CountingAllocator() noexcept {}
    template <class U>
    CountingAllocator(CountingAllocator<U> const &) noexcept {}

    T* allocate(std::size_t n) {
        const std::size_t bytes = n * sizeof(T);
        memstats::total_allocated().fetch_add(bytes, std::memory_order_relaxed);
        memstats::bytes_in_use().fetch_add(bytes, std::memory_order_relaxed);
        return static_cast<T*>(::operator new(bytes));
    }

    void deallocate(T* p, std::size_t n) noexcept {
        const std::size_t bytes = n * sizeof(T);
        memstats::total_deallocated().fetch_add(bytes, std::memory_order_relaxed);
        memstats::bytes_in_use().fetch_sub(bytes, std::memory_order_relaxed);
        ::operator delete(p);
    }

    template <class U>
    struct rebind { using other = CountingAllocator<U>; };
};

template <class T, class U>
inline bool operator==(CountingAllocator<T> const&, CountingAllocator<U> const&) noexcept {
    return true;
}
template <class T, class U>
inline bool operator!=(CountingAllocator<T> const& a, CountingAllocator<U> const& b) noexcept {
    return !(a == b);
}


