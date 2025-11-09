#ifndef CUSTOM_ALLOCATOR_H
#define CUSTOM_ALLOCATOR_H

#include <cstddef>
#include <memory>
#include <atomic>

// Global memory tracking counters
class MemoryTracker {
public:
    static std::atomic<size_t> current_bytes;
    static std::atomic<size_t> total_allocated;
    static std::atomic<size_t> total_deallocated;
    
    static void reset() {
        current_bytes = 0;
        total_allocated = 0;
        total_deallocated = 0;
    }
    
    static size_t bytes() { return current_bytes.load(); }
    static size_t allocated() { return total_allocated.load(); }
    static size_t deallocated() { return total_deallocated.load(); }
};

// Initialize static members
inline std::atomic<size_t> MemoryTracker::current_bytes{0};
inline std::atomic<size_t> MemoryTracker::total_allocated{0};
inline std::atomic<size_t> MemoryTracker::total_deallocated{0};

// Custom STL allocator that tracks memory usage
template <typename T>
class CustomAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    
    template <typename U>
    struct rebind {
        using other = CustomAllocator<U>;
    };
    
    CustomAllocator() noexcept = default;
    
    template <typename U>
    CustomAllocator(const CustomAllocator<U>&) noexcept {}
    
    pointer allocate(size_type n) {
        if (n == 0) {
            return nullptr;
        }
        
        size_type bytes = n * sizeof(T);
        pointer p = static_cast<pointer>(::operator new(bytes));
        
        // Track allocation
        MemoryTracker::current_bytes += bytes;
        MemoryTracker::total_allocated += bytes;
        
        return p;
    }
    
    void deallocate(pointer p, size_type n) noexcept {
        if (p == nullptr || n == 0) {
            return;
        }
        
        size_type bytes = n * sizeof(T);
        
        // Track deallocation
        MemoryTracker::current_bytes -= bytes;
        MemoryTracker::total_deallocated += bytes;
        
        ::operator delete(p);
    }
    
    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        ::new(static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }
    
    template <typename U>
    void destroy(U* p) {
        p->~U();
    }
};

// Comparison operators
template <typename T, typename U>
bool operator==(const CustomAllocator<T>&, const CustomAllocator<U>&) noexcept {
    return true;
}

template <typename T, typename U>
bool operator!=(const CustomAllocator<T>&, const CustomAllocator<U>&) noexcept {
    return false;
}

#endif // CUSTOM_ALLOCATOR_H
