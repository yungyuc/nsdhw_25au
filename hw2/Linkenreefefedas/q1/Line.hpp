#pragma once
#include <vector>
#include <cstddef>
#include <stdexcept>

class Line {
public:
    Line();                         // 預設 0 點
    Line(size_t size);              // 給定點數
    Line(Line const& other);        // 複製建構
    Line(Line&& other) noexcept;    // 移動建構
    Line& operator=(Line const&);   // 複製賦值
    Line& operator=(Line&&) noexcept; // 移動賦值
    ~Line() = default;              // vector 自動管理

    size_t size() const noexcept { return x_.size(); }

    float const& x(size_t i) const { return at(x_, i); }
    float&       x(size_t i)       { return at(x_, i); }
    float const& y(size_t i) const { return at(y_, i); }
    float&       y(size_t i)       { return at(y_, i); }

private:
    std::vector<float> x_, y_;

    static float& at(std::vector<float>& v, size_t i) {
#ifndef NDEBUG
        if (i >= v.size()) throw std::out_of_range("index out of range");
#endif
        return v[i];
    }
    static float const& at(std::vector<float> const& v, size_t i) {
#ifndef NDEBUG
        if (i >= v.size()) throw std::out_of_range("index out of range");
#endif
        return v[i];
    }
};
