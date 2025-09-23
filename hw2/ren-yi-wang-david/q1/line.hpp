#pragma once
#include <vector>
#include <cstddef>  // for size_t


class Line {
public:
    Line() = default;                                // 預設建構子
    Line(size_t size);                               // 帶參數建構子
    Line(const Line &other) = default;               // 複製建構子
    Line(Line &&other) noexcept = default;           // 移動建構子
    Line& operator=(const Line &other) = default;    // 複製指定運算子
    Line& operator=(Line &&other) noexcept = default;// 移動指定運算子
    ~Line() = default;                               // 解構子

    size_t size() const;

    float const& x(size_t it) const;
    float& x(size_t it);

    float const& y(size_t it) const;
    float& y(size_t it);

private:
    std::vector<float> _x, _y;
};
