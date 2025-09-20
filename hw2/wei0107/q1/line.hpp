#pragma once
#include <vector>
#include <cstddef>
#include <stdexcept>

class Line {
public:
    Line() = default;
    Line(const Line&) = default;
    Line(Line&&) noexcept = default;
    Line& operator=(const Line&) = default;
    Line& operator=(Line&&) noexcept = default;

    explicit Line(std::size_t size)
        : xs_(size, 0.0f), ys_(size, 0.0f) {}

    ~Line() = default;

    std::size_t size() const { return xs_.size(); }

    const float& x(std::size_t it) const { return xs_.at(it); }
    float&       x(std::size_t it)       { return xs_.at(it); }

    const float& y(std::size_t it) const { return ys_.at(it); }
    float&       y(std::size_t it)       { return ys_.at(it); }

private:
    std::vector<float> xs_;
    std::vector<float> ys_;
};