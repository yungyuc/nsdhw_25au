#include "Line.hpp"
#include <utility>

Line::Line() : x_(), y_() {}

Line::Line(size_t n) : x_(n, 0.0f), y_(n, 0.0f) {}

Line::Line(Line const& other) = default;

Line::Line(Line&& other) noexcept
    : x_(std::move(other.x_)), y_(std::move(other.y_)) {}

Line& Line::operator=(Line const& other) = default;

Line& Line::operator=(Line&& other) noexcept {
    if (this != &other) {
        x_ = std::move(other.x_);
        y_ = std::move(other.y_);
    }
    return *this;
}
