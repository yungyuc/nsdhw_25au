#include "Line.hpp"

Line::Line() = default;

Line::Line(size_t size) : xs(size), ys(size) {}

Line::Line(const Line&) = default;
Line::Line(Line&&) noexcept = default;
Line& Line::operator=(const Line&) = default;
Line& Line::operator=(Line&&) noexcept = default;

Line::~Line() = default;

size_t Line::size() const {return xs.size();}

float const& Line::x(size_t it) const {return xs[it];}
float& Line::x(size_t it) {return xs[it];}
float const& Line::y(size_t it) const {return ys[it];}
float& Line::y(size_t it) {return ys[it];}