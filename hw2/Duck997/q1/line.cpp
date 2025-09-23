#include "line.hpp"

#include <stdexcept>

Line::Line() = default;

Line::Line(Line const &) = default;

Line::Line(Line &&) noexcept = default;

Line & Line::operator=(Line const &) = default;

Line & Line::operator=(Line &&) noexcept = default;

Line::Line(std::size_t size)
  : m_x(size, 0.0f), m_y(size, 0.0f){}

Line::~Line() = default;

std::size_t Line::size() const{
    return m_x.size();
}

float const & Line::x(std::size_t it) const{
    if (it >= m_x.size()) throw std::out_of_range("x index out of range");
    return m_x[it];
}

float & Line::x(std::size_t it){
    if (it >= m_x.size()) throw std::out_of_range("x index out of range");
    return m_x[it];
}

float const & Line::y(std::size_t it) const{
    if (it >= m_y.size()) throw std::out_of_range("y index out of range");
    return m_y[it];
}

float & Line::y(std::size_t it){
    if (it >= m_y.size()) throw std::out_of_range("y index out of range");
    return m_y[it];
}


