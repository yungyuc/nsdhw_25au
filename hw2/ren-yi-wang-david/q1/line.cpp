#include "line.hpp"

Line::Line(size_t size) : _x(size), _y(size) {}

size_t Line::size() const { 
    return _x.size(); 
}

float const& Line::x(size_t it) const { 
    return _x[it]; 
}

float& Line::x(size_t it) { 
    return _x[it]; 
}

float const& Line::y(size_t it) const { 
    return _y[it]; 
}

float& Line::y(size_t it) { 
    return _y[it]; 
}
