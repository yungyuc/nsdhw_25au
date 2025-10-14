#include <cstddef>
#pragma once
#include <vector>

class Line
{
public:
    Line();

    Line(size_t size);

    Line(const Line&);
    Line(Line&&) noexcept;
    Line& operator=(const Line&);
    Line& operator=(Line&&) noexcept;

    ~Line();

    size_t size() const;
    
    float const& x(size_t it) const;
    float& x(size_t it);
    float const& y(size_t it) const;
    float& y(size_t it);

private:
    std::vector<float> xs;
    std::vector<float> ys;
};
