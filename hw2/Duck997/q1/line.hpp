#pragma once

#include <cstddef>
#include <vector>

class Line{
public:
    Line();
    Line(Line const &);
    Line(Line &&) noexcept;
    Line & operator=(Line const &);
    Line & operator=(Line &&) noexcept;
    explicit Line(std::size_t size);
    ~Line();

    std::size_t size() const;

    float const & x(std::size_t it) const;
    float & x(std::size_t it);

    float const & y(std::size_t it) const;
    float & y(std::size_t it);

private:
    std::vector<float> m_x;
    std::vector<float> m_y;
};


