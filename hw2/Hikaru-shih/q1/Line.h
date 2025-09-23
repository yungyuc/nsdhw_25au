#ifndef LINE_H
#define LINE_H

#include <vector>
#include <cstddef>

class Line {
public:
    Line() = default;
    Line(size_t size) : xs(size), ys(size) {}

    Line(const Line&) = default;
    Line(Line&&) noexcept = default;
    Line& operator=(const Line&) = default;
    Line& operator=(Line&&) noexcept = default;
    ~Line() = default;

    size_t size() const { return xs.size(); }

    float const & x(size_t i) const { return xs.at(i); }
    float       & x(size_t i)       { return xs.at(i); }
    float const & y(size_t i) const { return ys.at(i); }
    float       & y(size_t i)       { return ys.at(i); }

private:
    std::vector<float> xs;
    std::vector<float> ys;
};

#endif 
