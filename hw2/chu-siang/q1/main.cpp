#include <iostream>
#include <vector>
#include <cstddef>
#include <stdexcept>

class Line
{
public:
    Line() = default;
    Line(const Line&) = default;
    Line(Line&&) noexcept = default;
    Line& operator=(const Line&) = default;
    Line& operator=(Line&&) noexcept = default;
    explicit Line(std::size_t size) : xs_(size, 0.0f), ys_(size, 0.0f) {}
    ~Line() = default;

    std::size_t size() const noexcept { return xs_.size(); }

    float const& x(std::size_t it) const { check_index(it, xs_.size()); return xs_[it]; }
    float&       x(std::size_t it)       { check_index(it, xs_.size()); return xs_[it]; }

    float const& y(std::size_t it) const { check_index(it, ys_.size()); return ys_[it]; }
    float&       y(std::size_t it)       { check_index(it, ys_.size()); return ys_[it]; }

private:
    std::vector<float> xs_;
    std::vector<float> ys_;

    static void check_index(std::size_t it, std::size_t n) {
        if (it >= n) throw std::out_of_range("Line index out of range");
    }
}; /* end class Line */

int main(int, char **)
{
    Line line(3);
    line.x(0) = 0; line.y(0) = 1;
    line.x(1) = 1; line.y(1) = 3;
    line.x(2) = 2; line.y(2) = 5;

    Line line2(line);
    line2.x(0) = 9;

    std::cout << "line: number of points = " << line.size() << std::endl;
    for (size_t it=0; it<line.size(); ++it)
    {
        std::cout << "point " << it << ":"
                  << " x = " << line.x(it)
                  << " y = " << line.y(it) << std::endl;
    }

    std::cout << "line2: number of points = " << line.size() << std::endl;
    for (size_t it=0; it<line.size(); ++it)
    {
        std::cout << "point " << it << ":"
                  << " x = " << line2.x(it)
                  << " y = " << line2.y(it) << std::endl;
    }

    return 0;
}

