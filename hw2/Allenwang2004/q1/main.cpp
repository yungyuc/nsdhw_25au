#include <iostream>
#include <vector>

class Line
{
public:
    Line() = default;

    Line(size_t size)
        : _x(size), _y(size) {}

    Line(Line const& other)
        : _x(other._x), _y(other._y) {}

    Line(Line&& other) noexcept
        : _x(std::move(other._x)), _y(std::move(other._y)) {}

    Line& operator=(const Line&) = default;
    Line& operator=(Line&&) noexcept = default;

    ~Line() = default;

    size_t size() const { return _x.size(); }

    float const& x(size_t it) const { return _x.at(it); }
    float&       x(size_t it)       { return _x.at(it); }

    float const& y(size_t it) const { return _y.at(it); }
    float&       y(size_t it)       { return _y.at(it); }

private:
    std::vector<float> _x;
    std::vector<float> _y;
};


int main(int, char **)
{
    Line line(3); // Create a Line object with 3 points.
    line.x(0) = 0; line.y(0) = 1; // Set the coordinates of the first point.
    line.x(1) = 1; line.y(1) = 3; // Set the coordinates of the second point.
    line.x(2) = 2; line.y(2) = 5; // Set the coordinates of the third point.

    Line line2(line); // Copy constructor.
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