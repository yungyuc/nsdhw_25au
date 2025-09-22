#include <iostream>
#include <vector>

class Line
{
public:
    // Constructors
    Line() = default;
    explicit Line(size_t size) : m_x(size), m_y(size) {}

    // Copy / Move
    Line(const Line&) = default;
    Line(Line&&) noexcept = default;
    Line& operator=(const Line&) = default;
    Line& operator=(Line&&) noexcept = default;

    // Destructor
    ~Line() = default;

    // Size
    size_t size() const { return m_x.size(); }

    // Accessors for x
    float& x(size_t it) { return m_x.at(it); }
    const float& x(size_t it) const { return m_x.at(it); }

    // Accessors for y
    float& y(size_t it) { return m_y.at(it); }
    const float& y(size_t it) const { return m_y.at(it); }

private:
    std::vector<float> m_x;
    std::vector<float> m_y;
};


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