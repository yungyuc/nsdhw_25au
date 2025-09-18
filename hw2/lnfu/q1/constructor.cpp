#include <iostream>
#include <vector>
#include <tuple>

class Line
{
public:
    Line() = default;
    Line(Line const &);
    Line(Line &&);
    Line &operator=(Line const &);
    Line &operator=(Line &&);

    Line(size_t size) : m_size(size), m_coord(std::vector<std::tuple<float, float>>(size)) {}

    ~Line()
    {
        m_size = 0;
        m_coord.clear();
    }

    size_t size() const { return m_size; }
    const float &x(size_t it) const
    {
        check_range(it);
        return std::get<0>(m_coord[it]);
    }
    float &x(size_t it)
    {
        check_range(it);
        return std::get<0>(m_coord[it]);
    }
    const float &y(size_t it) const
    {
        check_range(it);
        return std::get<1>(m_coord[it]);
    }
    float &y(size_t it)
    {
        check_range(it);
        return std::get<1>(m_coord[it]);
    }

private:
    void check_range(size_t it) const
    {
        if (it >= m_size)
        {
            throw std::out_of_range("Line index out of range");
        }
    }
    size_t m_size = 0;
    std::vector<std::tuple<float, float>> m_coord;
};

Line::Line(Line const &other)
{
    if (!other.m_coord.empty())
    {
        m_size = other.m_size;
        m_coord = other.m_coord;
    }
    else
    {
        m_size = 0;
        m_coord.clear();
    }
}

Line::Line(Line &&other)
{
    if (!other.m_coord.empty())
    {
        m_size = other.m_size;
        m_coord = std::move(other.m_coord);
        other.m_size = 0;
        other.m_coord.clear();
    }
    else
    {
        m_size = 0;
        m_coord.clear();
    }
}

Line &Line::operator=(Line const &other)
{
    if (this != &other)
    {
        if (!other.m_coord.empty())
        {
            m_size = other.m_size;
            m_coord = other.m_coord;
        }
        else
        {
            m_size = 0;
            m_coord.clear();
        }
    }
    return *this;
}

Line &Line::operator=(Line &&other)
{
    if (this != &other)
    {
        if (!other.m_coord.empty())
        {
            m_size = other.m_size;
            m_coord = std::move(other.m_coord);
            other.m_size = 0;
            other.m_coord.clear();
        }
        else
        {
            m_size = 0;
            m_coord.clear();
        }
    }
    return *this;
}

int main(int, char **)
{
    Line line(3);
    line.x(0) = 0;
    line.y(0) = 1;
    line.x(1) = 1;
    line.y(1) = 3;
    line.x(2) = 2;
    line.y(2) = 5;

    Line line2(line);
    line2.x(0) = 9;

    std::cout << "line: number of points = " << line.size() << std::endl;
    for (size_t it = 0; it < line.size(); ++it)
    {
        std::cout << "point " << it << ":"
                  << " x = " << line.x(it)
                  << " y = " << line.y(it) << std::endl;
    }

    std::cout << "line2: number of points = " << line.size() << std::endl;
    for (size_t it = 0; it < line.size(); ++it)
    {
        std::cout << "point " << it << ":"
                  << " x = " << line2.x(it)
                  << " y = " << line2.y(it) << std::endl;
    }

    return 0;
}