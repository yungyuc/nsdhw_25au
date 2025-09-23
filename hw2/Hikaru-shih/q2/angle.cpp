#include <cmath>
#include <stdexcept>

double angle(double x1, double y1, double x2, double y2) {
    double dot = x1 * x2 + y1 * y2;
    double normA = std::sqrt(x1*x1 + y1*y1);
    double normB = std::sqrt(x2*x2 + y2*y2);

    if (normA == 0 || normB == 0) {
        throw std::invalid_argument("Zero-length vector not allowed");
    }

    double cosTheta = dot / (normA * normB);

    if (cosTheta > 1.0) cosTheta = 1.0;
    if (cosTheta < -1.0) cosTheta = -1.0;

    double result = std::acos(cosTheta);

    if (std::fabs(result) < 1e-7) {
        return 0.0;
    }

    return result;
}
