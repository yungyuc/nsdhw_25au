#include <cmath>      // for sqrt, acos
#include "vector.hpp"
#include <algorithm>  // for clamp
#include <stdexcept>  // for error
#include <limits>     // for numeric_limits

double vector_angle_rad(double _x1, double _y1, double _x2, double _y2) {
    // limit of double
    const double max_val = std::numeric_limits<double>::max();

    // input limit recheck
    if (std::isinf(_x1) || std::isinf(_y1) || std::isinf(_x2) || std::isinf(_y2) ||
        std::fabs(_x1) > max_val || std::fabs(_y1) > max_val ||
        std::fabs(_x2) > max_val || std::fabs(_y2) > max_val) {
        throw std::invalid_argument("Input exceeds double range");
    }

    // dot product and two norm
    double dot_product = _x1 * _x2 + _y1 * _y2;
    double v1_mag = std::sqrt(_x1 * _x1 + _y1 * _y1);
    double v2_mag = std::sqrt(_x2 * _x2 + _y2 * _y2);

    // check for zero vector
    if (v1_mag <1e-6 && v2_mag <1e-6) {
        throw std::invalid_argument("Both vectors are zero-length");
    }
    if (v1_mag == 0 || v2_mag == 0) {
        return 0.0;  // 單一零向量 → 回傳 0
    }

    // cos(theta)
    double cos_theta = dot_product / (v1_mag * v2_mag);

    // clamp it to avoid acos error
    cos_theta = std::clamp(cos_theta, -1.0, 1.0);

    return std::acos(cos_theta);
}
