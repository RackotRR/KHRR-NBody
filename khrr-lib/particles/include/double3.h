#pragma once

#include <cstdint>

namespace khrr_particles {

/**
 * @brief Simple 3D vector structure for coordinates and velocities.
 * Designed for compatibility with N-body and particle-in-cell solvers.
 */
struct double3 {
    double x;
    double y;
    double z;
};

} // namespace khrr_particles