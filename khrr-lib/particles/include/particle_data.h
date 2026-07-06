#pragma once
#include <vector>
#include <cstdint>
#include <khrr_types.h>

namespace khrr_particles {

using khrr_common::real;
using khrr_common::real3;

/**
 * @brief Container holding particle positions, velocities, and component counts.
 * Data is stored in a Structure of Arrays-like manner via separate vectors.
 */
struct ParticleData {
    std::vector<real3> positions;
    std::vector<real3> velocities;
    std::vector<real> masses;
    std::vector<real> eps2; // softening
    std::size_t n_stars = 0;
    std::size_t n_dm   = 0;

    std::size_t size() const noexcept { return positions.size(); }

    /// True if every sub-array has the same length and n_stars+n_dm == N.
    bool is_valid() const noexcept {
        const std::size_t n = positions.size();
        return !positions.empty()          &&
               velocities.size() == n      &&
               masses.size()     == n      &&
               eps2.size()       == n      &&
               (n_stars + n_dm)  == n;
    }
};

}  // namespace khrr_particles