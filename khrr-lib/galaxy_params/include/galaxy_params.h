#pragma once
#include <vector>
#include <cstdint>


namespace khrr_galaxy_params {

/**
 * @brief Stores physical and initialization parameters for a single galaxy.
 *
 * Contains counts of star and dark matter particles, their masses,
 * softening lengths, center-of-mass position/velocity, and disk inclination.
 */
struct GalaxyParams {
    int k_glx;                  ///< Galaxy index (expected to match iteration order)
    std::size_t N_s;            ///< Number of star particles (must be multiple of 1024)
    std::size_t N_dm;           ///< Number of dark matter particles (must be multiple of 1024)
    double Mass_s;              ///< Total mass of star component
    double Mass_dm;             ///< Total mass of dark matter component
    double eps_s;               ///< Gravitational softening for stars
    double eps_dm;              ///< Gravitational softening for dark matter
    double alpha_glx;           ///< Disk inclination angle in degrees
    double X_glx, Y_glx, Z_glx; ///< Center-of-mass position coordinates
    double Vx_glx, Vy_glx, Vz_glx; ///< Center-of-mass velocity components
};

/**
 * @brief Root container for galaxies input parameters.
 */
struct GalaxiesParams {
    int M_glx; ///< Total number of galaxies
    std::vector<GalaxyParams> galaxies;
};

} // namespace khrr_galaxy_params
