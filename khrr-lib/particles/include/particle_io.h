#pragma once

#include <vector>
#include <string>
#include <cstddef>
#include <cstdint>
#include <galaxy_params.h>
#include <particle_data.h>

namespace khrr_particles {

/**
 * @brief Saves particle snapshot to binary format.
 * @details Writes S_%5d.bin and DM_%5d.bin files with headers (int count, real time)
 * followed by continuous [x y z vx vy vz] data.
 *
 * @param directory    Output directory path.
 * @param step         Time step identifier (used in filename formatting).
 * @param time         Simulation time at this step.
 * @param star_pos     Star coordinates.
 * @param star_vel     Star velocities.
 * @param dm_pos       Dark matter coordinates.
 * @param dm_vel       Dark matter velocities.
 */
void save_binary(
    const std::string& directory,
    int step,
    real time,
    const ParticleData& particles,
    const khrr_galaxy_params::GalaxiesParams& galaxies_params
);

/**
 * @brief Loads particle snapshot from binary format.
 * @details Reads S_%5d.bin and DM_%5d.bin. Validates header count against expected values.
 * If mismatch, warns and reads min(expected, header, available).
 *
 * @param directory       Directory containing the files.
 * @param step            Time step identifier.
 * @param galaxies_params Galaxies parameters defining expected counts.
 * @return                Loaded ParticleData.
 */
ParticleData load_binary(
    const std::string& directory,
    int step,
    const khrr_galaxy_params::GalaxiesParams& galaxies_params
);

/**
 * @brief Loads initial conditions from text format.
 * @details Reads start_S%d.txt for stars and start_DM%d.txt for DM across multiple galaxies.
 * Concatenates stars first, then DM. Uses fast parsing for whitespace-separated values.
 *
 * @param directory    Directory containing text files.
 * @param galaxies_params   Galaxies parameters defining galaxy indices and expected counts.
 * @return             Concatenated ParticleData.
 */
ParticleData load_text_initial(
    const std::string& directory,
    const khrr_galaxy_params::GalaxiesParams& galaxies_params
);

} // namespace khrr_particles