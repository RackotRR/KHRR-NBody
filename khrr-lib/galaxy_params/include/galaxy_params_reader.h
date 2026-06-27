#pragma once
#include <string>

#include "galaxy_params.h"

namespace khrr_galaxy_params {

/**
 * @brief Parses the `__start_galaxies.ini` configuration file.
 *
 * Reads galaxy parameters line-by-line, stripping comments (starting with `----` or `#`).
 * Validates structural integrity, type compatibility, and physical constraints.
 * Uses `range-v3` lazy views for batch validation of numerical fields.
 *
 * @param filepath Path to the INI file.
 * @return GalaxiesParams Populated and validated structure.
 * @throws std::runtime_error If file is missing, malformed, or violates physical constraints.
 */
GalaxiesParams read_galaxy_params(const std::string& filepath);

} // namespace khrr_galaxy_params
