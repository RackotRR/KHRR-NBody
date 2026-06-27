#include "galaxy_params.h"
#include "galaxy_params_reader.h"
#include <khrr_parser.h>
#include <spdlog/spdlog.h>

#include <stdexcept>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <range/v3/algorithm/all_of.hpp>

namespace khrr_galaxy_params {
using khrr_parser::parse_tuple_values;

GalaxiesParams read_galaxy_params(const std::string& filepath) {
    spdlog::info("load galaxy params");
    std::vector<std::string> lines = khrr_parser::read_file(filepath);

    // Парсинг M_glx: количество элементов проверяется на этапе компиляции
    auto [M_glx] = parse_tuple_values<int>(lines[0]);
    if (M_glx <= 0) throw std::runtime_error("M_glx must be strictly positive.");

    GalaxiesParams params{M_glx};
    params.galaxies.reserve(M_glx);

    constexpr std::size_t LINES_PER_GALAXY = 7;
    const std::size_t expected_total = 1 + static_cast<std::size_t>(M_glx) * LINES_PER_GALAXY;
    if (lines.size() < expected_total) {
        throw std::runtime_error(fmt::format("File contains fewer configuration lines than required for M_glx={}", M_glx));
    }

    spdlog::info("galaxies count = {}", M_glx);
    for (int i = 0; i < M_glx; ++i) {
        GalaxyParams g{};
        std::size_t base = 1 + i * LINES_PER_GALAXY;

        std::tie(g.k_glx) = parse_tuple_values<int>(lines[base]);
        if (g.k_glx != i) {
            throw std::runtime_error(fmt::format("Mismatched galaxy index at line {} (expected {}, got {})", base + 1, i, g.k_glx));
        }

        std::tie(g.N_s, g.N_dm) = parse_tuple_values<std::size_t, std::size_t>(lines[base + 1]);
        if (g.N_s == 0 || g.N_dm == 0 || g.N_s % 1024 != 0 || g.N_dm % 1024 != 0) {
            throw std::runtime_error("N_s and N_dm must be >0 and multiples of 1024.");
        }

        std::tie(g.Mass_s, g.Mass_dm) = parse_tuple_values<double, double>(lines[base + 2]);
        if (g.Mass_s <= 0.0 || g.Mass_dm <= 0.0) throw std::runtime_error("Masses must be strictly positive.");

        std::tie(g.eps_s, g.eps_dm) = parse_tuple_values<double, double>(lines[base + 3]);
        if (g.eps_s < 0.0 || g.eps_dm < 0.0) throw std::runtime_error("Softening length cannot be negative.");

        std::tie(g.alpha_glx) = parse_tuple_values<double>(lines[base + 4]);

        std::tie(g.X_glx, g.Y_glx, g.Z_glx) = parse_tuple_values<double, double, double>(lines[base + 5]);

        std::tie(g.Vx_glx, g.Vy_glx, g.Vz_glx) = parse_tuple_values<double, double, double>(lines[base + 6]);

        std::vector<double> checks = {
            g.alpha_glx,
            g.Mass_s,
            g.Mass_dm,
            g.eps_s,
            g.eps_dm,
            g.X_glx, g.Y_glx, g.Z_glx,
            g.Vx_glx, g.Vy_glx, g.Vz_glx
        };

        if (!ranges::all_of(checks, [](double v){ return std::isfinite(v); })) {
            throw std::runtime_error(fmt::format("Detected NaN or Inf in galaxy parameters (index {})", i));
        }

        spdlog::info("\t galaxy {} (Star : Dark matter)", i);
        spdlog::info("\t -- N = {} ({} : {})", g.N_s + g.N_dm, g.N_s, g.N_dm);
        spdlog::info("\t -- eps = {} : {}", g.eps_s, g.eps_dm);
        spdlog::info("\t -- mass = {} : {}", g.Mass_s, g.Mass_dm);
        spdlog::info("\t -- alpha = {}", g.alpha_glx);
        spdlog::info("\t -- pos = ({}, {}, {})", g.X_glx, g.Y_glx, g.Z_glx);
        spdlog::info("\t -- vel = ({}, {}, {})", g.Vx_glx, g.Vy_glx, g.Vz_glx);
        params.galaxies.push_back(std::move(g));
    }

    return params;
}

} // namespace khrr_galaxy_params