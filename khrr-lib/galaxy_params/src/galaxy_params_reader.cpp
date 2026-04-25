#include "galaxy_params.h"
#include "galaxy_params_reader.h"
#include <fmt/format.h>

#include <stdexcept>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <range/v3/view/filter.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/range/conversion.hpp>
#include <range/v3/algorithm/all_of.hpp>

namespace khrr_galaxy_params {

namespace {
    /// Removes inline comments ("----" or "#") and trims whitespace.
    std::string clean_line(std::string line) {
        if (auto pos = line.find("----"); pos != std::string::npos) line.erase(pos);
        if (auto pos = line.find('#'); pos != std::string::npos) line.erase(pos);
        line.erase(0, line.find_first_not_of(" \t\r\n"));
        line.erase(line.find_last_not_of(" \t\r\n") + 1);
        return line;
    }

    /// Parses a line into numeric values, treating ',' and whitespace as delimiters.
    template<typename T>
    std::vector<T> parse_values(const std::string& line) {
        std::vector<T> result;
        std::string token;
        std::istringstream ss(line);

        while (std::getline(ss, token, ',')) {
            std::istringstream token_stream(token);
            T val;
            while (token_stream >> val) {
                result.push_back(val);
            }
        }
        return result;
    }
} // namespace


SimulationParams read_galaxy_params(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error(fmt::format("Cannot open configuration file: {}", filepath));
    }

    std::vector<std::string> lines;
    std::string raw;
    while (std::getline(file, raw)) {
        std::string cleaned = clean_line(raw);
        if (!cleaned.empty()) lines.push_back(std::move(cleaned));
    }

    if (lines.empty()) throw std::runtime_error("File is empty or contains only comments.");

    auto m_vals = parse_values<int>(lines[0]);
    if (m_vals.size() != 1) throw std::runtime_error("Invalid format on M_glx line.");
    int M_glx = m_vals[0];
    if (M_glx <= 0) throw std::runtime_error("M_glx must be strictly positive.");

    SimulationParams params{M_glx};
    params.galaxies.reserve(M_glx);

    constexpr std::size_t LINES_PER_GALAXY = 7;
    const std::size_t expected_total = 1 + static_cast<std::size_t>(M_glx) * LINES_PER_GALAXY;
    if (lines.size() < expected_total) {
        throw std::runtime_error(fmt::format("File contains fewer configuration lines than required for M_glx={}", M_glx));
    }

    for (int i = 0; i < M_glx; ++i) {
        GalaxyParams g{};
        std::size_t base = 1 + i * LINES_PER_GALAXY;

        auto idx_vals = parse_values<int>(lines[base]);
        g.k_glx = idx_vals[0];
        if (g.k_glx != i) {
            throw std::runtime_error(fmt::format("Mismatched galaxy index at line {} (expected {}, got {})", base + 1, i, g.k_glx));
        }

        auto n_vals = parse_values<std::size_t>(lines[base + 1]);
        if (n_vals.size() != 2) throw std::runtime_error(fmt::format("Expected N_s,N_dm at line {}", base + 2));
        g.N_s = n_vals[0];
        g.N_dm = n_vals[1];
        if (g.N_s == 0 || g.N_dm == 0 || g.N_s % 1024 != 0 || g.N_dm % 1024 != 0) {
            throw std::runtime_error("N_s and N_dm must be >0 and multiples of 1024.");
        }

        auto m_comp = parse_values<double>(lines[base + 2]);
        if (m_comp.size() != 2) throw std::runtime_error(fmt::format("Expected Mass_s,Mass_dm at line {}", base + 3));
        g.Mass_s = m_comp[0];
        g.Mass_dm = m_comp[1];
        if (g.Mass_s <= 0.0 || g.Mass_dm <= 0.0) throw std::runtime_error("Masses must be strictly positive.");

        auto eps = parse_values<double>(lines[base + 3]);
        if (eps.size() != 2) throw std::runtime_error(fmt::format("Expected eps_s,eps_dm at line {}", base + 4));
        g.eps_s = eps[0];
        g.eps_dm = eps[1];
        if (g.eps_s < 0.0 || g.eps_dm < 0.0) throw std::runtime_error("Softening lengths cannot be negative.");

        auto alpha = parse_values<double>(lines[base + 4]);
        if (alpha.size() != 1) throw std::runtime_error(fmt::format("Expected alpha_glx at line {}", base + 5));
        g.alpha_glx = alpha[0];

        auto pos = parse_values<double>(lines[base + 5]);
        if (pos.size() != 3) throw std::runtime_error(fmt::format("Expected X,Y,Z at line {}", base + 6));
        g.X_glx = pos[0]; g.Y_glx = pos[1]; g.Z_glx = pos[2];

        auto vel = parse_values<double>(lines[base + 6]);
        if (vel.size() != 3) throw std::runtime_error(fmt::format("Expected Vx,Vy,Vz at line {}", base + 7));
        g.Vx_glx = vel[0]; g.Vy_glx = vel[1]; g.Vz_glx = vel[2];

        // Lazy validation with range-v3
        std::vector<double> checks = {g.alpha_glx, g.Mass_s, g.Mass_dm, g.eps_s, g.eps_dm,
                                      g.X_glx, g.Y_glx, g.Z_glx, g.Vx_glx, g.Vy_glx, g.Vz_glx};
        if (!ranges::all_of(checks, [](double v){ return std::isfinite(v); })) {
            throw std::runtime_error(fmt::format("Detected NaN or Inf in galaxy parameters (index {})", i));
        }

        params.galaxies.push_back(std::move(g));
    }

    return params;
}

} // namespace nbody