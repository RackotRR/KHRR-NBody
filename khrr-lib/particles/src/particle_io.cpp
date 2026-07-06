#include "particle_io.h"
#include <khrr_types.h>

#include <cstdio>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <range/v3/all.hpp>

#include <spdlog/spdlog.h>
#include <fmt/format.h>

namespace khrr_particles {

    using khrr_galaxy_params::GalaxiesParams;
    using khrr_galaxy_params::GalaxyParams;
    using khrr_galaxy_params::count_dm;
    using khrr_galaxy_params::count_stars;

    constexpr const char* STAR_PREFIX = "S";
    constexpr const char* DM_PREFIX = "DM";

    // Helper for binary I/O
    static void read_binary_component(
        const std::string& directory,
        int step,
        const std::string& component_prefix,
        std::size_t expected_count,
        std::vector<real3>& out_pos,
        std::vector<real3>& out_vel,
        std::size_t& out_read_count
    )
    {
        std::string path = fmt::format("{}/{}_{:5d}.bin", directory, component_prefix, step);
        spdlog::info("Read binary component from {}", path);

        FILE* f = fopen(path.c_str(), "rb");
        if (!f) {
            out_read_count = 0;
            return;
        }

        // Read header: [int count][real time]
        int header_count_int = 0;
        real time = 0.0;
        if (fread(&header_count_int, sizeof(int),  1, f) != 1 ||
            fread(&time,             sizeof(real), 1, f) != 1
        ) {
            spdlog::error("Failed to read header from {}", path);
            fclose(f);
            out_read_count = 0;
            return;
        }

        std::size_t header_count = static_cast<std::size_t>(header_count_int);

        if (header_count != expected_count) {
            spdlog::error(
                "{} header says {} particles, but expected {}",
                path,
                header_count,
                expected_count
            );
        }

        // Limit is min(expected, header_available)
        // We rely on fread return to handle actual file size
        std::size_t limit = std::min(header_count, expected_count);

        // Buffer for reading chunks (6 doubles per particle)
        // 48 bytes per particle. 100k particles = ~4.8MB buffer.
        const std::size_t CHUNK_PARTICLES = 100'000;
        std::vector<real> buffer(CHUNK_PARTICLES * 6);

        std::size_t total_read = 0;
        // Reserve memory to avoid reallocations
        out_pos.reserve(out_pos.size() + limit);
        out_vel.reserve(out_vel.size() + limit);

        while (total_read < limit) {
            std::size_t to_read_now = std::min(CHUNK_PARTICLES, limit - total_read);
            std::size_t items_to_read = to_read_now * 6;
            std::size_t items_read = fread(buffer.data(), sizeof(real), items_to_read, f);
            if (items_read == 0) break; // EOF or error

            std::size_t particles_in_chunk = items_read / 6;
            for (std::size_t i = 0; i < particles_in_chunk; ++i) {
                std::size_t idx = i * 6;
                // X Y Z VX VY VZ
                out_pos.push_back({buffer[idx],     buffer[idx + 1], buffer[idx + 2]});
                out_vel.push_back({buffer[idx + 3], buffer[idx + 4], buffer[idx + 5]});
            }
            total_read += particles_in_chunk;

            // If we read incomplete particle set (not multiple of 6), stop.
            if (items_read % 6 != 0) {
                spdlog::error("Corrupt/Short binary data in {}", path);
                break;
            }
        }

        if (total_read != limit && header_count > 0) {
            spdlog::error(
                "Could only read {} particles from {} (expected {} based on header/limit)",
                total_read,
                path,
                limit
            );
        }

        fclose(f);
        out_read_count = total_read;
    }

    void save_binary(
        const std::string& directory,
        int step,
        real time,
        const ParticleData& particles,
        const GalaxiesParams& galaxies_params
    )
    {
        spdlog::info("Save binary");

        if (false == particles.is_valid()) {
            throw std::invalid_argument("Invalid particles passed");
        }

        // Create directory if needed
        std::filesystem::create_directories(directory);

        auto write_component = [&](
            const std::string& prefix,
            auto&& pos,
            auto&& vel
        ) {
            spdlog::info("Write component {}", prefix);

            std::string filepath = fmt::format("{}/{}_{:5d}.bin", directory, prefix, step);
            FILE* f = fopen(filepath.c_str(), "wb");
            if (!f) {
                throw std::runtime_error("Cannot open " + filepath);
            }

            int n = static_cast<int>(pos.size());
            if (fwrite(&n,    sizeof(int),  1, f) != 1 ||
                fwrite(&time, sizeof(real), 1, f) != 1
            ) {
                fclose(f);
                throw std::runtime_error("Failed to write header for " + filepath);
            }

            // Write [x y z vx vy vz] blocks
            for (auto&& [p, v] : ranges::views::zip(pos, vel)) {
                fwrite(&p.x, sizeof(real), 1, f);
                fwrite(&p.y, sizeof(real), 1, f);
                fwrite(&p.z, sizeof(real), 1, f);
                fwrite(&v.x, sizeof(real), 1, f);
                fwrite(&v.y, sizeof(real), 1, f);
                fwrite(&v.z, sizeof(real), 1, f);
            }
            fclose(f);
        };

        size_t N_s = count_stars(galaxies_params);
        size_t N_dm = count_dm(galaxies_params);
        size_t N = N_s + N_dm;
        write_component(
            STAR_PREFIX,
            particles.positions
                | ranges::views::slice(0ull, N_s),
            particles.velocities
                | ranges::views::slice(0ull, N_s)
        );
        write_component(
            "DM",
            particles.positions
                | ranges::views::slice(N_s, N),
            particles.velocities
                | ranges::views::slice(N_s, N)
        );
    }

    ParticleData load_binary(
        const std::string& directory,
        int step,
        const GalaxiesParams& galaxies_params
    )
    {
        ParticleData result;

        size_t expected_stars = count_stars(galaxies_params);
        size_t expected_dm = count_dm(galaxies_params);
        size_t expected_all = expected_stars + expected_dm;

        // Reserve to prevent fragmentation if expected is accurate
        result.positions.reserve(expected_all);
        result.velocities.reserve(expected_all);

        // Stars first
        read_binary_component(
            directory,
            step,
            "S",
            expected_stars,
            result.positions,
            result.velocities,
            result.n_stars
        );

        // DM next
        read_binary_component(
            directory,
            step,
            "DM",
            expected_dm,
            result.positions,
            result.velocities,
            result.n_dm
        );

        // fill in mass and eps

        result.masses.reserve(expected_all);
        result.eps2.reserve(expected_all);

        auto append_component_with_mass_eps = [&galaxies_params, &result](
            auto get_N,
            auto get_mass,
            auto get_eps
        )
        {
            for (const auto& galaxy : galaxies_params.galaxies) {
                size_t n = std::invoke(get_N, std::cref(galaxy));
                double mass = std::invoke(get_mass, std::cref(galaxy));
                double eps = std::invoke(get_eps, std::cref(galaxy));

                ranges::fill_n(
                    ranges::back_inserter(result.masses),
                    n,
                    mass
                );
                ranges::fill_n(
                    ranges::back_inserter(result.eps2),
                    n,
                    eps * eps
                );
            }
        };

        append_component_with_mass_eps(
            &GalaxyParams::N_s,
            &GalaxyParams::Mass_s,
            &GalaxyParams::eps_s
        );
        append_component_with_mass_eps(
            &GalaxyParams::N_dm,
            &GalaxyParams::Mass_dm,
            &GalaxyParams::eps_dm
        );

        return result;
    }

    ParticleData load_text_initial(
        const std::string& directory,
        const khrr_galaxy_params::GalaxiesParams& galaxies_params
    )
    {
        spdlog::info("load particles from txt");

        // Calculate rough total for reserve
        std::size_t total_est = 0;
        for(const auto& g : galaxies_params.galaxies) {
            total_est += g.N_s + g.N_dm;
        }
        spdlog::debug("particles count estimation: {}", total_est);

        ParticleData result;
        result.positions.reserve(total_est);
        result.velocities.reserve(total_est);

        auto read_text_file = [&](
            const std::string& filename,
            double eps2,
            double mass,
            std::size_t expected_count
        ) {
            std::string path = fmt::format("{}/{}", directory, filename);
            spdlog::info("read text file '{}'", path);
            spdlog::debug("expected particles count (galaxy-specified): {}", expected_count);

            FILE* f = fopen(path.c_str(), "r");
            if (!f) return (std::size_t)0; // File missing is acceptable (e.g. DM component missing)

            // Read header: [int count][real time] (matches binary format)
            int header_count = 0;
            real header_time = 0.0;
            if (fscanf(f, "%d %lf", &header_count, &header_time) != 2) {
                spdlog::error("Failed to read header");
                fclose(f);
                return (std::size_t)0;
            }
            spdlog::debug("header particles count: {}", header_count);

            if (static_cast<std::size_t>(header_count) != expected_count) {
                spdlog::error(
                    "Warning: header says {} particles, but expected {}",
                    header_count,
                    expected_count
                );
            }

            real x, y, z, vx, vy, vz;
            int ret;
            std::size_t read_count = 0;

            // Reserve chunk
            // We push directly to result vectors, but result might realloc.
            // Let's use temp buffer to push all at once if possible
            std::vector<real3> temp_pos, temp_vel;
            temp_pos.reserve(std::min(expected_count, (std::size_t)100000));
            temp_vel.reserve(std::min(expected_count, (std::size_t)100000));

            while ((ret = fscanf(f, "%lf %lf %lf %lf %lf %lf", &x, &y, &z, &vx, &vy, &vz)) == 6) {
                temp_pos.push_back({x, y, z});
                temp_vel.push_back({vx, vy, vz});
                ++read_count;
                if (read_count == expected_count) break;
            }

            result.positions.insert(result.positions.end(), temp_pos.begin(), temp_pos.end());
            result.velocities.insert(result.velocities.end(), temp_vel.begin(), temp_vel.end());

            std::fill_n(
                std::back_inserter(result.masses),
                read_count,
                mass
            );
            std::fill_n(
                std::back_inserter(result.eps2),
                read_count,
                eps2
            );

            fclose(f);

            if (read_count != expected_count) {
                spdlog::error(
                    "Warning: file has {} lines, expected {}",
                    read_count,
                    expected_count
                );
            }
            return read_count;
        };

        std::size_t total_stars = 0;
        // 1. Read all Stars
        for (int i = 0; i < galaxies_params.M_glx; ++i) {
            const auto& g = galaxies_params.galaxies[i];
            // Filename format start_S%d.txt. Using galaxy index (usually 1-based in filenames)
            // We check if k_glx is usable (if != 0 or similar) or fallback to i+1.
            // Given struct has k_glx, I'll assume it holds the intended ID.
            std::string fname = fmt::format("start_S{}.txt", g.k_glx);
            total_stars += read_text_file(
                fname,
                g.Mass_s,
                g.eps_s * g.eps_s,
                g.N_s
            );
        }
        result.n_stars = total_stars;
        spdlog::info("n stars: {}", result.n_stars);

        std::size_t total_dm = 0;
        // 2. Read all DM
        for (int i = 0; i < galaxies_params.M_glx; ++i) {
            const auto& g = galaxies_params.galaxies[i];
            std::string fname = fmt::format("start_DM{}.txt", g.k_glx);
            total_dm += read_text_file(
                fname,
                g.Mass_dm,
                g.eps_dm * g.eps_dm,
                g.N_dm
            );
        }
        result.n_dm = total_dm;
        spdlog::info("n dark matter: {}", result.n_dm);
        spdlog::info("n total: {}", result.n_stars + result.n_dm);

        return result;
    }

} // namespace khrr_particles