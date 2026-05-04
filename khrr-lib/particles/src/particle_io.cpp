#include "particle_io.h"
#include "double3.h"

#include <fmt/format.h>
#include <cstdio>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <filesystem>

namespace khrr_particles {

    // Helper for binary I/O
    static void read_binary_component(
        const std::string& directory,
        int step,
        const std::string& component_prefix,
        std::size_t expected_count,
        std::vector<double3>& out_pos,
        std::vector<double3>& out_vel,
        std::size_t& out_read_count
    )
    {
        std::string path = fmt::format("{}/{}_{:5d}.bin", directory, component_prefix, step);

        FILE* f = fopen(path.c_str(), "rb");
        if (!f) {
            out_read_count = 0;
            return;
        }

        // Read header: [int count][double time]
        int header_count_int = 0;
        double time = 0.0;
        size_t read_h1 = fread(&header_count_int, sizeof(int), 1, f);
        size_t read_h2 = fread(&time, sizeof(double), 1, f);

        if (read_h1 != 1 || read_h2 != 1) {
            std::cerr << "[Warning] Failed to read header from " << path << "\n";
            fclose(f);
            out_read_count = 0;
            return;
        }

        std::size_t header_count = static_cast<std::size_t>(header_count_int);

        if (header_count != expected_count) {
            std::cerr << "[Warning] " << path << " header says " << header_count
                      << " particles, but expected " << expected_count << ".\n";
        }

        // Limit is min(expected, header_available)
        // We rely on fread return to handle actual file size
        std::size_t limit = std::min(header_count, expected_count);

        // Buffer for reading chunks (6 doubles per particle)
        // 48 bytes per particle. 100k particles = ~4.8MB buffer.
        const std::size_t CHUNK_PARTICLES = 100000;
        std::vector<double> buffer(CHUNK_PARTICLES * 6);

        std::size_t total_read = 0;
        // Reserve memory to avoid reallocations
        out_pos.reserve(out_pos.size() + limit);
        out_vel.reserve(out_vel.size() + limit);

        while (total_read < limit) {
            std::size_t to_read_now = std::min(CHUNK_PARTICLES, limit - total_read);
            std::size_t items_to_read = to_read_now * 6;

            std::size_t items_read = fread(buffer.data(), sizeof(double), items_to_read, f);

            if (items_read == 0) break; // EOF or error

            std::size_t particles_in_chunk = items_read / 6;

            for (std::size_t i = 0; i < particles_in_chunk; ++i) {
                std::size_t idx = i * 6;
                // X Y Z VX VY VZ
                out_pos.push_back({buffer[idx], buffer[idx+1], buffer[idx+2]});
                out_vel.push_back({buffer[idx+3], buffer[idx+4], buffer[idx+5]});
            }
            total_read += particles_in_chunk;

            // If we read incomplete particle set (not multiple of 6), stop.
            if (items_read % 6 != 0) {
                 std::cerr << "[Warning] Corrupt/Short binary data in " << path << "\n";
                 break;
            }
        }

        if (total_read != limit && header_count > 0) {
             std::cerr << "[Warning] Could only read " << total_read << " particles from "
                       << path << " (expected " << limit << " based on header/limit).\n";
        }

        fclose(f);
        out_read_count = total_read;
    }

    void save_binary(
        const std::string& directory,
        int step,
        double time,
        const std::vector<double3>& star_pos,
        const std::vector<double3>& star_vel,
        const std::vector<double3>& dm_pos,
        const std::vector<double3>& dm_vel
    )
    {
        if (star_pos.size() != star_vel.size()) {
            throw std::invalid_argument("Star position and velocity size mismatch");
        }
        if (dm_pos.size() != dm_vel.size()) {
            throw std::invalid_argument("DM position and velocity size mismatch");
        }

        // Create directory if needed
        std::filesystem::create_directories(directory);

        auto write_component = [&](
            const std::string& prefix,
            const std::vector<double3>& pos,
            const std::vector<double3>& vel
        ) {
            std::string filepath = fmt::format("{}/{}_{:5d}.bin", directory, prefix, step);
            FILE* f = fopen(filepath.c_str(), "wb");
            if (!f) throw std::runtime_error("Cannot open " + filepath);

            int n = static_cast<int>(pos.size());
            if (fwrite(&n, sizeof(int), 1, f) != 1 || fwrite(&time, sizeof(double), 1, f) != 1) {
                fclose(f);
                throw std::runtime_error("Failed to write header for " + filepath);
            }

            // Write [x y z vx vy vz] blocks
            for (std::size_t i = 0; i < pos.size(); ++i) {
                fwrite(&pos[i].x, sizeof(double), 1, f);
                fwrite(&pos[i].y, sizeof(double), 1, f);
                fwrite(&pos[i].z, sizeof(double), 1, f);
                fwrite(&vel[i].x, sizeof(double), 1, f);
                fwrite(&vel[i].y, sizeof(double), 1, f);
                fwrite(&vel[i].z, sizeof(double), 1, f);
            }
            fclose(f);
        };

        write_component("S", star_pos, star_vel);
        write_component("DM", dm_pos, dm_vel);
    }

    ParticleData load_binary(
        const std::string& directory,
        int step,
        std::size_t expected_stars,
        std::size_t expected_dm
    )
    {
        ParticleData result;
        // Reserve to prevent fragmentation if expected is accurate
        result.positions.reserve(expected_stars + expected_dm);
        result.velocities.reserve(expected_stars + expected_dm);

        // Stars first
        read_binary_component(directory, step, "S", expected_stars,
                              result.positions, result.velocities, result.n_stars);

        // DM next
        read_binary_component(directory, step, "DM", expected_dm,
                              result.positions, result.velocities, result.n_dm);

        return result;
    }

    ParticleData load_text_initial(
        const std::string& directory,
        const khrr_galaxy_params::SimulationParams& sim_params
    )
    {
        // Calculate rough total for reserve
        std::size_t total_est = 0;
        for(const auto& g : sim_params.galaxies) {
            total_est += g.N_s + g.N_dm;
        }

        ParticleData result;
        result.positions.reserve(total_est);
        result.velocities.reserve(total_est);

        auto read_text_file = [&](
            const std::string& filename,
            std::size_t expected_count
        ) {
            std::string path = fmt::format("{}/{}", directory, filename);
            FILE* f = fopen(path.c_str(), "r");
            if (!f) return (std::size_t)0; // File missing is acceptable (e.g. DM component missing)

            // Read header: [int count][double time] (matches binary format)
            int header_count = 0;
            double header_time = 0.0;
            if (fscanf(f, "%d %lf", &header_count, &header_time) != 2) {
                std::cerr << "[Warning] Failed to read header from " << path << "\n";
                fclose(f);
                return (std::size_t)0;
            }

            if (static_cast<std::size_t>(header_count) != expected_count) {
                std::cerr << "[Warning] " << path << " header says " << header_count
                        << " particles, but expected " << expected_count << ".\n";
            }

            double x, y, z, vx, vy, vz;
            int ret;
            std::size_t read_count = 0;

            // Reserve chunk
            // We push directly to result vectors, but result might realloc.
            // Let's use temp buffer to push all at once if possible
            std::vector<double3> temp_pos, temp_vel;
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

            fclose(f);

            if (read_count != expected_count) {
                std::cerr << "[Warning] " << path << " has " << read_count
                          << " lines, expected " << expected_count << ".\n";
            }
            return read_count;
        };

        std::size_t total_stars = 0;
        // 1. Read all Stars
        for (int i = 0; i < sim_params.M_glx; ++i) {
            const auto& g = sim_params.galaxies[i];
            // Filename format start_S%d.txt. Using galaxy index (usually 1-based in filenames)
            // We check if k_glx is usable (if != 0 or similar) or fallback to i+1.
            // Given struct has k_glx, I'll assume it holds the intended ID.
            std::string fname = fmt::format("start_S{}.txt", g.k_glx);
            total_stars += read_text_file(fname, g.N_s);
        }
        result.n_stars = total_stars;

        std::size_t total_dm = 0;
        // 2. Read all DM
        for (int i = 0; i < sim_params.M_glx; ++i) {
            const auto& g = sim_params.galaxies[i];
            std::string fname = fmt::format("start_DM{}.txt", g.k_glx);
            total_dm += read_text_file(fname, g.N_dm);
        }
        result.n_dm = total_dm;

        return result;
    }

} // namespace khrr_particles