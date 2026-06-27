#include <galaxy_params.h>
#include <galaxy_params_reader.h>
#include <particle_io.h>
#include <filesystem>
#include <iostream>
#include <fmt/format.h>

namespace fs = std::filesystem;

auto read_sample_galaxy_params(fs::path path) {

    khrr_galaxy_params::GalaxiesParams galaxies_params = khrr_galaxy_params::read_galaxy_params(
        path.string()
    );

    std::cout << "read galaxy params:" << std::endl;
    std::cout << "galaxies count: " << galaxies_params.M_glx << std::endl;
    std::cout << "galaxies: " << std::endl;
    for (auto& galaxy : galaxies_params.galaxies) {
        std::cout << "\t galaxy id: " << galaxy.k_glx << std::endl;
        std::cout << "\t N stars: " << galaxy.N_s << std::endl;
        std::cout << "\t N dark matter: " << galaxy.N_dm << std::endl;
        std::cout << "\t mass stars: " << galaxy.Mass_s << std::endl;
        std::cout << "\t mass dark matter: " << galaxy.Mass_dm << std::endl;
        std::cout << "\t eps stars: " << galaxy.eps_s << std::endl;
        std::cout << "\t eps dark matter: " << galaxy.eps_dm << std::endl;
        std::cout << "\t alpha: " << galaxy.alpha_glx << std::endl;
        std::cout << "\t pos xyz: " << fmt::format("{}, {}, {}", galaxy.X_glx, galaxy.Y_glx, galaxy.Z_glx) << std::endl;
        std::cout << "\t vel xyz: " << fmt::format("{}, {}, {}", galaxy.Vx_glx, galaxy.Vy_glx, galaxy.Vz_glx) << std::endl;
        std::cout << std::endl;
    }

    return galaxies_params;
}

auto read_particles_text(const khrr_galaxy_params::GalaxiesParams& galaxies_params) {
    auto particles_data = khrr_particles::load_text_initial(fs::current_path().string(), galaxies_params);

    std::cout << "star particles loaded: " << particles_data.n_stars << std::endl;
    std::cout << "dm particles loaded: " << particles_data.n_dm << std::endl;
    size_t total = particles_data.n_dm + particles_data.n_stars;

    for (size_t i = 0; i < std::min(10ull, total); ++i) {
        std::cout << fmt::format(
            "{}: x {}, vx {}",
            i,
            particles_data.positions[i].x,
            particles_data.velocities[i].x
        ) << std::endl;
    }

    return 0;
}
auto read_particles_bin(const khrr_galaxy_params::GalaxiesParams& galaxies_params) {
    auto particles_data = khrr_particles::load_binary(
        fs::current_path().string(),
        10,
        1000,
        1000
    );

    std::cout << "star particles loaded: " << particles_data.n_stars << std::endl;
    std::cout << "dm particles loaded: " << particles_data.n_dm << std::endl;
    size_t total = particles_data.n_dm + particles_data.n_stars;

    for (size_t i = 0; i < std::min(10ull, total); ++i) {
        std::cout << fmt::format(
            "{}: x {}, vx {}",
            i,
            particles_data.positions[i].x,
            particles_data.velocities[i].x
        ) << std::endl;
    }

    return 0;
}

int main() {
    fs::path galaxy_params_path = fs::current_path() / "__start_galaxies.ini";
    auto sim_params = read_sample_galaxy_params(galaxy_params_path);
    auto particles = read_particles_text(sim_params);

    return 0;
}