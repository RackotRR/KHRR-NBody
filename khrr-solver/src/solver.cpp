#include "nbody_context.h"
#include "solver-log.h"
#include "solver-fs.h"

#include <grav_params.h>
#include <sim_params.h>
#include <galaxy_params_reader.h>
#include <particle_io.h>

using khrr_common::real3;

static khrr_nbody::ParticleData
make_particles(
    int         N,
    double      mass     = 1.0,
    double      eps2_val = 0.01,
    real3       pos0     = {0,0,0},
    real3       vel0     = {0,0,0})
{
    khrr_nbody::ParticleData p;
    p.n_stars = static_cast<std::size_t>(N);
    p.n_dm    = 0;
    for (int i = 0; i < N; ++i) {
        p.positions .push_back({pos0.x + i, pos0.y, pos0.z});
        p.velocities.push_back(vel0);
        p.masses    .push_back(mass);
        p.eps2      .push_back(eps2_val);
    }
    return p;
}



int main(void) {
    try {

        khrr_solver::log::setup_logging();
        auto project_path = khrr_solver::fs::get_projects_directory() / "base";
        auto grav_params = khrr_grav_params::load_grav_config(
            khrr_solver::fs::get_grav_params_path(project_path).string()
        );
        auto sim_params = khrr_sim_params::load_sim_config(
            khrr_solver::fs::get_sim_params_path(project_path).string()
        );
        auto galaxy_params = khrr_galaxy_params::read_galaxy_params(
            khrr_solver::fs::get_galaxies_params_path(project_path).string()
        );
        auto particles = khrr_particles::load_text_initial(
            khrr_solver::fs::get_project_ini_directory(project_path).string(),
            galaxy_params
        );

        khrr_nbody::NBodyContext ctx;
        ctx.upload(particles);

        std::size_t n_steps = std::max<std::size_t>(
            1ull,
            sim_params.dt_save / grav_params.dtgrav
        );
        std::size_t n_steps_total = std::max<std::size_t>(
            1ull,
            sim_params.tmax / grav_params.dtgrav
        );

        ctx.integrate(
            grav_params.dtgrav,
            n_steps,
            n_steps_total
        );

        auto ti = ctx.timing();
        spdlog::info("steps done: {}", ti.steps_done);
        spdlog::info("steps total: {}", ti.steps_total);
        spdlog::info("elapsed seconds: {}", ti.elapsed_seconds);
        spdlog::info("avg step seconds: {}", ti.avg_step_seconds);
        spdlog::info("estimated remaining: {}", ti.estimated_remaining);

    }
    catch(const std::exception& ex) {
        spdlog::error(ex.what());
    }

    return 0;
}
