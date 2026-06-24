#include "nbody_context.h"
#include "solver-log.h"

using khrr_common::real3;

static khrr_nbody::NBodyParticles
make_particles(
    int         N,
    double      mass     = 1.0,
    double      eps2_val = 0.01,
    real3       pos0     = {0,0,0},
    real3       vel0     = {0,0,0})
{
    khrr_nbody::NBodyParticles p;
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
    khrr_solver::log::setup_logging();

    khrr_nbody::NBodyContext ctx;
    ctx.upload(make_particles(4));
    ctx.integrate(0.01, 5, 20);

    auto ti = ctx.timing();
    spdlog::info("steps done: {}", ti.steps_done);
    spdlog::info("steps total: {}", ti.steps_total);
    spdlog::info("elapsed seconds: {}", ti.elapsed_seconds);
    spdlog::info("avg step seconds: {}", ti.avg_step_seconds);
    spdlog::info("estimated remaining: {}", ti.estimated_remaining);

    return 0;
}
