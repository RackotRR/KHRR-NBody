#include "nbody_context.h"
#include "nbody_accel.cuh"
#include "nbody_leapfrog.cuh"

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <stdexcept>

// Helper: throw on CUDA error
#define CUDA_CHECK(expr)                                                       \
    do {                                                                        \
        cudaError_t _e = (expr);                                               \
        if (_e != cudaSuccess)                                                 \
            throw std::runtime_error(std::string("CUDA error: ")              \
                                     + cudaGetErrorString(_e));                \
    } while (false)

namespace khrr_nbody {

// ─────────────────────────────────────────────────────────────────────────────
// Impl: owns all GPU state
// ─────────────────────────────────────────────────────────────────────────────
struct NBodyContext::Impl {
    // Current state on device
    thrust::device_vector<double3> d_pos;
    thrust::device_vector<double3> d_vel;
    thrust::device_vector<double3> d_acc;
    thrust::device_vector<double>  d_mass;
    thrust::device_vector<double>  d_eps2;

    // Predictor / corrector work buffers (reused each step)
    thrust::device_vector<double3> d_pos_tmp;
    thrust::device_vector<double3> d_vel_tmp;
    thrust::device_vector<double3> d_acc_tmp;

    std::size_t n_total = 0;
    std::size_t n_stars = 0;
    std::size_t n_dm    = 0;
    real        sim_time = 0.0;
    real        G        = 1.0;
    bool        ready    = false;

    TimingInfo  last_timing;

    explicit Impl(real G_) : G(G_) {}

    // ── Allocation ────────────────────────────────────────────────────────────
    void alloc(std::size_t N) {
        d_pos.resize(N);
        d_vel.resize(N);
        d_acc.resize(N);
        d_mass.resize(N);
        d_eps2.resize(N);
        d_pos_tmp.resize(N);
        d_vel_tmp.resize(N);
        d_acc_tmp.resize(N);
        n_total = N;
    }

    // ── Acceleration evaluation ───────────────────────────────────────────────
    void eval_accel() {
        const int N       = static_cast<int>(n_total);
        // blockDim MUST equal TILE_SIZE so shared-memory cooperative load works.
        const int threads = kernels::TILE_SIZE;
        const int blocks  = (N + threads - 1) / threads;

        kernels::compute_accel_tiled<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_pos .data()),
            thrust::raw_pointer_cast(d_mass.data()),
            thrust::raw_pointer_cast(d_eps2.data()),
            thrust::raw_pointer_cast(d_acc .data()),
            N, G);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void kick(double half_dt) {
        const int N       = static_cast<int>(n_total);
        const int threads = kernels::TILE_SIZE;
        const int blocks  = (N + threads - 1) / threads;

        kernels::leapfrog_kick<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_vel.data()),
            thrust::raw_pointer_cast(d_acc.data()),
            N, half_dt);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    void drift(double dt) {
        const int N       = static_cast<int>(n_total);
        const int threads = kernels::TILE_SIZE;
        const int blocks  = (N + threads - 1) / threads;

        kernels::leapfrog_drift<<<blocks, threads>>>(
            thrust::raw_pointer_cast(d_pos.data()),
            thrust::raw_pointer_cast(d_vel.data()),
            N, dt);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    // ── KDK Leapfrog step ─────────────────────────────────────────────────────
    //
    //   1. Kick:  v_{n+½} = v_n + a_n · dt/2
    //   2. Drift: x_{n+1} = x_n + v_{n+½} · dt
    //   3. Eval:  a_{n+1} = f(x_{n+1})          ← only ONE force eval per step
    //   4. Kick:  v_{n+1} = v_{n+½} + a_{n+1} · dt/2
    //
    void step(double dt) {
        const double half_dt = 0.5 * dt;
        kick (half_dt);   // 1
        drift(dt);        // 2
        eval_accel();     // 3
        kick (half_dt);   // 4
        sim_time += dt;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// NBodyContext public API
// ─────────────────────────────────────────────────────────────────────────────

NBodyContext::NBodyContext(real G)
    : impl_(std::make_unique<Impl>(G)) {}

NBodyContext::~NBodyContext() = default;

// ── upload ────────────────────────────────────────────────────────────────────
void NBodyContext::upload(const NBodyParticles& p)
{
    if (p.size() == 0)
        throw std::invalid_argument(
            "NBodyContext::upload: particle set is empty");
    if (!p.is_valid())
        throw std::invalid_argument(
            "NBodyContext::upload: particle arrays have inconsistent sizes "
            "or n_stars + n_dm != N");

    impl_->alloc(p.size());
    impl_->n_stars   = p.n_stars;
    impl_->n_dm      = p.n_dm;
    impl_->sim_time  = 0.0;

    // real3 (host) vs double3 (cuda) <-- these are the same
    const auto& h_positions = reinterpret_cast<const std::vector<double3>&>(p.positions);
    const auto& h_velocities = reinterpret_cast<const std::vector<double3>&>(p.velocities);
    thrust::copy(h_positions.begin(), h_positions.end(), impl_->d_pos.begin());
    thrust::copy(h_velocities.begin(), h_velocities.end(), impl_->d_vel.begin());
    thrust::copy(p.masses.begin(),     p.masses.end(),     impl_->d_mass.begin());
    thrust::copy(p.eps2.begin(),       p.eps2.end(),       impl_->d_eps2.begin());

    // Seed a_0 so the first kick has valid accelerations.
    impl_->eval_accel();

    impl_->ready = true;
}

// ── download ──────────────────────────────────────────────────────────────────
NBodyParticles NBodyContext::download() const
{
    if (!impl_->ready)
        throw std::logic_error(
            "NBodyContext::download: no particles loaded; call upload() first");

    NBodyParticles out;
    std::size_t N = impl_->n_total;
    out.positions.resize(N);
    out.velocities.resize(N);
    out.masses.resize(N);
    out.eps2.resize(N);
    out.n_stars = impl_->n_stars;
    out.n_dm    = impl_->n_dm;

    // real3 (host) vs double3 (cuda) <-- these are the same
    auto& h_positions = reinterpret_cast<std::vector<double3>&>(out.positions);
    auto& h_velocities = reinterpret_cast<std::vector<double3>&>(out.velocities);
    thrust::copy(impl_->d_pos.begin(),  impl_->d_pos.end(),  h_positions.begin());
    thrust::copy(impl_->d_vel.begin(),  impl_->d_vel.end(),  h_velocities.begin());
    thrust::copy(impl_->d_mass.begin(), impl_->d_mass.end(), out.masses.begin());
    thrust::copy(impl_->d_eps2.begin(), impl_->d_eps2.end(), out.eps2.begin());
    return out;
}

// ── integrate ─────────────────────────────────────────────────────────────────
void NBodyContext::integrate(
    real dt,
    std::size_t n_steps,
    std::size_t total_steps
)
{
    if (!impl_->ready)
        throw std::logic_error(
            "NBodyContext::integrate: call upload() before integrate()");
    if (dt <= 0.0)
        throw std::invalid_argument(
            "NBodyContext::integrate: dt must be > 0");
    if (n_steps == 0) return;

    using Clock     = std::chrono::steady_clock;
    using Seconds   = std::chrono::duration<real>;

    auto& ti = impl_->last_timing;
    ti = TimingInfo{};
    ti.steps_total = total_steps;

    const auto t_start = Clock::now();

    for (std::size_t s = 0; s < n_steps; ++s) {
        impl_->step(dt);
        ++ti.steps_done;

        ti.elapsed_seconds  = Seconds(Clock::now() - t_start).count();
        ti.avg_step_seconds = ti.elapsed_seconds / static_cast<real>(ti.steps_done);

        const std::size_t global_done =
            (total_steps > 0) ? (total_steps - n_steps + ti.steps_done) : ti.steps_done;

        if (total_steps > 0 && global_done < total_steps) {
            ti.estimated_remaining =
                ti.avg_step_seconds * static_cast<real>(total_steps - global_done);
        }

        // Progress printout every 10 steps and on the final step
        if ((s + 1) % 10 == 0 || s + 1 == n_steps) {
            if (total_steps > 0) {
                std::printf(
                    "[NBody] step %5zu / %5zu | sim_t = %.4g | "
                    "elapsed = %.2f s | ETA = %.2f s\n",
                    global_done, total_steps,
                    impl_->sim_time,
                    ti.elapsed_seconds, ti.estimated_remaining);
            } else {
                std::printf(
                    "[NBody] step %5zu | sim_t = %.4g | elapsed = %.2f s\n",
                    ti.steps_done, impl_->sim_time, ti.elapsed_seconds);
            }
            std::fflush(stdout);
        }
    }
}

// ── accessors ─────────────────────────────────────────────────────────────────
real        NBodyContext::time()        const noexcept { return impl_->sim_time; }
std::size_t NBodyContext::n_particles() const noexcept { return impl_->n_total;  }
TimingInfo  NBodyContext::timing()      const noexcept { return impl_->last_timing; }

} // namespace khrr_nbody