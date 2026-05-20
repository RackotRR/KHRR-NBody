#include "nbody_context.h"
#include "nbody_kernels.cuh"

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
    void eval_accel(
        const thrust::device_vector<double3>& pos,
        thrust::device_vector<double3>&       acc)
    {
        int N       = static_cast<int>(n_total);
        int threads = 256;
        int blocks  = (N + threads - 1) / threads;

        kernels::compute_accel<<<blocks, threads>>>(
            thrust::raw_pointer_cast(pos.data()),
            thrust::raw_pointer_cast(d_mass.data()),
            thrust::raw_pointer_cast(d_eps2.data()),
            thrust::raw_pointer_cast(acc.data()),
            N, G);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ── Single PC step ────────────────────────────────────────────────────────
    void step(real dt) {
        int N       = static_cast<int>(n_total);
        int threads = 256;
        int blocks  = (N + threads - 1) / threads;

        auto* p_pos     = thrust::raw_pointer_cast(d_pos.data());
        auto* p_vel     = thrust::raw_pointer_cast(d_vel.data());
        auto* p_acc     = thrust::raw_pointer_cast(d_acc.data());
        auto* p_pos_tmp = thrust::raw_pointer_cast(d_pos_tmp.data());
        auto* p_vel_tmp = thrust::raw_pointer_cast(d_vel_tmp.data());
        auto* p_acc_tmp = thrust::raw_pointer_cast(d_acc_tmp.data());

        // 1. Predict → d_pos_tmp, d_vel_tmp
        kernels::predict<<<blocks, threads>>>(
            p_pos, p_vel, p_acc,
            p_pos_tmp, p_vel_tmp,
            N, dt);
        CUDA_CHECK(cudaDeviceSynchronize());

        // 2. a* = f(x*) → d_acc_tmp
        eval_accel(d_pos_tmp, d_acc_tmp);

        // 3. Correct: (x_n, v_n, a_n, a*) → (d_pos_tmp, d_vel_tmp) reused as output
        //    Note: pos_out != pos_n  (d_pos_tmp vs d_pos) ✓
        kernels::correct<<<blocks, threads>>>(
            p_pos, p_vel, p_acc, p_acc_tmp,
            p_pos_tmp, p_vel_tmp,           // output (overwrite predicted)
            N, dt);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Commit corrected state
        thrust::copy(d_pos_tmp.begin(), d_pos_tmp.end(), d_pos.begin());
        thrust::copy(d_vel_tmp.begin(), d_vel_tmp.end(), d_vel.begin());

        // 4. Recompute a_{n+1} = f(x_{n+1}) for next step seed
        eval_accel(d_pos, d_acc);

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

    // Pre-compute a_0 so the first predict step has valid acceleration.
    impl_->eval_accel(impl_->d_pos, impl_->d_acc);

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