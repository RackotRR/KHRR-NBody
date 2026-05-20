#pragma once

#include <vector>
#include <cstddef>
#include <memory>
#include <stdexcept>

#include <khrr_types.h>

namespace khrr_nbody {

using khrr_common::real;
using khrr_common::real3;

// ─────────────────────────────────────────────────────────────────────────────
// Host-side particle container
// Layout mirrors khrr_particles::ParticleData for easy conversion (see nbody_compat.h).
// ─────────────────────────────────────────────────────────────────────────────
struct NBodyParticles {
    std::vector<real3> positions;   ///< [N] world-space positions
    std::vector<real3> velocities;  ///< [N] velocities
    std::vector<real>  masses;      ///< [N] masses (must be >= 0)
    std::vector<real>  eps2;        ///< [N] per-particle softening² (>= 0)
    std::size_t          n_stars = 0;
    std::size_t          n_dm    = 0;

    std::size_t size() const noexcept { return positions.size(); }

    /// True iff every sub-array has the same length and n_stars+n_dm == N.
    bool is_valid() const noexcept {
        const std::size_t n = positions.size();
        return !positions.empty()          &&
               velocities.size() == n      &&
               masses.size()     == n      &&
               eps2.size()       == n      &&
               (n_stars + n_dm)  == n;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Timing information produced by integrate()
// ─────────────────────────────────────────────────────────────────────────────
struct TimingInfo {
    real      elapsed_seconds     = 0.0;
    real      avg_step_seconds    = 0.0;
    real      estimated_remaining = 0.0; ///< only valid when total_steps was set
    std::size_t steps_done          = 0;
    std::size_t steps_total         = 0;   ///< as passed by caller (0 = unknown)
};

// ─────────────────────────────────────────────────────────────────────────────
// NBodyContext
//
// Owns all GPU memory.  Typical usage:
//   NBodyContext ctx;
//   ctx.upload(initial_particles);
//   ctx.integrate(dt, n_steps, total_steps);
//   auto result = ctx.download();
// ─────────────────────────────────────────────────────────────────────────────
class NBodyContext {
public:
    /// @param G  Gravitational constant (default 1.0 for dimensionless units).
    explicit NBodyContext(real G = 1.0);
    ~NBodyContext();

    NBodyContext(const NBodyContext&)             = delete;
    NBodyContext& operator=(const NBodyContext&)  = delete;
    NBodyContext(NBodyContext&&)                  = default;
    NBodyContext& operator=(NBodyContext&&)       = default;

    // ── Data transfer ────────────────────────────────────────────────────────

    /// Upload particles to GPU and reset simulation time to 0.
    /// Computes initial accelerations immediately.
    /// @throws std::invalid_argument  particles.is_valid() == false OR empty.
    void upload(const NBodyParticles& particles);

    /// Download current particle state from GPU to host.
    /// @throws std::logic_error  if upload() was never called.
    NBodyParticles download() const;

    // ── Integration ──────────────────────────────────────────────────────────

    /// Advance simulation using a 2nd-order Predictor-Corrector scheme.
    ///
    /// Each step:
    ///   1. Predict:   x* = x + v·dt + ½·a·dt²,  v* = v + a·dt
    ///   2. Eval:      a* = f(x*)
    ///   3. Correct:   v' = v + ½(a+a*)·dt,  x' = x + ½(v+v')·dt
    ///   4. Re-eval:   a_{n+1} = f(x')   (seed for next step)
    ///
    /// @param dt           Time-step (must be > 0).
    /// @param n_steps      Steps to execute  (0 == no-op).
    /// @param total_steps  Grand total steps expected (for ETA; 0 = unknown).
    /// @throws std::logic_error       if upload() was never called.
    /// @throws std::invalid_argument  if dt <= 0.
    void integrate(real dt, std::size_t n_steps, std::size_t total_steps = 0);

    // ── State queries ────────────────────────────────────────────────────────

    real      time()        const noexcept; ///< Current simulation time.
    std::size_t n_particles() const noexcept; ///< 0 before first upload().
    TimingInfo  timing()      const noexcept; ///< From last integrate() call.

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace khrr_nbody