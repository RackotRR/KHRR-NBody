// Unit tests for khrr_nbody using Catch2 v3.
//
// Coverage:
//   Invalid-input / boundary cases  → guard against crashes & silent errors.
//   Integration correctness         → physical sanity checks.

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <cmath>
#include "nbody_context.h"

using namespace khrr_nbody;
using Catch::Matchers::WithinRel;
using Catch::Matchers::WithinAbs;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static ParticleData make_particles(
    int         N,
    double      mass     = 1.0,
    double      eps2_val = 0.01,
    real3       pos0     = {0,0,0},
    real3       vel0     = {0,0,0})
{
    ParticleData p;
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

// ─────────────────────────────────────────────────────────────────────────────
// Section 1: ParticleData validation
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("ParticleData::is_valid rejects inconsistent data", "[particles]")
{
    SECTION("empty → invalid") {
        ParticleData p;
        CHECK_FALSE(p.is_valid());
    }

    SECTION("mismatched velocities") {
        auto p = make_particles(4);
        p.velocities.pop_back();
        CHECK_FALSE(p.is_valid());
    }

    SECTION("mismatched masses") {
        auto p = make_particles(4);
        p.masses.pop_back();
        CHECK_FALSE(p.is_valid());
    }

    SECTION("mismatched eps2") {
        auto p = make_particles(4);
        p.eps2.pop_back();
        CHECK_FALSE(p.is_valid());
    }

    SECTION("n_stars + n_dm != N") {
        auto p = make_particles(4);
        p.n_dm = 1; // now 5 != 4
        CHECK_FALSE(p.is_valid());
    }

    SECTION("well-formed → valid") {
        auto p = make_particles(3);
        CHECK(p.is_valid());
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 2: upload() error handling
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("NBodyContext::upload throws on bad input", "[upload]")
{
    NBodyContext ctx;

    SECTION("empty particle set") {
        ParticleData empty;
        empty.n_stars = 0; empty.n_dm = 0;
        CHECK_THROWS_AS(ctx.upload(empty), std::invalid_argument);
    }

    SECTION("inconsistent arrays") {
        auto p = make_particles(5);
        p.masses.pop_back();           // now size 4, not 5
        CHECK_THROWS_AS(ctx.upload(p), std::invalid_argument);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 3: integrate() / download() without prior upload
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("NBodyContext::integrate / download without upload throws", "[state]")
{
    NBodyContext ctx;

    SECTION("integrate without upload") {
        CHECK_THROWS_AS(ctx.integrate(0.01, 1), std::logic_error);
    }

    SECTION("download without upload") {
        CHECK_THROWS_AS(ctx.download(), std::logic_error);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 4: integrate() bad dt
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("NBodyContext::integrate rejects non-positive dt", "[integrate]")
{
    NBodyContext ctx;
    ctx.upload(make_particles(2));

    SECTION("dt == 0") {
        CHECK_THROWS_AS(ctx.integrate(0.0, 10), std::invalid_argument);
    }

    SECTION("dt < 0") {
        CHECK_THROWS_AS(ctx.integrate(-1.0, 10), std::invalid_argument);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 5: zero steps is a no-op
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("integrate with n_steps=0 is a no-op", "[integrate]")
{
    NBodyContext ctx;
    ctx.upload(make_particles(3));

    REQUIRE_NOTHROW(ctx.integrate(0.1, 0));
    CHECK(ctx.time() == 0.0);

    auto out = ctx.download();
    auto ref = make_particles(3);
    for (std::size_t i = 0; i < out.size(); ++i) {
        CHECK_THAT(out.positions[i].x, WithinAbs(ref.positions[i].x, 1e-12));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 6: upload / download round-trip (identity check before any steps)
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("upload then immediate download preserves positions and velocities", "[roundtrip]")
{
    auto ref = make_particles(5, 2.5, 0.1, {1.0, 2.0, 3.0}, {-0.5, 0.0, 0.5});

    NBodyContext ctx;
    ctx.upload(ref);
    auto out = ctx.download();

    REQUIRE(out.size()    == ref.size());
    REQUIRE(out.n_stars   == ref.n_stars);
    REQUIRE(out.n_dm      == ref.n_dm);

    for (std::size_t i = 0; i < ref.size(); ++i) {
        CHECK_THAT(out.positions [i].x, WithinAbs(ref.positions [i].x, 1e-14));
        CHECK_THAT(out.positions [i].y, WithinAbs(ref.positions [i].y, 1e-14));
        CHECK_THAT(out.positions [i].z, WithinAbs(ref.positions [i].z, 1e-14));
        CHECK_THAT(out.velocities[i].x, WithinAbs(ref.velocities[i].x, 1e-14));
        CHECK_THAT(out.masses    [i],   WithinAbs(ref.masses    [i],   1e-14));
        CHECK_THAT(out.eps2      [i],   WithinAbs(ref.eps2      [i],   1e-14));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 7: single isolated particle — free drift
//
// With N=1 there is no gravitational source, so a=0 at all times.
// The integrator must reproduce uniform rectilinear motion exactly.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("Single particle drifts with constant velocity (no force)", "[physics]")
{
    ParticleData p;
    p.n_stars = 1; p.n_dm = 0;
    p.positions .push_back({0.0, 0.0, 0.0});
    p.velocities.push_back({1.0, 0.0, 0.0}); // vx = 1
    p.masses    .push_back(1.0);
    p.eps2      .push_back(0.01);

    NBodyContext ctx;
    ctx.upload(p);

    const double dt      = 0.01;
    const int    n_steps = 100;
    ctx.integrate(dt, n_steps);

    const double T   = dt * n_steps;       // expected time = 1.0
    const double x_expected = 1.0 * T;     // v·t

    auto out = ctx.download();
    CHECK_THAT(ctx.time(),              WithinAbs(T, 1e-12));
    CHECK_THAT(out.positions[0].x,      WithinAbs(x_expected, 1e-10));
    CHECK_THAT(out.positions[0].y,      WithinAbs(0.0,        1e-14));
    CHECK_THAT(out.velocities[0].x,     WithinAbs(1.0,        1e-12));
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 8: two-body momentum conservation
//
// Total linear momentum P = Σ m_i v_i must be conserved to integration accuracy.
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("Two-body system conserves total linear momentum", "[physics]")
{
    ParticleData p;
    p.n_stars = 2; p.n_dm = 0;
    // Place on x-axis, same mass, symmetric velocities
    p.positions .push_back({-1.0, 0.0, 0.0});
    p.positions .push_back({ 1.0, 0.0, 0.0});
    p.velocities.push_back({ 0.0, 0.5, 0.0});
    p.velocities.push_back({ 0.0,-0.5, 0.0});
    p.masses    .push_back(1.0);
    p.masses    .push_back(1.0);
    p.eps2      .push_back(0.1);
    p.eps2      .push_back(0.1);

    // Initial momentum
    double px0 = 0.0, py0 = 0.0, pz0 = 0.0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        px0 += p.masses[i] * p.velocities[i].x;
        py0 += p.masses[i] * p.velocities[i].y;
        pz0 += p.masses[i] * p.velocities[i].z;
    }

    NBodyContext ctx(1.0);
    ctx.upload(p);
    ctx.integrate(0.001, 500);

    auto out = ctx.download();
    double px = 0.0, py = 0.0, pz = 0.0;
    for (std::size_t i = 0; i < out.size(); ++i) {
        px += out.masses[i] * out.velocities[i].x;
        py += out.masses[i] * out.velocities[i].y;
        pz += out.masses[i] * out.velocities[i].z;
    }

    CHECK_THAT(px, WithinAbs(px0, 1e-10));
    CHECK_THAT(py, WithinAbs(py0, 1e-10));
    CHECK_THAT(pz, WithinAbs(pz0, 1e-10));
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 9: simulation time advances correctly
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("Simulation time advances by dt * n_steps", "[time]")
{
    NBodyContext ctx;
    ctx.upload(make_particles(2));

    ctx.integrate(0.1, 7);
    CHECK_THAT(ctx.time(), WithinAbs(0.7, 1e-12));

    ctx.integrate(0.05, 4);
    CHECK_THAT(ctx.time(), WithinAbs(0.7 + 0.2, 1e-12));
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 10: TimingInfo is populated after integrate()
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("TimingInfo is populated after integrate()", "[timing]")
{
    NBodyContext ctx;
    ctx.upload(make_particles(4));
    ctx.integrate(0.01, 5, 20);

    auto ti = ctx.timing();
    CHECK(ti.steps_done       == 5);
    CHECK(ti.steps_total      == 20);
    CHECK(ti.elapsed_seconds  >  0.0);
    CHECK(ti.avg_step_seconds >  0.0);
    CHECK(ti.estimated_remaining >= 0.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Section 11: re-upload resets state
// ─────────────────────────────────────────────────────────────────────────────
TEST_CASE("Re-uploading resets simulation time and particle count", "[state]")
{
    NBodyContext ctx;
    ctx.upload(make_particles(3));
    ctx.integrate(0.1, 10);
    REQUIRE(ctx.time() > 0.0);

    // Re-upload with different particle count
    ctx.upload(make_particles(5));
    CHECK(ctx.time()        == 0.0);
    CHECK(ctx.n_particles() == 5);
}