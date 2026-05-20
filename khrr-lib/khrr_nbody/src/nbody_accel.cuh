#pragma once
#include <vector_types.h>

namespace khrr_nbody::kernels {

// ─────────────────────────────────────────────────────────────────────────────
// Tile size for shared-memory blocking.
// Must match blockDim.x used at the call site.
// ─────────────────────────────────────────────────────────────────────────────
constexpr int TILE_SIZE = 256;

// ─────────────────────────────────────────────────────────────────────────────
// Tiled O(N²) gravitational acceleration with per-particle softening.
//
//   a_i = G · Σ_{j≠i}  m_j (r_j−r_i) / (|r_j−r_i|² + ½(ε²_i+ε²_j))^{3/2}
//
// Each thread block cooperatively loads TILE_SIZE source particles (pos + mass
// + eps2) into shared memory, then every thread computes its contribution from
// that tile before moving to the next one.
// Global memory reads per particle: O(N / TILE_SIZE) rounds × 1 coalesced load
// instead of N scattered reads → bandwidth drops by ~TILE_SIZE×.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void compute_accel_tiled(
    const double3* __restrict__ pos,
    const double*  __restrict__ mass,
    const double*  __restrict__ eps2,
    double3*       __restrict__ acc,
    int   N,
    double G)
{
    // Shared-memory tile: positions, masses, softening of source particles.
    __shared__ double3 sh_pos [TILE_SIZE];
    __shared__ double  sh_mass[TILE_SIZE];
    __shared__ double  sh_eps2[TILE_SIZE];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    double ax = 0.0, ay = 0.0, az = 0.0;

    // Cache target particle data in registers (safe even if i >= N; the
    // accumulation result won't be written in that case).
    const double3 ri  = (i < N) ? pos [i] : double3{0,0,0};
    const double  ei2 = (i < N) ? eps2[i] : 0.0;

    // Walk over tiles of source particles.
    for (int tile_start = 0; tile_start < N; tile_start += TILE_SIZE)
    {
        // ── Cooperative load ──────────────────────────────────────────────
        const int src = tile_start + threadIdx.x;
        if (src < N) {
            sh_pos [threadIdx.x] = pos [src];
            sh_mass[threadIdx.x] = mass[src];
            sh_eps2[threadIdx.x] = eps2[src];
        } else {
            // Pad with a zero-mass particle so arithmetic stays safe.
            sh_pos [threadIdx.x] = {0, 0, 0};
            sh_mass[threadIdx.x] = 0.0;
            sh_eps2[threadIdx.x] = 1.0; // non-zero to avoid 0/0
        }
        __syncthreads();

        // ── Accumulate forces from this tile ──────────────────────────────
        if (i < N) {
            const int tile_end = min(TILE_SIZE, N - tile_start);
            #pragma unroll 8
            for (int t = 0; t < tile_end; ++t)
            {
                const int j_global = tile_start + t;
                if (j_global == i) continue;   // skip self

                const double dx = sh_pos[t].x - ri.x;
                const double dy = sh_pos[t].y - ri.y;
                const double dz = sh_pos[t].z - ri.z;

                const double r2    = dx*dx + dy*dy + dz*dz
                                     + 0.5 * (ei2 + sh_eps2[t]);
                const double r3inv = rsqrt(r2 * r2 * r2);
                const double Gmj   = G * sh_mass[t];

                ax += Gmj * dx * r3inv;
                ay += Gmj * dy * r3inv;
                az += Gmj * dz * r3inv;
            }
        }
        __syncthreads(); // protect shared memory before next tile load
    }

    if (i < N)
        acc[i] = {ax, ay, az};
}

// ─────────────────────────────────────────────────────────────────────────────
// Direct O(N²) gravitational acceleration with pairwise softening.
//
//   a_i = G · Σ_{j≠i}  m_j · (r_j − r_i) / (|r_j − r_i|² + ½(ε²_i+ε²_j))^{3/2}
// ─────────────────────────────────────────────────────────────────────────────
__global__ void compute_accel(
    const double3* __restrict__ pos,
    const double*  __restrict__ mass,
    const double*  __restrict__ eps2,
    double3*       __restrict__ acc,
    int   N,
    double G)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double ax = 0.0, ay = 0.0, az = 0.0;
    const double3 ri  = pos[i];
    const double  ei2 = eps2[i];

    for (int j = 0; j < N; ++j) {
        if (j == i) continue;
        const double dx = pos[j].x - ri.x;
        const double dy = pos[j].y - ri.y;
        const double dz = pos[j].z - ri.z;
        const double r2     = dx*dx + dy*dy + dz*dz + 0.5*(ei2 + eps2[j]);
        const double r3inv  = rsqrt(r2 * r2 * r2);
        const double Gmj    = G * mass[j];
        ax += Gmj * dx * r3inv;
        ay += Gmj * dy * r3inv;
        az += Gmj * dz * r3inv;
    }
    acc[i] = {ax, ay, az};
}

} // namespace khrr_nbody::kernels