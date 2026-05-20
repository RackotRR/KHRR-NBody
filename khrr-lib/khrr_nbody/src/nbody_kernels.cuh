#pragma once
#include <vector_types.h>

namespace khrr_nbody {
namespace kernels  {

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

// ─────────────────────────────────────────────────────────────────────────────
// Predictor:
//   x* = x + v·dt + ½·a·dt²
//   v* = v + a·dt
// ─────────────────────────────────────────────────────────────────────────────
__global__ void predict(
    const double3* __restrict__ pos,
    const double3* __restrict__ vel,
    const double3* __restrict__ acc,
    double3*       __restrict__ pos_pred,
    double3*       __restrict__ vel_pred,
    int    N,
    double dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const double hdt2 = 0.5 * dt * dt;
    pos_pred[i] = { pos[i].x + vel[i].x*dt + acc[i].x*hdt2,
                    pos[i].y + vel[i].y*dt + acc[i].y*hdt2,
                    pos[i].z + vel[i].z*dt + acc[i].z*hdt2 };
    vel_pred[i] = { vel[i].x + acc[i].x*dt,
                    vel[i].y + acc[i].y*dt,
                    vel[i].z + acc[i].z*dt };
}

// ─────────────────────────────────────────────────────────────────────────────
// Corrector:
//   v' = v_n + ½·(a_n + a*)·dt
//   x' = x_n + ½·(v_n + v')·dt
// ─────────────────────────────────────────────────────────────────────────────
__global__ void correct(
    const double3* __restrict__ pos_n,
    const double3* __restrict__ vel_n,
    const double3* __restrict__ acc_n,
    const double3* __restrict__ acc_pred,
    double3*       __restrict__ pos_out,   // must NOT alias pos_n
    double3*       __restrict__ vel_out,   // must NOT alias vel_n
    int    N,
    double dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const double hdt = 0.5 * dt;
    double3 vn1;
    vn1.x = vel_n[i].x + (acc_n[i].x + acc_pred[i].x) * hdt;
    vn1.y = vel_n[i].y + (acc_n[i].y + acc_pred[i].y) * hdt;
    vn1.z = vel_n[i].z + (acc_n[i].z + acc_pred[i].z) * hdt;

    pos_out[i] = { pos_n[i].x + (vel_n[i].x + vn1.x) * hdt,
                   pos_n[i].y + (vel_n[i].y + vn1.y) * hdt,
                   pos_n[i].z + (vel_n[i].z + vn1.z) * hdt };
    vel_out[i] = vn1;
}

} // namespace kernels
} // namespace khrr_nbody