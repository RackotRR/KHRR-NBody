#pragma once
#include <vector_types.h>

namespace khrr_nbody::kernels {

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

} // namespace khrr_nbody::kernels