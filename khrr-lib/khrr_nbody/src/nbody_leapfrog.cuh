#pragma once
#include <vector_types.h>

namespace khrr_nbody::kernels  {

// ─────────────────────────────────────────────────────────────────────────────
// Leapfrog — Kick  (half-step velocity update)
//
//   v += a · (dt/2)
// ─────────────────────────────────────────────────────────────────────────────
__global__ void leapfrog_kick(
    double3*       __restrict__ vel,
    const double3* __restrict__ acc,
    int    N,
    double half_dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    vel[i].x += acc[i].x * half_dt;
    vel[i].y += acc[i].y * half_dt;
    vel[i].z += acc[i].z * half_dt;
}

// ─────────────────────────────────────────────────────────────────────────────
// Leapfrog — Drift  (full-step position update)
//
//   x += v · dt
// ─────────────────────────────────────────────────────────────────────────────
__global__ void leapfrog_drift(
    double3*       __restrict__ pos,
    const double3* __restrict__ vel,
    int    N,
    double dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    pos[i].x += vel[i].x * dt;
    pos[i].y += vel[i].y * dt;
    pos[i].z += vel[i].z * dt;
}

} // namespace khrr_nbody::kernels