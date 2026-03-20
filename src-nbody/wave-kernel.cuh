#pragma once
#include "common.cuh"


__global__ void wave_diss_iteration(
    double* phi_new,
    const double* phi,
    const double* phi_old,
    const double* mass
)
{
	const int NX =              wave_eq_data_.nx_wave;
	const double DX =           wave_eq_data_.dx_wave;
	const double DISS_BASE =    wave_eq_data_.diss_base;
	const double DISS_EXTRA =   wave_eq_data_.diss_extra;
	const double SIM_L =        wave_eq_data_.sim_l;
	const double BC_L =         wave_eq_data_.bc_l;
	const double C =            wave_eq_data_.c_wave;
	const double DT =           wave_eq_data_.dt_wave;

    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int iz = threadIdx.z + blockIdx.z * blockDim.z;

    int is_valid_idx =
           ix < NX
        && iy < NX
        && iz < NX;
    if (!is_valid_idx) return;


    double x = ix * DX;
    double y = iy * DX;
    double z = iz * DX;

	double rho = mass[at(ix, iy, iz)] / (DX * DX * DX);
	const double G = 1.;
	double f = 4 * PI * G * rho;

    double diss = DISS_BASE;
#define _DISS_FUNC(x) (DISS_EXTRA * (x) * (x))

    if (x > -(SIM_L - BC_L)) {
        const double right = SIM_L - BC_L;
        diss += _DISS_FUNC(fabs(x - right));
    }
    else if (x < -(SIM_L - BC_L)) {
        const double left = -(SIM_L - BC_L);
        diss += _DISS_FUNC(fabs(x - left));
    }

    if (y > (SIM_L - BC_L)) {
        const double top = SIM_L - BC_L;
        diss += _DISS_FUNC(fabs(y - top));
    }
    else if (y < -(SIM_L - BC_L)) {
        const double bottom = -(SIM_L - BC_L);
        diss += _DISS_FUNC(fabs(y - bottom));
    }

    if (z > (SIM_L - BC_L)) {
        const double far = SIM_L - BC_L;
        diss += _DISS_FUNC(fabs(z - far));
    }
    else if (z < -(SIM_L - BC_L)) {
        const double near = -(SIM_L - BC_L);
        diss += _DISS_FUNC(fabs(z - near));
    }


#define _DIM 3
#define _CSQR (C * C)
#define _DT2 (DT * DT)
#define _DX2 (DX * DX)
#define _Q (diss * C * DT * 0.5)
#define _W (1 + _Q)
#define _K (_CSQR * _DT2 / _DX2)
#define _K_MAIN (_K / _W)
#define _K_F (-_CSQR * _DT2 / _W)
#define _K_ACTUAL (2 * (1. - _DIM * _K) / _W)
#define _K_OLD ((_Q - 1.) / _W)

#define _IS_I_EDGE(i) (i == 0 || i == NX - 1)
#define _IS_X_EDGE _IS_I_EDGE(ix)
#define _IS_Y_EDGE _IS_I_EDGE(iy)
#define _IS_Z_EDGE _IS_I_EDGE(iz)

    if (_IS_X_EDGE || _IS_Y_EDGE || _IS_Z_EDGE) {
        phi_new[at(ix, iy, iz)] = 0.;
    }
    else {
        phi_new[at(ix, iy, iz)] =
            phi[at(ix - 1, iy, iz)] * _K_MAIN +
            phi[at(ix + 1, iy, iz)] * _K_MAIN +
            phi[at(ix, iy - 1, iz)] * _K_MAIN +
            phi[at(ix, iy + 1, iz)] * _K_MAIN +
            phi[at(ix, iy, iz - 1)] * _K_MAIN +
            phi[at(ix, iy, iz + 1)] * _K_MAIN +
            phi[at(ix, iy, iz)] * _K_ACTUAL +
            phi_old[at(ix, iy, iz)] * _K_OLD +
            f * _K_F;
    }
}
