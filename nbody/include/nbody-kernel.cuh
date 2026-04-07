#pragma once
#include "common.cuh"

//Author: S.S. Khrapov
//Parallel Nbody Code OpenMP-CUDA 4GPU

//-----device function-----
__device__ real3 dev_fex(real3 p, real t){
	real3 f;
	real rr, rr3, forcehalo, rrbcore1, root1, forceblg1, forcesph, rrcore;

	rr = sqrt(
		p.x*p.x +
		p.y*p.y +
		p.z*p.z
	);

	if (rr > 0.0) {
		rr3 = rr*rr*rr;
		rrcore = rr / dd.a;
		rrbcore1 = rr / dd.b;
		root1 = sqrt(1.0 + rrbcore1*rrbcore1);

		//--Halo--
		if (rr<dd.Rh2) forcehalo = -dd.con *(rrcore - atan(rrcore)) / rr3;
		else  forcehalo = -dd.Mh_inf / rr3;
		//--Bulge--
		if (rr<dd.Rb) forceblg1 = -dd.const1*(dd.b*log(rrbcore1 + root1) - rr / root1) / rr3;
		else  forceblg1 = -dd.Mb / rr3;

		forcesph = forceblg1 + forcehalo;
		f.x = forcesph * p.x;
		f.y = forcesph * p.y;
		f.z = forcesph * p.z;
	}
	else {
		f = make_real3(0.0, 0.0, 0.0);
	}
	return f;
}

//-----Phi_Nbody kernel--------
__global__ void PHI_kernel(
	real* phi,
	const real3* pos_i,
	const real3* pos_j,
	const real* mass_j,
	const real* eps2_j
)
{
	__shared__ real3 pos_other[BLOCK_SIZE];
	__shared__ real mass_other[BLOCK_SIZE];
	__shared__ real eps2_other[BLOCK_SIZE];
	cuda::std::memset(pos_other, 0, sizeof(real3) * BLOCK_SIZE);
	cuda::std::memset(mass_other, 0, sizeof(real) * BLOCK_SIZE);
	cuda::std::memset(eps2_other, 0, sizeof(real) * BLOCK_SIZE);

	real phi_sum = 0.;
	int i_curr_global = threadIdx.x + blockIdx.x * blockDim.x;

	real3 p_curr;
	real eps2_curr;
	if (i_curr_global < dd.NN) {
		p_curr = pos_i[i_curr_global];
		eps2_curr = eps2_j[i_curr_global];
	}
	else {
		p_curr = make_real3(0., 0., 0.);
		eps2_curr = 0.;
	}

	for (int block = 0; block < gridDim.x; ++block) {
		int i_other_global = threadIdx.x + block * blockDim.x;
		if (i_other_global < dd.NN) {
			pos_other[threadIdx.x] = pos_j[i_other_global];
			mass_other[threadIdx.x] = mass_j[i_other_global];
			eps2_other[threadIdx.x] = eps2_j[i_other_global];
		}

		__syncthreads();

		for (int i_other_local = 0; i_other_local < blockDim.x; ++i_other_local) {
			real3 dp = make_real3(
				pos_other[i_other_local].x - p_curr.x,
				pos_other[i_other_local].y - p_curr.y,
				pos_other[i_other_local].z - p_curr.z
			);

			phi_sum += mass_other[i_other_local] / sqrt(
				dot3(dp, dp) +
				0.5 * (eps2_curr + eps2_other[i_other_local])
			);
		}

		__syncthreads();

	}

	if (i_curr_global < dd.NN) {
		// fix self-gravity
		double mass_curr = i_curr_global < dd.NN
			? mass_j[i_curr_global]
			: 0.;
		double self_grav = mass_curr / sqrt(eps2_curr);
		phi[i_curr_global] -= phi_sum - self_grav;
	}
}

//-----Force_Nbody kernel--------
__global__ void ACCEL_kernel(
	real3* acc,   // f_ij
	const real3* pos_i, // r_i
	const real3* pos_j, // r_j
	const real* mass_j, // G * m_j
	const real* eps2_j
)
{
	__shared__ real3 pos_other[BLOCK_SIZE];
	__shared__ real mass_other[BLOCK_SIZE];
	__shared__ real eps2_other[BLOCK_SIZE];
	cuda::std::memset(pos_other, 0, sizeof(real3) * BLOCK_SIZE);
	cuda::std::memset(mass_other, 0, sizeof(real) * BLOCK_SIZE);
	cuda::std::memset(eps2_other, 0, sizeof(real) * BLOCK_SIZE);

	real3 f_sum = make_real3(0.0, 0.0, 0.0);
	int i_curr_global = threadIdx.x + blockIdx.x * blockDim.x;

	real3 p_curr;
	real eps2_curr;
	if (i_curr_global < dd.NN) {
		p_curr = pos_i[i_curr_global];
		eps2_curr = eps2_j[i_curr_global];
	}
	else {
		p_curr = make_real3(0., 0., 0.);
		eps2_curr = 0.;
	}

	for (int block = 0; block < gridDim.x; block++) {
		int i_other_global = threadIdx.x + block * blockDim.x;
		if (i_other_global < dd.NN) {
			pos_other[threadIdx.x] = pos_j[i_other_global];
			mass_other[threadIdx.x] = mass_j[i_other_global];
			eps2_other[threadIdx.x] = eps2_j[i_other_global];
		}

		__syncthreads();

		for (int i_other_local = 0; i_other_local < blockDim.x; ++i_other_local) {
			real3 dp = make_real3(
				pos_other[i_other_local].x - p_curr.x,
				pos_other[i_other_local].y - p_curr.y,
				pos_other[i_other_local].z - p_curr.z
			);
			real denominator = sqrt(
				dot3(dp, dp) +
				0.5 * (eps2_curr + eps2_other[i_other_local])
			);
			real k = mass_other[i_other_local] / cube(denominator);
			f_sum.x += dp.x * k;
			f_sum.y += dp.y * k;
			f_sum.z += dp.z * k;
		}

		__syncthreads();

	}

	if (i_curr_global < dd.NN) {
		acc[i_curr_global] = make_real3(
			acc[i_curr_global].x + f_sum.x,
			acc[i_curr_global].y + f_sum.y,
			acc[i_curr_global].z + f_sum.z
		);
	}
}

__global__ void kernelNbody_integTime(
	const real3* acc,
	real3* pos_t,
	real3* vel_t,
	const real3* pos,
	const real3* vel,
	real dt,
	int istep,
	real t,
	real3* acc0
)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= dd.NN) {
		return;
	}

	real3 v = vel[i];
	real3 r = pos[i];
	real3 vt;
	real3 f, fex;

	fex = dev_fex(r, t);
	f.x = acc[i].x + fex.x;
	f.y = acc[i].y + fex.y;
	f.z = acc[i].z + fex.z;

	//----predictor-------------------- q = q(t), q_t = q(t+dt) , dt = dt
	if (istep == 0) {
		//----Velosity vx, vy, vz----------------------
		vt.x = v.x + dt * f.x;
		vt.y = v.y + dt * f.y;
		vt.z = v.z + dt * f.z;
		//----Position x,y,z----------------------
		pos_t[i].x = r.x + 0.5 * dt * (v.x + vt.x);
		pos_t[i].y = r.y + 0.5 * dt * (v.y + vt.y);
		pos_t[i].z = r.z + 0.5 * dt * (v.z + vt.z);

		vel_t[i] = vt;
	}
	else {
		//------corrector---------------------- q_t = q(t), q = q(t+dt) - predictor, dt = dt
		real3 vtt;
		vt = vel_t[i];

		//----Velosity vz----------------------
		vtt.x = 0.5*(vt.x + v.x + dt * f.x);
		vtt.y = 0.5*(vt.y + v.y + dt * f.y);
		vtt.z = 0.5*(vt.z + v.z + dt * f.z);

		pos_t[i] = r;
		vel_t[i] = vtt;
		acc0[i] = acc[i];
	}
}