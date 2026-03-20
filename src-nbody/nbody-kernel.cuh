#pragma once
#include "common.cuh"

//Author: S.S. Khrapov
//Parallel Nbody Code OpenMP-CUDA 4GPU

//-----device function-----
__device__ real3 dev_fex(real4 p, real t){
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

//-----Force_Nbody kernel--------
__global__ void ACC_Zero(real3 *ACC)
{
	int ii = threadIdx.x + blockIdx.x * blockDim.x;
	ACC[ii] = make_real3(0.0, 0.0, 0.0);
}
__global__ void PSI_Zero(real *PSI)
{
	int ii = threadIdx.x + blockIdx.x * blockDim.x;
	PSI[ii] = 0.0;
}
__global__ void ACCEL(
	real3 *ACC,   // f_ij
	real4 *Pos_i, // r_i
	real4 *Pos_j, // r_j
	real2 *Mhp_j, // G * m_j
	real *eps2_pj
)
{
	__shared__ real4 sp[BLOCK_SIZE];
	__shared__ real eps2[BLOCK_SIZE];

	int ii = threadIdx.x + blockIdx.x * BLOCK_SIZE;
	real4 ps = Pos_i[ii];
	real3 f = make_real3(0.0, 0.0, 0.0);
	int block, j, jj;
	real s, eps2_ii = eps2_pj[ii];
	real3 r;

	for (block = 0; block < gridDim.x; block++) {
		jj = block * BLOCK_SIZE + threadIdx.x;
		sp[threadIdx.x] = make_real4(
			Pos_j[jj].x,
			Pos_j[jj].y,
			Pos_j[jj].z,
			Mhp_j[jj].x
		);
		eps2[threadIdx.x] = eps2_pj[jj];
		__syncthreads();
		for (j = 0; j < BLOCK_SIZE; j++)
		{
			r.x = sp[j].x - ps.x;
			r.y = sp[j].y - ps.y;
			r.z = sp[j].z - ps.z;
			s = 1.0 / sqrt(
				r.x*r.x +
				r.y*r.y +
				r.z*r.z +
				0.5*(eps2_ii+eps2[j])
			);
			s = s*s*s * sp[j].w;
			f.x += r.x*s;
			f.y += r.y*s;
			f.z += r.z*s;
		}
		__syncthreads();
	}
	ACC[ii] = make_real3(
		ACC[ii].x + f.x,
		ACC[ii].y + f.y,
		ACC[ii].z + f.z
	);
	//ACC[ii] = f;
}

//-----Psi_Nbody kernel--------
__global__ void PSI_kernel(real *PSI, real4 *Pos_i, real4 *Pos_j, real2 *Mhp_j, real *eps2_pj)
{
	__shared__ real4 sp[BLOCK_SIZE];
	__shared__ real eps2[BLOCK_SIZE];

	int ind = blockIdx.x * blockDim.x;
	int ii = threadIdx.x + ind;
	real4 ps = Pos_i[ii];
	int i, j, jj;
	real s, eps2_ii = eps2_pj[ii];
	real3 r;
  s=0.0;

	ind = 0;
	for (i = 0; i < gridDim.x; i++, ind += BLOCK_SIZE) //(0)
	{
		jj = ind + threadIdx.x;
		sp[threadIdx.x] = make_real4(
			Pos_j[jj].x,
			Pos_j[jj].y,
			Pos_j[jj].z,
			Mhp_j[jj].x
		);
		eps2[threadIdx.x] = eps2_pj[jj];
		__syncthreads();
		for (j = 0; j < BLOCK_SIZE; j++)
		{
			r.x = sp[j].x - ps.x;
			r.y = sp[j].y - ps.y;
			r.z = sp[j].z - ps.z;
			s += sp[j].w / sqrt(
				r.x*r.x +
				r.y*r.y +
				r.z*r.z +
				0.5*(eps2_ii+eps2[j])
			);
		}
		__syncthreads();
	}
	PSI[ii] = PSI[ii] - s;
}

__global__ void kernelNbody_integTime(real3 *ACC, real4 *Pos_t, real4 *Vel_t, real4 *Pos, real4 *Vel, real dt, int istep, real t, real3 *ACC0)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	real4 v = Vel[i];
	real4 r = Pos[i];
	real4 vt;
	real3 f, fex;

	fex = dev_fex(r, t);
	f.x = ACC[i].x + fex.x;
	f.y = ACC[i].y + fex.y;
	f.z = ACC[i].z + fex.z;

	//----predictor-------------------- q = q(t), q_t = q(t+dt) , dt = dt
	if (istep == 0){
		//----Velosity vx, vy, vz----------------------
		vt.x = v.x + dt*f.x;
		vt.y = v.y + dt*f.y;
		vt.z = v.z + dt*f.z;
		//----Position x,y,z----------------------
		Pos_t[i].z = r.z + 0.5*dt*(v.z + vt.z);
		Pos_t[i].x = r.x + 0.5*dt*(v.x + vt.x);
		Pos_t[i].y = r.y + 0.5*dt*(v.y + vt.y);

		Vel_t[i] = vt;
	} else{
		//------corrector---------------------- q_t = q(t), q = q(t+dt) - predictor, dt = dt
		real4 vtt;
		vt = Vel_t[i];
		//----Velosity vz----------------------
		vtt.z = 0.5*(vt.z + v.z + dt*f.z);
		vtt.x = 0.5*(vt.x + v.x + dt*f.x);
		vtt.y = 0.5*(vt.y + v.y + dt*f.y);

		Pos_t[i] = r;
		Vel_t[i] = vtt;
		ACC0[i] = ACC[i];
	}
}