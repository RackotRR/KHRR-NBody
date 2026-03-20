//Author: S.S. Khrapov
//Parallel Nbody Code OpenMP-CUDA 4GPU

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <numeric>

#include "RRCU.cuh"

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <tuple>

#define PI 3.14159265358979
#define BLOCK_SIZE 512
#define BLOCK_SIZE_b 512
#define DEG2RAD (PI / 180.0)
#define TIME2SEC (0.001f)

#define real double
#define real2 double2
#define real3 double3
#define real4 double4_32a
#define make_real3 make_double3
#define make_real4 make_double4_32a
#define make_real2 make_double2

using namespace RR::CUDA;

real Z_max, E0;
real3 Imp0, L0;

typedef struct{
    int count; // число частиц в ячейке
    int start_id; // накопленная сумма частиц (префиксная сумма) - идекс в массиве частиц
} CellInfo;

typedef struct {
    int cell_id; // индекс ячейки
    int id_in_cell; // индекс частицы в ячейке
} ParticleCellInfo;

template<typename T>
constexpr T intlog2(T size) {
	return size > 1
        ? 1 + intlog2(size >> 1)
        : 0;
}

struct DataBlock{
	int		Ns;
	int		NN;
	real	Mh;
	real	Mh_inf;
	real    a;
	real    Rh;
	real    Mb;
	real    b;
	real    Rb;
	real    Rh2;
	real    con;
	real    const1;
	real    c_psi_h;
	real    c_psi_b;
	real    eps2;
    real4 	domainMin;
    real4 	domainMax;
    real4 	cellSize;
    int3	gridSize;
	double  dt_wave;
	double  c_wave;
	double  dx_wave;
	int 	nx_wave;
	double diss_base;
	double diss_extra;
	double sim_l;
	double bc_l;
};

DataBlock d;
__constant__ DataBlock dd;

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

template<typename T>
__host__ __device__ T clamp(T val, T min_val, T max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
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

// ====================================================
// ЯДРО 1: ИНИЦИАЛИЗАЦИЯ ВСПОМОГАТЕЛЬНЫХ МАССИВОВ
// ====================================================
__global__ void initSortingArrays(
    CellInfo* cellInfo,     // [TOTAL_CELLS] информация о ячейках
    int* cellParticleCount, // [TOTAL_CELLS] счётчик для atomicAdd
    int* maxPBC,            // [TOTAL_CELLS/BLOCK_SIZE] частиц в блоке
    const int numCells,
	const int numBlocks
)
{
	int i_cell = threadIdx.x + blockIdx.x * blockDim.x;
    if (i_cell >= numCells) return;

    // Инициализация нулями
    cellInfo[i_cell].count = 0;
    cellInfo[i_cell].start_id = 0;
    cellParticleCount[i_cell] = 0;

    // Инициализация maxPBC для каждого блока (делает только первый поток)
    if (i_cell % BLOCK_SIZE == 0) {
        if (blockIdx.x < numBlocks) {
            maxPBC[blockIdx.x] = 0;
        }
    }
}

// ====================================================
// ЯДРО 2: ИНИЦИАЛИЗАЦИЯ ВСПОМОГАТЕЛЬНЫХ МАССИВОВ
// ====================================================
__global__ void assignParticlesToCells(
	const real4* particles_pos, // [N] исходные частицы
    ParticleCellInfo* particleCellInfo, // [N] инфо: x=ячейка, y=индекс в ячейке
    CellInfo* cellInfo,          // [TOTAL_CELLS] для подсчёта частиц
    int* cellParticleCount, // [TOTAL_CELLS] счётчик для atomicAdd
    const int numParticles
)
{
    int i_part = threadIdx.x + blockIdx.x * blockDim.x;
    if (i_part >= numParticles) return;

    real4 p = particles_pos[i_part];

    // Вычисление индексов ячейки
    int ix = clamp((int)((p.x - dd.domainMin.x) / dd.cellSize.x), 0, (int)(dd.gridSize.x - 1));
    int iy = clamp((int)((p.y - dd.domainMin.y) / dd.cellSize.y), 0, (int)(dd.gridSize.y - 1));
    int iz = clamp((int)((p.z - dd.domainMin.z) / dd.cellSize.z), 0, (int)(dd.gridSize.y - 1));

    // Линейный индекс ячейки (Morton-подобный порядок для пространственной локальности)
    // Используем чередование битов для лучшей когерентности
    int i_cell = iz * dd.gridSize.x * dd.gridSize.y + iy * dd.gridSize.x + ix;

    // Сохраняем номер ячейки для частицы
    particleCellInfo[i_part].cell_id = i_cell;

    // Атомарно увеличиваем счётчик частиц в ячейке
    int pos = atomicAdd(&cellParticleCount[i_cell], 1);

    // Сохраняем позицию внутри ячейки
    particleCellInfo[i_part].id_in_cell = pos;

    // Атомарно обновляем количество частиц в ячейке
	atomicAdd(&cellInfo[i_cell].count, 1);

}

// ====================================================
// ЯДРО 3: ВЫЧИСЛЕНИЕ ЧАСТИЧНЫХ СУММ (КАСКАДНЫЙ АЛГОРИТМ)
// ====================================================
__global__ void computePrefixSums(
    CellInfo* cellInfo,    // [TOTAL_CELLS]
    int* maxPBC,           // [TOTAL_CELLS/BLOCK_SIZE]
    const int numCells)
{
    __shared__ int shared[BLOCK_SIZE];
    __shared__ int sharedPrev[BLOCK_SIZE];

    int i_cell = threadIdx.x + blockIdx.x * blockDim.x;
    int i_local = threadIdx.x;
    int i_block = blockIdx.x;

    // Загружаем количество частиц в ячейке
    if (i_cell >= numCells) {
        shared[i_local] = 0;
        sharedPrev[i_local] = 0;
    }
    else {
        int count = cellInfo[i_cell].count;
        shared[i_local]     = count;
        sharedPrev[i_local] = count;
    }

	__syncthreads();
    // Каскадное суммирование (параллельное сканирование)
    for (int stride = 1; stride < BLOCK_SIZE; stride <<= 1) {
        if (i_local + stride < BLOCK_SIZE) {
            shared[i_local + stride] += sharedPrev[i_local];
        }
		__syncthreads();

        // Обновляем предыдущие значения
        sharedPrev[i_local] = shared[i_local];
		__syncthreads();
    }

    // Первый поток в блоке сохраняет общую сумму блока
    if (i_local == 0) {
        int particles_in_block = shared[BLOCK_SIZE - 1];
        for (int i = i_block; i < gridDim.x; ++i) {
            atomicAdd(&(maxPBC[i]), particles_in_block);
        }
    }
}

// ====================================================
// ЯДРО 4: КОРРЕКТИРОВКА ГЛОБАЛЬНЫХ ПРЕФИКСНЫХ СУММ
// ====================================================
__global__ void adjustGlobalPrefixSums(
    CellInfo* cellInfo,     // [TOTAL_CELLS]
    const int* maxPBC,      // [TOTAL_CELLS/BLOCK_SIZE]
    const int numCells)
{
    int i_cell = threadIdx.x + blockIdx.x * blockDim.x;
    int i_block = blockIdx.x;
    if (i_cell >= numCells) return;

    // Добавляем сумму всех предыдущих блоков
    if (i_block > 0) {
        cellInfo[i_cell].start_id += maxPBC[i_block - 1];
    }
}

// ====================================================
__global__ void computeCellMassesUnsorted(
    const real2* particle_mass,   	   // [N] упорядоченные частицы
    ParticleCellInfo* particleCellInfo,// [N] инфо: x=ячейка, y=индекс в ячейке
    double* cellMasses,                // [TOTAL_CELLS] результат
    const int numParticles)
{
    int i_particle = threadIdx.x + blockIdx.x * blockDim.x;
    if (i_particle >= numParticles) return;

    int i_cell = particleCellInfo[i_particle].cell_id;
	atomicAdd(&cellMasses[i_cell], particle_mass[i_particle].x);
}

// ====================================================
__global__ void countParticles(
    double* cell_phi,                // [TOTAL_CELLS] результат
    CellInfo* cellInfo,     // [TOTAL_CELLS]
	const int numCells
)
{
	int i_cell = threadIdx.x + blockIdx.x * blockDim.x;
	if (i_cell > numCells) return;

	cell_phi[i_cell] = cellInfo[i_cell].count;
}
// ====================================================
__global__ void zeroCellPhi(
    double* cell_phi,                // [TOTAL_CELLS] результат
	const int numCells,
	double value
)
{
	int i_cell = threadIdx.x + blockIdx.x * blockDim.x;
	if (i_cell > numCells) return;

	cell_phi[i_cell] = value;
}
__global__ void computeCellPhiUnsorted(
    const real* particle_phi,   	   // [N] упорядоченные частицы
    const ParticleCellInfo* particleCellInfo,// [N] инфо: x=ячейка, y=индекс в ячейке
    const CellInfo* cellInfo,     		 // [TOTAL_CELLS]
    double* cell_phi,                // [TOTAL_CELLS] результат
    const int numParticles)
{
    int i_particle = threadIdx.x + blockIdx.x * blockDim.x;
    if (i_particle >= numParticles) return;

    int i_cell = particleCellInfo[i_particle].cell_id;
	atomicAdd(&cell_phi[i_cell], particle_phi[i_particle] / cellInfo[i_cell].count);
	// if (particleCellInfo[i_particle].id_in_cell == 0) {
	// 	cell_phi[i_cell] = particle_phi[i_particle];
	// }
}

#define at(x, y, z) ((x) + (y) * (NX) + (z) * (NX) * (NX))
__global__ void acceleration_field(
	real3* acceleration,
	const real* phi,
	const real* mass,
	const int NX,
	const double DX
)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int iz = threadIdx.z + blockIdx.z * blockDim.z;

	int xyz = at(ix, iy, iz);

	real3 dphi;

	if (ix == 0) {
		dphi.x = phi[at(ix + 1, iy, iz)] - phi[xyz];
	}
	else if (ix == NX - 1) {
		dphi.x = phi[xyz] - phi[at(ix - 1, iy, iz)];
	}
	else {
		dphi.x = 0.5 * (phi[at(ix + 1, iy, iz)] - phi[at(ix - 1, iy, iz)]);
	}

	if (iy == 0) {
		dphi.y = phi[at(ix, iy + 1, iz)] - phi[xyz];
	}
	else if (iy == NX - 1) {
		dphi.y = phi[xyz] - phi[at(ix, iy - 1, iz)];
	}
	else {
		dphi.y = 0.5 * (phi[at(ix, iy + 1, iz)] - phi[at(ix, iy - 1, iz)]);
	}

	if (iz == 0) {
		dphi.z = phi[at(ix, iy, iz + 1)] - phi[xyz];
	}
	else if (iz == NX - 1) {
		dphi.z = phi[xyz] - phi[at(ix, iy, iz - 1)];
	}
	else {
		dphi.z = 0.5 * (phi[at(ix, iy, iz + 1)] - phi[at(ix, iy, iz - 1)]);
	}

	dphi.x /= DX;
	dphi.y /= DX;
	dphi.z /= DX;
	acceleration[xyz] = dphi;
}

__global__ void apply_acceleration_to_particles(
	real3* particle_acceleration,
	const real3* cell_acceleration,
    ParticleCellInfo* particleCellInfo,
    const int numParticles
)
{
    int i_particle = threadIdx.x + blockIdx.x * blockDim.x;
    if (i_particle >= numParticles) return;

    int i_cell = particleCellInfo[i_particle].cell_id;
	particle_acceleration[i_particle] = cell_acceleration[i_cell];
}
__global__ void apply_psi_to_particles(
	real* particle_psi,
	const real* cell_psi,
    CellInfo* cellInfo,     // [TOTAL_CELLS]
    ParticleCellInfo* particleCellInfo,
    const int numParticles
)
{
    int i_particle = threadIdx.x + blockIdx.x * blockDim.x;
    if (i_particle >= numParticles) return;

    int i_cell = particleCellInfo[i_particle].cell_id;
	// particle_psi[i_particle] = cell_psi[i_cell];
	particle_psi[i_particle] = cellInfo[i_cell].count;
}

__global__ void init_phi(
	real* mass,
	real* phi0,
	real* phi1,
	real* phi2,
	const int NX
)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int iz = threadIdx.z + blockIdx.z * blockDim.z;

    int is_valid_idx =
           ix < NX
        && iy < NX
        && iz < NX;
    if (!is_valid_idx) return;

	mass[at(ix, iy, iz)] = 0;
	phi0[at(ix, iy, iz)] = 0;
	phi1[at(ix, iy, iz)] = 0;
	phi2[at(ix, iy, iz)] = 0;
}

// ====================================================
__global__ void wave_diss_iteration(
    double* phi_new,
    const double* phi,
    const double* phi_old,
    const double* mass
)
{
	const int NX = dd.nx_wave;
	const double DX = dd.dx_wave;
	const double DISS_BASE = dd.diss_base;
	const double DISS_EXTRA = dd.diss_extra;
	const double SIM_L = dd.sim_l;
	const double BC_L = dd.bc_l;
	const double C = dd.c_wave;
	const double DT = dd.dt_wave;

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

// double* calc_phi(
//     double* phi_new,
//     double* phi,
//     double* phi_old,
//     const double* mass,
// 	const int NX,
// 	const int NX_LOCAL,
// 	const double DX,
// 	const double DISS_BASE,
// 	const double DISS_EXTRA,
// 	const double SIM_L,
// 	const double BC_L,
// 	const double C,
// 	const double DT,
// 	const int iterations
// )
// {
// 	cudaDeviceSynchronize();
// 	unsigned threads_count = static_cast<unsigned>(NX_LOCAL);// 8;
// 	unsigned blocks_count = static_cast<unsigned>(NX + threads_count - 1) / threads_count;
// 	dim3 threads_per_block{
// 		threads_count,
// 		threads_count,
// 		threads_count
// 	};
// 	dim3 blocks_per_grid{
// 		blocks_count,
// 		blocks_count,
// 		blocks_count
// 	};
// 	for (int iter = 0; iter < iterations; ++iter) {
// 		printf("wave_diss_iteration iter: %d\n", iter);
// 		wave_diss_iteration<<<blocks_per_grid, threads_per_block>>>(
// 			phi_new,
// 			phi,
// 			phi_old,
// 			mass,
// 			NX,
// 			DX,
// 			DISS_BASE,
// 			DISS_EXTRA,
// 			SIM_L,
// 			BC_L,
// 			C,
// 			DT
// 		);

// 		auto err = cudaDeviceSynchronize();
// 		if (err != cudaSuccess) {
// 			printf("wave_diss_iteration err: %d (%s)\n", (int)err, cudaGetErrorName(err));
// 		}

// 		phi_old = phi;
// 		phi = phi_new;
// 		phi_new = phi_old;
// 	}

// 	return phi;
// }

void check_energy(
	const std::vector<double>& cell_mass,
	const std::vector<double>& cell_phi,
	int _num_cells
)
{
	double ep = 0;
	for (size_t i = 0; i < _num_cells; ++i) {
		ep += 0.5 * cell_mass[i] * cell_phi[i];
	}
	printf("ep: %g\n", ep);
}

void check_cell_particle_info_match(
	const std::vector<CellInfo>& cell_info,
	const std::vector<ParticleCellInfo>& particle_info,
	int _num_particles,
	int _num_cells
) {
	// check cell info & particle cell info
	std::vector<CellInfo> cell_info_manual(_num_cells, CellInfo{ 0, 0 });
	for (size_t i = 0; i < _num_particles; ++i) {
		size_t c = particle_info[i].cell_id;
		cell_info_manual[c].count++;
	}

	for (size_t i = 0; i < _num_cells; ++i) {
		if (cell_info_manual[i].count != cell_info[i].count) {
			std::cout << "cell info mismatch!!! " << cell_info_manual[i].count << " : " << cell_info[i].count << std::endl;
		}
	}
}

auto check_particles_distribution(
	const std::vector<CellInfo>& cell_info
)
{
	std::map<int, int> dist;
	std::ofstream stream{"particles_in_cell.txt"};
	stream << "n_particles, cells_with_so_many_particles" << std::endl;

	for (const auto& info : cell_info) {
		dist[info.count]++;
	}

	std::vector<double> partial_counts;
	double partial_counts_prev = 0.;
	for (auto iter = dist.rbegin(); iter != dist.rend(); ++iter) {
		auto [count, d] = *iter;
		stream << count << ", " << d << std::endl;

		partial_counts.push_back(partial_counts_prev + count * d);
		partial_counts_prev = partial_counts.back();
	}

	std::ofstream stream_partial_sum{ "particles_in_cell_partial_sum.txt" };
	for (size_t i = 1; i < partial_counts.size(); ++i) {
		stream_partial_sum << i << ", " << partial_counts[i] << std::endl;
	}
}

auto check_mass(
	const std::vector<double>& cell_mass,
	const std::vector<double2>& particles_mass,
	double _dx,
	double _domain_l,
	int NX
)
{
	int IY = NX / 2;
	int IZ = NX / 2;

	std::ofstream stream{ "mass_part.txt" };
	stream << "x, mass" << std::endl;
	for (size_t i = 0; i < NX; ++i) {
		double x = -_domain_l + i * _dx;
		stream << std::format("{}, {:.6f}", x, cell_mass[at(i, IY, IZ)]) << std::endl;
	}

	std::cout
		<< "total mass in cells: "
		<< std::accumulate(cell_mass.begin(), cell_mass.end(), 0.)
		<< std::endl;


	std::cout << "total mass in particles: "
		<< std::accumulate(
			particles_mass.begin(),
			particles_mass.end(),
			0.,
			[](double sum, real2 v) {
				return sum + v.x;
			}
		)
		<< std::endl;
}

auto convert_particles(
	CuDarray<real>& particles_phi,
	const CuDarray<real4>& particles_pos,
	const CuDarray<real2>& particles_mass,
	CuDarray<CellInfo>& cell_info,
	CuDarray<int>& cell_particles_count,
	CuDarray<int>& particles_in_block,
	CuDarray<ParticleCellInfo>& particles_cell_info,
	CuDarray<real>& cell_mass,
	CuDarray<real>& cell_phi_prev,
	CuDarray<real>& cell_phi_curr,
	CuDarray<real>& cell_phi_next,
	CuDarray<real3>& cell_acceleration,
	int _over_cells,
	int _over_blocks,
	int _over_particles,
	int _num_cells,
	int _num_cell_blocks,
	int _num_particles,
	real _domain_l,
	int _nx,
	real _dx,
	int iterations,
	bool need_init_phi
)
{
	cudaDeviceSynchronize();

	unsigned threads_count = 8;
	unsigned blocks_count = static_cast<unsigned>(_nx + threads_count - 1) / threads_count;
	dim3 threads_per_block{ threads_count, threads_count, threads_count };
	dim3 blocks_per_grid{ blocks_count, blocks_count, blocks_count };

	if (need_init_phi) {
		cell_mass.set_zero();
		cell_phi_prev.set_zero();
		cell_phi_curr.set_zero();
		cell_phi_next.set_zero();
	}

	cell_info.set_zero();
	cell_particles_count.set_zero();
	particles_in_block.set_zero();

    CuCall(assignParticlesToCells, _over_particles, _over_blocks) (
        particles_pos,
        particles_cell_info,
        cell_info,
        cell_particles_count,
        _num_particles
    );
    CuCall(computePrefixSums, _over_cells, _over_blocks) (
        cell_info,
        particles_in_block,
        _num_cells
    );
    CuCall(adjustGlobalPrefixSums, _over_cells, _over_blocks) (
        cell_info,
        particles_in_block,
        _num_cells
    );
	CuCall(computeCellMassesUnsorted, _over_particles, _over_blocks) (
		particles_mass,
		particles_cell_info,
		cell_mass,
		_num_particles
	);

	for (int iter = 0; iter < iterations; ++iter) {
		if (iter && iter % 500 == 0) {
			std::cout << "wave_diss_iteration iter: " << iter << std::endl;
		}
		CuCall(wave_diss_iteration, blocks_per_grid, threads_per_block) (
			cell_phi_next,
			cell_phi_curr,
			cell_phi_prev,
			cell_mass
		);
		swap(cell_phi_prev, cell_phi_curr);
		swap(cell_phi_curr, cell_phi_next);
	}

	std::vector<double> cell_mass_host(_num_cells);
	std::vector<double> cell_phi_host(_num_cells);
	std::vector<CellInfo> cell_info_host(_num_cells);
	std::vector<ParticleCellInfo> particle_info_host(_num_particles);
	cudaMemcpy(cell_mass_host.data(), cell_mass, _num_cells * sizeof(real), cudaMemcpyDeviceToHost);
	cudaMemcpy(cell_phi_host.data(), cell_phi_curr, _num_cells * sizeof(real), cudaMemcpyDeviceToHost);
	cudaMemcpy(cell_info_host.data(), cell_info, _num_cells * sizeof(CellInfo), cudaMemcpyDeviceToHost);
	cudaMemcpy(particle_info_host.data(), particles_cell_info, _num_particles * sizeof(ParticleCellInfo), cudaMemcpyDeviceToHost);

	check_cell_particle_info_match(
		cell_info_host,
		particle_info_host,
		_num_particles,
		_num_cells
	);

	check_energy(
		cell_mass_host,
		cell_phi_host,
		_num_cells
	);

	check_particles_distribution(
		cell_info_host
	);

	check_mass(
		cell_mass_host,
		particles_mass.to_vector(),
		_dx,
		_domain_l,
		_nx
	);

	const int NX = _nx;
	const int IY = _nx / 2 ;
	const int IZ = _nx / 2 ;

	// particle_counts
	{
		CuDarray<double> particle_counts_(_num_cells);
		CuCall(countParticles, _over_cells, _over_blocks) (
			particle_counts_,
			cell_info,
			_num_cells
		);

		auto particle_counts = particle_counts_.to_vector();

		std::ofstream stream{ "particle_counts.txt" };
		stream << "xi, counts" << std::endl;
		for (size_t i = 0; i < _nx; ++i) {
			stream << i << ", " << particle_counts[at(i, IY, IZ)] << std::endl;
		}
	}

	// phi cell
	{
		std::ofstream stream{ "phi_cell.txt" };
		stream << "x, phi" << std::endl;
		for (size_t i = 0; i < _nx; ++i) {
			double x = -_domain_l + i * _dx;
			stream << std::format("{}, {:.7f}", x, cell_phi_host[at(i, IY, IZ)]) << std::endl;
		}
	}

	// phi from particles
	{
		std::ofstream stream{ "phi_part.txt" };
		stream << "x, phi" << std::endl;

		CuDarray<double> phi_(_num_cells);
		CuCall(computeCellPhiUnsorted, _over_particles, _over_blocks) (
			particles_phi,
			particles_cell_info,
			cell_info,
			phi_, // compute
			_num_particles
		);
		auto phi = phi_.to_vector();

		for (size_t i = 0; i < _nx; ++i) {
			double x = -_domain_l + i * _dx;
			stream << std::format("{}, {:.7f}", x, phi[at(i, IY, IZ)]) << std::endl;
		}
	}


	// phi from particles cycle
	// {
	// 	zeroCellPhi<<<_over_particles, _over_blocks>>>(
	// 		particles_phi,
	// 		_num_particles,
	// 		0.
	// 	);
	// 	zeroCellPhi<<<_over_cells, _over_blocks>>>(
	// 		cell_phi_dev,
	// 		_num_cells,
	// 		100.
	// 	);
	// 	cudaDeviceSynchronize();
	// 	apply_psi_to_particles<<<_over_particles, _over_blocks>>>(
	// 		particles_phi,
	// 		cell_phi_dev,
	// 		cell_info,
	// 		particles_cell_info,
	// 		_num_particles
	// 	);
	// 	auto err = cudaDeviceSynchronize();
	// 	printf("apply_psi_to_particles err: %d (%s)\n", (int)err, cudaGetErrorName(err));

	// 	zeroCellPhi<<<_over_cells, _over_blocks>>>(
	// 		cell_phi_dev,
	// 		_num_cells,
	// 		0.
	// 	);
	// 	err = cudaDeviceSynchronize();
	// 	printf("zeroCellPhi err: %d (%s)\n", (int)err, cudaGetErrorName(err));

	// 	computeCellPhiUnsorted<<<_over_particles, _over_blocks>>>(
	// 		particles_phi,
	// 		particles_cell_info,
	// 		cell_info,
	// 		cell_phi_dev,
	// 		_num_particles
	// 	);
	// 	err = cudaDeviceSynchronize();
	// 	printf("computeCellPhiUnsorted err: %d (%s)\n", (int)err, cudaGetErrorName(err));
	// 	cudaMemcpy(cell_phi_host.data(), cell_phi_dev, _num_cells * sizeof(real), cudaMemcpyDeviceToHost);

	// 	std::ofstream stream{ "phi_part_cycle.txt" };
	// 	stream << "x, phi" << std::endl;
	// 	const int NX = _nx;
	// 	for (size_t i = 0; i < _nx; ++i) {
	// 		int ix = i;
	// 		int iy = IY;
	// 		int iz = IZ;
	// 		double x = -_domain_l + ix * _dx;
	// 		stream << x << ", " << cell_phi_host[at(ix, iy, iz)] << std::endl;
	// 	}
	// }

	// mass

	// acceleration_field<<<blocks_per_grid, threads_per_block>>>(
	// 	cell_acceleration,
	// 	phi_dev,
	// 	cell_mass,
	// 	_nx,
	// 	_dx
	// );
	// err = cudaDeviceSynchronize();
	// printf("acceleration_field err: %d (%s)\n", (int)err, cudaGetErrorName(err));

	// if (phi_dev != cell_phi_next) {
	// 	cudaMemcpy(cell_phi_next, phi_dev, _num_cells * sizeof(real), cudaMemcpyDeviceToDevice);
	// }
}

//---Host Function----
__host__ void print_particles_bin(
	const char* name,
	int i0,
	int icount,
	const std::vector<real4>& pos,
	const std::vector<real4>& vel,
	int it,
	real t
)
{
	printf("print particles bin %s\n", name);
	FILE *outf;
	char buffer[24];
	int i;

	sprintf(buffer, "bin/%s_%5d.bin", name, it);
	outf = fopen(buffer, "wb");
	fwrite(&icount, sizeof(int), 1, outf);
	fwrite(&t, sizeof(double), 1, outf);
	for (i = i0; i < i0 + icount; ++i) {
		fwrite(&pos[i].x, sizeof(double), 1, outf);
		fwrite(&pos[i].y, sizeof(double), 1, outf);
		fwrite(&pos[i].z, sizeof(double), 1, outf);
		fwrite(&vel[i].x, sizeof(double), 1, outf);
		fwrite(&vel[i].y, sizeof(double), 1, outf);
		fwrite(&vel[i].z, sizeof(double), 1, outf);
	}
	fclose(outf);
	printf("print particles fin\n");
}

__host__ void  result(
	const std::vector<real4>& pos,
	const std::vector<real4>& vel,
	const std::vector<real2>& mass,
	int it,
	real t,
	const std::vector<real>& PSI,
	int it_all
)
{
	printf("print result begin\n");

	FILE *outf;
	int Ns = d.Ns;
	int NN = d.NN;
	int i;
	real vfi, vr, vx, vy, vz, x, y, z, r;
	real Vr_max = 0.0, Vfi_max = 0.0, Vz_max = 0.0, R_max = 0.0, Z_max = 0.0;
	real Ek = 0.0; // kinetic energy
	real Ep = 0.0; // potential energy
	real E = 0.0; // total energy
	real3 Imp = make_real3(0.0, 0.0, 0.0); // momentum
	real3 L = make_real3(0.0, 0.0, 0.0); // angular momentum

	//---Star------
	print_particles_bin("S", 0, Ns, pos, vel, it, t);

	//---DM------
  	if(NN > Ns) {
		int Ndm = NN - Ns;
		print_particles_bin("DM", Ns, Ndm, pos, vel, it, t);
	}

  	Ek = 0.0;
	Ep = 0.0;
	for (i = 0; i < NN; i++) {
		x = pos[i].x;
		y = pos[i].y;
		z = pos[i].z;

		vx = vel[i].x;
		vy = vel[i].y;
		vz = vel[i].z;

		r = sqrt(x * x + y * y);
		if (r > 0.0) {
			vr = (vx * x + vy * y) / r;
			vfi = (vy * x - vx * y) / r;
		}
		else {
			vr = 0.0;
			vfi = 0.0;
		}

		L.x += (vz*y - vy*z)*mass[i].x;
		L.y += (vx*z - vz*x)*mass[i].x;
		L.z += (vy*x - vx*y)*mass[i].x;
		Imp.x += mass[i].x * vx;
		Imp.y += mass[i].x * vy;
		Imp.z += mass[i].x * vz;
		Ek += mass[i].x * (vx * vx + vy * vy + vz * vz);
		Ep += mass[i].x * PSI[i];

		Vr_max = max(Vr_max, abs(vr));
		Vfi_max = max(Vfi_max, vfi);
		Vz_max = max(Vz_max, abs(vz));
		R_max = max(R_max, r);
		Z_max = max(Z_max, abs(z));
	}
  	E = 0.5 * (Ek + Ep);

	if (it == 0) {
		L0.x = L.x;
		L0.y = L.y;
		L0.z = L.z;
		Imp0.x = Imp.x;
		Imp0.y = Imp.y;
		Imp0.z = Imp.z;
    	E0=E;
		outf = fopen("LIE_0.bin", "wb");
		fwrite(&L0, sizeof(real3), 1, outf);
		fwrite(&Imp0, sizeof(real3), 1, outf);
      	fwrite(&E0, sizeof(real), 1, outf);
		fclose(outf);
	}

	real LL = sqrt(
		L.x * L.x +
		L.y * L.y +
		L.z * L.z
	);
	real LL0 = sqrt(
		L0.x * L0.x +
		L0.y * L0.y +
		L0.z * L0.z
	);


	printf("           ***Star***");
	printf("\n R_max = %g  Z_max = %g \n",
		R_max,
		Z_max
	);
	printf("Vr_max = %g  Vfi_max = %g  Vz_max = %g \n",
		Vr_max,
		Vfi_max,
		Vz_max
	);

	printf("           ***Conservation Laws***\n");
	printf("----Imp0--- = %g; %g; %g  \n",
		Imp0.x,
		Imp0.y,
		Imp0.z
	);
	printf("----dImp--- = %g; %g; %g  \n",
		Imp.x - Imp0.x,
		Imp.y - Imp0.y,
		Imp.z - Imp0.z
	);
	printf("Lz = %g  dLz = %g \n",
		L.z,
		L.z / L0.z - 1.0
	);
	printf("LL = %g  dLL = %g \n",
		LL,
		LL / LL0 - 1.0
	);
  	printf("E = %g  dE = %g \n",
		E,
		E / E0 - 1.0
	);

	outf = (it == 0)
		? fopen("Lz(t).dat", "w")
		: fopen("Lz(t).dat", "a");
	fprintf(outf, "%d %f %1.15f %g %g\n",
		it_all,
		t,
		L.z,
		L.z / L0.z - 1.0,
		fabs(L.z / L0.z - 1.0)
	);
	fclose(outf);

	outf = (it == 0)
		? fopen("LL(t).dat", "w")
		: fopen("LL(t).dat", "a");
	fprintf(outf, "%d %f %1.15f %g %g\n",
		it_all,
		t,
		LL,
		LL / LL0 - 1.0,
		fabs(LL / LL0 - 1.0)
	);
	fclose(outf);

	outf = (it == 0)
		? fopen("Imp(t).dat", "w")
		: fopen("Imp(t).dat", "a");
	real Impls = sqrt(
		Imp.x * Imp.x +
		Imp.y * Imp.y +
		Imp.z * Imp.z
	);
	real Impls0 = sqrt(
		Imp0.x * Imp0.x +
		Imp0.y * Imp0.y +
		Imp0.z * Imp0.z
	);
	fprintf(outf, "%d %f %1.15f %g %g %g\n",
		it_all,
		t,
		Impls,
		Impls - Impls0,
		Impls / Impls0 - 1.0,
		fabs(Impls / Impls0 - 1.0)
	);
	fclose(outf);

  	outf = (it == 0)
		? fopen("E(t).dat", "w")
		: fopen("E(t).dat", "a");
	fprintf(outf, "%d %f %1.15f %g %g %g %g\n",
		it_all,
		t,
		E,
		E / E0 - 1.0,
		fabs(E / E0 - 1.0),
		0.5 * Ek,
		0.5 * Ep
	);
	fclose(outf);

	printf("print result end\n");
}
__host__ auto read_start_info(const char* filename) {
	int i_cont = 0;
	double tmax = 0.;
	double dtsave = 0.;

	char temp[FILENAME_MAX];
	FILE* outf = fopen(filename, "r");
	fscanf(outf, "%d  %[^\n]", &i_cont, temp);
	fscanf(outf, "%lf  %[^\n]", &tmax, temp);
	fscanf(outf, "%lf  %[^\n]", &dtsave, temp);
	fclose(outf);

	printf("i_cont: %d\n", i_cont);
	printf("tmax: %f\n", tmax);
	printf("dtsave %f\n", dtsave);
	printf("tsave %f\n", dtsave);

	return std::make_tuple(
		i_cont,
		tmax,
		dtsave,
		dtsave
	);
}
__host__ auto read_galaxies(const char* filename) {
	int M_glx = 0;
	int k_glx = 0;
	int Ns = 0;
	int Ndm = 0;
	int NN = 0;

	char temp[FILENAME_MAX];
	FILE* outf = fopen(filename, "r");
	fscanf(outf, "%d %[^\n]", &M_glx, temp);

	auto N_s = new int[M_glx];
	auto N_dm = new int[M_glx];
	auto Mass_s = new double[M_glx];
	auto Mass_dm = new double[M_glx];
	auto mp_s = new double[M_glx];
	auto mp_dm = new double[M_glx];
	auto X_glx = new double[M_glx];
	auto Y_glx = new double[M_glx];
	auto Z_glx = new double[M_glx];
	auto Vx_glx = new double[M_glx];
	auto Vy_glx = new double[M_glx];
	auto Vz_glx = new double[M_glx];
	auto alpha_glx = new double[M_glx];
	auto eps_s = new double[M_glx];
	auto eps_dm = new double[M_glx];

	for(int k = 0; k < M_glx; k++) {
		fscanf(outf, "%d %[^\n]", &k_glx, temp);
		fscanf(outf, "%d,%d %[^\n]", &N_s[k], &N_dm[k], temp);
		fscanf(outf, "%lf,%lf %[^\n]", &Mass_s[k], &Mass_dm[k], temp);
		fscanf(outf, "%lf,%lf %[^\n]", &eps_s[k], &eps_dm[k], temp);
		fscanf(outf, "%lf %[^\n]", &alpha_glx[k], temp);
		fscanf(outf, "%lf,%lf,%lf %[^\n]", &X_glx[k], &Y_glx[k], &Z_glx[k], temp);
		fscanf(outf, "%lf,%lf,%lf %[^\n]", &Vx_glx[k], &Vy_glx[k], &Vz_glx[k], temp);
		if (Mass_s[k] == 0.0 || N_s[k] == 0) {
			Mass_s[k] = 0.0;
			N_s[k] = 0;
		}
		if (Mass_dm[k] == 0.0 || N_dm[k] == 0) {
			Mass_dm[k] = 0.0;
			N_dm[k] = 0;
		}

		mp_s[k] = (N_s[k] > 0)
			? Mass_s[k] / N_s[k]
			: 0.0;
		mp_dm[k] = (N_dm[k] > 0)
			? Mass_dm[k] / N_dm[k]
			: 0.0;

		Ns += N_s[k];
		Ndm += N_dm[k];
		alpha_glx[k] *= DEG2RAD;
	}

	NN = Ns + Ndm;
	printf("NN = %d, Ns = %d, Ndm = %d\n", NN, Ns, Ndm);
	printf("mp_s[0] = %g, mp_dm[0] = %g \n", mp_s[0], mp_dm[0]);

	return std::make_tuple(
		M_glx,
		Ns,
		Ndm,
		NN,
		N_s,
		N_dm,
		Mass_s,
		Mass_dm,
		mp_s,
		mp_dm,
		X_glx,
		Y_glx,
		Z_glx,
		Vx_glx,
		Vy_glx,
		Vz_glx,
		alpha_glx,
		eps_s,
		eps_dm
	);
}

int main(int argc, char * argv[])
{
	FILE *outf;
	real r, vr, vfi, fi;
  	char str[24];

	//----GPU device------------------------------------------------
	int deviceCount, nGPU;
	cudaDeviceProp prop;
	int j, nthr, i;

	nthr = omp_get_num_threads();
	printf("cpuThreads = %d\n", nthr);

	cudaGetDeviceCount(&deviceCount);
	printf("deviceCount = %d\n", deviceCount);
	for (i = 0; i < deviceCount; i++){
		cudaGetDeviceProperties(&prop, i);
		printf("gpuID = %d, gpuName = %s\n", i, prop.name);
	}

  	char name[FILENAME_MAX];
	outf = fopen("__GPUs.ini", "r");
	fscanf(outf, "%d  %[^\n]", &nGPU, name);

    int *deviceId = new int[nGPU];
	for (i = 0; i < nGPU; i++) {
		fscanf(outf, "%d  %[^\n]", &deviceId[i], name);
		if (deviceId[i] > deviceCount - 1) {
			printf("\n Net takogo nomera device GPU");
			return 0;
		}
	}
	fclose(outf);

	for (i = 0; i < nGPU; i++)
		printf("deviceId[%d] = %d\n", i, deviceId[i]);

	int can_access_peer, itmp;
	for (i = 0; i < nGPU; i++) {
		cudaSetDevice(deviceId[i]);
		for (j = 0; j < nGPU; j++) {
			if (j != i) {
				cudaDeviceCanAccessPeer(&can_access_peer, deviceId[i], deviceId[j]);
				printf("can_access_peer=%d, %d, %d\n", can_access_peer, deviceId[i], deviceId[j]);
				if (can_access_peer == 0) {
					printf("ERROR! -- can_access_peer = 0 for deviceId = %d  and  deviceId = %d\n", deviceId[i], deviceId[j]);
					printf("Press any key + Enter\n");
					scanf("%d", &itmp);
					exit(0);
				}
			}
		}
	}

	for (i = 0; i < nGPU; i++) {
		cudaSetDevice(deviceId[i]);
		for (j = 0; j < nGPU; j++) {
			if (j != i) {
				cudaDeviceEnablePeerAccess(deviceId[j], 0);
			}
		}
	}

	cudaEvent_t start, stop, start1, stop1;
	float gpuTime = 0.0; //
	float gpuTime_GFC = 0.0; // time between saves
	float gpuTime_US = 0.0; // zero only ???
	float gpuTime1 = 0.0; // time for single iteration
	//-----Unitial State---------------------------------------------------------
	real t = 0.0; // current time
	real tmax = 0.0; // max simulation time
	real tsave = 0.0;
	real dtsave = 0.0;
	real dtgrav=0.001, tgrav;
	real Mh, a, Rh, Mb, b, Rb, eps2;
	int i_cont = 0; // iteration to continue from
	real K_m, K_r;
	int Ns = 0; // total star particles
	int Ndm = 0; // total dark matter particles
	int NN = 0; // total particles
	int *N_s, *N_dm; // particles count [galaxy num]
	int M_glx = 0; // galaxies count
	int k = 0; // galaxies iterator
	// int k_glx = 0; // galaxy number
	double *Mass_s, *Mass_dm; // mass [galaxy num]
	double *eps_s, *eps_dm; // gravitational softening length [galaxy num]
	double *mp_s, *mp_dm; // particles mass in the galaxy [galaxy num]
	double *alpha_glx; // galaxy angle [galaxy num]
	double *X_glx, *Y_glx, *Z_glx; // galaxy mass center [galaxy num]
	double *Vx_glx, *Vy_glx, *Vz_glx; // galaxy mass center [galaxy num]

	std::tie(
		M_glx,
		Ns,
		Ndm,
		NN,
		N_s,
		N_dm,
		Mass_s,
		Mass_dm,
		mp_s,
		mp_dm,
		X_glx,
		Y_glx,
		Z_glx,
		Vx_glx,
		Vy_glx,
		Vz_glx,
		alpha_glx,
		eps_s,
		eps_dm
	) = read_galaxies("__start_galaxies.ini");

	std::tie(
		i_cont,
		tmax,
		dtsave,
		tsave
	) = read_start_info("__start_nbody.ini");

	printf("Ns / BLOCK_SIZE_b = %d\n", Ns / BLOCK_SIZE);
	printf("Ndm / BLOCK_SIZE_b = %d\n", Ns / BLOCK_SIZE);
	printf("NN / BLOCK_SIZE_b = %d\n", NN / BLOCK_SIZE);

	outf = fopen("__gr_par.ini", "r");
	fscanf(outf, "%lf  %[^\n]", &Mh, name);
	fscanf(outf, "%lf  %[^\n]", &a, name);
	fscanf(outf, "%lf  %[^\n]", &Rh, name);
	fscanf(outf, "%lf  %[^\n]", &Mb, name);
	fscanf(outf, "%lf  %[^\n]", &b, name);
	fscanf(outf, "%lf  %[^\n]", &Rb, name);
	fscanf(outf, "%lf  %[^\n]", &eps2, name);
	fscanf(outf, "%lf  %[^\n]", &dtgrav, name);
	fscanf(outf, "%lf  %[^\n]", &K_m, name);    // K_m = Md/(10^{10}*Msun)
	fscanf(outf, "%lf  %[^\n]", &K_r, name);    // K_r = L_r / 10 кпк
	fclose(outf);
	printf("Mh\t= %f\n", Mh);
	printf("a\t= %f\n", a);
	printf("Rh\t= %f\n", Rh);
	printf("Mb\t= %f\n", Mb);
	printf("b\t= %f\n", b);
	printf("Rb\t= %f\n", Rb);
	printf("eps\t= %f\n", eps2);
	eps2 *= eps2;
	printf("eps2\t= %f\n", eps2);
	printf("dtgrav\t= %f\n", dtgrav);

	const int is_grav = (int)(dtsave / dtgrav + 0.5); // in-frame iterations max (between saves)
	int it_grav = 0; // in-frame iterations (between saves)

	const real Rh2 = 3.0*Rh;
	const real rcore1 = Rh / a;
	const real rbcore1 = 1.0 / b;
	const real rbcore2 = rbcore1 * rbcore1;
	const real root1 = sqrt(1.0 + (Rb*Rb)*rbcore2);
	const real con = Mh / (rcore1 - atan(rcore1));
	const real const1 = Mb / (b*log(Rb*rbcore1 + root1) - Rb / root1);
	const real c_psi_h = con / a*(0.5*log(Rh2*Rh2 / a / a + 1.0) + atan(Rh2 / a)*a / Rh2) + Mh / Rh2;
	const real c_psi_b = Mb / Rb - const1*log(Rb / b + root1) / Rb;
	const real Mh_inf = Mh * (Rh2 / a - atan(Rh2 / a)) / (rcore1 - atan(rcore1));

	printf("*****Rh2 = %g \n", Rh2);
	printf("*****con = %g \n", con);
	printf("*****const1 = %g \n", const1);
	printf("*****c_psi_h = %g \n", c_psi_h);
	printf("*****c_psi_b = %g \n", c_psi_b);

    constexpr int _nx = 200;
    constexpr double _dx = 0.1;
    constexpr double _domain_l = 0.5 * _nx * _dx;
	constexpr double _c_wave = 4574.337022617616;
	const double _dt_wave = 0.5 * _dx / (_c_wave * sqrt(3));
	const double _diss_base = 0.1;
	const double _diss_extra = 0.01;
	const double _sim_l = _domain_l;
	const double _bc_l = _domain_l / 10.;
	const int _iterations = 500;

    real4 _domain_min{
        -_domain_l,
        -_domain_l,
        -_domain_l,
		0.
    };
    real4 _domain_max{
        _domain_l,
        _domain_l,
        _domain_l,
		0.
    };
    real4 _cell_size{
        _dx,
        _dx,
        _dx,
		0.
    };
    int3 _grid_size{
        _nx,
        _nx,
        _nx
    };
    int _num_particles = NN;
    int _num_cells = _nx * _nx * _nx;
    int _num_cell_blocks = (_num_cells + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int _num_particle_blocks = (_num_particles + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int _over_cells = _num_cell_blocks;
    int _over_particles = _num_particle_blocks;
    int _over_blocks = BLOCK_SIZE;
	printf("_num_particles: %d\n", _num_particles);
	printf("_num_cells: %d\n", _num_cells);
	printf("_num_cell_blocks: %d\n", _num_cell_blocks);
	printf("_num_particle_blocks: %d\n", _num_particle_blocks);
	printf("_over_cells: %d\n", _over_cells);
	printf("_over_particles: %d\n", _over_particles);
	printf("_over_blocks: %d\n", _over_blocks);
	printf("_grid_size: (%d, %d, %d)\n", _grid_size.x, _grid_size.y, _grid_size.z);

	//-----Allocate Massiv Host-----------------------------
	printf("allocate memory HOST\n");
	std::vector<real4> pos_host(NN, make_real4(0., 0., 0., 0.));
	std::vector<real4> vel_host(NN, make_real4(0., 0., 0., 0.));
	std::vector<real2> mass_host(NN, make_real2(0., 0.));
	std::vector<real> eps2_host(NN, 0.);
	std::vector<real> psi_host(NN, 0.);

	std::vector<real> cell_mass_host(_num_cells, 0.);
	std::vector<real> cell_phi_host(_num_cells,  0.);

	//-----------------------------------------------------

	d.Ns = Ns;
	d.NN = NN;
	d.Mh = Mh;
	d.Mh_inf = Mh_inf;
	d.a = a;
	d.Rh = Rh;
	d.Mb = Mb;
	d.b = b;
	d.Rb = Rb;
	d.Rh2 = Rh2;
	d.con = con;
	d.const1 = const1;
	d.c_psi_h = c_psi_h;
	d.c_psi_b = c_psi_b;
	d.eps2 = eps2;
	d.domainMin = _domain_min;
	d.domainMax = _domain_max;
	d.cellSize = _cell_size;
	d.gridSize  = _grid_size;
	d.dx_wave = _dx;
	d.dt_wave = _dt_wave;
	d.c_wave = _c_wave;
	d.nx_wave = _nx;
	d.diss_base = _diss_base;
	d.diss_extra = _diss_extra;
	d.sim_l = _sim_l;
	d.bc_l = _bc_l;

	int it = 1, // save num
		itt = 1,
		ittg = 0,
		itg = 1;

	printf("***Input Data***\n");

  	if (i_cont > 0) {
		//---Stars---
		sprintf(str, "bin/S_%5d.bin", i_cont);
      	FILE* outf = fopen(str, "rb");
      	fread(&Ns, sizeof(int), 1, outf);
		fread(&t, sizeof(double), 1, outf);
		int n0 = 0;
		for(k = 0; k < M_glx; k++) {
        	for (i = n0; i < n0 + N_s[k]; i++) {
				fread(&pos_host[i].x, sizeof(double), 1, outf);
				fread(&pos_host[i].y, sizeof(double), 1, outf);
				fread(&pos_host[i].z, sizeof(double), 1, outf);
				fread(&vel_host[i].x, sizeof(double), 1, outf);
				fread(&vel_host[i].y, sizeof(double), 1, outf);
				fread(&vel_host[i].z, sizeof(double), 1, outf);
				mass_host[i].x = mp_s[k];
				eps2_host[i] = eps_s[k]*eps_s[k];
			}
			n0 += N_s[k];
		}
		fclose(outf);
      	//---DM---
		if(Ndm>0) {
			i = sprintf(str, "bin/DM_%5d.bin", i_cont);
			outf = fopen(str, "rb");
        	fread(&Ndm, sizeof(int), 1, outf);
          	fread(&t, sizeof(double), 1, outf);
			n0=0;
			for(k=0; k<M_glx; k++) {
           		for (i = n0+Ns; i < n0+Ns+N_dm[k]; i++) {
					fread(&pos_host[i].x, sizeof(double), 1, outf);
					fread(&pos_host[i].y, sizeof(double), 1, outf);
					fread(&pos_host[i].z, sizeof(double), 1, outf);
					fread(&vel_host[i].x, sizeof(double), 1, outf);
					fread(&vel_host[i].y, sizeof(double), 1, outf);
					fread(&vel_host[i].z, sizeof(double), 1, outf);
					mass_host[i].x = mp_dm[k];
					eps2_host[i] = eps_dm[k]*eps_dm[k];
				}
				n0 += N_dm[k];
			}
        	fclose(outf);
		}
		it = (int)(t / dtsave);
		printf("Start time = %f  it = %d\n", t, it);
		printf("***Start result t***\n");

		outf = fopen("LIE_0.bin", "rb");
		fread(&L0, sizeof(real3), 1, outf);
		fread(&Imp0, sizeof(real3), 1, outf);
		fread(&E0, sizeof(real), 1, outf);
		fclose(outf);

		it++;
		itg = (int)(t / dtgrav) + 1;
		tsave = t + dtsave;
		tgrav = t + dtgrav;
	}
	else {
    	printf("t = %f\n",t);
		//----Stars-------------------------------
		int n0 = 0;
      	for(k = 0; k < M_glx; k++) {
      		if(N_s[k] > 0) {
				FILE* outf = NULL;
        		sprintf(str, "start_S%1d.txt", k);
				outf = fopen(str, "r");
          		if (NULL == outf) {
					printf("Error OF -- %s ",str);
					return 0;
				}
          		else {
					int itmp = 0;
					real rtmp = 0.0;
            		fscanf(outf, "%d %lf", &itmp, &rtmp);
              		printf("N_s[%d] = %d, t = %f\n", k, itmp, rtmp);
              		for(i = n0; i < n0 + N_s[k]; ++i) {
						fscanf(outf, "%lf %lf %lf %lf %lf %lf",
							&pos_host[i].x,
							&pos_host[i].y,
							&pos_host[i].z,
							&vel_host[i].x,
							&vel_host[i].y,
							&vel_host[i].z
						);
						mass_host[i].x = mp_s[k];

						pos_host[i].x = X_glx[k]
						 	+ pos_host[i].x * cos(alpha_glx[k])
							+ pos_host[i].z * sin(alpha_glx[k]);
						pos_host[i].y += Y_glx[k];
						pos_host[i].z = Z_glx[k]
							+ pos_host[i].z * cos(alpha_glx[k])
							- pos_host[i].x * sin(alpha_glx[k]);

						vel_host[i].x = Vx_glx[k]
							+ vel_host[i].x * cos(alpha_glx[k])
							+ vel_host[i].z * sin(alpha_glx[k]);
						vel_host[i].y += Vy_glx[k];
						vel_host[i].z = Vz_glx[k]
							+ vel_host[i].z * cos(alpha_glx[k])
							- vel_host[i].x * sin(alpha_glx[k]);

						eps2_host[i] = eps_s[k]*eps_s[k];
              		}
          		}
				fclose(outf);
          		n0 += N_s[k];
        	}
		}
		//----DM-------------------------------
		for (k = 0; k < M_glx; k++) {
			if (N_dm[k] > 0) {
				FILE* outf = NULL;
				sprintf(str, "start_DM%1d.txt", k);
				outf = fopen(str, "r");
          		if (NULL == outf) {
					printf("Error OF -- %s ",str);
					return 0;
				}
				else {
					int itmp = 0;
					real rtmp = 0.0;
					fscanf(outf, "%d %lf", &itmp, &rtmp);
					printf("N_dm[%d] = %d, t = %f\n", k, itmp, rtmp);
					for(i = n0; i < n0 + N_dm[k]; ++i) {
						fscanf(outf, "%lf %lf %lf %lf %lf %lf",
							&pos_host[i].x,
							&pos_host[i].y,
							&pos_host[i].z,
							&vel_host[i].x,
							&vel_host[i].y,
							&vel_host[i].z
						);

						mass_host[i].x = mp_dm[k];

						pos_host[i].x = X_glx[k]
							+ pos_host[i].x * cos(alpha_glx[k])
							+ pos_host[i].z * sin(alpha_glx[k]);
						pos_host[i].y += Y_glx[k];
						pos_host[i].z = Z_glx[k]
							+ pos_host[i].z * cos(alpha_glx[k])
							- pos_host[i].x * sin(alpha_glx[k]);

						vel_host[i].x = Vx_glx[k]
							+ vel_host[i].x * cos(alpha_glx[k])
							+ vel_host[i].z * sin(alpha_glx[k]);
						vel_host[i].y += Vy_glx[k];
						vel_host[i].z = Vz_glx[k]
							+ vel_host[i].z * cos(alpha_glx[k])
							- vel_host[i].x * sin(alpha_glx[k]);

						eps2_host[i] = eps_dm[k]*eps_dm[k];
					}
				}
				fclose(outf);
				n0 += N_dm[k];
			}
		}
		tsave = dtsave;
		tgrav = dtgrav;
		t = 0.0;
    }

	printf("***Start GPU***\n");

	//-----Allocate Massiv GPU--------------------------------------

	printf("allocate memory GPU %d\n", i);
	CuDarray<real4> pos_(pos_host);
	CuDarray<real4> vel_(vel_host);
	CuDarray<real2> mass_(mass_host);
	CuDarray<real> eps2_(eps2_host);
	CuDarray<real> psi_(psi_host);
	CuDarray<real4> post_(_num_particles);
	CuDarray<real4> velt_(_num_particles);
	CuDarray<real3> acc_(_num_particles);
	CuDarray<real3> acct_(_num_particles);

	int Nk = NN / nGPU;

	CuDarray<CellInfo> cell_info_(_num_cells);
	CuDarray<int> cell_particles_count_(_num_cells);
	CuDarray<real> cell_mass_(_num_cells);
	CuDarray<real> cell_phi_prev_(_num_cells);
	CuDarray<real> cell_phi_curr_(_num_cells);
	CuDarray<real> cell_phi_next_(_num_cells);
	CuDarray<real3> cell_acceleration_(_num_cells);

	CuDarray<ParticleCellInfo> particles_cell_info_(_num_particles);

	CuDarray<int> particles_in_block_(_num_cell_blocks);

	CuCopyToSymbol(d, dd, ToDevice);
	CuDeviceSync();

	//------Расчет грав. сил-----------------------------------------------------
	printf("calc grav forces\n");
	CuCall(PSI_kernel, Nk / BLOCK_SIZE, BLOCK_SIZE) (
		psi_,
		pos_,
		pos_,
		mass_,
		eps2_
	);

	convert_particles(
		psi_,
		pos_,
		mass_,
		cell_info_,
		cell_particles_count_,
		particles_in_block_,
		particles_cell_info_,
		cell_mass_,
		cell_phi_prev_,
		cell_phi_curr_,
		cell_phi_next_,
		cell_acceleration_,
		_over_cells,
		_over_blocks,
		_over_particles,
		_num_cells,
		_num_cell_blocks,
		_num_particles,
		_domain_l,
		_nx,
		_dx,
		_iterations,
		true
	);

	// apply_acceleration_to_particles<<<_over_particles, _over_blocks>>>(
	// 	acc_,
	// 	cell_acceleration_,
	// 	particles_cell_info_,
	// 	_num_particles
	// );
	// auto err = cudaDeviceSynchronize();
	// printf("apply_acceleration_to_particles err: %d (%s)\n", (int)err, cudaGetErrorName(err));

	// apply_psi_to_particles<<<_over_particles, _over_blocks>>>(
	// 	psi_,
	// 	cell_phi_next_,
	// 	cell_info_,
	// 	particles_cell_info_,
	// 	_num_particles
	// );
	// err = cudaDeviceSynchronize();
	// printf("apply_psi_to_particles err: %d (%s)\n", (int)err, cudaGetErrorName(err));

	printf("calc grav forces: copy to host\n");
	psi_.to_vector(psi_host);

	if(it == 1) {
		printf("***Start result t=0***\n");
		result(pos_host, vel_host, mass_host, 0, 0.0, psi_host, 0);
	}

	for (i = 0; i < nGPU; i++){
		cudaSetDevice(deviceId[i]);
		for (j = 0; j < nGPU; j++){
			if (j != i) cudaDeviceDisablePeerAccess(deviceId[j]);
		}
	}

	return 0;
}