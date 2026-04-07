#pragma once
#include "common.cuh"

// алгоритм основан на статье
// Khrapov, Khoperskov 2017 - Smoothed-Particle Hydrodynamics Models: Implementation Features on GPUs


// ====================================================
// ЯДРО 2: ИНИЦИАЛИЗАЦИЯ ВСПОМОГАТЕЛЬНЫХ МАССИВОВ
// ====================================================
__global__ void assignParticlesToCells(
	const real3* particles_pos, // [N] исходные частицы
    ParticleCellInfo* particles_cell_info, // [N] инфо: x=ячейка, y=индекс в ячейке
    CellInfo* cellInfo,          // [TOTAL_CELLS] для подсчёта частиц
    int* cellParticleCount, // [TOTAL_CELLS] счётчик для atomicAdd
    const int numParticles
)
{
    int i_part = threadIdx.x + blockIdx.x * blockDim.x;
    if (i_part >= numParticles) return;

    real3 p = particles_pos[i_part];

    // Вычисление индексов ячейки
    int ix = cuda::std::clamp(
        (int)((p.x - wave_eq_data_.domainMin.x) / wave_eq_data_.cellSize.x),
        (int)0,
        (int)(wave_eq_data_.gridSize.x - 1));
    int iy = cuda::std::clamp(
        (int)((p.y - wave_eq_data_.domainMin.y) / wave_eq_data_.cellSize.y),
        (int)0,
        (int)(wave_eq_data_.gridSize.y - 1));
    int iz = cuda::std::clamp(
        (int)((p.z - wave_eq_data_.domainMin.z) / wave_eq_data_.cellSize.z),
        (int)0,
        (int)(wave_eq_data_.gridSize.y - 1));

    // Линейный индекс ячейки (Morton-подобный порядок для пространственной локальности)
    // Используем чередование битов для лучшей когерентности
    int sx = wave_eq_data_.gridSize.x;
    int sy = wave_eq_data_.gridSize.y;
    int sz = wave_eq_data_.gridSize.z;
    int i_cell = iz * sx * sy + iy * sz + ix;

    // Сохраняем номер ячейки для частицы
    particles_cell_info[i_part].cell_id = i_cell;

    // Атомарно увеличиваем счётчик частиц в ячейке
    int pos = atomicAdd(&cellParticleCount[i_cell], 1);

    // Сохраняем позицию внутри ячейки
    particles_cell_info[i_part].id_in_cell = pos;

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
    const real* particle_mass,   	   // [N] упорядоченные частицы
    ParticleCellInfo* particles_cell_info,// [N] инфо: x=ячейка, y=индекс в ячейке
    double* cellMasses,                // [TOTAL_CELLS] результат
    const int numParticles)
{
    int i_particle = threadIdx.x + blockIdx.x * blockDim.x;
    if (i_particle >= numParticles) return;

    int i_cell = particles_cell_info[i_particle].cell_id;
	atomicAdd(&cellMasses[i_cell], particle_mass[i_particle]);
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
__global__ void computeCellPhiUnsorted(
    const real* particle_phi,   	   // [N] упорядоченные частицы
    const ParticleCellInfo* particles_cell_info,// [N] инфо: x=ячейка, y=индекс в ячейке
    const CellInfo* cellInfo,     		 // [TOTAL_CELLS]
    double* cell_phi,                // [TOTAL_CELLS] результат
    const int numParticles)
{
    int i_particle = threadIdx.x + blockIdx.x * blockDim.x;
    if (i_particle >= numParticles) return;

    int i_cell = particles_cell_info[i_particle].cell_id;
	atomicAdd(&cell_phi[i_cell], particle_phi[i_particle] / cellInfo[i_cell].count);
	// if (particles_cell_info[i_particle].id_in_cell == 0) {
	// 	cell_phi[i_cell] = particle_phi[i_particle];
	// }
}

__global__ void acc_field(
	real3* cell_acc, // сила на единицу массы
	real*  cell_acc_abs,  // сила на единицу массы
	const real* cell_phi,
	const int NX,
	const double DX
)
{
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int iz = threadIdx.z + blockIdx.z * blockDim.z;

	int xyz = AT(ix, iy, iz);

	real3 dphi;

	if (ix == 0) {
		dphi.x = cell_phi[AT(ix + 1, iy, iz)] - cell_phi[xyz];
	}
	else if (ix == NX - 1) {
		dphi.x = cell_phi[xyz] - cell_phi[AT(ix - 1, iy, iz)];
	}
	else {
		dphi.x = 0.5 * (cell_phi[AT(ix + 1, iy, iz)] - cell_phi[AT(ix - 1, iy, iz)]);
	}

	if (iy == 0) {
		dphi.y = cell_phi[AT(ix, iy + 1, iz)] - cell_phi[xyz];
	}
	else if (iy == NX - 1) {
		dphi.y = cell_phi[xyz] - cell_phi[AT(ix, iy - 1, iz)];
	}
	else {
		dphi.y = 0.5 * (cell_phi[AT(ix, iy + 1, iz)] - cell_phi[AT(ix, iy - 1, iz)]);
	}

	if (iz == 0) {
		dphi.z = cell_phi[AT(ix, iy, iz + 1)] - cell_phi[xyz];
	}
	else if (iz == NX - 1) {
		dphi.z = cell_phi[xyz] - cell_phi[AT(ix, iy, iz - 1)];
	}
	else {
		dphi.z = 0.5 * (cell_phi[AT(ix, iy, iz + 1)] - cell_phi[AT(ix, iy, iz - 1)]);
	}

    real coef = -1. / DX;
	dphi.x *= coef;
	dphi.y *= coef;
	dphi.z *= coef;

	cell_acc[xyz] = dphi;

    if (cell_acc_abs) {
        cell_acc_abs[xyz] = norm3(dphi);
    }
}

__global__ void apply_acceleration_to_particles(
	real3* particle_acceleration,
	const real3* cell_acc,
    ParticleCellInfo* particles_cell_info,
    const int numParticles
)
{
    int i_particle = threadIdx.x + blockIdx.x * blockDim.x;
    if (i_particle >= numParticles) return;

    int i_cell = particles_cell_info[i_particle].cell_id;
	particle_acceleration[i_particle].x = cell_acc[i_cell].x;
	particle_acceleration[i_particle].y = cell_acc[i_cell].y;
	particle_acceleration[i_particle].z = cell_acc[i_cell].z;
}
__global__ void apply_cell_to_particles(
	real* particle_value,
	const real* cell_value,
    ParticleCellInfo* particles_cell_info,
    const int numParticles
)
{
    int i_particle = threadIdx.x + blockIdx.x * blockDim.x;
    if (i_particle >= numParticles) return;

    int i_cell = particles_cell_info[i_particle].cell_id;
	particle_value[i_particle] = cell_value[i_cell];
}

__global__ void apply_cell_to_particles_avg(
	real* particle_value,
	const real* cell_value,
    ParticleCellInfo* particles_cell_info,
    const CellInfo* cellInfo,
    const int numParticles
)
{
    int i_particle = threadIdx.x + blockIdx.x * blockDim.x;
    if (i_particle >= numParticles) return;

    int i_cell = particles_cell_info[i_particle].cell_id;
	particle_value[i_particle] = cellInfo[i_cell].count > 0
        ? cell_value[i_cell] / cellInfo[i_cell].count
        : -1.;
}

__global__ void apply_particle_to_cell_weighted(
    double* cell_value,
    const real* cell_mass,
    const real* particle_mass,
    const real* particle_value,
    const ParticleCellInfo* particles_cell_info,
    const int numParticles)
{
    int i_particle = threadIdx.x + blockIdx.x * blockDim.x;
    if (i_particle >= numParticles) return;

    int i_cell = particles_cell_info[i_particle].cell_id;
    real weight = particle_mass[i_particle] / cell_mass[i_cell];
	atomicAdd(&cell_value[i_cell], particle_value[i_particle] * weight);
}

__global__ void acc_abs(
    real* acc_abs,
    const real3* acc,
    const int N
)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N) return;

    acc_abs[i] = norm3(acc[i]);
}