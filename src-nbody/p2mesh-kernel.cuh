#pragma once
#include "common.cuh"

// алгоритм основан на статье
// Khrapov, Khoperskov 2017 - Smoothed-Particle Hydrodynamics Models: Implementation Features on GPUs

template<typename T>
__host__ __device__ T clamp(T val, T min_val, T max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
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
    int ix = clamp(
        (int)((p.x - wave_eq_data_.domainMin.x) / wave_eq_data_.cellSize.x),
        (int)0,
        (int)(wave_eq_data_.gridSize.x - 1));
    int iy = clamp(
        (int)((p.y - wave_eq_data_.domainMin.y) / wave_eq_data_.cellSize.y),
        (int)0,
        (int)(wave_eq_data_.gridSize.y - 1));
    int iz = clamp(
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