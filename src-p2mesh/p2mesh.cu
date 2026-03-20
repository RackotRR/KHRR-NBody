
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <RR/RRCU.cuh>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <format>
#include <span>

#define PI 3.14159265358979
#define BLOCK_SIZE 512


#define real double
#define real2 double2
#define real3 double3
#define real4 double4_32a
#define make_real3 make_double3
#define make_real4 make_double4_32a
#define make_real2 make_double2

struct CellInfo{
    int count; // число частиц в ячейке
    int start_id; // накопленная сумма частиц (префиксная сумма) - идекс в массиве частиц
};

struct ParticleCellInfo {
    int cell_id; // индекс ячейки
    int id_in_cell; // индекс частицы в ячейке
};

typedef struct {
    double x, y, z;  // координаты частицы
    double mass;     // масса частицы
} Particle;

template<typename T>
constexpr T intlog2(T size) {
	return size > 1
        ? 1 + intlog2(size >> 1)
        : 0;
}

struct DataBlock{
    real3 	domainMin;
    real3 	domainMax;
    real3 	cellSize;
    int3	gridSize;
};
__constant__ DataBlock dd;

auto read_particles(
    const std::filesystem::path& path,
    double total_mass
)
{
    std::ifstream stream{ path };
    size_t N = 0;
    double t = 0;
    stream >> N >> t;
    std::cout << std::format("read {} particles on time {}", N, t) << std::endl;

    double vx, vy, vz;
    const double particle_mass = total_mass / N;
    std::cout << std::format("particles mass: {}", particle_mass) << std::endl;

    Particle particle;
    particle.mass = particle_mass;

    std::vector<Particle> particles;
    particles.reserve(N);
    while (
        stream >> particle.x >> particle.y >> particle.z >> vx >> vy >> vz
    )
    {
        particles.push_back(particle);
        if (particles.size() % 10000 == 0) {
            std::cout << std::format("{} / {}", particles.size(), N) << std::endl;
        }

        if (particles.size() == N) {
            break;
        }
    }

    double3 domain_min{
        0., 0., 0.
    };
    double3 domain_max{
        0., 0., 0.
    };
    for (const auto& particle : particles) {
        domain_min.x = std::min(domain_min.x, particle.x);
        domain_min.y = std::min(domain_min.y, particle.y);
        domain_min.z = std::min(domain_min.z, particle.z);
        domain_max.x = std::max(domain_max.x, particle.x);
        domain_max.y = std::max(domain_max.y, particle.y);
        domain_max.z = std::max(domain_max.z, particle.z);
    }

    std::cout << std::format("domain min: ({},\t{},\t{})",domain_min.x, domain_min.y, domain_min.z) << std::endl;
    std::cout << std::format("domain max: ({},\t{},\t{})",domain_max.x, domain_max.y, domain_max.z) << std::endl;

    return particles;
}

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

int main(int argc, char * argv[]) {

    constexpr int nx = 200;
    constexpr double dx = 0.1;
    constexpr double domain_l = 0.5 * nx * dx;
    double3 domain_min{
        -domain_l,
        -domain_l,
        -domain_l
    };
    double3 domain_max{
        domain_l,
        domain_l,
        domain_l
    };
    double3 cell_size{
        dx,
        dx,
        dx
    };
    int3 grid_size{
        nx,
        nx,
        nx
    };

    auto particles = read_particles("./start_S0.txt", 1.0);

    int num_particles = particles.size();
    int num_cells = nx * nx * nx;
    int block_size = 256;
    int num_cell_blocks = (num_cells + block_size - 1) / block_size;
    int num_particle_blocks = (num_particles + block_size - 1) / block_size;
    int over_cells = num_cell_blocks;
    int over_particles = num_particle_blocks;
    int over_blocks = block_size;

    using namespace RR::CUDA;
    auto particles_pos_ = CuDarray<real4>(num_particles);
    auto particles_cell_info_ = CuDarray<ParticleCellInfo>(num_particles);
    auto cell_mass_ = CuDarray<real>(num_cells);
    auto cell_info_ = CuDarray<CellInfo>(num_cells);
    auto cell_particles_count_ = CuDarray<int>(num_cells);
    auto particles_in_block_ = CuDarray<int>(num_cells);
    std::cout << "total allocated: " << CuDarrayBase::get_total_allocated_mb() << std::endl;

    DataBlock data_block;
    data_block.cellSize = cell_size;
    data_block.domainMax = domain_max;
    data_block.domainMin = domain_min;
    data_block.gridSize = grid_size;
    CuCopyToSymbol(data_block, dd, ToDevice);
    CuDeviceSync();

    CuCall(assignParticlesToCells, over_particles, over_blocks) (
        particles_pos_,
        particles_cell_info_,
        cell_info_,
        cell_particles_count_,
        num_particles
    );
    CuCall(computePrefixSums, over_cells, over_blocks) (
        cell_info_,
        particles_in_block_,
        num_cells
    );
    CuCall(adjustGlobalPrefixSums, over_cells, over_blocks) (
        cell_info_,
        particles_in_block_,
        num_cells
    );

    // particles cell info check
    {
        std::vector<ParticleCellInfo> particles_cell_info = particles_cell_info_.to_vector();
        std::vector<CellInfo> cell_info = cell_info_.to_vector();
        std::vector<CellInfo> cell_info_manual(num_cells, CellInfo{ 0, 0 });
        for (size_t i = 0; i < num_particles; ++i) {
            size_t c = particles_cell_info[i].cell_id;
            cell_info_manual[c].count++;
        }

        for (size_t i = 0; i < num_cells; ++i) {
            if (cell_info_manual[i].count != cell_info[i].count) {
                std::cout << "cell info mismatch!!! " << cell_info_manual[i].count << " : " << cell_info[i].count << std::endl;
            }
        }

        std::cout << "full match" << std::endl;
    }

    std::cout << "success!" << std::endl;
    return 0;
}