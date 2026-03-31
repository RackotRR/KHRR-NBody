#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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
	real    c_phi_h;
	real    c_phi_b;
	real    eps2;
};

struct WaveEqData {
    real3 domainMin;
    real3 domainMax;
    real3 cellSize;
    int3 gridSize;

	int nx_wave;
	double dt_wave;
	double c_wave;
	double dx_wave;

	double diss_base;
	double diss_extra;
	double sim_l;
	double bc_l;
};

__constant__ DataBlock dd;
__constant__ WaveEqData wave_eq_data_;

#define AT(x, y, z) ((x) + (y) * (NX) + (z) * (NX) * (NX))

inline __host__ __device__ real dot2(const real2& vec1, const real2& vec2) {
	return
		vec1.x * vec2.x +
		vec1.y * vec2.y;
}
inline __host__ __device__ real dot3(const real3& vec1, const real3& vec2) {
	return
		vec1.x * vec2.x +
		vec1.y * vec2.y +
		vec1.z * vec2.z;
}

inline __host__ __device__ real norm2(const real2& vec) {
	return sqrt(dot2(vec, vec));
}
inline __host__ __device__ real norm3(const real3& vec) {
	return sqrt(dot3(vec, vec));
}

inline __host__ __device__ real cube(real x) {
	return x * x * x;
}

template<typename T>
__host__ __device__ T clamp(T val, T min_val, T max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}