//Author: S.S. Khrapov
//Parallel Nbody Code OpenMP-CUDA 4GPU

#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>
#include <map>
#include <numeric>

#include <RR/RRCU.cuh>

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <tuple>

#include "target_dir.h"
#include "common.cuh"
#include "nbody-kernel.cuh"
#include "p2mesh-kernel.cuh"
#include "wave-kernel.cuh"

using namespace RR::CUDA;

real Z_max, E0;
real3 Imp0, L0;

DataBlock d;
WaveEqData wave_eq_data;


// ====================================================

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
	std::ofstream stream{ DEBUG_PATH / "particles_in_cell.txt"};
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

	std::ofstream stream_partial_sum{ DEBUG_PATH / "particles_in_cell_partial_sum.txt" };
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

	std::ofstream stream{ DEBUG_PATH / "mass_part.txt" };
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

		std::ofstream stream{ DEBUG_PATH / "particle_counts.txt" };
		stream << "xi, counts" << std::endl;
		for (size_t i = 0; i < _nx; ++i) {
			stream << i << ", " << particle_counts[at(i, IY, IZ)] << std::endl;
		}
	}

	// phi cell
	{
		std::ofstream stream{ DEBUG_PATH / "phi_cell.txt" };
		stream << "x, phi" << std::endl;
		for (size_t i = 0; i < _nx; ++i) {
			double x = -_domain_l + i * _dx;
			stream << std::format("{}, {:.7f}", x, cell_phi_host[at(i, IY, IZ)]) << std::endl;
		}
	}

	// phi from particles
	{
		std::ofstream stream{ DEBUG_PATH / "phi_part.txt" };
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

	// 	std::ofstream stream{ DEBUG_PATH / "phi_part_cycle.txt" };
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
	int i;

	auto path = BIN_PATH / std::format("{}_{:5}.bin", name, it);
	auto path_str = path.string();

	FILE *outf = fopen(path_str.c_str(), "wb");
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

		auto path = BIN_PATH / "LIE_0.bin";
		auto path_str = path.string();
		outf = fopen(path_str.c_str(), "wb");
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

	{
		auto path = OUT_PATH / "Lz(t).dat";
		auto path_str = path.string();
		FILE* outf = (it == 0)
			? fopen(path_str.c_str(), "w")
			: fopen(path_str.c_str(), "a");
		fprintf(outf, "%d %f %1.15f %g %g\n",
			it_all,
			t,
			L.z,
			L.z / L0.z - 1.0,
			fabs(L.z / L0.z - 1.0)
		);
		fclose(outf);
	}

	{
		auto path = OUT_PATH / "LL(t).dat";
		auto path_str = path.string();
		FILE* outf = (it == 0)
			? fopen(path_str.c_str(), "w")
			: fopen(path_str.c_str(), "a");
		fprintf(outf, "%d %f %1.15f %g %g\n",
			it_all,
			t,
			LL,
			LL / LL0 - 1.0,
			fabs(LL / LL0 - 1.0)
		);
		fclose(outf);
	}

	{
		auto path = OUT_PATH / "Imp(t).dat";
		auto path_str = path.string();
		FILE* outf = (it == 0)
			? fopen(path_str.c_str(), "w")
			: fopen(path_str.c_str(), "a");
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
	}

	{
		auto path = OUT_PATH / "Imp(t).dat";
		auto path_str = path.string();
		FILE* outf = (it == 0)
			? fopen(path_str.c_str(), "w")
			: fopen(path_str.c_str(), "a");
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
	}

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

	fclose(outf);
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

void check_computing_units_info() {
	// get threads count
	int cpuThreads = omp_get_num_threads();
	printf("cpuThreads = %d\n", cpuThreads);

	// get device count
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	printf("deviceCount = %d\n", deviceCount);

	if (deviceCount == 0) {
		exit(-1);
	}

	// print devices properties
	cudaDeviceProp prop;
	for (int i = 0; i < deviceCount; i++){
		cudaGetDeviceProperties(&prop, i);
		printf("gpuID = %d, gpuName = %s\n", i, prop.name);
	}

	cudaSetDevice(0);
}

int main(int argc, char * argv[]) {
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << std::endl;
	check_computing_units_info();

	char temp[512];

	real r, vr, vfi, fi;
  	char str[24];

	//----GPU device------------------------------------------------
	int j, i;


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

	{
		auto start_galaxies_path = INI_PATH / "__start_galaxies.ini";
		auto start_galaxies_path_str = start_galaxies_path.string();
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
		) = read_galaxies(start_galaxies_path_str.c_str());
		printf("Ns / BLOCK_SIZE_b = %d\n", Ns / BLOCK_SIZE);
		printf("Ndm / BLOCK_SIZE_b = %d\n", Ns / BLOCK_SIZE);
		printf("NN / BLOCK_SIZE_b = %d\n", NN / BLOCK_SIZE);
	}

	{
		auto start_nbody_path = INI_PATH / "__start_nbody.ini";
		auto start_nbody_path_str = start_nbody_path.string();
		std::tie(
			i_cont,
			tmax,
			dtsave,
			tsave
		) = read_start_info(start_nbody_path_str.c_str());
	}

	{
		auto gr_par_path = INI_PATH / "__gr_par.ini";
		auto gr_par_path_str = gr_par_path.string();
		FILE* outf = fopen(gr_par_path_str.c_str(), "r");
		memset(temp, 0, sizeof(temp));
		fscanf(outf, "%lf  %[^\n]", &Mh, temp);
		fscanf(outf, "%lf  %[^\n]", &a, temp);
		fscanf(outf, "%lf  %[^\n]", &Rh, temp);
		fscanf(outf, "%lf  %[^\n]", &Mb, temp);
		fscanf(outf, "%lf  %[^\n]", &b, temp);
		fscanf(outf, "%lf  %[^\n]", &Rb, temp);
		fscanf(outf, "%lf  %[^\n]", &eps2, temp);
		fscanf(outf, "%lf  %[^\n]", &dtgrav, temp);
		fscanf(outf, "%lf  %[^\n]", &K_m, temp);    // K_m = Md/(10^{10}*Msun)
		fscanf(outf, "%lf  %[^\n]", &K_r, temp);    // K_r = L_r / 10 кпк
		fclose(outf);
		printf("Mh\t= %lf\n", Mh);
		printf("a\t= %lf\n", a);
		printf("Rh\t= %lf\n", Rh);
		printf("Mb\t= %lf\n", Mb);
		printf("b\t= %lf\n", b);
		printf("Rb\t= %lf\n", Rb);
		printf("eps\t= %lf\n", eps2);
		eps2 *= eps2;
		printf("eps2\t= %lf\n", eps2);
		printf("dtgrav\t= %lf\n", dtgrav);
	}

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
	CuCopyToSymbol(d, dd, ToDevice);

	wave_eq_data.domainMin = _domain_min;
	wave_eq_data.domainMax = _domain_max;
	wave_eq_data.cellSize = _cell_size;
	wave_eq_data.gridSize  = _grid_size;
	wave_eq_data.nx_wave = _nx;
	wave_eq_data.dx_wave = _dx;
	wave_eq_data.dt_wave = _dt_wave;
	wave_eq_data.c_wave = _c_wave;
	wave_eq_data.diss_base = _diss_base;
	wave_eq_data.diss_extra = _diss_extra;
	wave_eq_data.sim_l = _sim_l;
	wave_eq_data.bc_l = _bc_l;
	CuCopyToSymbol(wave_eq_data, wave_eq_data_, ToDevice);

	int it = 1, // save num
		itt = 1,
		ittg = 0,
		itg = 1;

	printf("***Input Data***\n");

  	if (i_cont > 0) {

		//---Stars---
		{
			auto path = BIN_PATH / std::format("S_%5d.bin", i_cont);
			auto path_str = path.string();
			FILE* outf = fopen(path_str.c_str(), "rb");
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
		}

      	//---DM---
		if(Ndm > 0) {
			auto path = BIN_PATH / std::format("DM_%5d.bin", i_cont);
			auto path_str = path.string();
			FILE* outf = fopen(path_str.c_str(), "rb");
        	fread(&Ndm, sizeof(int), 1, outf);
          	fread(&t, sizeof(double), 1, outf);
			int n0=0;
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

		{
			auto path = BIN_PATH / std::format("LIE_0.bin", i_cont);
			auto path_str = path.string();
			FILE* outf = fopen(path_str.c_str(), "rb");
			fread(&L0, sizeof(real3), 1, outf);
			fread(&Imp0, sizeof(real3), 1, outf);
			fread(&E0, sizeof(real), 1, outf);
			fclose(outf);
		}

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
				auto path = INI_PATH / std::format("start_S{}.txt", k);
				auto path_str = path.string();
				FILE* outf = fopen(path_str.c_str(), "r");
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
				auto path = INI_PATH / std::format("start_DM{}.txt", k);
				auto path_str = path.string();
				outf = fopen(path_str.c_str(), "r");
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

	CuDarray<CellInfo> cell_info_(_num_cells);
	CuDarray<int> cell_particles_count_(_num_cells);
	CuDarray<real> cell_mass_(_num_cells);
	CuDarray<real> cell_phi_prev_(_num_cells);
	CuDarray<real> cell_phi_curr_(_num_cells);
	CuDarray<real> cell_phi_next_(_num_cells);
	CuDarray<real3> cell_acceleration_(_num_cells);

	CuDarray<ParticleCellInfo> particles_cell_info_(_num_particles);

	CuDarray<int> particles_in_block_(_num_cell_blocks);

	//------Расчет грав. сил-----------------------------------------------------
	printf("calc grav forces\n");
	CuCall(PSI_kernel, NN / BLOCK_SIZE, BLOCK_SIZE) (
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

	return 0;
}