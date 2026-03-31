//Author: S.S. Khrapov
//Parallel Nbody Code OpenMP-CUDA 4GPU

#include <cstdlib>
#include <cassert>
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
#include "parse.cuh"
#include "nbody-kernel.cuh"
#include "p2mesh-kernel.cuh"
#include "wave-kernel.cuh"

using namespace RR::CUDA;

real Z_max, E0;
real3 Imp0, L0;

DataBlock d;
WaveEqData wave_eq_data;

void check_energy(
	const std::vector<double>& cell_mass,
	const std::vector<double>& cell_phi,
	int num_cells
)
{
	double ep = 0;
	for (size_t i = 0; i < num_cells; ++i) {
		ep += 0.5 * cell_mass[i] * cell_phi[i];
	}
	printf("ep: %g\n", ep);
}

void check_cell_particle_info_match(
	const std::vector<CellInfo>& cell_info,
	const std::vector<ParticleCellInfo>& particle_info,
	int num_particles,
	int num_cells
) {
	// check cell info & particle cell info
	std::vector<CellInfo> cell_info_manual(num_cells, CellInfo{ 0, 0 });
	for (size_t i = 0; i < num_particles; ++i) {
		size_t c = particle_info[i].cell_id;
		cell_info_manual[c].count++;
	}

	for (size_t i = 0; i < num_cells; ++i) {
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
	const std::vector<double>& particles_mass,
	double dx,
	double domain_l,
	int NX
)
{
	int IY = NX / 2;
	int IZ = NX / 2;

	std::ofstream stream{ DEBUG_PATH / "mass_cell.txt" };
	stream << "x, mass" << std::endl;
	for (size_t i = 0; i < NX; ++i) {
		double x = -domain_l + i * dx;
		stream << std::format("{}, {:.6f}", x, cell_mass[AT(i, IY, IZ)]) << std::endl;
	}

	std::cout
		<< "total mass in cells: "
		<< std::accumulate(cell_mass.begin(), cell_mass.end(), 0.)
		<< std::endl;


	std::cout << "total mass in particles: "
		<< std::accumulate(
			particles_mass.begin(),
			particles_mass.end(),
			0.
		)
		<< std::endl;
}

struct ConvertParticlesParams {
	int num_particles;
	int num_cells;
	int num_blocks;
	int over_particles;
	int over_cells;
	int over_blocks;
	int nx;
};

auto convert_particles_to_grid(
	const ConvertParticlesParams& params,
	const CuDarray<real3>& particles_pos_,
	const CuDarray<real>& particles_mass_,
	CuDarray<ParticleCellInfo>& particles_cell_info_,
	CuDarray<CellInfo>& cell_info_,
	CuDarray<int>& cell_particles_count_,
	CuDarray<int>& particles_in_block_,
	CuDarray<real>& cell_mass_
)
{
	CuDeviceSync();

	particles_cell_info_.set_zero();
	cell_info_.set_zero();
	cell_particles_count_.set_zero();
	particles_in_block_.set_zero();
	cell_mass_.set_zero();

    CuCall(assignParticlesToCells, params.over_particles, params.over_blocks) (
        particles_pos_,
        particles_cell_info_,
        cell_info_,
        cell_particles_count_,
        params.num_particles
    );
    CuCall(computePrefixSums, params.over_cells, params.over_blocks) (
        cell_info_,
        particles_in_block_,
        params.num_cells
    );
    CuCall(adjustGlobalPrefixSums, params.over_cells, params.over_blocks) (
        cell_info_,
        particles_in_block_,
        params.num_cells
    );
	CuCall(computeCellMassesUnsorted, params.over_particles, params.over_blocks) (
		particles_mass_,
		particles_cell_info_,
		cell_mass_,
		params.num_particles
	);
}

auto calc_acceleration_by_grid(
	const ConvertParticlesParams& params,
	const CuDarray<real3>& particles_pos_,
	const CuDarray<real>& particles_mass_,
	CuDarray<real3> particles_acc_,
	CuDarray<real> particles_phi_,
	CuDarray<real> cell_phi_prev_,
	CuDarray<real> cell_phi_curr_,
	int iterations
)
{
	static CuDarray<ParticleCellInfo> particles_cell_info_(params.num_particles);
	static CuDarray<CellInfo> cell_info_(params.num_cells);
	static CuDarray<int> cell_particles_count_(params.num_cells);
	static CuDarray<int> particles_in_block_(params.num_blocks);
	static CuDarray<real> cell_mass_(params.num_cells);
	static CuDarray<real3> cell_acc_(params.num_cells);
	static CuDarray<real> cell_phi_next_(params.num_cells);

	static std::once_flag once_flag;
	std::call_once(once_flag, []{
		std::cout << "GPU memory allocated total: " << CuDarrayBase::get_total_allocated_mb() << " mb" << std::endl;
	});

	CuTimer timer;
	timer.start();
	convert_particles_to_grid(
		params,
		particles_pos_,
		particles_mass_,
		particles_cell_info_,
		cell_info_,
		cell_particles_count_,
		particles_in_block_,
		cell_mass_
	);
	timer.stop();
	float time_convert_particles_to_grid = timer.elapsedMilliseconds();

	unsigned threads_count = 8;
	unsigned blocks_count = static_cast<unsigned>(params.nx + threads_count - 1) / threads_count;
	dim3 threads_per_block{ threads_count, threads_count, threads_count };
	dim3 blocks_per_grid{ blocks_count, blocks_count, blocks_count };
	timer.start();
	for (int iter = 0; iter < iterations; ++iter) {
		if (iter && iter % 500 == 0) {
			std::cout << "wave_diss_iteration iter: " << iter << std::endl;
		}
		CuCall(wave_diss_iteration, blocks_per_grid, threads_per_block) (
			cell_phi_next_,
			cell_phi_curr_,
			cell_phi_prev_,
			cell_mass_
		);
		swap(cell_phi_prev_, cell_phi_curr_);
		swap(cell_phi_curr_, cell_phi_next_);
	}
	timer.stop();
	float time_wave_diss = timer.elapsedMilliseconds();

	timer.start();
	CuCall(acc_field, params.over_cells, params.over_blocks) (
		cell_acc_,
		nullptr,
		cell_phi_curr_,
		wave_eq_data.nx_wave,
		wave_eq_data.dx_wave
	);
	CuCall(apply_acceleration_to_particles, params.over_particles, params.over_blocks) (
		particles_acc_,
		cell_acc_,
		particles_cell_info_,
		params.num_particles
	);
	CuCall(apply_cell_to_particles, params.over_particles, params.over_blocks) (
		particles_phi_,
		cell_phi_curr_,
		particles_cell_info_,
		params.num_particles
	);
	timer.stop();
	float time_acc = timer.elapsedMilliseconds();

	std::cout << "timer convert_particles_to_grid : "
		<< time_convert_particles_to_grid
		<< " ms"
		<< std::endl;
	std::cout << "timer wave_diss_iteration : "
		<< time_wave_diss
		<< " ms"
		<< std::endl;
	std::cout << "timer forces : "
		<< time_acc
		<< " ms"
		<< std::endl;

	return std::make_tuple(
		std::move(particles_acc_),
		std::move(particles_phi_),
		std::move(cell_phi_prev_),
		std::move(cell_phi_curr_)
	);
}

auto convert_particles(
	const CuDarray<real>& particles_phi_,
	const CuDarray<real3>& particles_acc_,
	const CuDarray<real3>& particles_pos_,
	const CuDarray<real>& particles_mass_,
	int over_cells,
	int over_blocks,
	int over_particles,
	int num_cells,
	int num_cell_blocks,
	int num_particles,
	real domain_l,
	int NX,
	real dx,
	int iterations,
	bool need_init_phi
)
{
	cudaDeviceSynchronize();

	static CuDarray<ParticleCellInfo> particles_cell_info_(num_particles);
	static CuDarray<CellInfo> cell_info_(num_cells);
	static CuDarray<int> cell_particles_count_(num_cells);
	static CuDarray<int> particles_in_block_(num_cell_blocks);
	static CuDarray<real> cell_mass_(num_cells);
	static CuDarray<real3> cell_acc_(num_cells);
	static CuDarray<real> cell_phi_next_(num_cells);
	static CuDarray<real> cell_phi_curr_(num_cells);
	static CuDarray<real> cell_phi_prev_(num_cells);

	unsigned threads_count = 8;
	unsigned blocks_count = static_cast<unsigned>(NX + threads_count - 1) / threads_count;
	dim3 threads_per_block{ threads_count, threads_count, threads_count };
	dim3 blocks_per_grid{ blocks_count, blocks_count, blocks_count };

	if (need_init_phi) {
		cell_mass_.set_zero();
		cell_phi_prev_.set_zero();
		cell_phi_curr_.set_zero();
		cell_phi_next_.set_zero();
	}

	cell_info_.set_zero();
	cell_particles_count_.set_zero();
	particles_in_block_.set_zero();

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
	CuCall(computeCellMassesUnsorted, over_particles, over_blocks) (
		particles_mass_,
		particles_cell_info_,
		cell_mass_,
		num_particles
	);

	for (int iter = 0; iter < iterations; ++iter) {
		if (iter && iter % 500 == 0) {
			std::cout << "wave_diss_iteration iter: " << iter << std::endl;
		}
		CuCall(wave_diss_iteration, blocks_per_grid, threads_per_block) (
			cell_phi_next_,
			cell_phi_curr_,
			cell_phi_prev_,
			cell_mass_
		);
		swap(cell_phi_prev_, cell_phi_curr_);
		swap(cell_phi_curr_, cell_phi_next_);
	}

	CuDarray<double> cell_acc_abs_(num_cells);
	CuCall(acc_field, over_cells, over_blocks) (
		cell_acc_,
		cell_acc_abs_,
		cell_phi_curr_,
		NX,
		dx
	);

	std::vector<double> cell_mass = cell_mass_.to_vector();
	std::vector<double> cell_phi = cell_phi_curr_.to_vector();
	std::vector<CellInfo> cell_info = cell_info_.to_vector();
	std::vector<ParticleCellInfo> particles_cell_info = particles_cell_info_.to_vector();

	check_cell_particle_info_match(
		cell_info,
		particles_cell_info,
		num_particles,
		num_cells
	);

	check_energy(
		cell_mass,
		cell_phi,
		num_cells
	);

	check_particles_distribution(
		cell_info
	);

	check_mass(
		cell_mass,
		particles_mass_.to_vector(),
		dx,
		domain_l,
		NX
	);

	const int IY = NX / 2 ;
	const int IZ = NX / 2 ;

	// particle_counts
	[
		NX, IY, IZ,
		num_cells, over_cells, over_blocks
	]
	{
		CuDarray<double> _particle_counts(num_cells);
		CuCall(countParticles, over_cells, over_blocks) (
			_particle_counts,
			cell_info_,
			num_cells
		);

		auto particle_counts = _particle_counts.to_vector();

		std::ofstream stream{ DEBUG_PATH / "particle_counts.txt" };
		stream << "xi, counts" << std::endl;
		for (size_t i = 0; i < NX; ++i) {
			stream << i << ", " << particle_counts[AT(i, IY, IZ)] << std::endl;
		}
	}();

	// phi cell
	[
		NX, domain_l, dx, IY, IZ,
		&cell_phi
	]
	{
		std::ofstream stream{ DEBUG_PATH / "phi_cell.txt" };
		stream << "x, phi" << std::endl;
		for (size_t i = 0; i < NX; ++i) {
			double x = -domain_l + i * dx;
			stream << std::format("{}, {:.7f}", x, cell_phi[AT(i, IY, IZ)]) << std::endl;
		}
	}();

	// phi from particles (NBODY)
	[
		NX, domain_l, dx, IY, IZ,
		num_cells, num_particles, over_particles, over_blocks,
		&particles_phi_
	]
	{
		std::ofstream stream{ DEBUG_PATH / "phi_part.txt" };
		stream << "x, phi" << std::endl;

		CuDarray<double> cell_phi_from_particles_(num_cells);
		CuCall(computeCellPhiUnsorted, over_particles, over_blocks) (
			particles_phi_,
			particles_cell_info_,
			cell_info_,
			cell_phi_from_particles_, // compute
			num_particles
		);
		auto phi = cell_phi_from_particles_.to_vector();

		for (size_t i = 0; i < NX; ++i) {
			double x = -domain_l + i * dx;
			stream << std::format("{}, {:.7f}", x, phi[AT(i, IY, IZ)]) << std::endl;
		}
	}();

	// acceleration abs in cell
	[
		domain_l, dx, NX, IY, IZ,
		&cell_acc_abs_
	]
	{
		std::ofstream stream{ DEBUG_PATH / "acc_cell.txt" };
		stream << "x, acc" << std::endl;

		std::vector<double> cell_acc_abs = cell_acc_abs_.to_vector();
		for (size_t i = 0; i < NX; ++i) {
			double x = -domain_l + i * dx;
			stream << std::format("{}, {:.7f}", x, cell_acc_abs[AT(i, IY, IZ)]) << std::endl;
		}
	}();

	// acceleration (particles_phi -> cell_phi -> cell_acceleration)
	[
		domain_l, dx, NX, IY, IZ, over_blocks,
		num_cells, over_cells,
		num_particles, over_particles,
		&particles_phi_
	]
	{
		std::ofstream stream{ DEBUG_PATH / "acc_part.txt" };
		stream << "x, acc" << std::endl;

		CuDarray<double> _cell_phi_from_particles(num_cells);
		CuCall(computeCellPhiUnsorted, over_particles, over_blocks) (
			particles_phi_,
			particles_cell_info_,
			cell_info_,
			_cell_phi_from_particles, // compute
			num_particles
		);
		CuDarray<real3> _cell_acc_from_particles(num_cells);
		CuDarray<double> _cell_acc_abs_from_particles(num_cells);
		CuCall(acc_field, over_cells, over_blocks) (
			_cell_acc_from_particles,
			_cell_acc_abs_from_particles,
			_cell_phi_from_particles,
			NX,
			dx
		);

		std::vector<double> cell_acceleration_abs = _cell_acc_abs_from_particles.to_vector();
		for (size_t i = 0; i < NX; ++i) {
			double x = -domain_l + i * dx;
			stream << std::format("{}, {:.7f}", x, cell_acceleration_abs[AT(i, IY, IZ)]) << std::endl;
		}
	}();

	// cell acc from nbody
	[
		NX, domain_l, dx, IY, IZ,
		num_cells, num_particles, over_particles, over_blocks,
		&particles_acc_,
		&particles_mass_
	] {
		std::ofstream stream{ DEBUG_PATH / "nbody_cell_acc.txt" };
		stream << "x, acc" << std::endl;

		CuDarray<double> _nbody_part_acc(num_particles);
		CuCall(acc_abs, over_particles, over_blocks) (
			_nbody_part_acc,
			particles_acc_,
			num_particles
		);

		CuDarray<real> _nbody_cell_acc(num_cells);
		CuCall(apply_particle_to_cell_weighted, over_particles, over_blocks) (
			_nbody_cell_acc,
			cell_mass_,
			particles_mass_,
			_nbody_part_acc,
			particles_cell_info_,
			num_particles
		);

		std::vector<double> nbody_cell_acc = _nbody_cell_acc.to_vector();
		for (size_t i = 0; i < NX; ++i) {
			double x = -domain_l + i * dx;
			stream << std::format("{}, {:.7f}", x, nbody_cell_acc[AT(i, IY, IZ)]) << std::endl;
		}
	}();

	// cell acc to particles and back to cell
	[
		num_particles, over_particles, over_blocks, NX, num_cells,
		domain_l, dx, IY, IZ,
		&particles_mass_
	]
	{
		std::ofstream stream{ DEBUG_PATH / "acc_cell_cycle.txt" };
		stream << "x, acc" << std::endl;

		CuDarray<real3> _part_acc(num_particles);
		CuDarray<real> _part_acc_abs(num_particles);
		CuCall(apply_acceleration_to_particles, over_particles, over_blocks) (
			_part_acc,
			cell_acc_,
			particles_cell_info_,
			num_particles
		);
		CuCall(acc_abs, over_particles, over_blocks) (
			_part_acc_abs,
			_part_acc,
			num_particles
		);
		CuDarray<real> _cell_acc_cycle(num_cells);
		CuCall(apply_particle_to_cell_weighted, over_particles, over_blocks) (
			_cell_acc_cycle,
			cell_mass_,
			particles_mass_,
			_part_acc_abs,
			particles_cell_info_,
			num_particles
		);

		std::vector<double> cell_acc_cycle = _cell_acc_cycle.to_vector();
		for (size_t i = 0; i < NX; ++i) {
			double x = -domain_l + i * dx;
			stream << std::format("{}, {:.7f}", x, cell_acc_cycle[AT(i, IY, IZ)]) << std::endl;
		}
	}();


	// particle acc from cell
	[
		num_particles, over_particles, over_blocks, NX,
		&particles_mass_
	] {
		std::ofstream stream{ DEBUG_PATH / "cell_part_acc.txt" };
		stream << "i, acc" << std::endl;

		CuDarray<real3> _cell_part_acc(num_particles);
		CuCall(apply_acceleration_to_particles, over_particles, over_blocks) (
			_cell_part_acc,
			cell_acc_,
			particles_cell_info_,
			num_particles
		);

		CuDarray<real> _cell_part_acc_abs(num_particles);
		CuCall(acc_abs, over_particles, over_blocks) (
			_cell_part_acc_abs,
			_cell_part_acc,
			num_particles
		);
		std::vector<double> cell_part_acc = _cell_part_acc_abs.to_vector();
		std::cout << "cell part acc sum: " << std::accumulate(cell_part_acc.begin(), cell_part_acc.end(), 0.) << std::endl;
		for (size_t i = 0; i < num_particles; ++i) {
			stream << std::format("{}, {:.7f}", i, cell_part_acc[i]) << std::endl;
		}
	}();

	// particle acc from nbody
	[
		num_particles, over_particles, over_blocks, NX,
		&particles_acc_
	] {
		std::ofstream stream{ DEBUG_PATH / "nbody_part_acc.txt" };
		stream << "i, acc" << std::endl;

		CuDarray<double> _nbody_part_acc(num_particles);
		CuCall(acc_abs, over_particles, over_blocks) (
			_nbody_part_acc,
			particles_acc_,
			num_particles
		);
		std::vector<double> nbody_part_acc = _nbody_part_acc.to_vector();
		std::cout << "nbody part acc sum: " << std::accumulate(nbody_part_acc.begin(), nbody_part_acc.end(), 0.) << std::endl;
		for (size_t i = 0; i < num_particles; ++i) {
			stream << std::format("{}, {:.7f}", i, nbody_part_acc[i]) << std::endl;
		}
	}();

	// mass from cell
	[
		num_particles, over_particles, over_blocks, NX
	] {
		std::ofstream stream{ DEBUG_PATH / "mpart_from_cell.txt" };
		stream << "i, acc" << std::endl;

		CuDarray<real> _mass_from_cell(num_particles);
		CuCall(apply_cell_to_particles_avg, over_particles, over_blocks) (
			_mass_from_cell,
			cell_mass_,
			particles_cell_info_,
			cell_info_,
			num_particles
		);

		std::vector<double> mass_from_cell = _mass_from_cell.to_vector();
		std::cout << "total mpart from cell: "
			<< std::accumulate(mass_from_cell.begin(), mass_from_cell.end(), 0.)
			<< std::endl;

		for (size_t i = 0; i < num_particles; ++i) {
			stream << std::format("{}, {:.7f}", i, mass_from_cell[i]) << std::endl;
		}
	}();

	// mass particles
	[
		num_particles, over_particles, over_blocks, NX,
		&particles_mass_
	] {
		std::ofstream stream{ DEBUG_PATH / "mpart.txt" };
		stream << "i, acc" << std::endl;

		std::vector<double> mass = particles_mass_.to_vector();

		std::cout << "total mpart: "
			<< std::accumulate(
				mass.begin(),
				mass.end(),
				0.
			)
			<< std::endl;

		for (size_t i = 0; i < num_particles; ++i) {
			stream << std::format("{}, {:.7f}", i, mass[i]) << std::endl;
		}
	}();

}

//---Host Function----
__host__ void print_cell_phi(
	size_t iter,
	const ConvertParticlesParams& params,
	const CuDarray<real>& cell_phi_
)
{
	static std::vector<double> cell_phi(params.num_cells, 0.);
	cell_phi_.to_vector(cell_phi);

	auto dir = OUT_PATH / "phi";
	std::filesystem::create_directory(dir);
	std::ofstream stream{ dir / std::format("phi_{:5}.csv", iter) };
	stream << "x, phi" << std::endl;
	int NX = params.nx;
	int IY = NX / 2;
	int IZ = NX / 2;
	for (size_t i = 0; i < NX; ++i) {
		double x = -wave_eq_data.sim_l + i * wave_eq_data.dx_wave;
		stream << std::format("{}, {:.7f}", x, cell_phi[AT(i, IY, IZ)]) << std::endl;
	}
}

__host__ void print_particles_bin(
	const char* name,
	int i0,
	int icount,
	const std::vector<real3>& pos,
	const std::vector<real3>& vel,
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

__host__ std::ios::openmode get_openmode(int it) {
	return it == 0
		? std::ios::out
		: std::ios::app;
}

__host__ void  result(
	const std::vector<real3>& pos,
	const std::vector<real3>& vel,
	const std::vector<real>& mass,
	int it,
	real t,
	const std::vector<real>& PSI,
	int it_all
)
{
	printf("print result begin\n");

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

	if (it > 1 && it % 10 == 0) {
		//---Star------
		print_particles_bin("S", 0, Ns, pos, vel, it, t);

		//---DM------
		if(NN > Ns) {
			int Ndm = NN - Ns;
			print_particles_bin("DM", Ns, Ndm, pos, vel, it, t);
		}
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

		L.x += (vz*y - vy*z)*mass[i];
		L.y += (vx*z - vz*x)*mass[i];
		L.z += (vy*x - vx*y)*mass[i];
		Imp.x += mass[i] * vx;
		Imp.y += mass[i] * vy;
		Imp.z += mass[i] * vz;
		Ek += mass[i] * (vx * vx + vy * vy + vz * vz);
		Ep += mass[i] * PSI[i];

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

		std::ofstream stream{ BIN_PATH / "LIE_0.bin", std::ios::out | std::ios::binary };
		stream.write(reinterpret_cast<const char*>(&L0), sizeof(real3));
		stream.write(reinterpret_cast<const char*>(&Imp0), sizeof(real3));
		stream.write(reinterpret_cast<const char*>(&E0), sizeof(real3));
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
		std::ofstream stream{ OUT_PATH / "Lz(t).csv", get_openmode(it) };
		if (0 == it) {
			stream << "it, t, Lz, Lz_rel_err, Lz_rel_err_magnitude" << std::endl;
		}

		stream
			<< it_all << ", "
			<< t << ", "
			<< std::format("{:.15f}", L.z) << ", "
			<< (L.z / L0.z - 1.0) << ", "
			<< std::fabs(L.z / L0.z - 1.0) << std::endl;
	}

	{
		std::ofstream stream{ OUT_PATH / "LL(t).csv", get_openmode(it) };
		if (0 == it) {
			stream << "it, t, LL, LL_rel_err, LL_rel_err_magnitude" << std::endl;
		}

		stream
			<< it_all << ", "
			<< t << ", "
			<< std::format("{:.15f}", LL) << ", "
			<< (LL / LL0 - 1.0) << ", "
			<< std::fabs(LL / LL0 - 1.0) << std::endl;
	}

	{
		std::ofstream stream{ OUT_PATH / "Imp(t).csv", get_openmode(it) };
		if (0 == it) {
			stream << "it, t, Imp, Imp_abs_err, Imp_rel_err, Imp_rel_err_magnitude" << std::endl;
		}

		real Impls = norm3(Imp);
		real Impls0 = norm3(Imp0);

		stream
			<< it_all << ", "
			<< t << ", "
			<< std::format("{:.15f}", Impls) << ", "
			<< (Impls - Impls0) << ", "
			<< (Impls / Impls0 - 1.0) << ", "
			<< std::fabs(Impls / Impls0 - 1.0) << std::endl;
	}

	{
		std::ofstream stream{ OUT_PATH / "E(t).csv", get_openmode(it) };
		if (0 == it) {
			stream << "it, t, E, E_rel_err, E_rel_err_magnitude, Ek, Ep" << std::endl;
		}

		stream
			<< it_all << ", "
			<< t << ", "
			<< std::format("{:.15f}", E) << ", "
			<< (E / E0 - 1.0) << ", "
			<< std::fabs(E / E0 - 1.0) << ", "
			<< (0.5 * Ek) << ", "
			<< (0.5 * Ep) << std::endl;
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

void run() {

	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << std::endl;
	check_computing_units_info();

	char temp[512];

  	char str[24];

	//----GPU device------------------------------------------------
	int i;

	float gpuTime = 0.0; //
	//-----Initial State---------------------------------------------------------
	real t = 0.0; // current time
	real tmax = 0.0; // max simulation time
	real tsave = 0.0;
	real dtsave = 0.0;
	real dtgrav=0.001, tgrav;
	real Mh, a, Rh, Mb, b, Rb, eps2;
	int i_cont = 0; // iteration to continue from
	real K_m, K_r;

	auto galaxy_properties = read_galaxy_properties(INI_PATH / "__start_galaxies.ini");
	if (galaxy_properties.empty()) {
		throw std::runtime_error{ "No galaxy properties read" };
	}

	int N_star = std::accumulate(
		galaxy_properties.begin(),
		galaxy_properties.end(),
		0,
		[](int N, const GalaxyProperties& galaxy) {
			return N + galaxy.n_star_particles;
		}
	);
	int N_dark = std::accumulate(
		galaxy_properties.begin(),
		galaxy_properties.end(),
		0,
		[](int N, const GalaxyProperties& galaxy) {
			return N + galaxy.n_dark_particles;
		}
	);
	int N_total = N_star + N_dark;

	std::cout << "galaxies count: " << galaxy_properties.size() << std::endl;

	std::cout << "total particles: " << N_total << std::endl;
	std::cout << "star particles: " << N_star << std::endl;
	std::cout << "dark particles: " << N_dark << std::endl;

	assert(false == galaxy_properties.empty());
	std::cout << "mass of star particles: " << galaxy_properties.front().get_mass_star_particle();
	std::cout << "mass of dark particles: " << galaxy_properties.front().get_mass_dark_particle();

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
	const real c_phi_h = con / a*(0.5*log(Rh2*Rh2 / a / a + 1.0) + atan(Rh2 / a)*a / Rh2) + Mh / Rh2;
	const real c_phi_b = Mb / Rb - const1*log(Rb / b + root1) / Rb;
	const real Mh_inf = Mh * (Rh2 / a - atan(Rh2 / a)) / (rcore1 - atan(rcore1));

	printf("*****Rh2 = %g \n", Rh2);
	printf("*****con = %g \n", con);
	printf("*****const1 = %g \n", const1);
	printf("*****c_phi_h = %g \n", c_phi_h);
	printf("*****c_phi_b = %g \n", c_phi_b);

	enum class NBodySolver {
		Nbody,
		Wave
	};
	NBodySolver solver = NBodySolver::Wave;

    constexpr int _nx = 200;
    constexpr double _dx = 0.2;
    constexpr double _domain_l = 0.5 * _nx * _dx;
	constexpr double _c_wave = 4574.337022617616;
	const double _dt_wave = 0.5 * _dx / (_c_wave * sqrt(3));
	const double _diss_base = 0.1;
	const double _diss_extra = 0.01;
	const double _sim_l = _domain_l;
	const double _bc_l = _domain_l / 10.;
	const int _iterations_to_setup = 20000;
	std::cout << "dt_wave: " << _dt_wave << std::endl;

    real3 _domain_min{
        -_domain_l,
        -_domain_l,
        -_domain_l
    };
    real3 _domain_max{
        _domain_l,
        _domain_l,
        _domain_l
    };
    real3 _cell_size{
        _dx,
        _dx,
        _dx
    };
    int3 _grid_size{
        _nx,
        _nx,
        _nx
    };
    int _num_particles = N_total;
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
	std::vector<real3> pos_host(N_total, make_real3(0., 0., 0.));
	std::vector<real3> vel_host(N_total, make_real3(0., 0., 0.));
	std::vector<real> mass_host(N_total, 0.);
	std::vector<real> eps2_host(N_total, 0.);
	std::vector<real> phi_host(N_total, 0.);
	std::vector<real3> acc_host(N_total, make_real3(0., 0., 0.));

	std::vector<real> cell_mass_host(_num_cells, 0.);
	std::vector<real> cell_phi_host(_num_cells,  0.);

	//-----------------------------------------------------

	ConvertParticlesParams convert_particles_params;
	convert_particles_params.num_blocks = _num_cell_blocks;
	convert_particles_params.num_cells = _num_cells;
	convert_particles_params.num_particles = _num_particles;
	convert_particles_params.nx = _nx;
	convert_particles_params.over_blocks = _over_blocks;
	convert_particles_params.over_cells = _over_cells;
	convert_particles_params.over_particles = _over_particles;

	d.Ns = N_star;
	d.NN = N_total;
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
	d.c_phi_h = c_phi_h;
	d.c_phi_b = c_phi_b;
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

	int M_glx = galaxy_properties.size();
	int k = 0;
	int Ns = 0;
	int Ndm = 0;
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
				for (i = n0; i < n0 + galaxy_properties[k].get_n_star_particles(); i++) {
					fread(&pos_host[i].x, sizeof(double), 1, outf);
					fread(&pos_host[i].y, sizeof(double), 1, outf);
					fread(&pos_host[i].z, sizeof(double), 1, outf);
					fread(&vel_host[i].x, sizeof(double), 1, outf);
					fread(&vel_host[i].y, sizeof(double), 1, outf);
					fread(&vel_host[i].z, sizeof(double), 1, outf);
					mass_host[i] = galaxy_properties[k].get_mass_star_particle();
					eps2_host[i] = galaxy_properties[k].eps_stars * galaxy_properties[k].eps_stars;
				}
				n0 += galaxy_properties[k].get_n_star_particles();
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
           		for (i = n0+Ns; i < n0+Ns+galaxy_properties[k].get_n_dark_particles(); i++) {
					fread(&pos_host[i].x, sizeof(double), 1, outf);
					fread(&pos_host[i].y, sizeof(double), 1, outf);
					fread(&pos_host[i].z, sizeof(double), 1, outf);
					fread(&vel_host[i].x, sizeof(double), 1, outf);
					fread(&vel_host[i].y, sizeof(double), 1, outf);
					fread(&vel_host[i].z, sizeof(double), 1, outf);
					mass_host[i] = galaxy_properties[k].get_mass_dark_particle();
					eps2_host[i] = galaxy_properties[k].eps_dark * galaxy_properties[k].eps_dark;
				}
				n0 += galaxy_properties[k].get_n_dark_particles();
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
      		if(galaxy_properties[k].get_n_star_particles() > 0) {
				auto path = INI_PATH / std::format("start_S{}.txt", k);
				auto path_str = path.string();
				FILE* outf = fopen(path_str.c_str(), "r");
          		if (NULL == outf) {
					printf("Error OF -- %s ",str);
					exit(0);
				}
          		else {
					int itmp = 0;
					real rtmp = 0.0;
            		fscanf(outf, "%d %lf", &itmp, &rtmp);
              		printf("N_s[%d] = %d, t = %f\n", k, itmp, rtmp);
              		for(i = n0; i < n0 + galaxy_properties[k].get_n_star_particles(); ++i) {
						fscanf(outf, "%lf %lf %lf %lf %lf %lf",
							&pos_host[i].x,
							&pos_host[i].y,
							&pos_host[i].z,
							&vel_host[i].x,
							&vel_host[i].y,
							&vel_host[i].z
						);
						mass_host[i] = galaxy_properties[k].get_mass_star_particle();

						pos_host[i].x = galaxy_properties[k].pos_center.x
						 	+ pos_host[i].x * cos(galaxy_properties[k].get_angle_rad())
							+ pos_host[i].z * sin(galaxy_properties[k].get_angle_rad());
						pos_host[i].y += galaxy_properties[k].pos_center.y;
						pos_host[i].z = galaxy_properties[k].pos_center.z
							+ pos_host[i].z * cos(galaxy_properties[k].get_angle_rad())
							- pos_host[i].x * sin(galaxy_properties[k].get_angle_rad());

						vel_host[i].x = galaxy_properties[k].vel_center.x
							+ vel_host[i].x * cos(galaxy_properties[k].get_angle_rad())
							+ vel_host[i].z * sin(galaxy_properties[k].get_angle_rad());
						vel_host[i].y += galaxy_properties[k].vel_center.y;
						vel_host[i].z = galaxy_properties[k].vel_center.z
							+ vel_host[i].z * cos(galaxy_properties[k].get_angle_rad())
							- vel_host[i].x * sin(galaxy_properties[k].get_angle_rad());

						eps2_host[i] = galaxy_properties[k].eps_stars * galaxy_properties[k].eps_stars;
              		}
          		}
				fclose(outf);
          		n0 += galaxy_properties[k].get_n_star_particles();
        	}
		}
		//----DM-------------------------------
		for (k = 0; k < M_glx; k++) {
			if (galaxy_properties[k].get_n_dark_particles() > 0) {
				FILE* outf = NULL;
				auto path = INI_PATH / std::format("start_DM{}.txt", k);
				auto path_str = path.string();
				outf = fopen(path_str.c_str(), "r");
          		if (NULL == outf) {
					printf("Error OF -- %s ",str);
					exit(0);
				}
				else {
					int itmp = 0;
					real rtmp = 0.0;
					fscanf(outf, "%d %lf", &itmp, &rtmp);
					printf("N_dm[%d] = %d, t = %f\n", k, itmp, rtmp);
					for(i = n0; i < n0 + galaxy_properties[k].get_n_dark_particles(); ++i) {
						fscanf(outf, "%lf %lf %lf %lf %lf %lf",
							&pos_host[i].x,
							&pos_host[i].y,
							&pos_host[i].z,
							&vel_host[i].x,
							&vel_host[i].y,
							&vel_host[i].z
						);

						mass_host[i] = galaxy_properties[k].get_mass_dark_particle();

						pos_host[i].x = galaxy_properties[k].pos_center.x
						 	+ pos_host[i].x * cos(galaxy_properties[k].get_angle_rad())
							+ pos_host[i].z * sin(galaxy_properties[k].get_angle_rad());
						pos_host[i].y += galaxy_properties[k].pos_center.y;
						pos_host[i].z = galaxy_properties[k].pos_center.z
							+ pos_host[i].z * cos(galaxy_properties[k].get_angle_rad())
							- pos_host[i].x * sin(galaxy_properties[k].get_angle_rad());

						vel_host[i].x = galaxy_properties[k].vel_center.x
							+ vel_host[i].x * cos(galaxy_properties[k].get_angle_rad())
							+ vel_host[i].z * sin(galaxy_properties[k].get_angle_rad());
						vel_host[i].y += galaxy_properties[k].vel_center.y;
						vel_host[i].z = galaxy_properties[k].vel_center.z
							+ vel_host[i].z * cos(galaxy_properties[k].get_angle_rad())
							- vel_host[i].x * sin(galaxy_properties[k].get_angle_rad());

						eps2_host[i] = galaxy_properties[k].eps_dark * galaxy_properties[k].eps_dark;
					}
				}
				fclose(outf);
				n0 += galaxy_properties[k].get_n_dark_particles();
			}
		}
		tsave = dtsave;
		tgrav = dtgrav;
		t = 0.0;
    }

	printf("***Start GPU***\n");

	//-----Allocate Massiv GPU--------------------------------------

	printf("allocate memory GPU %d\n", i);
	CuDarray<real3> pos_(pos_host);
	CuDarray<real3> vel_(vel_host);
	CuDarray<real> mass_(mass_host);
	CuDarray<real> eps2_(eps2_host);
	CuDarray<real> phi_(phi_host);
	CuDarray<real3> post_(_num_particles);
	CuDarray<real3> velt_(_num_particles);
	CuDarray<real3> acc_(_num_particles);
	CuDarray<real3> acct_(_num_particles);

	std::cout << "GPU memory allocated for nbody particles: " << CuDarrayBase::get_total_allocated_mb() << " mb" << std::endl;

	CuDarray<real> cell_phi_prev_(_num_cells);
	CuDarray<real> cell_phi_curr_(_num_cells);


	//------Расчет грав. сил-----------------------------------------------------
	printf("calc grav forces\n");

	CuTimer timer_setup;
	timer_setup.start();
	if (solver == NBodySolver::Wave) {
		std::tie(
			acc_,
			phi_,
			cell_phi_prev_,
			cell_phi_curr_
		) = calc_acceleration_by_grid(
			convert_particles_params,
			pos_,
			mass_,
			std::move(acc_),
			std::move(phi_),
			std::move(cell_phi_prev_),
			std::move(cell_phi_curr_),
			_iterations_to_setup
		);
		print_cell_phi(0, convert_particles_params, cell_phi_curr_);
	}
	else {
		CuCall(PHI_kernel, N_total / BLOCK_SIZE, BLOCK_SIZE) (
			phi_,
			pos_,
			pos_,
			mass_,
			eps2_
		);
		CuCall(ACCEL, N_total / BLOCK_SIZE, BLOCK_SIZE) (
			acc_,
			pos_,
			pos_,
			mass_,
			eps2_
		);

	}
	timer_setup.stop();
	std::cout << "time for setup: " << timer_setup.elapsedSeconds() << " seconds " << std::endl;

	// convert_particles(
	// 	phi_,
	// 	acc_,
	// 	pos_,
	// 	mass_,
	// 	_over_cells,
	// 	_over_blocks,
	// 	_over_particles,
	// 	_num_cells,
	// 	_num_cell_blocks,
	// 	_num_particles,
	// 	_domain_l,
	// 	_nx,
	// 	_dx,
	// 	_iterations,
	// 	true
	// );


	printf("calc grav forces: copy to host\n");
	phi_.to_vector(phi_host);

	if(it == 1) {
		printf("***Start result t=0***\n");
		result(pos_host, vel_host, mass_host, 0, 0.0, phi_host, 0);
	}

	CuTimer timer_between_saves;
	timer_between_saves.start();

	do {
		// --- Nbody и самогравитация
		CuTimer timer_base_cycle;
		timer_base_cycle.start();

		//------Nbody predictor (tn+dtgrav)----------------------------------------------------------------------------
		printf("Nbody predictor %d-%d/%d (%lf - %lf / %lf)\n", it, itt, is_grav, t, tgrav, tsave);
		CuCall(kernelNbody_integTime, _num_particles / BLOCK_SIZE, BLOCK_SIZE) (
			acc_,
			post_,
			velt_,
			pos_,
			vel_,
			dtgrav,
			0,
			tgrav - dtgrav,
			acc_
		);

		//------Расчет самогравитации Nbody частиц-----------------------------------------------------
		printf("Nbody grav %d-%d/%d (%lf - %lf / %lf)\n", it, itt, is_grav, t, tgrav, tsave);

		if (solver == NBodySolver::Wave) {
			std::cout << "Wave eq iterations: " << (int)(dtgrav / _dt_wave) << std::endl;
			acct_.set_zero();
			std::tie(
				acct_,
				phi_,
				cell_phi_prev_,
				cell_phi_curr_
			) = calc_acceleration_by_grid(
				convert_particles_params,
				post_,
				mass_,
				std::move(acct_),
				std::move(phi_),
				std::move(cell_phi_prev_),
				std::move(cell_phi_curr_),
				dtgrav / _dt_wave
			);
		}
		else {
			CuCall(ACCEL, N_total / BLOCK_SIZE, BLOCK_SIZE) (
				acct_,
				post_,
				post_,
				mass_,
				eps2_
			);
		}


		//------Nbody corrector (tn+dtgrav)----------------------------------------------------------------------------
		printf("Nbody corrector %d-%d/%d (%lf - %lf / %lf)\n", it, itt, is_grav, t, tgrav, tsave);
		CuCall(kernelNbody_integTime, _num_particles / BLOCK_SIZE, BLOCK_SIZE) (
			acct_,
			pos_,
			vel_,
			post_,
			velt_,
			dtgrav,
			1,
			tgrav,
			acc_
		);


		timer_base_cycle.stop();
		float time_base_cycle_sec = timer_base_cycle.elapsedSeconds();
		//----------------------------------------------------------------------------------------------------------------------

		printf("predictor-grav-corrector time: %lf s\n", time_base_cycle_sec);

		ittg++;
		itg++;
		t = tgrav;
		tgrav = itg * dtgrav;
		it_grav++;

		if (it_grav >= is_grav) {
			tsave = tgrav;
			//Copy data GPU to CPU

			if (solver == NBodySolver::Nbody) {
				printf("Copy data GPU to CPU: redo PSI_kernel\n");
				phi_.set_zero();
				CuCall(PHI_kernel, _num_particles / BLOCK_SIZE, BLOCK_SIZE) (
					phi_,
					pos_,
					pos_,
					mass_,
					eps2_
				);

				static CuDarray<ParticleCellInfo> particles_cell_info_(_num_particles);
				static CuDarray<CellInfo> cell_info_(_num_cells);
				static CuDarray<int> cell_particles_count_(_num_cells);
				static CuDarray<int> particles_in_block_(_num_cell_blocks);
				static CuDarray<real> cell_mass_(_num_cells);

				convert_particles_to_grid(
					convert_particles_params,
					pos_,
					mass_,
					particles_cell_info_,
					cell_info_,
					cell_particles_count_,
					particles_in_block_,
					cell_mass_
				);
				CuCall(computeCellPhiUnsorted, _over_particles, _over_blocks) (
					phi_,
					particles_cell_info_,
					cell_info_,
					cell_phi_curr_,
					_num_particles
				);
			}


			printf("Copy data GPU to CPU\n");
			pos_.to_vector(pos_host);
			vel_.to_vector(vel_host);
			mass_.to_vector(mass_host);
			phi_.to_vector(phi_host);
			acc_.to_vector(acc_host);
			timer_between_saves.stop();
			float time_between_saves_sec = timer_between_saves.elapsedSeconds();
			timer_between_saves.start();

			printf("time between saves: %g s", time_between_saves_sec);

			gpuTime += time_between_saves_sec;
			printf("--------------------------------------------------------------------------------\n");

			printf("<Time_frame> = %.3f s, frame = %d, iter = %d, iter_g = %d\n",
				gpuTime,
				it,
				itt,
				ittg
			);
			printf("Time = (%g,  %g) ---  dt = (%g,  %g)\n",
				t,
				tgrav - dtgrav,
				dtsave / itt,
				dtgrav
			);

			print_cell_phi(itt * it, convert_particles_params, cell_phi_curr_);

			// print sample particle:
			{
				std::fstream::openmode openmode = it == 1
					? std::fstream::out
					: std::fstream::app;

				std::ofstream stream_sample{ OUT_PATH / "sample.csv", openmode };

				if (1 == it) {
					stream_sample << "it, t, i, r, fi, z, acc, phi, ax, ay, az" << std::endl;
				}

				int i = 1;
				real fi = atan2(pos_host[i].y, pos_host[i].x);
				real r = norm2(make_real2(pos_host[i].x, pos_host[i].y));
				real vr = (vel_host[i].x*pos_host[i].x + vel_host[i].y*pos_host[i].y) / r;
				real vfi = (vel_host[i].y*pos_host[i].x - vel_host[i].x*pos_host[i].y) / r;
				real acc_abs = norm3(acc_host[i]);

				stream_sample
					<< it * itt << ", "
					<< t << ", "
					<< i << ", "
					<< r << ", "
					<< fi << ", "
					<< pos_host[i].z << ", "
					<< acc_abs << ", "
					<< phi_host[i] << ", "
					<< acc_host[i].x << ", "
					<< acc_host[i].y << ", "
					<< acc_host[i].z << std::endl;
			}


			result(pos_host, vel_host, mass_host, it, t, phi_host, it*itt);

			tsave += dtsave; it++;
			itt = 0; ittg = 0;
			it_grav = 0;
		}
		itt++;
	} while (t < tmax);

}

int main(int argc, char * argv[]) {

	try {
		run();
	}
	catch (const std::exception& ex) {
		std::cerr << "Error: " << ex.what() << std::endl;
	}
	return 0;
}