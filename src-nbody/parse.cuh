#pragma once
#include <cstdlib>
#include <cassert>
#include <cmath>

#include <vector>
#include <filesystem>
#include <string>
#include <functional>

#include "common.cuh"

// enum class GalaxyComponent {
// 	Star,
// 	Dark
// };


struct GalaxyProperties {
	int n_star_particles = 0;
	int n_dark_particles = 0;
	double mass_stars = 0.;
	double mass_dark = 0.;
	real3 pos_center = make_real3(0., 0., 0.);
	real3 vel_center = make_real3(0., 0., 0.);
	double angle = 0.;
	double eps_stars = 0.;
	double eps_dark = 0.;

	int get_n_particles() const {
		return get_n_star_particles() + get_n_dark_particles();
	}
	int get_n_star_particles() const {
		return mass_stars > 0.
			? n_star_particles
			: 0;
	}
	int get_n_dark_particles() const {
		return mass_dark > 0.
			? n_dark_particles
			: 0;
	}
	double get_mass_star_component() const {
		return n_star_particles > 0
			? mass_stars
			: 0.;
	}
	double get_mass_dark_component() const {
		return n_dark_particles > 0
			? mass_dark
			: 0.;
	}
	double get_mass_star_particle() const {
		return mass_stars / get_n_star_particles();
	}
	double get_mass_dark_particle() const {
		return mass_dark / get_n_dark_particles();
	}
	double get_angle_rad() const {
		return DEG2RAD * angle;
	}
};

// using read_double3_func = std::function<double3(FILE*)>;
// using read_double_func = std::function<double(FILE*)>;
// using read_int_func = std::function<int(FILE*)>;

// struct ReadDumpParams {
// 	read_double3_func read_double3;
// 	read_double_func read_double;
// 	read_int_func read_int;
// 	std::string openmode;
// };

// int read_int_binary(FILE* file) {
// 	int result = 0;
// 	assert(file);
// 	fread(&result, sizeof(int), 1, file);
// 	return result;
// };
// double read_double_binary(FILE* file) {
// 	double result = 0.;
// 	assert(file);
// 	fread(&result, sizeof(double), 1, file);
// 	return result;
// };
// double3 read_double3_binary(FILE* file) {
// 	double3 result = make_double3(0., 0., 0.);
// 	assert(file);
// 	fread(&result.x, sizeof(double), 1, file);
// 	fread(&result.y, sizeof(double), 1, file);
// 	fread(&result.z, sizeof(double), 1, file);
// 	return result;
// };

// using calc_num_func = std::function<int(int, const GalaxyProperties&)>;
// int calc_n_stars(int N, const GalaxyProperties& galaxy) {
// 	return N + galaxy.n_star_particles;
// }
// int calc_n_dark(int N, const GalaxyProperties& galaxy) {
// 	return N + galaxy.n_dark_particles;
// }

// ReadDumpParams read_binary_params{
// 	read_double3_binary,
// 	read_double_binary,
// 	read_int_binary,
// 	"rb"
// };

// __host__ void read_particles_from_galaxies_bin(
// 	const std::filesystem::path& path,
// 	const ReadDumpParams& read_params,
// 	const std::vector<GalaxyProperties>& galaxy_properties,
// 	GalaxyComponent component,
// 	std::vector<real3>& pos,
// 	std::vector<real3>& vel,
// 	std::vector<real>& mass,
// 	std::vector<real>& eps2
// )
// {
// 	auto path_str = path.string();
// 	FILE* outf = fopen(path_str.c_str(), read_params.openmode.c_str());

// 	int N_in_file_header = read_params.read_int(outf);
// 	double t = read_params.read_double(outf);

// 	using calc_num_func = std::function<int(int, const GalaxyProperties&)>;
// 	const calc_num_func calc_num_component =
// 		(component == GalaxyComponent::Star)
// 		? calc_n_stars
// 		: calc_n_dark;

// 	using get_component_mass_func = std::function<double(const GalaxyProperties&)>;
// 	const get_component_mass_func get_particle_mass_component =
// 		(component == GalaxyComponent::Star)
// 		? GalaxyProperties::get_mass_star_particle
// 		: GalaxyProperties::get_mass_dark_particle;

// 	using get_component_eps_func = std::function<double(const GalaxyProperties&)>;
// 	const get_component_eps_func get_particle_eps_component =
// 		(component == GalaxyComponent::Star)
// 		? std::mem_fn(&GalaxyProperties::eps_stars)
// 		: std::mem_fn(&GalaxyProperties::eps_dark);


// 	int N_particles_in_component = std::accumulate(
// 		galaxy_properties.begin(),
// 		galaxy_properties.end(),
// 		0,
// 		calc_num_component
// 	);

// 	if (N_in_file_header < N_particles_in_component) {
// 		throw std::runtime_error{
// 			std::format(
// 				"Expected {} particles, but file header says there're only {} particles",
// 				N_particles_in_component,
// 				N_in_file_header
// 			)
// 		};
// 	}

// 	int n0 = 0;
// 	for (const GalaxyProperties& galaxy : galaxy_properties) {
// 		const double m_part = get_particle_mass_component(galaxy);
// 		const double e_part = get_particle_eps_component(galaxy);

// 		for (int i = 0; i < N_particles_in_component; ++i) {
// 			pos.push_back(read_params.read_double3(outf));
// 			vel.push_back(read_params.read_double3(outf));
// 			mass.push_back(m_part);
// 			eps2.push_back(e_part * e_part);
// 		}
// 	}
// 	fclose(outf);
// }

// __host__ auto read_particles_from_galaxies_txt(
// 	const std::vector<GalaxyProperties>& galaxy_properties,
// 	std::vector<real3>& pos,
// 	std::vector<real3>& vel,
// 	std::vector<real>& mass,
// 	std::vector<real>& eps2
// )
// {
// 	std::cout << "read particles from galaxies" << std::endl;

// 	pos.clear();
// 	vel.clear();
// 	mass.clear();
// 	eps2.clear();

// 	for(size_t galaxy_id = 0; galaxy_id < galaxy_properties.size(); ++galaxy_id) {
// 		const auto& galaxy = galaxy_properties[galaxy_id];

// 		if (galaxy.get_n_star_particles() < 0) {
// 			std::cout << "skip empty galaxy " << galaxy_id << std::endl;
// 			continue;
// 		}

// 		auto path = INI_PATH / std::format("start_S{}.txt", galaxy_id);
// 		auto path_str = path.string();

// 		FILE* outf = fopen(path_str.c_str(), "r");
// 		if (NULL == outf) {
// 			throw std::runtime_error{ "EOF on start stars read" };
// 		}

// 		int N_in_file = 0;
// 		int t_in_file = 0.;
// 		fscanf(outf, "%d %lf", &N_in_file, &t_in_file);
// 		std::cout << std::format("particles in galaxy: {}, t: {}", N_in_file, t_in_file) << std::endl;

// 		int N_in_galaxy = galaxy.get_n_star_particles();
// 		if (N_in_file != N_in_galaxy) {
// 			std::cout << "WARNING: particles count mismatch!" << std::endl;
// 			std::cout << "\t galaxy properties: " << N_in_galaxy << std::endl;
// 			std::cout << "\t galaxy file: " << N_in_file << std::endl;
// 		}

// 		const int n_star_galaxy = galaxy.get_n_star_particles();
// 		const double m_galaxy = galaxy.get_mass_star_particle();
// 		const double e_galaxy = galaxy.eps_stars;
// 		const double3& p_galaxy = galaxy.pos_center;
// 		const double3& v_galaxy = galaxy.vel_center;
// 		const double sin_galaxy = std::sin(galaxy.get_angle_rad());
// 		const double cos_galaxy = std::cos(galaxy.get_angle_rad());

// 		for (int i = 0; i < n_star_galaxy; ++i) {
// 			auto& p = pos.emplace_back();
// 			auto& v = vel.emplace_back();
// 			fscanf(outf, "%lf %lf %lf %lf %lf %lf",
// 				&p.x,
// 				&p.y,
// 				&p.z,
// 				&v.x,
// 				&v.y,
// 				&v.z
// 			);
// 			mass.push_back(galaxy_properties[galaxy_id].get_mass_star_particle());
// 			eps2.push_back(e_galaxy * e_galaxy);

// 			p.x = p_galaxy.x
// 				+ p.x * cos_galaxy
// 				+ p.z * sin_galaxy;
// 			p.y += p_galaxy.y;
// 			p.z = p_galaxy.z
// 				+ p.z * cos_galaxy
// 				- p.x * sin_galaxy;

// 			v.x = v_galaxy.x
// 				+ v.x * cos_galaxy
// 				+ v.z * sin_galaxy;
// 			v.y += v_galaxy.y;
// 			v.z = v_galaxy.z
// 				+ v.z * cos_galaxy
// 				- v.x * sin_galaxy;
// 		}

// 		fclose(outf);
// 	}

// 	//----DM-------------------------------
// 	for (k = 0; k < M_glx; k++) {
// 		if (N_dm[k] > 0) {
// 			FILE* outf = NULL;
// 			auto path = INI_PATH / std::format("start_DM{}.txt", k);
// 			auto path_str = path.string();
// 			outf = fopen(path_str.c_str(), "r");
// 			if (NULL == outf) {
// 				printf("Error OF -- %s ",str);
// 				exit(0);
// 			}
// 			else {
// 				int itmp = 0;
// 				real rtmp = 0.0;
// 				fscanf(outf, "%d %lf", &itmp, &rtmp);
// 				printf("N_dm[%d] = %d, t = %f\n", k, itmp, rtmp);
// 				for(i = n0; i < n0 + N_dm[k]; ++i) {
// 					fscanf(outf, "%lf %lf %lf %lf %lf %lf",
// 						&pos_host[i].x,
// 						&pos_host[i].y,
// 						&pos_host[i].z,
// 						&vel_host[i].x,
// 						&vel_host[i].y,
// 						&vel_host[i].z
// 					);

// 					mass_host[i] = mp_dm[k];

// 					pos_host[i].x = X_glx[k]
// 						+ pos_host[i].x * cos(alpha_glx[k])
// 						+ pos_host[i].z * sin(alpha_glx[k]);
// 					pos_host[i].y += Y_glx[k];
// 					pos_host[i].z = Z_glx[k]
// 						+ pos_host[i].z * cos(alpha_glx[k])
// 						- pos_host[i].x * sin(alpha_glx[k]);

// 					vel_host[i].x = Vx_glx[k]
// 						+ vel_host[i].x * cos(alpha_glx[k])
// 						+ vel_host[i].z * sin(alpha_glx[k]);
// 					vel_host[i].y += Vy_glx[k];
// 					vel_host[i].z = Vz_glx[k]
// 						+ vel_host[i].z * cos(alpha_glx[k])
// 						- vel_host[i].x * sin(alpha_glx[k]);

// 					eps2_host[i] = eps_dm[k]*eps_dm[k];
// 				}
// 			}
// 			fclose(outf);
// 			n0 += N_dm[k];
// 		}
// 	}
// 	tsave = dtsave;
// 	tgrav = dtgrav;
// 	t = 0.0;
// }

__host__ std::vector<GalaxyProperties> read_galaxy_properties(const std::filesystem::path& path) {
	char temp[256];

	std::string filename = path.string();
	FILE* outf = fopen(filename.c_str(), "r");

	int galaxies_count = 0;
	int galaxy_id = 0;
	fscanf(outf, "%d %[^\n]", &galaxies_count, temp);

	std::vector<GalaxyProperties> galaxy_properties(galaxies_count);
	for (auto& galaxy : galaxy_properties) {
		fscanf(outf, "%d %[^\n]", &galaxy_id, temp); // номер галактики не нужен
		fscanf(outf, "%d,%d %[^\n]",
			&galaxy.n_star_particles,
			&galaxy.n_dark_particles,
			temp);
		fscanf(outf, "%lf,%lf %[^\n]",
			&galaxy.mass_stars,
			&galaxy.mass_dark,
			temp);
		fscanf(outf, "%lf,%lf %[^\n]",
			&galaxy.eps_stars,
			&galaxy.eps_dark,
			temp);
		fscanf(outf, "%lf %[^\n]",
			&galaxy.angle,
			temp);
		fscanf(outf, "%lf,%lf,%lf %[^\n]",
			&galaxy.pos_center.x,
			&galaxy.pos_center.y,
			&galaxy.pos_center.z,
			temp);
		fscanf(outf, "%lf,%lf,%lf %[^\n]",
			&galaxy.vel_center.x,
			&galaxy.vel_center.y,
			&galaxy.vel_center.z,
			temp);
	}
	fclose(outf);

	return galaxy_properties;
}
