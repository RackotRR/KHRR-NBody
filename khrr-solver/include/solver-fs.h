#pragma once
#include <filesystem>

namespace khrr_solver::fs {

    constexpr const char* APP_NAME = "KHRR-Solver";

    std::filesystem::path get_app_directory();

    std::filesystem::path get_log_directory();

    std::filesystem::path get_projects_directory();

    std::filesystem::path get_grav_params_path(
        const std::filesystem::path& project_dir
    );

    std::filesystem::path get_sim_params_path(
        const std::filesystem::path& project_dir
    );

    std::filesystem::path get_galaxies_params_path(
        const std::filesystem::path& project_dir
    );

    void create_directory_if_not_exists(const std::filesystem::path& path);
}