#include "solver-fs.h"
#include "solver-log.h"

namespace khrr_solver::fs {

    std::filesystem::path get_app_directory() {
        std::filesystem::path dir;

#ifdef _WIN32
        const char* userprofile = std::getenv("USERPROFILE");
        if (userprofile) {
            dir = std::filesystem::path{ userprofile } / APP_NAME;
        }
        else {
            dir = std::filesystem::current_path() / APP_NAME;
        }
#else
        const char* home = std::getenv("HOME");
        if (home) {
            dir = std::filesystem::path{ home } / APP_NAME;
        }
        else {
            dir = fs::current_path() / APP_NAME;
        }
#endif

        return dir;

    }


    std::filesystem::path get_log_directory() {
        return get_app_directory() / "logs";
    }

    std::filesystem::path get_projects_directory() {
        return get_app_directory() / "projects";
    }

    std::filesystem::path get_project_ini_directory(
        const std::filesystem::path& project_dir
    )
    {
        return project_dir / "ini";
    }

    std::filesystem::path get_grav_params_path(
        const std::filesystem::path& project_dir
    )
    {
        return get_project_ini_directory(project_dir) / "__gr_par.ini";
    }

    std::filesystem::path get_sim_params_path(
        const std::filesystem::path& project_dir
    )
    {
        return get_project_ini_directory(project_dir) / "__start_nbody.ini";
    }

    std::filesystem::path get_galaxies_params_path(
        const std::filesystem::path& project_dir
    )
    {
        return get_project_ini_directory(project_dir) / "__start_galaxies.ini";
    }


    void create_directory_if_not_exists(const std::filesystem::path& path) {
        try {
            if (false == std::filesystem::exists(path)) {
                std::filesystem::create_directories(path);
                spdlog::info("Directory created: {}", path.string());
            }
        }
        catch (const std::filesystem::filesystem_error& e) {
            spdlog::error("Failed to create directory {}: {}", path.string(), e.what());
        }
    }
}