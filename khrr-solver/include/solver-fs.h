#pragma once
#include <filesystem>

namespace khrr_solver::fs {

    constexpr const char* APP_NAME = "KHRR-Solver";

    std::filesystem::path get_app_directory();

    std::filesystem::path get_log_directory();

    void create_directory_if_not_exists(const std::filesystem::path& path);
}