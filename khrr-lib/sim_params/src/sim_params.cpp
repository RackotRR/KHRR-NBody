#include <khrr_parser.h>
#include <stdexcept>
#include <tuple>
#include <spdlog/spdlog.h>
#include "sim_params.h"

namespace khrr_sim_params {

/// Вспомогательная функция: извлекает строго одно значение заданного типа из строки.
template<typename T>
inline T parse_single(std::string_view line) {
    return std::get<0>(khrr_parser::parse_tuple_values<T>(line));
}

SimConfig load_sim_config(std::string_view filepath) {
    spdlog::info("load sim config");
    const auto lines = khrr_parser::read_file(std::string(filepath));

    if (lines.size() < 3) {
        throw std::runtime_error("Simulation config must contain at least 3 lines: i_cont, tmax, dt_save");
    }

    const std::size_t i_cont = parse_single<std::size_t>(lines[0]);
    const double tmax = parse_single<double>(lines[1]);
    const double dt_save = parse_single<double>(lines[2]);

    // Логическая валидация
    if (tmax <= 0.0) {
        throw std::runtime_error("'tmax' must be strictly positive");
    }
    if (dt_save <= 0.0) {
        throw std::runtime_error("'dt_save' must be strictly positive");
    }
    if (dt_save > tmax) {
        throw std::runtime_error("'dt_save' cannot exceed 'tmax'");
    }

    spdlog::info("\t -- i continue = {}", i_cont);
    spdlog::info("\t -- tmax = {}", tmax);
    spdlog::info("\t -- dt save = {}", dt_save);
    return SimConfig{i_cont, tmax, dt_save};
}

} // namespace khrr_sim_params