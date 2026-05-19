#pragma once
#include <string_view>
#include <cstddef>

namespace khrr_sim_params {

/// @brief Конфигурация запуска и сохранения N-body симуляции.
/// Содержит контрольные параметры цикла интегрирования и I/O.
struct SimConfig {
    std::size_t i_cont; ///< Номер итерации/сохранения для продолжения расчёта (0 = новый запуск)
    double tmax;        ///< Максимальное время моделирования (безразмерные единицы)
    double dt_save;     ///< Временной шаг между сохранениями состояния системы в файлы
};

/// @brief Загружает, парсит и валидирует параметры симуляции из "__start_nbody.ini".
/// @param filepath Путь к файлу конфигурации.
/// @return Структура SimConfig (возвращается по значению, RVO/move).
/// @throws std::runtime_error при ошибках I/O, парсинга или физической/логической некорректности.
SimConfig load_sim_config(std::string_view filepath);

} // namespace khrr_sim_params