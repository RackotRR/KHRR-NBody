#pragma once
#include <string_view>

namespace khrr_grav_params {

/// @brief Конфигурация гравитационных параметров галактики (Halo + Bulge).
/// Содержит сырые параметры из входного файла и заранее вычисленные константы
/// для эффективного использования в CUDA-ядрах (минимизация трансцендентных операций на устройстве).
struct GravConfig {
    // --- Raw parameters ---
    double Mh;       ///< Масса гало (Dark Matter Halo)
    double a;        ///< Масштабный радиус гало
    double Rh;       ///< Радиус обрезки гало
    double Mb;       ///< Масса балджа (Bulge)
    double b;        ///< Масштабный радиус балджа
    double Rb;       ///< Радиус обрезки балджа
    double eps;      ///< Длина гравитационного сглаживания (softening)
    double dtgrav;   ///< Шаг интегрирования гравитации
    double K_m;      ///< Коэффициент масштабирования массы для сетки (PM)
    double K_r;      ///< Коэффициент масштабирования радиуса для сетки (PM)

    // --- Precomputed constants for device code ---
    double Rh2;      ///< 3.0 * Rh
    double rcore1;   ///< Rh / a
    double rbcore1;  ///< 1.0 / b
    double rbcore2;  ///< rbcore1^2
    double root1;    ///< sqrt(1 + (Rb*b)^-2)
    double con;      ///< Нормировка силы гало
    double const1;   ///< Нормировка силы балджа
    double c_phi_h;  ///< Потенциальная константа гало
    double c_phi_b;  ///< Потенциальная константа балджа
    double Mh_inf;   ///< Эффективная масса гало за пределами Rh2
};

/// @brief Загружает, парсит и валидирует параметры из файла конфигурации.
/// @param filepath Путь к файлу (обычно "__gr_par.ini").
/// @return Структура GravConfig с вычисленными константами (возвращается по значению, RVO/move).
/// @throws std::runtime_error при ошибках I/O, парсинга или физической некорректности.
GravConfig load_grav_config(std::string_view filepath);

} // namespace khrr_grav_params