#include <khrr_parser.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include "grav_params.h"

namespace khrr_grav_params {

// Вспомогательная функция для извлечения одного double из строки
inline double extract_double(std::string_view line) {
    return std::get<0>(khrr_parser::parse_tuple_values<double>(line));
}

GravConfig load_grav_config(std::string_view filepath) {
    const auto lines = khrr_parser::read_file(std::string(filepath));

    if (lines.size() < 10) {
        throw std::runtime_error("Configuration file must contain at least 10 parameter lines");
    }

    const double Mh     = extract_double(lines[0]);
    const double a      = extract_double(lines[1]);
    const double Rh     = extract_double(lines[2]);
    const double Mb     = extract_double(lines[3]);
    const double b      = extract_double(lines[4]);
    const double Rb     = extract_double(lines[5]);
    const double eps    = extract_double(lines[6]);
    const double dtgrav = extract_double(lines[7]);
    const double K_m    = extract_double(lines[8]);
    const double K_r    = extract_double(lines[9]);

    // Физическая валидация: предотвращает деление на ноль и NaN в dev_fex
    if (a <= 0.0) throw std::runtime_error("'a' must be strictly positive");
    if (b <= 0.0) throw std::runtime_error("'b' must be strictly positive");
    if (Rh <= 0.0) throw std::runtime_error("'Rh' must be strictly positive");
    if (Rb <= 0.0) throw std::runtime_error("'Rb' must be strictly positive");
    if (dtgrav <= 0.0) throw std::runtime_error("'dtgrav' must be strictly positive");
    if (eps < 0.0) throw std::runtime_error("'eps' must be non-negative");

    // Предвычисление констант (функциональный стиль, только const)
    const double Rh2 = 3.0 * Rh;
    const double rcore1 = Rh / a;
    const double rbcore1 = 1.0 / b;
    const double rbcore2 = rbcore1 * rbcore1;
    const double root1 = std::sqrt(1.0 + (Rb * Rb) * rbcore2);

    const double halo_denom = rcore1 - std::atan(rcore1);
    const double con = (Mh != 0.0) ? Mh / halo_denom : 0.0;

    const double bulge_log_term = std::log(Rb * rbcore1 + root1);
    const double bulge_denom = b * bulge_log_term - Rb / root1;
    const double const1 = (Mb != 0.0 && std::abs(bulge_denom) > 1e-15) ? Mb / bulge_denom : 0.0;

    const double c_phi_h = con / a * (0.5 * std::log(Rh2 * Rh2 / a / a + 1.0) + std::atan(Rh2 / a) * a / Rh2) + Mh / Rh2;
    const double c_phi_b = (Mb != 0.0) ? (Mb / Rb - const1 * bulge_log_term / Rb) : 0.0;
    const double Mh_inf = Mh * (Rh2 / a - std::atan(Rh2 / a)) / halo_denom;

    // Возврат по значению (компилятор применит RVO/move)
    return GravConfig{
        Mh, a, Rh, Mb, b, Rb, eps, dtgrav, K_m, K_r,
        Rh2, rcore1, rbcore1, rbcore2, root1, con, const1, c_phi_h, c_phi_b, Mh_inf
    };
}

} // namespace khrr_grav_params