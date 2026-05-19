#pragma once
#include <tuple>
#include <string_view>
#include <sstream>
#include <stdexcept>
#include <vector>


namespace khrr_parser {

/// Removes inline comments ("----" or "#") and trims whitespace.
std::string clean_line(std::string line);

/// @brief Парсит строку, разделённую запятыми или пробелами, в std::tuple.
/// @tparam Ts Ожидаемые типы значений в строке.
/// @param line Очищенная строка конфигурации.
/// @return Кортеж с распарсенными значениями.
/// @throws std::runtime_error при несоответствии количества/типов или наличии лишних данных.
template<typename... Ts>
std::tuple<Ts...> parse_tuple_values(std::string_view line) {
    // Заменяем запятые на пробелы для унификации разделителей.
    // Для типичных строк конфигурации (менее 16-23 символов) используется SSO (нулевая аллокация).
    std::string buffer(line);
    for (char& c : buffer) {
        if (c == ',') c = ' ';
    }

    std::istringstream iss(buffer);
    std::tuple<Ts...> result;

    // Пытаемся прочитать все значения
    std::apply(
        [&](auto&... elements) {
            (iss >> ... >> elements);
        },
        result
    );

    if (!iss) {
        throw std::runtime_error("Failed to parse expected values from line");
    }

    // Проверяем, не осталось ли лишних значений
    std::string leftover;
    if (iss >> leftover) {
        throw std::runtime_error("Extra unexpected values found in line");
    }

    return result;
}

/// Извлекает строго одно значение заданного типа из строки.
template<typename T>
inline T parse_single(std::string_view line) {
    return std::get<0>(khrr_parser::parse_tuple_values<T>(line));
}

/// Читает файл в вектор строк по указанному пути
std::vector<std::string> read_file(std::string_view filepath);

} // namespace khrr_parser