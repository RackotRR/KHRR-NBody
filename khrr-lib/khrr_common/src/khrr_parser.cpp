#include "khrr_parser.h"
#include <fmt/format.h>
#include <fstream>

namespace khrr_parser {


std::string clean_line(std::string line) {
    if (auto pos = line.find("----"); pos != std::string::npos) line.erase(pos);
    if (auto pos = line.find('#'); pos != std::string::npos) line.erase(pos);
    line.erase(0, line.find_first_not_of(" \t\r\n"));
    line.erase(line.find_last_not_of(" \t\r\n") + 1);
    return line;
}

std::vector<std::string> read_file(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error(fmt::format("Cannot open configuration file: {}", filepath));
    }

    std::vector<std::string> lines;
    std::string raw;
    while (std::getline(file, raw)) {
        std::string cleaned = clean_line(raw);
        if (!cleaned.empty()) lines.push_back(std::move(cleaned));
    }

    if (lines.empty()) throw std::runtime_error("File is empty or contains only comments.");

    return lines;
}

}