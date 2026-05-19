#include <catch2/catch_test_macros.hpp>
#include <fstream>
#include <filesystem>
#include <grav_params.h>

namespace fs = std::filesystem;

// Простой RAII-хелпер для временных файлов
struct TempFile {
    fs::path path;
    TempFile(const std::string& name, const std::string& content)
        : path(fs::temp_directory_path() / name)
    {
        std::ofstream ofs(path);
        ofs << content;
    }
    ~TempFile() {
        if (fs::exists(path)) fs::remove(path);
    }
    std::string str() const { return path.string(); }
};

TEST_CASE("GravParams loads valid config correctly", "[grav_params]") {
    const TempFile f("valid_grav.ini",
        "1.0 --- Mh\n0.5 --- a\n2.0 --- Rh\n0.5 --- Mb\n0.1 --- b\n1.0 --- Rb\n"
        "0.01 --- eps\n0.001 --- dtgrav\n3.0 --- K_m\n0.8 --- K_r\n)");

    const auto cfg = khrr_grav_params::load_grav_config(f.str());

    CHECK(cfg.Mh == 1.0);
    CHECK(cfg.a == 0.5);
    CHECK(cfg.Rh2 == 6.0);
    CHECK(cfg.con > 0.0);
}

TEST_CASE("GravParams throws on missing lines", "[grav_params]") {
    const TempFile f("missing_lines.ini", "1.0 --- Mh\n0.5 --- a\n");
    REQUIRE_THROWS_AS(khrr_grav_params::load_grav_config(f.str()), std::runtime_error);
}

TEST_CASE("GravParams throws on invalid type", "[grav_params]") {
    const TempFile f("invalid_type.ini",
        "1.0\nabc\n2.0\n0.0\n0.1\n1.0\n0.01\n0.001\n3.0\n0.8\n");
    REQUIRE_THROWS_AS(khrr_grav_params::load_grav_config(f.str()), std::runtime_error);
}

TEST_CASE("GravParams throws on extra values in line", "[grav_params]") {
    const TempFile f("extra_values.ini",
        "1.0 2.0\n0.5\n2.0\n0.0\n0.1\n1.0\n0.01\n0.001\n3.0\n0.8\n");
    REQUIRE_THROWS_AS(khrr_grav_params::load_grav_config(f.str()), std::runtime_error);
}

TEST_CASE("GravParams throws on zero scale radius", "[grav_params]") {
    const TempFile f("zero_a.ini",
        "1.0\n0.0\n2.0\n0.0\n0.1\n1.0\n0.01\n0.001\n3.0\n0.8\n");
    REQUIRE_THROWS_AS(khrr_grav_params::load_grav_config(f.str()), std::runtime_error);
}

TEST_CASE("GravParams handles zero mass gracefully", "[grav_params]") {
    const TempFile f("zero_mass.ini",
        "0.0\n0.5\n2.0\n0.0\n0.1\n1.0\n0.01\n0.001\n3.0\n0.8\n");

    const auto cfg = khrr_grav_params::load_grav_config(f.str());

    CHECK(cfg.Mh == 0.0);
    CHECK(cfg.con == 0.0);
    CHECK(cfg.Mb == 0.0);
    CHECK(cfg.const1 == 0.0);
}