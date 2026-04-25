#include <catch2/catch_all.hpp>
#include <galaxy_params_reader.h>
#include <fstream>
#include <filesystem>
#include <range/v3/range/conversion.hpp>
#include <range/v3/view/transform.hpp>
#include <range/v3/view/iota.hpp>
#include <range/v3/algorithm/fold_left.hpp>
#include <range/v3/utility/functional.hpp>

namespace fs = std::filesystem;

// RAII-хелпер для автоматической очистки временной директории
struct TempTestDir {
    fs::path path;
    TempTestDir() :
        path{ fs::temp_directory_path() / "khrr_galaxy_params_test.ini" }
    {
        fs::create_directories(path);
    }
    ~TempTestDir() { fs::remove_all(path); }
};

std::string write_ini(const fs::path& dir, const std::string& content) {
    std::ofstream(dir / "galaxies.ini") << content;
    return (dir / "galaxies.ini").string();
}

TEST_CASE("Valid Single Galaxy", "[galaxy_params]") {
    TempTestDir dir;
    auto path = write_ini(dir.path,
        "1\n0\n1024,2048\n1.0,6.699356\n0.004,0.004\n0.0\n0.0,0.0,0.0\n10.5,0.0,-5.2\n"
    );
    auto p = khrr_galaxy_params::read_galaxy_params(path);

    REQUIRE(p.M_glx == 1);
    REQUIRE(p.galaxies.size() == 1);
    CHECK(p.galaxies[0].k_glx == 0);
    CHECK(p.galaxies[0].Vx_glx == 10.5);
}

TEST_CASE("Valid Multiple Galaxies with Range-v3", "[galaxy_params]") {
    TempTestDir dir;
    std::string header = "3\n";

    // Ленивая генерация блоков через range-v3
    auto galaxies_lines = ranges::views::iota(0, 3)
        | ranges::views::transform([](int i) -> std::string {
            std::ostringstream oss;
            oss << i << "\n"
                << (1024 + i * 1024) << "," << (2048 + i * 2048) << "\n"
                << "1.0,2.0\n0.1,0.1\n" << (i * 15.0) << "\n"
                << "0.0,0.0,0.0\n0.0,0.0,0.0\n";
            return oss.str();
        });
    std::string content = ranges::fold_left(
        galaxies_lines,
        std::string{},
        ranges::plus{}
    );

    auto path = write_ini(dir.path, header + content);
    auto p = khrr_galaxy_params::read_galaxy_params(path);

    REQUIRE(p.galaxies.size() == 3);
    CHECK(p.galaxies[2].alpha_glx == 30.0);
}

TEST_CASE("File Not Found", "[galaxy_params][error]") {
    REQUIRE_THROWS_AS(khrr_galaxy_params::read_galaxy_params("/nonexistent/path.ini"), std::runtime_error);
}

TEST_CASE("Zero Galaxies", "[galaxy_params][error]") {
    TempTestDir dir;
    REQUIRE_THROWS_AS(khrr_galaxy_params::read_galaxy_params(write_ini(dir.path, "0\n")), std::runtime_error);
}

TEST_CASE("Negative Mass", "[galaxy_params][error]") {
    TempTestDir dir;
    REQUIRE_THROWS_AS(khrr_galaxy_params::read_galaxy_params(write_ini(dir.path,
        "1\n0\n1024,1024\n-5.0,2.0\n0.1,0.1\n0.0\n0.0,0.0,0.0\n0.0,0.0,0.0\n")), std::runtime_error);
}

TEST_CASE("Particle Count Not Aligned to 1024", "[galaxy_params][error]") {
    TempTestDir dir;
    REQUIRE_THROWS_AS(khrr_galaxy_params::read_galaxy_params(write_ini(dir.path,
        "1\n0\n1000,2048\n1.0,2.0\n0.1,0.1\n0.0\n0.0,0.0,0.0\n0.0,0.0,0.0\n")), std::runtime_error);
}

TEST_CASE("Index Mismatch", "[galaxy_params][error]") {
    TempTestDir dir;
    REQUIRE_THROWS_AS(khrr_galaxy_params::read_galaxy_params(write_ini(dir.path,
        "1\n42\n1024,1024\n1.0,2.0\n0.1,0.1\n0.0\n0.0,0.0,0.0\n0.0,0.0,0.0\n")), std::runtime_error);
}

TEST_CASE("NaN in Parameters", "[galaxy_params][error]") {
    TempTestDir dir;
    REQUIRE_THROWS_AS(khrr_galaxy_params::read_galaxy_params(write_ini(dir.path,
        "1\n0\n1024,1024\n1.0,2.0\n0.1,0.1\nNaN\n0.0,0.0,0.0\n0.0,0.0,0.0\n")), std::runtime_error);
}