// khrr-lib/particles/tests/test_particles.cpp
#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "particle_io.h"
#include <khrr_types.h>

#include <filesystem>
#include <stdexcept>
#include <cstdio>

namespace fs = std::filesystem;

// Helper to create temp directory for tests
std::string get_temp_test_dir() {
    std::string path = "/tmp/khrr_particle_test_" + std::to_string(std::hash<std::string>{}(std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())));
    return path;
}

TEST_CASE("double3 structure defaults", "[double3]") {
    khrr_common::real3 v{1.0, 2.0, 3.0};
    REQUIRE(v.x == 1.0);
    REQUIRE(v.y == 2.0);
    REQUIRE(v.z == 3.0);
}

TEST_CASE("Binary save and load - normal", "[binary]") {
    std::string dir = get_temp_test_dir() + "/bin1";
    fs::create_directories(dir);

    // Data: 2 Stars, 1 DM
    std::vector<khrr_common::real3> s_pos = {{0,1,2}, {3,4,5}};
    std::vector<khrr_common::real3> s_vel = {{0.1,1.1,2.1}, {3.1,4.1,5.1}};
    std::vector<khrr_common::real3> dm_pos = {{6,7,8}};
    std::vector<khrr_common::real3> dm_vel = {{6.6,7.7,8.8}};

    // Save
    REQUIRE_NOTHROW(khrr_particles::save_binary(dir, 10, 123.45, s_pos, s_vel, dm_pos, dm_vel));

    // Verify files exist
    REQUIRE(fs::exists(dir + "/S_   10.bin"));
    REQUIRE(fs::exists(dir + "/DM_   10.bin"));

    // Load with expected counts matching file
    auto data = khrr_particles::load_binary(dir, 10, 2, 1);

    CHECK(data.n_stars == 2);
    CHECK(data.n_dm == 1);
    REQUIRE(data.positions.size() == 3);
    REQUIRE(data.velocities.size() == 3);

    // Check Star 0
    CHECK(data.positions[0].x == 0.0);
    CHECK(data.positions[0].y == 1.0);
    CHECK(data.positions[0].z == 2.0);
    CHECK(data.velocities[0].x == Catch::Approx(0.1));

    // Check Star 1
    CHECK(data.positions[1].x == 3.0);

    // Check DM 0 (index 2)
    CHECK(data.positions[2].x == 6.0);
    CHECK(data.velocities[2].y == Catch::Approx(7.7));
}

TEST_CASE("Binary load - header mismatch and missing file", "[binary]") {
    std::string dir = get_temp_test_dir() + "/bin2";
    fs::create_directories(dir);

    std::vector<khrr_common::real3> s_pos = {{10,20,30}};
    std::vector<khrr_common::real3> s_vel = {{0.1,0.2,0.3}};

    // Save 1 Star
    REQUIRE_NOTHROW(khrr_particles::save_binary(dir, 5, 0.0, s_pos, s_vel, {}, {}));

    // Remove DM file if it exists (save_binary with empty might not create it, but good to check)
    if (fs::exists(dir + "/DM_    5.bin")) fs::remove(dir + "/DM_    5.bin");

    // Load with expected 2 Stars (header says 1). Should warn and load min(1,2)=1.
    auto data = khrr_particles::load_binary(dir, 5, 2, 0);

    CHECK(data.n_stars == 1);
    CHECK(data.n_dm == 0);
}

TEST_CASE("Text load initial - multiple files", "[text]") {
    std::string dir = get_temp_test_dir() + "/text1";
    fs::create_directories(dir);

    // Create start_S1.txt for Galaxy 1
    std::string f1 = dir + "/start_S1.txt";
    FILE* fp = std::fopen(f1.c_str(), "w");
    REQUIRE(fp != nullptr);
    std::fprintf(fp, "1 0.0\n1.1 2.2 3.3 4.4 5.5 6.6\n");
    std::fclose(fp);

    // Create start_S2.txt for Galaxy 2
    std::string f2 = dir + "/start_S2.txt";
    fp = std::fopen(f2.c_str(), "w");
    REQUIRE(fp != nullptr);
    std::fprintf(fp, "1 0.0\n10.0 20.0 30.0 40.0 50.0 60.0\n");
    std::fclose(fp);

    // Create start_DM1.txt
    std::string f3 = dir + "/start_DM1.txt";
    fp = std::fopen(f3.c_str(), "w");
    REQUIRE(fp != nullptr);
    std::fprintf(fp, "1 0.0\n100.0 200.0 300.0 400.0 500.0 600.0\n");
    std::fclose(fp);

    // No start_DM2.txt (should be handled gracefully)

    // Setup params
    khrr_galaxy_params::GalaxiesParams params;
    params.M_glx = 2;
    params.galaxies.resize(2);

    params.galaxies[0].k_glx = 1;
    params.galaxies[0].N_s = 1;
    params.galaxies[0].N_dm = 1;

    params.galaxies[1].k_glx = 2;
    params.galaxies[1].N_s = 1;
    params.galaxies[1].N_dm = 0;

    auto result = khrr_particles::load_text_initial(dir, params);

    // Total 2 stars, 1 DM
    CHECK(result.n_stars == 2);
    CHECK(result.n_dm == 1);

    REQUIRE(result.positions.size() == 3);
    REQUIRE(result.velocities.size() == 3);

    // Order: Stars (Galaxy 1, Galaxy 2), then DM (Galaxy 1)

    // Star from Galaxy 1
    CHECK(result.positions[0].x == 1.1);
    CHECK(result.velocities[0].z == 6.6);

    // Star from Galaxy 2
    CHECK(result.positions[1].x == 10.0);
    CHECK(result.velocities[1].y == 50.0);

    // DM from Galaxy 1
    CHECK(result.positions[2].x == 100.0);
    CHECK(result.velocities[2].x == 400.0);
}

TEST_CASE("Text load initial - missing galaxy files", "[text]") {
    std::string dir = get_temp_test_dir() + "/text2";
    fs::create_directories(dir);
    // No files created

    khrr_galaxy_params::GalaxiesParams params;
    params.M_glx = 1;
    params.galaxies.resize(1);
    params.galaxies[0].k_glx = 5;
    params.galaxies[0].N_s = 10;
    params.galaxies[0].N_dm = 0;

    auto res = khrr_particles::load_text_initial(dir, params);
    CHECK(res.n_stars == 0);
    CHECK(res.n_dm == 0);
}