cmake_minimum_required(VERSION 3.20)
project(khrr_particles LANGUAGES CXX)

add_library(khrr_particles STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/particle_io.cpp
)
target_include_directories(khrr_particles
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(khrr_particles
    PUBLIC
        khrr_galaxy_params
        khrr_common
    PRIVATE
        spdlog::spdlog
        fmt::fmt
)
set_target_properties(khrr_particles PROPERTIES
    CXX_STANDARD 17
)

# Тесты
add_executable(test_particles
    tests/test_particles.cpp
)
target_link_libraries(test_particles
    PUBLIC
        fmt::fmt
        khrr_particles
        Catch2::Catch2WithMain
)

# Тестовая программа, которая читает файл и выводит информацию о нём
add_executable(sample_particles
    tests/sample_reader.cpp
)
target_link_libraries(sample_particles
    PUBLIC
        fmt::fmt
        khrr_particles
)