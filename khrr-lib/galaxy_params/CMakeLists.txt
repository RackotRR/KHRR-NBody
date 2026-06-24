cmake_minimum_required(VERSION 3.20)
project(khrr_galaxy_params LANGUAGES CXX)

# Основная библиотека
add_library(khrr_galaxy_params STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/galaxy_params_reader.cpp
)
target_include_directories(khrr_galaxy_params
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(khrr_galaxy_params
    PRIVATE
        khrr_common
        range-v3::range-v3
        fmt::fmt
)
set_target_properties(khrr_galaxy_params PROPERTIES
    CXX_STANDARD 17
)

# Тесты
add_executable(test_galaxy_params
    tests/test_galaxy_params_reader.cpp
)
target_link_libraries(test_galaxy_params
    PRIVATE
        khrr_galaxy_params
        range-v3::range-v3
        Catch2::Catch2WithMain
)
