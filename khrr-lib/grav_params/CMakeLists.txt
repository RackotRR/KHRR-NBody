cmake_minimum_required(VERSION 3.20)
project(khrr_grav_params LANGUAGES CXX)

# Основная библиотека
add_library(khrr_grav_params STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/grav_params.cpp
)
target_include_directories(khrr_grav_params
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(khrr_grav_params
    PRIVATE
        khrr_common
)
set_target_properties(khrr_grav_params PROPERTIES
    CXX_STANDARD 17
)

# Тесты
add_executable(test_grav_params
    tests/test_grav_params.cpp
)
target_link_libraries(test_grav_params
    PRIVATE
        khrr_grav_params
        Catch2::Catch2WithMain
)
