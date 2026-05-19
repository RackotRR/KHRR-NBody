cmake_minimum_required(VERSION 3.20)
project(khrr_sim_params LANGUAGES CXX)

# Основная библиотека
add_library(khrr_sim_params STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sim_params.cpp
)
target_include_directories(khrr_sim_params
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(khrr_sim_params
    PRIVATE
    khrr_common
)
set_target_properties(khrr_sim_params PROPERTIES
    CXX_STANDARD 17
)
