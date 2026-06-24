cmake_minimum_required(VERSION 3.20)
project(khrr_solver CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(khrr_solver
    src/solver.cpp
    src/solver-fs.cpp
    src/solver-log.cpp
)

target_include_directories(khrr_solver
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(khrr_solver
    PRIVATE
        khrr_nbody
        khrr_grav_params
        khrr_sim_params
        khrr_galaxy_params
        spdlog::spdlog
        fmt::fmt
)

set_target_properties(khrr_solver PROPERTIES
    CXX_STANDARD 17
    CUDA_ARCHITECTURES          OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION  ON
)