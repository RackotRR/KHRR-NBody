cmake_minimum_required(VERSION 3.18)
project(khrr_nbody LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_library(khrr_nbody STATIC
    src/nbody_context.cu
)

target_include_directories(khrr_nbody
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Adjust target name to match your project structure.
target_link_libraries(khrr_nbody
    PUBLIC
        khrr_common
        khrr_particles
    PRIVATE
        CUDA::cudart
        spdlog::spdlog
)

target_compile_definitions(khrr_nbody PRIVATE FMT_UNICODE=0)

set_target_properties(khrr_nbody PROPERTIES
    CUDA_SEPARABLE_COMPILATION  ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_ARCHITECTURES          native
    CXX_STANDARD                17
    CUDA_STANDARD               17
)

option(KHRR_NBODY_BUILD_TESTS "Build khrr_nbody unit tests" ON)
if(KHRR_NBODY_BUILD_TESTS)
    enable_testing()
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tests)
endif()