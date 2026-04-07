cmake_minimum_required(VERSION 3.15)
project(KHRR-Nbody LANGUAGES C CXX CUDA)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CXX_STANDARD 20)
set(RR_CUDA_ARCH "89")

# Add MSVC-specific flags for standard-conforming preprocessor (required for CCCL)
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif()

add_subdirectory(extern/RR)
add_subdirectory(nbody)
add_subdirectory(p2mesh)

configure_file("${CMAKE_SOURCE_DIR}/target_dir.h.in" "${CMAKE_SOURCE_DIR}/nbody/include/target_dir.h")