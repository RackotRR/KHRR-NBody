cmake_minimum_required(VERSION 3.15)
project(nbody LANGUAGES CXX CUDA)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CXX_STANDARD 20)
set(RR_CUDA_ARCH "89")

# Add MSVC-specific flags for standard-conforming preprocessor (required for CCCL)
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>)
endif()

add_subdirectory(extern/RR)

# p2mesh
add_executable(p2mesh
    "src-p2mesh/p2mesh.cu"
)
target_include_directories(p2mesh PRIVATE "include")
target_link_options(p2mesh PRIVATE /NODEFAULTLIB:LIBCMT)
target_link_libraries(p2mesh PRIVATE RR)
set_target_properties(p2mesh PROPERTIES CUDA_ARCHITECTURES ${RR_CUDA_ARCH})

#nbody
add_executable(nbody
    "src-nbody/nbody.cu"
)
target_include_directories(nbody PRIVATE "include")
target_link_options(nbody PRIVATE /NODEFAULTLIB:LIBCMT)
target_link_libraries(nbody PRIVATE RR)
set_target_properties(nbody PROPERTIES CUDA_ARCHITECTURES ${RR_CUDA_ARCH})

configure_file("${CMAKE_SOURCE_DIR}/target_dir.h.in" "${CMAKE_SOURCE_DIR}/src-nbody/target_dir.h")