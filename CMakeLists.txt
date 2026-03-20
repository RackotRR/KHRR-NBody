cmake_minimum_required(VERSION 3.15)
project(nbody LANGUAGES CXX CUDA)

# p2mesh
add_executable(p2mesh
    "src-p2mesh/p2mesh.cu"
)
target_include_directories(p2mesh PRIVATE
    "include"
    "extern/RR/include"
)
target_link_options(p2mesh PRIVATE /NODEFAULTLIB:LIBCMT)
set_target_properties(p2mesh PROPERTIES
    CUDA_ARCHITECTURES "89"
    CXX_STANDARD "20"
)

# nbody
add_executable(nbody
    "src-nbody/nbody.cu"
)
target_include_directories(nbody PRIVATE
    "include"
    "extern/RR/include"
)
target_link_options(nbody PRIVATE /NODEFAULTLIB:LIBCMT)
set_target_properties(nbody PROPERTIES
    CUDA_ARCHITECTURES "89"
    CXX_STANDARD "20"
)

configure_file("${CMAKE_SOURCE_DIR}/target_dir.h.in" "${CMAKE_SOURCE_DIR}/src-nbody/target_dir.h")