cmake_minimum_required(VERSION 3.15)

#nbody
add_executable(nbody
    "src/nbody.cu"
)

target_include_directories(nbody PRIVATE "include")
target_link_options(nbody PRIVATE /NODEFAULTLIB:LIBCMT)
target_link_libraries(nbody PRIVATE RR)
set_target_properties(nbody PROPERTIES CUDA_ARCHITECTURES ${RR_CUDA_ARCH})