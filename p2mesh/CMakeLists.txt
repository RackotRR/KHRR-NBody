cmake_minimum_required(VERSION 3.15)

# p2mesh
add_executable(p2mesh
    "src/p2mesh.cu"
)

target_include_directories(p2mesh PRIVATE "include")
target_link_options(p2mesh PRIVATE /NODEFAULTLIB:LIBCMT)
target_link_libraries(p2mesh PRIVATE RR)
set_target_properties(p2mesh PROPERTIES CUDA_ARCHITECTURES ${RR_CUDA_ARCH})