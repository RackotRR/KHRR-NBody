cmake_minimum_required(VERSION 3.18)
project(test_nbody LANGUAGES CXX)


add_executable(test_nbody test_nbody.cpp)
target_link_libraries(test_nbody
    PRIVATE
        khrr_nbody
        Catch2::Catch2WithMain
)
set_target_properties(test_nbody PROPERTIES
    CXX_STANDARD 17
    CUDA_ARCHITECTURES          OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_SEPARABLE_COMPILATION  ON
)