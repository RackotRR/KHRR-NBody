cmake_minimum_required(VERSION 3.20)
project(khrr_common LANGUAGES CXX)

# Основная библиотека
add_library(khrr_common STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/khrr_parser.cpp
)
target_include_directories(khrr_common
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(khrr_common
    PRIVATE
    fmt::fmt
)
set_target_properties(khrr_common PROPERTIES
    CXX_STANDARD 17
)
