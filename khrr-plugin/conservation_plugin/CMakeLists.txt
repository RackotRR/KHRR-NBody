cmake_minimum_required(VERSION 3.20)
project(khrr_conservation_plugin LANGUAGES CXX)

add_library(${PROJECT_NAME} STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/conservation_plugin.cpp
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        khrr_common
        khrr_particles
        range-v3::range-v3
        spdlog::spdlog
)