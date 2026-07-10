cmake_minimum_required(VERSION 3.20)
project(khrr-plugin CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (not TARGET khrr-lib)
message("khrr-lib required for khrr-plugins")
endif()

add_subdirectory(consistency_plugin)