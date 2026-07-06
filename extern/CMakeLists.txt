cmake_minimum_required(VERSION 3.15)


# Скажем spdlog использовать внешнюю библиотеку fmt
set(SPDLOG_FMT_EXTERNAL ON CACHE BOOL "Use external fmt library" FORCE)
add_subdirectory(fmt)
add_subdirectory(spdlog)

add_subdirectory(json)
add_subdirectory(range-v3)
add_subdirectory(eigen)
add_subdirectory(Catch2)

add_subdirectory(RR)
