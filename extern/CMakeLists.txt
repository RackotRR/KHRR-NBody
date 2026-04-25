cmake_minimum_required(VERSION 3.15)
add_subdirectory(RR)

include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)

if(NOT TARGET fmt)
    FetchContent_Declare(
        fmt
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG master
        GIT_PROGRESS TRUE
    )
    FetchContent_MakeAvailable(fmt)
endif()

if(NOT TARGET range-v3)
    FetchContent_Declare(
        range-v3
        GIT_REPOSITORY https://github.com/ericniebler/range-v3.git
        GIT_TAG        master
        GIT_PROGRESS   TRUE
    )
    FetchContent_MakeAvailable(range-v3)
endif()

if (NOT TARGET Catch2)
    FetchContent_Declare(
        Catch2
        GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        GIT_TAG        v3.7.1
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(Catch2)
endif()