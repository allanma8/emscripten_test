cmake_minimum_required(VERSION 3.29)
project(zozofit_peach_game_sdk)

# C 17 and C++ 23
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 23)

# Disable extensions - fully conformant clib
set(CMAKE_CXX_STANDARD_REQUIRED)
set(CMAKE_CXX_EXTENSIONS)

add_library(
    ${PROJECT_NAME}
    STATIC
        library.cpp
)


