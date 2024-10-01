cmake_minimum_required(VERSION 3.29)

project(onnxruntime_emscripten_module)

if (NOT DEFINED EMSCRIPTEN)
    message(FATAL_ERROR "NOT BUILDING WITH EMSCRIPTEN - please use dev container")
endif ()

# We are stuck with 17 with this version of emscripten, we technically can compile with 20 but the
# clang modules parser isn't included so we get a build error. When we update emscripten we can use 20.
set(CMAKE_CXX_STANDARD 17)

# Disable extensions - fully conformant libstdc++
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# We need to make this an executable target even though it's meant to be a library. This is because emscripten
# doesn't generate .js files for libraries and only produces a static library.
set(
    SOURCE_FILES
        source/library.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# Link onnxruntime - oh im sorry anyone on windows, y'all gonna have fun linking this 50 mb binary 👹
# TODO: once we know this shit works, make a builder for the .a and upload the binary/pull using CMake
target_include_directories(
    ${PROJECT_NAME}
    PRIVATE
        include
        onnx/include
)

target_link_directories(
    ${PROJECT_NAME}
    PRIVATE
        onnx/lib
)

target_link_libraries(
    ${PROJECT_NAME}
    PRIVATE
        embind
        onnxruntime_webassembly
)

# Custom EMSCRIPTEN settings
# TODO: Change depending on debug/release etc.
# Note: more flags and shit found here
#   - https://emscripten.org/docs/tools_reference/emcc.html
#   - https://emscripten.org/docs/tools_reference/settings_reference.html?highlight=environment

set(
    COMPILE_FLAGS
        -O3
        -flto
)

set(
    LINKER_FLAGS
        -O3
        -flto
        -s MODULARIZE=1
        -s DISABLE_EXCEPTION_CATCHING=0
        -s ASSERTIONS=0
        -s ALLOW_MEMORY_GROWTH=1
        --emit-tsd ${PROJECT_NAME}.d.ts
)

string (REPLACE ";" " " COMPILER_FLAGS_STR "${COMPILER_FLAGS}")
string (REPLACE ";" " " LINKER_FLAGS_STR "${LINKER_FLAGS}")

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${COMPILER_FLAGS_STR}")
set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS    "${LINKER_FLAGS_STR}")