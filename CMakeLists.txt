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

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build.emscripten)

# We need to make this an exectuable target even though it's meant to be a library. This is because emscripten
# doesn't generate .js files for libraries and only produces a static library.
set(
    SOURCE_FILES
        library.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# Custom EMSCRIPTEN settings
# Note: once we fix node not liking our dev container install tsc and add `--emit-tsd` to our link flags
# Note: more flags and shit found here
#   - https://emscripten.org/docs/tools_reference/emcc.html
#   - https://emscripten.org/docs/tools_reference/settings_reference.html?highlight=environment

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-O3 -flto -fno-rtti -fno-exceptions -DEMSCRIPTEN_HAS_UNBOUND_TYPE_NAMES=0")
set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS    "-O3 -flto -fno-rtti -fno-exceptions -DEMSCRIPTEN_HAS_UNBOUND_TYPE_NAMES=0 -s WASM=1 -s STANDALONE_WASM -s MODULARIZE=1 --no-entry --bind")