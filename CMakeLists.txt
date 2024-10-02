cmake_minimum_required(VERSION 3.29)

option(BUILD_TESTS          "Include and build the test suite." OFF)
option(BUILD_TESTS_MODULAR  "Enable MODULARIZE for tests - allows us to run them on a web browser." OFF)

message(STATUS "BUILD_TESTS: ${BUILD_TESTS}")
message(STATUS "BUILD_TESTS_MODULAR: ${BUILD_TESTS_MODULAR}")

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

#
# Custom EMSCRIPTEN settings
# Note: more flags and shit found here
#   - https://emscripten.org/docs/tools_reference/emcc.html
#   - https://emscripten.org/docs/tools_reference/settings_reference.html?highlight=environment
#

set(
    COMPILER_FLAGS
        -flto
        -fno-rtti
        -pthread
        -msimd128
        -DEMSCRIPTEN_HAS_UNBOUND_TYPE_NAMES=0
)

set(
    LINKER_FLAGS
        -Wno-pthreads-mem-growth
        -s ERROR_ON_UNDEFINED_SYMBOLS=1
        -s ALLOW_MEMORY_GROWTH=1
        -s MAXIMUM_MEMORY=4GB
#        -s DEFAULT_PTHREAD_STACK_SIZE=131072
        -s INITIAL_MEMORY=1024MB
        -s STACK_SIZE=64MB
        -s PTHREAD_POOL_SIZE=Module[\\\"numThreads\\\"]-1
#        -s EXIT_RUNTIME=0
        -s WASM_BIGINT=1
#        -s FILESYSTEM=0
#        -s TOTAL_MEMORY=1024MB
        --preload-file ${CMAKE_SOURCE_DIR}/data@/data
)

if (NOT BUILD_TESTS OR BUILD_TESTS_MODULAR)
    # Only modularise our tests when we want to run them in browser for testing
    # Most times if we want to run in node, we will have to run them un modularised
    # TODO: create a js that loads the modularised `*.js` for tests - this way no need for flag
    set(
        LINKER_FLAGS
        ${LINKER_FLAGS}
            -s MODULARIZE=1
    )
endif ()

if (BUILD_TESTS)
    # Only include test data if building tests
    set(
        LINKER_FLAGS
        ${LINKER_FLAGS}
            --preload-file ${CMAKE_SOURCE_DIR}/data_test@/data_test
    )
else ()
    # Only generate definitions if building library
    set(
        LINKER_FLAGS
        ${LINKER_FLAGS}
            --emit-tsd ${PROJECT_NAME}.d.ts
    )
endif ()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    # Optimisation over size and debug-ness
    set(
        COMPILER_FLAGS
        ${COMPILER_FLAGS}
            -O3
    )

    set(
        LINKER_FLAGS
        ${LINKER_FLAGS}
            -s ASSERTIONS=0
            -s SAFE_HEAP=0
            -s STACK_OVERFLOW_CHECK=0
    )
else ()
    # GAF about speed here
    set(
        COMPILER_FLAGS
        ${COMPILER_FLAGS}
            -g2
    )

    set(
        LINKER_FLAGS
        ${LINKER_FLAGS}
            -s ASSERTIONS=1
            -s NO_DISABLE_EXCEPTION_CATCHING
    )
endif ()

# Merge compiler flags into linker flags
set(
    LINKER_FLAGS
        ${COMPILER_FLAGS}
        ${LINKER_FLAGS}
)

string (REPLACE ";" " " COMPILER_FLAGS_STR "${COMPILER_FLAGS}")
string (REPLACE ";" " " LINKER_FLAGS_STR "${LINKER_FLAGS}")

#
#
#

set(
    SOURCE_FILES
        source/library.cpp
)

# We need to make this an executable target even though it's meant to be a library. This is because emscripten
# doesn't generate .js files for libraries and only produces a static library.
if (BUILD_TESTS)
    # Build a linkable library, we will include tests
    add_library(${PROJECT_NAME} STATIC ${SOURCE_FILES})
    add_subdirectory(test)
else ()
    # Build library as standalone
    add_executable(${PROJECT_NAME} ${SOURCE_FILES})
endif ()

# Link onnxruntime - oh im sorry anyone on windows, y'all gonna have fun linking this 90 mb binary 👹
# TODO: once we know this shit works, make a builder for the .a and upload the binary/pull using CMake
target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
        include
        onnx/include
)

target_link_directories(
    ${PROJECT_NAME}
    PUBLIC
        onnx/lib
)

target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC
        embind
        onnxruntime
)

#
# Set compile settings - this has to be done like this, other ways don't set options correctly
#
set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${COMPILER_FLAGS_STR}")
set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS    "${LINKER_FLAGS_STR}")