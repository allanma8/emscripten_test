#cmake_minimum_required(VERSION 3.29)
#
#project(onnxruntime_emscripten_module)
#
#if (NOT DEFINED EMSCRIPTEN)
#    message(FATAL_ERROR "NOT BUILDING WITH EMSCRIPTEN - please use dev container")
#endif ()
#
## We are stuck with 17 with this version of emscripten, we technically can compile with 20 but the
## clang modules parser isn't included so we get a build error. When we update emscripten we can use 20.
#set(CMAKE_CXX_STANDARD 17)
#
## Disable extensions - fully conformant libstdc++
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS OFF)
#
## We need to make this an executable target even though it's meant to be a library. This is because emscripten
## doesn't generate .js files for libraries and only produces a static library.
#set(
#    SOURCE_FILES
#        source/library.cpp
#)
#
#add_executable(${PROJECT_NAME} ${SOURCE_FILES})
#
## Link onnxruntime - oh im sorry anyone on windows, y'all gonna have fun linking this 50 mb binary 👹
## TODO: once we know this shit works, make a builder for the .a and upload the binary/pull using CMake
#target_include_directories(
#    ${PROJECT_NAME}
#    PRIVATE
#        include
#        onnx/include
#)
#
#target_link_directories(
#    ${PROJECT_NAME}
#    PRIVATE
#        onnx/lib
#)
#
#target_link_libraries(
#    ${PROJECT_NAME}
#    PRIVATE
#        embind
#        onnxruntime
#)
#
## Custom EMSCRIPTEN settings
## TODO: Change depending on debug/release etc.
## Note: more flags and shit found here
##   - https://emscripten.org/docs/tools_reference/emcc.html
##   - https://emscripten.org/docs/tools_reference/settings_reference.html?highlight=environment
#
#set(
#    COMPILER_FLAGS
#        -O3
#        -flto
##        -fexceptions
##        -pthread
#        -msimd128
##        -Wno-pthreads-mem-growth
##        -Wbad-function-cast
##        -Wcast-function-type
#)
#
#set(
#    LINKER_FLAGS
#        -O3
#        -flto
##        -fexceptions
##        -pthread
#        -msimd128
##        -Wno-pthreads-mem-growth
##        -s ERROR_ON_UNDEFINED_SYMBOLS=1
#        -s MODULARIZE=1
##        -s ASSERTIONS=0
##        -s SAFE_HEAP=0
##        -s STACK_OVERFLOW_CHECK=0
##        -s ALLOW_MEMORY_GROWTH=1
##        -s MAXIMUM_MEMORY=4294967296
##        -s DEFAULT_PTHREAD_STACK_SIZE=131072
##        -s STACK_SIZE=16777216
##        -s PTHREAD_POOL_SIZE=Module[\\\"numThreads\\\"]-1
##        -s EXIT_RUNTIME=0
##        -s WASM_BIGINT=1
##        -s FILESYSTEM=0
##        -s TOTAL_MEMORY=1024MB
#        --emit-tsd ${PROJECT_NAME}.d.ts
#)
#
#string (REPLACE ";" " " COMPILER_FLAGS_STR "${COMPILER_FLAGS}")
#string (REPLACE ";" " " LINKER_FLAGS_STR "${LINKER_FLAGS}")
#
#set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${COMPILER_FLAGS_STR}")
#set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS    "${LINKER_FLAGS_STR}")

cmake_minimum_required(VERSION 3.26)

project(tests)

add_executable(${PROJECT_NAME}
        ${CMAKE_SOURCE_DIR}/data/test_main.cc
        ${CMAKE_SOURCE_DIR}/data/test_inference.cc
)

set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "\
    -g2 \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s ASSERTIONS=1 \
    --preload-file ${CMAKE_SOURCE_DIR}/data/testdata@/testdata \
    -s EXIT_RUNTIME=1 \
    -s NO_DISABLE_EXCEPTION_CATCHING \
    -s MODULARIZE=1 \
    --emit-tsd ${PROJECT_NAME}.d.ts\
")

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/519beb0e52c842729b4b53731d27c0e0c32ab4a2.zip
        FIND_PACKAGE_ARGS NAMES GTest
        URL_HASH SHA1=4b3c37972e4c1bef1185d46f702082f8772ee73f
)
FetchContent_MakeAvailable(googletest)
target_link_libraries(${PROJECT_NAME} GTest::gtest)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/onnx/include)
target_link_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/onnx/lib)
target_link_libraries(${PROJECT_NAME} onnxruntime)

enable_testing()
add_test(NAME ${PROJECT_NAME} COMMAND node ${PROJECT_NAME}.js)