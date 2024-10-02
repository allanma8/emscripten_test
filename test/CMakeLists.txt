message(STATUS "Adding tests")
message(STATUS "=============================================")

project(tests)

#
# Fetch GTest
# Todo: check if we can build and use catch2 on wasm
#

include(FetchContent)

FetchContent_Declare(
    googletest
        URL https://github.com/google/googletest/archive/519beb0e52c842729b4b53731d27c0e0c32ab4a2.zip
        FIND_PACKAGE_ARGS NAMES GTest
        URL_HASH SHA1=4b3c37972e4c1bef1185d46f702082f8772ee73f
)

FetchContent_MakeAvailable(googletest)

set_target_properties(gtest PROPERTIES COMPILE_FLAGS "${COMPILER_FLAGS_STR}")
set_target_properties(gtest PROPERTIES LINK_FLAGS    "${LINKER_FLAGS_STR}")

#
# Create Test Binary
#

add_executable(
    ${PROJECT_NAME}
        ./test_inference.cc
        ./test_main.cc
)

set_target_properties(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "${COMPILER_FLAGS_STR}")
set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS    "${LINKER_FLAGS_STR}")

target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC
        gtest
        onnxruntime_emscripten_module
)

enable_testing()

add_test(NAME ${PROJECT_NAME} COMMAND node ${PROJECT_NAME}.js)
