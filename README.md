# emscripten_test

## Requirements

- Devpods: https://devpod.sh/
- Docker: https://www.docker.com/

## Building

### Building onnxruntime wasm binaries (tested with 1.19.2)

**Note: you must build onnxruntime outside the dev container (i.e: on your host machine)**

**Note: this is not required, you can skip this step if you're not updating onnxruntime**

Statically linkable wasm binaries are not provided by Microsoft, we need to build them our selves from source.
Official steps for setting up the build environment can be found [here](https://onnxruntime.ai/docs/build/web.html). 

The build command used to produce the artifact should be: 

- `./build.sh --config Release --build_wasm_static_lib --skip_tests --enable_wasm_simd --enable_wasm_threads --disable_wasm_exception_catching --disable_rtti --parallel`

Relevant header files should also be copied

- `include/onnxruntime/core/session/onnxruntime_c_api.h`
- `include/onnxruntime/core/session/onnxruntime_cxx_api.h`
- `include/onnxruntime/core/session/onnxruntime_cxx_inline.h`
- `include/onnxruntime/core/session/onnxruntime_float16.h`

Relevant static objects should be `libonnxruntime_webassembly.a` (renamed to `libonnxruntime.a`)

### Building the library

- Start the DevPod and connect to it.
- Make sure you set `-DCMAKE_TOOLCHAIN_FILE=/emsdk/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake`
- Build through your IDE or the CLI.

## Adding new models/resources

Wasm doesn't support IO operations by default. Emscripten emulates a virtual filesystem during runtime to get around this. 

The repo is set up to include files from the `data` folder in the virtual filesystem. The runtime path to any file inside the virtual
file system have the root directory `data`. Eg: `./data/mul_1.onnx` has a runtime path of `data/mul_1.onnx`. 
