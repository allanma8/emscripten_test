# zozofit peach game sdk

## Requirements

- Devpods: https://devpod.sh/
- Docker: https://www.docker.com/

## Building

### Building onnxruntime wasm binaries

Statically linkable wasm binaries are not provided by Microsoft, we need to build them our selves from source.
Follow [these step](https://onnxruntime.ai/docs/build/web.html) but make the following changes:

- change `latest` in `./emsdk install latest` and `./emsdk activate latest` to the same version as defined in `builder.dockerfile`
- always use `--enable_wasm_threads	--enable_wasm_simd`
- change `--build_wasm` to `--build_wasm_static_lib`

The complete build command should be: 

- `./build.sh --config Release --build_wasm_static_lib --skip_tests --disable_wasm_exception_catching --disable_rtti --enable_wasm_threads	--enable_wasm_simd`

Relevant header files should also be copied

- `include/onnxruntime/core/session/onnxruntime_c_api.h`
- `include/onnxruntime/core/session/onnxruntime_cxx_api.h`
- `include/onnxruntime/core/session/onnxruntime_cxx_inline.h`

Relevant static objects should be `libonnxruntime_webassembly.a`