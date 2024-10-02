# emscripten_test

## Requirements

- Devpods: https://devpod.sh/
- Docker: https://www.docker.com/

## Building

### Building onnxruntime wasm binaries (tested with 1.19.2)

Statically linkable wasm binaries are not provided by Microsoft, we need to build them our selves from source.
Follow [these step](https://onnxruntime.ai/docs/build/web.html) but make the following changes:

- change `latest` in `./emsdk install latest` and `./emsdk activate latest` to the same version as defined in `builder.dockerfile`
- always use `--enable_wasm_threads	--enable_wasm_simd`
- change `--build_wasm` to `--build_wasm_static_lib`

The complete build command should be: 

- `./build.sh --config Release --build_wasm_static_lib --skip_tests --enable_wasm_simd --enable_wasm_threads --disable_wasm_exception_catching --disable_rtti --parallel`

Relevant header files should also be copied

- `include/onnxruntime/core/session/onnxruntime_c_api.h`
- `include/onnxruntime/core/session/onnxruntime_cxx_api.h`
- `include/onnxruntime/core/session/onnxruntime_cxx_inline.h`
- `include/onnxruntime/core/session/onnxruntime_float16.h`

Relevant static objects should be `libonnxruntime_webassembly.a`

### Building this library

- Start the DevPod and connect to it.
- Make sure you set `-DCMAKE_TOOLCHAIN_FILE=/emsdk/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake`
- Build

## Example

Create a Javascript file and paste the following code

```javascript
const liblerp = require('<path to wasm module>.js');

// Instantiate liblerp by calling it. Promise returns an instance
// which contains your exported function.
liblerp()
.then(instance => {
  console.log(`${instance.lerp(100, 200, 0.5)}`);
  console.log(`${instance.lerp(10, 20, 0.5)}`);
  console.log(`${instance.lerp(1, 2, 0.5)}`);
});
```

Then run using node: `node ./<file>.js`
