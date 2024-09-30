#include <emscripten/bind.h>
#include <onnxruntime_cxx_api.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>

using namespace emscripten;

std::optional<Ort::Session> get_session(const std::filesystem::path& model_path) {
    if (model_path.empty()) {
        std::cout << "Hello Empty Shit \n";
        return std::nullopt;
    }

    const Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "example-model-explorer");
    const Ort::SessionOptions session_options;

    return Ort::Session(env, model_path.c_str(), session_options);
}

float lerp(const float a, const float b, const float t) {
    [[maybe_unused]] const auto session = get_session("");
    return (1 - t) * a + t * b;
}

EMSCRIPTEN_BINDINGS(my_module) {
    function("lerp", &lerp);
}