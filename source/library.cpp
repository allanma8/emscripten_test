#include <emscripten/bind.h>
#include <onnxruntime_cxx_api.h>

#include <library.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>

void YoloModel::load_model() {
    // This is just temp shit to see if it works or what ever
    // todo: implement properly
    if (m_inputBuffer.m_buffer == nullptr ||
        m_outputBuffer.m_buffer == nullptr) {
        return;
    }

    std::swap(m_inputBuffer.m_x, m_outputBuffer.m_x);
    std::swap(m_inputBuffer.m_y, m_outputBuffer.m_y);
    std::swap(m_inputBuffer.m_z, m_outputBuffer.m_z);

    std::swap(m_inputBuffer.m_buffer, m_outputBuffer.m_buffer);
}

void YoloModel::create_input_buffer(const size_t x, const size_t y, const size_t z) {
    m_inputBuffer = IOBuffer(x, y, z);
}

void YoloModel::create_output_buffer(const size_t x, const size_t y, const size_t z) {
    m_outputBuffer = IOBuffer(x, y, z);
}

emscripten::val YoloModel::get_input_buffer_raw() const {
    return emscripten::val(emscripten::typed_memory_view(
        m_inputBuffer.size_bytes(),
        m_inputBuffer.m_buffer.get()
    ));
}

emscripten::val YoloModel::get_output_buffer_raw() const {
    return emscripten::val(emscripten::typed_memory_view(
        m_outputBuffer.size_bytes(),
        m_outputBuffer.m_buffer.get()
    ));
}

EMSCRIPTEN_BINDINGS(yolo_model) {
    emscripten::class_<YoloModel>("YoloModel")
        .constructor<>()
        .function("create_input_buffer", &YoloModel::create_input_buffer)
        .function("create_output_buffer", &YoloModel::create_output_buffer)
        .function("get_input_buffer_raw", &YoloModel::get_input_buffer_raw)
        .function("get_output_buffer_raw", &YoloModel::get_output_buffer_raw);
}

//
//
//
//
//

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
    emscripten::function("lerp", &lerp);
}