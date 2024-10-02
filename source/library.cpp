#include <emscripten/bind.h>

#include <library.hpp>

#include <iostream>

// https://github.com/emscripten-core/emscripten/issues/16305
// https://github.com/DmitriyValetov/onnx_wasm_example/blob/main/src/CMakeLists.txt
// https://github.com/dm33tri/wasm-onnx-opencv-demo?tab=readme-ov-file
// https://github.com/csukuangfj/onnxruntime-libs?tab=readme-ov-file

namespace {
    constexpr size_t INPUT_IMG_X = 640;
    constexpr size_t INPUT_IMG_Y = 640;

    constexpr size_t PIXEL_DEPTH = 3;
}

//
// DEBUG SHIT FOR EXCEPTIONS COZ EMSCRIPTEN DOESNT ADD THIS FOR SOME REASON???
//
std::string getExceptionMessage(int exceptionPtr) {
    return {reinterpret_cast<std::exception *>(exceptionPtr)->what()};
}

EMSCRIPTEN_BINDINGS(my_module) {
    emscripten::function("getExceptionMessage", &getExceptionMessage);
}

//
// Shitty test library for our POC
//

YoloModel::YoloModel(const std::filesystem::path &model_path)
    : m_input_buffer(INPUT_IMG_X * INPUT_IMG_Y * PIXEL_DEPTH, 0)
    , m_output_buffer(INPUT_IMG_X * INPUT_IMG_Y * PIXEL_DEPTH, 0) {

    // Global threading options
    // Note: setting to 0 will determine automatically
    Ort::ThreadingOptions threading_options;
    threading_options.SetGlobalIntraOpNumThreads(0);

    // Force full optimisation - might be a bit slow to init ??
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    Ort::Env environment(threading_options, ORT_LOGGING_LEVEL_WARNING);
    environment.DisableTelemetryEvents();

    m_session       = std::make_unique<Ort::Session>(environment, model_path.c_str(), session_options);
    m_memory_info   = std::make_optional(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU));

    // Get names of inputs from model
    Ort::AllocatorWithDefaultOptions allocator;

    const size_t input_node_count = m_session->GetInputCount();

    for (size_t i = 0; i < input_node_count; i++) {
        const Ort::AllocatedStringPtr input_node_name   = m_session->GetInputNameAllocated(i, allocator);
        const auto temp_buf                             = new char[50]; // cleaned up in destructor

        std::strcpy(temp_buf, input_node_name.get());
        m_input_node_names.push_back(temp_buf);
    }

    const size_t output_node_count = m_session->GetOutputCount();

    for (size_t i = 0; i < output_node_count; i++) {
        const Ort::AllocatedStringPtr output_node_name  = m_session->GetOutputNameAllocated(i, allocator);
        const auto temp_buf                             = new char[10]; // cleaned up in destructor

        std::strcpy(temp_buf, output_node_name.get());
        m_output_node_names.push_back(temp_buf);
    }

    m_output_tensor_dim = warm_up();

    for (size_t i = 0; i < m_output_tensor_dim.size(); i++) {
        std::cout << i << ": val = " << m_output_tensor_dim.at(i) << "\n";
    }
}

YoloModel::~YoloModel() {
    for (const auto ptr: m_input_node_names) {
        delete[] ptr;
    }

    for (const auto ptr: m_output_node_names) {
        delete[] ptr;
    }
}

void YoloModel::update_input_buffer_size(
    const size_t width,
    const size_t height,
    const size_t bytes_per_pixel
) {
    // Reshape the input buffer
    m_input_buffer.clear();
    m_input_buffer.reserve(width * height * bytes_per_pixel);
}

emscripten::val YoloModel::get_input_buffer() const {
    return emscripten::val(emscripten::typed_memory_view(
        m_input_buffer.size(),
        m_input_buffer.data()
    ));
}

emscripten::val YoloModel::get_output_buffer() const {
    return emscripten::val(emscripten::typed_memory_view(
        m_output_buffer.size(),
        m_output_buffer.data()
    ));
}

std::vector<int64_t> YoloModel::warm_up() const {
    // Use an empty array at first just to "warm up" the model
    // We also use this to get the size of our output tensor :)

    std::array<float, INPUT_IMG_Y * INPUT_IMG_X * PIXEL_DEPTH> input_data   = {};
    std::array<int64_t, 4> input_shape                                      = {1, PIXEL_DEPTH, INPUT_IMG_Y, INPUT_IMG_X};

    const Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        m_memory_info.value(),
        input_data.data(),
        input_data.size(),
        input_shape.data(), input_shape.size()
    );

    const auto output = m_session->Run(
        Ort::RunOptions{nullptr},
        m_input_node_names.data(),
        &input_tensor, m_input_node_names.size(),
        m_output_node_names.data(),
        m_output_node_names.size()
    );

    const auto& type_info   = output.front().GetTypeInfo();
    const auto& tensor_info = type_info.GetTensorTypeAndShapeInfo();

    return tensor_info.GetShape();
}

EMSCRIPTEN_BINDINGS(my_module_2) {
    emscripten::class_<YoloModel>("YoloModel")
        .constructor<const std::filesystem::path>()
        .function("update_input_buffer_size",   &YoloModel::update_input_buffer_size)
        .function("get_input_buffer",           &YoloModel::get_input_buffer)
        .function("get_output_buffer",          &YoloModel::get_output_buffer)
        ;
};