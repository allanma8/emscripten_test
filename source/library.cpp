#include <emscripten/bind.h>

#include <library.hpp>

#include <iostream>

#include <chrono>
#include <thread>


// https://niekdeschipper.com/projects/emscripten.html
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

YoloModel::YoloModel(const size_t num_threads_intra, const size_t num_threads_inter)
    : m_input_buffer(INPUT_IMG_X * INPUT_IMG_Y * PIXEL_DEPTH, 0)
    , m_input_image_size(INPUT_IMG_X, INPUT_IMG_Y) {

    // Global threading options
    // Note: setting to 0 will determine automatically
    Ort::ThreadingOptions threading_options;
    threading_options.SetGlobalInterOpNumThreads(num_threads_inter);
    threading_options.SetGlobalIntraOpNumThreads(num_threads_intra);

    // Force full optimisation - might be a bit slow to init ??
    Ort::SessionOptions session_options;
    session_options.DisablePerSessionThreads();
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);

    m_environment   = std::make_optional<Ort::Env>(threading_options, ORT_LOGGING_LEVEL_WARNING);
    m_session       = std::make_unique<Ort::Session>(*m_environment, "data/yolov8n-pose.onnx", session_options);

    // Get names of inputs from model
    Ort::AllocatorWithDefaultOptions allocator;

    const size_t input_node_count = m_session->GetInputCount();

    for (size_t i = 0; i < input_node_count; i++) {
        const Ort::AllocatedStringPtr input_node_name = m_session->GetInputNameAllocated(i, allocator);
        const auto temp_buf = new char[50]; // cleaned up in destructor

        std::strcpy(temp_buf, input_node_name.get());
        m_input_node_names.push_back(temp_buf);
    }

    const size_t output_node_count = m_session->GetOutputCount();

    for (size_t i = 0; i < output_node_count; i++) {
        const Ort::AllocatedStringPtr output_node_name = m_session->GetOutputNameAllocated(i, allocator);
        const auto temp_buf = new char[10]; // cleaned up in destructor

        std::strcpy(temp_buf, output_node_name.get());
        m_output_node_names.push_back(temp_buf);
    }

    m_output_tensor_dim = warm_up();

    for (size_t i = 0; i < m_output_tensor_dim.size(); i++) {
        std::cout << i << ": val = " << m_output_tensor_dim.at(i) << "\n";
    }

    // Allocate the output buffer based on output tensor dimensions
    // Todo: sanity check that we have 3 elements at all lol
    m_output_buffer.resize(m_output_tensor_dim.at(0) * m_output_tensor_dim.at(1) * m_output_tensor_dim.at(2) * sizeof(float));
}

YoloModel::~YoloModel() {
    for (const auto ptr: m_input_node_names) {
        delete[] ptr;
    }

    for (const auto ptr: m_output_node_names) {
        delete[] ptr;
    }
}

void YoloModel::update_input_image_size(const size_t width, const size_t height) {
    m_input_image_size = {width, height};
}

void YoloModel::update_input_buffer_size(const size_t bytes) {
    // Reshape the input buffer
    m_input_buffer.clear();
    m_input_buffer.resize(bytes);
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

    std::array<float, INPUT_IMG_Y * INPUT_IMG_X * PIXEL_DEPTH> input_data = {};
    std::array<int64_t, 4> input_shape = {1, PIXEL_DEPTH, INPUT_IMG_Y, INPUT_IMG_X};

    const Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
        input_data.data(), input_data.size(),
        input_shape.data(), input_shape.size()
    );

    const auto output = m_session->Run(
        Ort::RunOptions{nullptr},
        m_input_node_names.data(),
        &input_tensor, m_input_node_names.size(),
        m_output_node_names.data(),
        m_output_node_names.size()
    );

    const auto &type_info = output.front().GetTypeInfo();
    const auto &tensor_info = type_info.GetTensorTypeAndShapeInfo();

    return tensor_info.GetShape();
}

void YoloModel::run() {

    // Todo: normalise and rescale input to be 640 x 640 using OpenCV if not
    // Todo: process output into joint locations

    constexpr float inverse_scale = 1.f / 255.f;

    const auto [width, height] = m_input_image_size;
    const auto row_stride = width * 4;

    std::vector<float> input_data(width * height * 3);

    for (size_t y = 0, i = 0; y < height; y++) {
        const size_t row_offset = row_stride * y;

        for (size_t x = 0; x < width; x++) {
            const size_t pixel_offset = row_offset + x * 4;

            input_data[i]                       = static_cast<float>(m_input_buffer[pixel_offset]) * inverse_scale;
            input_data[i + width * height]      = static_cast<float>(m_input_buffer[pixel_offset + 1]) * inverse_scale;
            input_data[i + 2 * width * height]  = static_cast<float>(m_input_buffer[pixel_offset + 2]) * inverse_scale;

            i++;
        }
    }

    std::array<int64_t, 4> input_shape = {1, PIXEL_DEPTH, INPUT_IMG_Y, INPUT_IMG_X};

    const Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
        input_data.data(),input_data.size(),
        input_shape.data(), input_shape.size()
    );

    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU),
        reinterpret_cast<float*>(m_output_buffer.data()), m_output_buffer.size() * sizeof(float),
        m_output_tensor_dim.data(), m_output_tensor_dim.size()
    );

    m_session->Run(
        Ort::RunOptions{nullptr},
        m_input_node_names.data(), &input_tensor, m_input_node_names.size(),
        m_output_node_names.data(), &output_tensor, m_output_node_names.size()
    );
}

EMSCRIPTEN_BINDINGS(my_module_2) {
    emscripten::class_<YoloModel>("YoloModel")
            .constructor<size_t, size_t>()
            .function("update_input_image_size", &YoloModel::update_input_image_size)
            .function("update_input_buffer_size", &YoloModel::update_input_buffer_size)
            .function("get_input_buffer", &YoloModel::get_input_buffer)
            .function("get_output_buffer", &YoloModel::get_output_buffer)
            .function("run", &YoloModel::run);
};
