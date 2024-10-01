#include <emscripten/bind.h>

#include <library.hpp>

#include <iostream>

// https://github.com/emscripten-core/emscripten/issues/16305
// https://github.com/DmitriyValetov/onnx_wasm_example/blob/main/src/CMakeLists.txt
// https://github.com/dm33tri/wasm-onnx-opencv-demo?tab=readme-ov-file

namespace {
    constexpr size_t INPUT_IMG_X = 640;
    constexpr size_t INPUT_IMG_Y = 640;
}

YoloModel::~YoloModel() {

    for (const auto ptr: m_inputNodeNames) {
        delete[] ptr;
    }

    for (const auto ptr: m_outputNodeNames) {
        delete[] ptr;
    }
}

void YoloModel::load_model() {

    if (m_modelBinary.empty()) {
        std::cout << "model binary vector is empty. fix it. \n";
        return;
    }

    Ort::SessionOptions session_options;
    session_options.DisablePerSessionThreads();
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

    Ort::ThreadingOptions threading_options;
    threading_options.SetGlobalIntraOpNumThreads(10);

    Ort::Env environment(threading_options, ORT_LOGGING_LEVEL_WARNING, "hello");
    environment.DisableTelemetryEvents();

    m_session = std::make_unique<Ort::Session>(environment, m_modelBinary.data(), m_modelBinary.size(), session_options);
    std::cout << "warming up\n";

    Ort::AllocatorWithDefaultOptions allocator;

    const size_t input_node_count = m_session->GetInputCount();

    std::cout << "Input:\n";

    for (size_t i = 0; i < input_node_count; i++) {
        const Ort::AllocatedStringPtr input_node_name   = m_session->GetInputNameAllocated(i, allocator);
        const auto temp_buf                             = new char[50]; // cleaned up in destructor

        std::strcpy(temp_buf, input_node_name.get());
        m_inputNodeNames.push_back(temp_buf);

        std::cout << "\t -" << temp_buf << "\n";
    }

    const size_t output_node_count = m_session->GetOutputCount();

    std::cout << "Output:\n";

    for (size_t i = 0; i < output_node_count; i++) {
        const Ort::AllocatedStringPtr output_node_name  = m_session->GetOutputNameAllocated(i, allocator);
        const auto temp_buf                             = new char[10]; // cleaned up in destructor

        std::strcpy(temp_buf, output_node_name.get());
        m_outputNodeNames.push_back(temp_buf);

        std::cout << "\t -" << temp_buf << "\n";
    }

    warm_up();
}

emscripten::val YoloModel::model_data_handle(const size_t numBytes) {

    m_modelBinary.clear();
    m_modelBinary.resize(numBytes);

    return emscripten::val(emscripten::typed_memory_view(
        m_modelBinary.size(),
        m_modelBinary.data()
    ));
}

void YoloModel::warm_up() const {

    constexpr size_t BYTES_PER_PIXEL = 3;
    const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    std::array<float, INPUT_IMG_X * INPUT_IMG_Y * BYTES_PER_PIXEL> input_data = {};
    std::array<int64_t, 4> input_shape_array                    = {
        1, 3, INPUT_IMG_X, INPUT_IMG_Y
    };

    const Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_data.data(), input_data.size(),
        input_shape_array.data(), input_shape_array.size()
    );

    // const std::vector<Ort::Value> output_tensors = m_session->Run(
    //     Ort::RunOptions{nullptr},
    //     m_inputNodeNames.data(),
    //     &input_tensor,
    //     1,
    //     m_outputNodeNames.data(),
    //     m_outputNodeNames.size()
    // );
}

EMSCRIPTEN_BINDINGS(yolo_model) {
    emscripten::class_<YoloModel>("YoloModel")
        .constructor<>()
        .function("load_model", &YoloModel::load_model)
        .function("model_data_handle", &YoloModel::model_data_handle);
}

//
//
//
//
//

float lerp(const float a, const float b, const float t) {
    return (1 - t) * a + t * b;
}

// DEBUG SHIT FOR EXCEPTIONS COZ EMSCRIPTEN DOESNT ADD THIS FOR SOME REASON???
std::string getExceptionMessage(int exceptionPtr) {
    return {reinterpret_cast<std::exception *>(exceptionPtr)->what()};
}

EMSCRIPTEN_BINDINGS(my_module) {
    emscripten::function("lerp", &lerp);
    emscripten::function("getExceptionMessage", &getExceptionMessage);
}