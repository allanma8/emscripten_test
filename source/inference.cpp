#include <inference.hpp>

#include <emscripten.h>
#include <emscripten/bind.h>

#define PRINT_INFO 1

#if PRINT_INFO
#include <iostream>
#endif

namespace {
    constexpr size_t POSE_BATCH = 1;
    constexpr size_t POSE_IMG_X = 640;
    constexpr size_t POSE_IMG_Y = 640;
    constexpr size_t POSE_IMG_DEPTH = 3;

    constexpr std::array<int64_t, 4> POSE_INPUT_SHAPE = {POSE_BATCH, POSE_IMG_DEPTH, POSE_IMG_Y, POSE_IMG_X};

    constexpr size_t NMS_0 = 56;
    constexpr size_t NMS_1 = 8400;

    constexpr std::array<int64_t, 2> NMS_INPUT_SHAPE = {NMS_0, NMS_1};

    [[nodiscard]] int64_t accumulate_shape(const std::vector<int64_t>& shape) {
        int64_t out = 0;
        for (const auto i: shape) {
            out += i;
        }
        return out;
    }
}

Inference::Inference(const size_t num_threads_intra, const size_t num_threads_inter)
    : m_input_buffer_size(POSE_IMG_X, POSE_IMG_Y, 4)
    , m_input_buffer(std::get<0>(m_input_buffer_size) * std::get<1>(m_input_buffer_size) * std::get<2>(m_input_buffer_size)) {
    Ort::ThreadingOptions threading_options;

    // Note: some of these options depend on parallel execution to be enabled in session options
    threading_options.SetGlobalIntraOpNumThreads(static_cast<int>(num_threads_intra));
    threading_options.SetGlobalInterOpNumThreads(static_cast<int>(num_threads_inter));

    m_environment = std::make_shared<Ort::Env>(threading_options, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING);

    // Set up the two different models we currently use - this might be more later??
    m_yolo_pose_session = std::make_unique<InferenceSession>(m_environment, "data/yolov8n-pose.onnx");
    m_yolo_nms_session = std::make_unique<InferenceSession>(m_environment, "data/yolov8nms-pose.onnx");

    // This will run things once with null data just to get the gears turning.
    warm_up();

    // Allocate output buffers because we know size and shit now.
    m_output_boxes.resize(accumulate_shape(m_yolo_nms_session->get_output_tensor_dimensions().at(0)));
    m_output_classes.resize(accumulate_shape(m_yolo_nms_session->get_output_tensor_dimensions().at(1)));
    m_output_features.resize(accumulate_shape(m_yolo_nms_session->get_output_tensor_dimensions().at(2)));
}

void Inference::set_input_image_size(
    const size_t width,
    const size_t height,
    const size_t channels
) {
    // NOTE: THIS SHOULD NEVER BE CALLED RIGHT NOW!!
    // TODO: WAIT FOR RESCALE TO BE IMPLEMENTED BEFORE REMOVING THROW
    auto& [ m_width, m_height, m_channels ] = m_input_buffer_size;

    // Don't get angry if we are setting as the same
    if (m_width == width && m_height == height && m_channels == channels) {
        return;
    }

    throw std::runtime_error("don't call this function with different sizes asshole");

    // this is unreachable now but once we implement scaling it will need to call this.

    m_width = width;
    m_height = height;
    m_channels = channels;

    m_input_buffer.resize(width * height * channels);
}

void Inference::run_frame() {
    // TODO: IMPLEMENT RESCALE USING OPENCV IF NOT 640/640
    // TODO: IMPLEMENT NORMALISE USING OPENCV
    // NOTE: THIS WHOLE FUNCTION IS SKETCHY AS SHIT REWRITE IT PROPERLY LATER
    const auto [width, height, channels] = m_input_buffer_size;

    if (width != POSE_IMG_X || height != POSE_IMG_Y || channels != 4) {
        throw std::runtime_error("you somehow managed to break this. well done!");
    }

    constexpr float inverse_scale = 1.f / 255.f;

    const auto row_stride = POSE_IMG_X * channels;

    // Normalised and scaled data - this should be done with opencv
    std::vector<float> input_data(POSE_IMG_X * POSE_IMG_Y * POSE_IMG_DEPTH);

    for (size_t y = 0, i = 0; y < POSE_IMG_Y; y++) {
        const size_t row_offset = row_stride * y;

        for (size_t x = 0; x < POSE_IMG_X; x++) {
            const size_t pixel_offset = row_offset + x * 4;

            input_data[i]                               = static_cast<float>(m_input_buffer[pixel_offset]) * inverse_scale;
            input_data[i + POSE_IMG_X * POSE_IMG_Y]     = static_cast<float>(m_input_buffer[pixel_offset + 1]) * inverse_scale;
            input_data[i + 2 * POSE_IMG_X * POSE_IMG_Y] = static_cast<float>(m_input_buffer[pixel_offset + 2]) * inverse_scale;

            i++;
        }
    }

    const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    std::vector<Ort::Value> pose_input_tensor;
    pose_input_tensor.reserve(1);

    pose_input_tensor.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info,
        input_data.data(), POSE_IMG_X * POSE_IMG_Y * POSE_IMG_DEPTH,
        POSE_INPUT_SHAPE.data(), POSE_INPUT_SHAPE.size()
    ));

    auto pose_output_tensors = m_yolo_pose_session->run(pose_input_tensor);
    auto& pose_output_tensor = pose_output_tensors.at(0);

    std::vector<Ort::Value> nms_input_tensor;
    nms_input_tensor.reserve(1);

    nms_input_tensor.emplace_back(Ort::Value::CreateTensor<float>(
        memory_info,
        pose_output_tensor.GetTensorMutableData<float>(), NMS_0 * NMS_1,
        NMS_INPUT_SHAPE.data(), NMS_INPUT_SHAPE.size()
    ));

    const auto nms_output_tensors = m_yolo_nms_session->run(nms_input_tensor);

    // This memcpy is un-needed - we can get rid of it by supplying output tensors
    std::memcpy(m_output_boxes.data(), nms_output_tensors.at(0).GetTensorData<float>(), m_output_boxes.size());
    std::memcpy(m_output_classes.data(), nms_output_tensors.at(1).GetTensorData<float>(), m_output_classes.size());
    std::memcpy(m_output_features.data(), nms_output_tensors.at(2).GetTensorData<float>(), m_output_features.size());
}

size_t Inference::get_input_image_size_width() const {
    return std::get<0>(m_input_buffer_size);
}

size_t Inference::get_input_image_size_height() const {
    return std::get<1>(m_input_buffer_size);
}

size_t Inference::get_input_image_size_channels() const {
    return std::get<2>(m_input_buffer_size);
}

emscripten::val Inference::get_input_image_buffer() const {
    return emscripten::val(emscripten::typed_memory_view(
        m_input_buffer.size(),
        m_input_buffer.data()
    ));
}

emscripten::val Inference::get_output_boxes() const {
    return emscripten::val(emscripten::typed_memory_view(
        m_output_boxes.size(),
        m_output_boxes.data()
    ));
}

emscripten::val Inference::get_output_classes() const {
    return emscripten::val(emscripten::typed_memory_view(
        m_output_classes.size(),
        m_output_classes.data()
    ));
}

emscripten::val Inference::get_output_features() const {
    return emscripten::val(emscripten::typed_memory_view(
        m_output_features.size(),
        m_output_features.data()
    ));
}

void Inference::warm_up() const {
    const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    // Pose
    {
        // Dummy data - nothing just to run once
        std::vector<float> data(POSE_IMG_X * POSE_IMG_Y * POSE_IMG_DEPTH, 0.f);

        std::vector<Ort::Value> input_tensor;
        input_tensor.reserve(1);

        input_tensor.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info,
            data.data(), data.size(),
            POSE_INPUT_SHAPE.data(), POSE_INPUT_SHAPE.size()
        ));

        [[maybe_unused]] const auto output_tensor = m_yolo_pose_session->run(input_tensor);

#if PRINT_INFO
        for (const auto input_name: m_yolo_pose_session->get_input_node_names()) {
            std::cout << "INPUT NODE: " << input_name << "\n";
        }

        for (const auto output_name: m_yolo_pose_session->get_output_node_names()) {
            std::cout << "OUTPUT_NODE: " << output_name << "\n";
        }
#endif
    }

#if PRINT_INFO
    std::cout << "\n";
#endif

    // NMS
    {
        // Dummy data - nothing just to run once;
        std::vector<float> data(NMS_0 * NMS_1, 0.f);

        std::vector<Ort::Value> input_tensor;
        input_tensor.reserve(1);

        input_tensor.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info,
            data.data(), data.size(),
            NMS_INPUT_SHAPE.data(), NMS_INPUT_SHAPE.size()
        ));

        [[maybe_unused]] const auto output_tensor = m_yolo_nms_session->run(input_tensor);

#if PRINT_INFO
        for (const auto input_name: m_yolo_nms_session->get_input_node_names()) {
            std::cout << "INPUT NODE: " << input_name << "\n";
        }

        for (const auto output_name: m_yolo_nms_session->get_output_node_names()) {
            std::cout << "OUTPUT_NODE: " << output_name << "\n";
        }
#endif
    }
}

EMSCRIPTEN_BINDINGS(inference_module) {
    emscripten::class_<Inference>("Inference")
        .constructor<size_t, size_t>()
        .function("set_input_image_size", &Inference::set_input_image_size)
        .function("run_frame", &Inference::run_frame)
        .function("get_input_image_size_width", &Inference::get_input_image_size_width)
        .function("get_input_image_size_height", &Inference::get_input_image_size_height)
        .function("get_input_image_size_channels", &Inference::get_input_image_size_channels)
        .function("get_input_image_buffer", &Inference::get_input_image_buffer)
        .function("get_output_boxes", &Inference::get_output_boxes)
        .function("get_output_classes", &Inference::get_output_classes)
        .function("get_output_features", &Inference::get_output_features);
}