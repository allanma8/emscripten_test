#include <inference_yolo.hpp>

#include <helper/tensor.hpp>

namespace {
    constexpr size_t POSE_BATCH = 1;
    constexpr size_t POSE_IMG_X = 640;
    constexpr size_t POSE_IMG_Y = 640;
    constexpr size_t POSE_IMG_DEPTH = 3;

    constexpr size_t NMS_0 = 56;
    constexpr size_t NMS_1 = 8400;

    constexpr std::array<int64_t, 4> POSE_INPUT_SHAPE = {POSE_BATCH, POSE_IMG_DEPTH, POSE_IMG_Y, POSE_IMG_X};
    constexpr std::array<int64_t, 3> POSE_OUTPUT_SHAPE = {POSE_BATCH, NMS_0, NMS_1};

    constexpr std::array<int64_t, 2> NMS_INPUT_SHAPE = {NMS_0, NMS_1};

    // Note: NMS will output a dynamic number of detections however we only care about the first
    constexpr std::array<int64_t, 2> NMS_OUT_FEATURES_SHAPE = {1, 51};
}

Inference_Yolo::Inference_Yolo(const size_t num_threads_intra, const size_t num_threads_inter)
    : Inference(num_threads_intra, num_threads_inter)
    , m_image_size({POSE_IMG_X, POSE_IMG_Y, 4}) {

    // Set up the two different models we currently use - this might be more later??
    m_yolo_pose_session = std::make_unique<InferenceSession>(get_environment(), "data/yolov8n-pose.onnx");
    m_yolo_nms_session = std::make_unique<InferenceSession>(get_environment(), "data/yolov8nms-pose.onnx");

    // Allocate output buffers because we know size and shit now.
    m_pose_tensor_data.resize(helper::tensor_data_size(POSE_OUTPUT_SHAPE));

    set_input_buffer_size(m_image_size.size());
    set_output_buffer_size(helper::tensor_data_size(NMS_OUT_FEATURES_SHAPE));

    // Call inference at least once to warm up
    run_frame();
}

void Inference_Yolo::set_input_image_size(
    const size_t width,
    const size_t height,
    const size_t channels
) {
    // NOTE: THIS SHOULD NEVER BE CALLED RIGHT NOW!!
    // TODO: WAIT FOR RESCALE TO BE IMPLEMENTED BEFORE REMOVING THROW

    // Don't get angry if we are setting as the same
    if (m_image_size.width == width && m_image_size.height == height && m_image_size.channels == channels) {
        return;
    }

    throw std::runtime_error("don't call this function with different sizes asshole");

    // this is unreachable now but once we implement scaling it will need to call this.

    m_image_size = { width, height, channels };
    set_input_buffer_size(m_image_size.size());
}

void Inference_Yolo::run_frame() {
    // NOTE: THIS WHOLE FUNCTION IS SKETCHY AS SHIT REWRITE IT PROPERLY LATER
    if (m_image_size.width != POSE_IMG_X || m_image_size.height != POSE_IMG_Y || m_image_size.channels != 4) {
        throw std::runtime_error("you somehow managed to break this. well done!");
    }

    constexpr float inverse_scale = 1.f / 255.f;
    const auto row_stride = POSE_IMG_X * m_image_size.channels;

    // Normalised and scaled data - this should be done with opencv
    const auto& input_buffer = get_input_buffer();
    std::array<float, helper::tensor_data_size(POSE_INPUT_SHAPE)> input_data = {};

    for (size_t y = 0, i = 0; y < POSE_IMG_Y; y++) {
        const size_t row_offset = row_stride * y;

        for (size_t x = 0; x < POSE_IMG_X; x++) {
            const size_t pixel_offset = row_offset + x * 4;

            input_data[i]                               = static_cast<float>(input_buffer[pixel_offset]) * inverse_scale;
            input_data[i + POSE_IMG_X * POSE_IMG_Y]     = static_cast<float>(input_buffer[pixel_offset + 1]) * inverse_scale;
            input_data[i + 2 * POSE_IMG_X * POSE_IMG_Y] = static_cast<float>(input_buffer[pixel_offset + 2]) * inverse_scale;

            i++;
        }
    }

    const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    //
    // Pose
    //

    const std::array<Ort::Value, 1> pose_input_tensor = {
        Ort::Value::CreateTensor<float>(
            memory_info,
            input_data.data(), POSE_IMG_X * POSE_IMG_Y * POSE_IMG_DEPTH,
            POSE_INPUT_SHAPE.data(), POSE_INPUT_SHAPE.size()
        )
    };

    std::array<Ort::Value, 1> pose_output_tensor = {
        Ort::Value::CreateTensor<float>(
            memory_info,
            m_pose_tensor_data.data(), m_pose_tensor_data.size(),
            POSE_OUTPUT_SHAPE.data(), POSE_OUTPUT_SHAPE.size()
        )
    };

    m_yolo_pose_session->run(pose_input_tensor, pose_output_tensor);

    //
    // NMS
    //

    const std::array<Ort::Value, 1> nms_input_tensor = {
        Ort::Value::CreateTensor<float>(
            memory_info,
            m_pose_tensor_data.data(), m_pose_tensor_data.size(),
            NMS_INPUT_SHAPE.data(), NMS_INPUT_SHAPE.size()
        )
    };

    std::array<Ort::Value, 3> nms_output_tensor = {
        Ort::Value{nullptr},
        Ort::Value{nullptr},
        Ort::Value{nullptr}
    };

    m_yolo_nms_session->run(nms_input_tensor, nms_output_tensor);

    // All outputs of this model use the format [detections, _]
    const auto num_detections = helper::tensor_shape(nms_output_tensor.at(0)).at(0);

    if (num_detections > 0) {
        auto &output_buffer = get_output_buffer();
        std::copy_n(
            nms_output_tensor.at(2).GetTensorMutableData<float>(),
            output_buffer.size(),
            output_buffer.begin()
        );
    }
}

type::image_size_t Inference_Yolo::get_input_image_size() const {
    return m_image_size;
}

emscripten::val Inference_Yolo::get_input_buffer_val() {
    return emscripten::val(emscripten::typed_memory_view(
        get_input_buffer().size(),
        get_input_buffer().data()
    ));
}

emscripten::val Inference_Yolo::get_output_buffer_val() {
    return emscripten::val(emscripten::typed_memory_view(
        get_output_buffer().size(),
        get_output_buffer().data()
    ));
}

EMSCRIPTEN_BINDINGS(inference_module) {
    emscripten::class_<Inference_Yolo>("Inference_Yolo")
            .constructor<size_t, size_t>()
            .function("set_input_image_size", &Inference_Yolo::set_input_image_size)
            .function("run_frame", &Inference_Yolo::run_frame)
            .function("get_input_image_size", &Inference_Yolo::get_input_image_size)
            .function("get_input_buffer_val", &Inference_Yolo::get_input_buffer_val)
            .function("get_output_buffer_val", &Inference_Yolo::get_output_buffer_val);
}
