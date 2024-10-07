#include <inference_mediapipe.hpp>
#include <helper/tensor_utils.hpp>

namespace {
    constexpr size_t IMG_INPUT_X = 256;
    constexpr size_t IMG_INPUT_Y = 256;

    constexpr std::array<int64_t, 4> MP_INPUT_SHAPE = {1, IMG_INPUT_X, IMG_INPUT_Y, 3};
    constexpr std::array<int64_t, 2> MP_OUTPUT_SHAPE = {1, 195};
}

Inference_Mediapipe::Inference_Mediapipe(
    const size_t num_threads_intra,
    const size_t num_threads_inter,
    const bool use_lite
)
    : Inference(num_threads_intra, num_threads_inter)
    , m_image_size({IMG_INPUT_X, IMG_INPUT_Y, 4}) {

    if (use_lite) {
        m_session = std::make_unique<InferenceSession>(get_environment(), "data/pose_landmark_lite.onnx");
    } else {
        m_session = std::make_unique<InferenceSession>(get_environment(), "data/pose_landmark_full.onnx");
    }

    set_input_buffer_size(m_image_size.size());
    set_output_buffer_size(helper::accumulate_shape(MP_OUTPUT_SHAPE));

    // Run inference at least once to allocate some of the backing buffers - aka warm up
    run_frame();
}

void Inference_Mediapipe::set_input_image_size(
    const size_t width,
    const size_t height,
    const size_t channels
) {
    // NOTE: THIS SHOULD NEVER BE CALLED RIGHT NOW!!
    // TODO: WAIT FOR RESCALE TO BE IMPLEMENTED BEFORE REMOVING THROW

    // Don't get angry if we are setting as the same
    if (m_image_size.width == width &&
        m_image_size.height == height &&
        m_image_size.channels == channels) {
        return;
    }

    throw std::runtime_error("don't call this function with different sizes asshole");

    // this is unreachable now but once we implement scaling it will need to call this.
    m_image_size = { width, height, channels };
    set_input_buffer_size(m_image_size.size());
}

void Inference_Mediapipe::run_frame() {
    if (m_image_size.width != IMG_INPUT_X || m_image_size.height != IMG_INPUT_Y || m_image_size.channels != 4) {
        throw std::runtime_error("you somehow managed to break this. well done!");
    }

    constexpr float inverse_scale = 1.f / 255.f;
    const auto row_stride = IMG_INPUT_X * m_image_size.channels;

    // Normalised and scaled data - this should be done with opencv
    const auto& input_buffer = get_input_buffer();
    std::array<float, helper::accumulate_shape(MP_INPUT_SHAPE)> input_data = {};

    for (size_t y = 0, i = 0; y < IMG_INPUT_Y; y++) {
        const size_t row_offset = row_stride * y;

        for (size_t x = 0; x < IMG_INPUT_X; x++) {
            const size_t pixel_offset = row_offset + x * 4;

            input_data[i]                                   = static_cast<float>(input_buffer[pixel_offset]) * inverse_scale;
            input_data[i + IMG_INPUT_X * IMG_INPUT_Y]       = static_cast<float>(input_buffer[pixel_offset + 1]) * inverse_scale;
            input_data[i + 2 * IMG_INPUT_X * IMG_INPUT_Y]   = static_cast<float>(input_buffer[pixel_offset + 2]) * inverse_scale;

            i++;
        }
    }

    const auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    const std::array<Ort::Value, 1> input_tensor = {
        Ort::Value::CreateTensor<float>(
            memory_info,
            input_data.data(), input_data.size(),
            MP_INPUT_SHAPE.data(), MP_INPUT_SHAPE.size()
        )
    };

    std::array<Ort::Value, 5> output_tensor = {
        Ort::Value::CreateTensor<float>(
            memory_info,
            get_output_buffer().data(), get_output_buffer().size(),
            MP_OUTPUT_SHAPE.data(), MP_OUTPUT_SHAPE.size()
        ),
        Ort::Value{nullptr},
        Ort::Value{nullptr},
        Ort::Value{nullptr},
        Ort::Value{nullptr}
    };

    m_session->run(input_tensor, output_tensor);
}

size_t Inference_Mediapipe::get_input_image_size_width() const {
    return m_image_size.width;
}

size_t Inference_Mediapipe::get_input_image_size_height() const {
    return m_image_size.height;
}

size_t Inference_Mediapipe::get_input_image_size_channels() const {
    return m_image_size.channels;
}

emscripten::val Inference_Mediapipe::get_input_buffer_val() {
    return emscripten::val(emscripten::typed_memory_view(
        get_input_buffer().size(),
        get_input_buffer().data()
    ));
}

emscripten::val Inference_Mediapipe::get_output_buffer_val() {
    return emscripten::val(emscripten::typed_memory_view(
        get_output_buffer().size(),
        get_output_buffer().data()
    ));
}

EMSCRIPTEN_BINDINGS(inference_module) {
    emscripten::class_<Inference_Mediapipe>("Inference_Mediapipe")
            .constructor<size_t, size_t, bool>()
            .function("set_input_image_size", &Inference_Mediapipe::set_input_image_size)
            .function("run_frame", &Inference_Mediapipe::run_frame)
            .function("get_input_image_size_width", &Inference_Mediapipe::get_input_image_size_width)
            .function("get_input_image_size_height", &Inference_Mediapipe::get_input_image_size_height)
            .function("get_input_image_size_channels", &Inference_Mediapipe::get_input_image_size_channels)
            .function("get_input_buffer_val", &Inference_Mediapipe::get_input_buffer_val)
            .function("get_output_buffer_val", &Inference_Mediapipe::get_output_buffer_val);
}
