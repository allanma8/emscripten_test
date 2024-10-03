#include <inference.hpp>

#include <emscripten.h>
#include <emscripten/bind.h>

Inference::Inference(const size_t num_threads_intra, const size_t num_threads_inter) {

    Ort::ThreadingOptions threading_options;

    // Note: some of these options depend on parallel execution to be enabled in session options
    threading_options.SetGlobalIntraOpNumThreads(static_cast<int>(num_threads_intra));
    threading_options.SetGlobalInterOpNumThreads(static_cast<int>(num_threads_inter));

    m_environment = std::make_shared<Ort::Env>(threading_options, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING);

    // Set up the two different models we currently use - this might be more later??
    m_yolo_pose_session = std::make_unique<InferenceSession>(m_environment, "data/yolov8n-pose.onnx");
    m_yolo_nms_session  = std::make_unique<InferenceSession>(m_environment, "data/yolov8nms-pose.onnx");
}

Inference::~Inference() {

}
