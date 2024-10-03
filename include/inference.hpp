#pragma once

#include <onnxruntime_cxx_api.h>

#include <inference_session.hpp>

//! Main inference class - this will hold all the "black box" logic that our front end will interact with
class Inference {
public:
    explicit Inference(size_t num_threads_intra, size_t num_threads_inter);
    ~Inference();

private:
    std::shared_ptr<Ort::Env>           m_environment;
    std::unique_ptr<InferenceSession>   m_yolo_pose_session;
    std::unique_ptr<InferenceSession>   m_yolo_nms_session;

    // These buffers contain tensor data. They MUST be the initial input and final output
    std::vector<uint8_t> m_input_buffer;
    std::vector<uint8_t> m_output_buffer;
};