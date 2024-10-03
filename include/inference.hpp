#pragma once

#include <emscripten/bind.h>

#include <onnxruntime_cxx_api.h>

#include <inference_session.hpp>

//! Main inference class - this will hold all the "black box" logic that our front end will interact with
class Inference {
public:
    explicit Inference(size_t num_threads_intra, size_t num_threads_inter);

    //! Set the buffer size from the javascript side.
    //! \note: this assumes everything is in bytes.
    //! \param width image width.
    //! \param height image height.
    //! \param channels number of channels per pixel.
    void set_input_image_size(size_t width, size_t height, size_t channels);

    //! Run all inference steps. Call this SYNCHRONOUSLY.
    //! \note this will use the latest data from the input buffer and write to output buffer.
    void run_frame();

    //! Get the current image buffer width.
    [[nodiscard]] size_t get_input_image_size_width() const;

    //! Get the current image buffer height.
    [[nodiscard]] size_t get_input_image_size_height() const;

    //! Get the current image buffer channels.
    [[nodiscard]] size_t get_input_image_size_channels() const;

    //! Get the image input buffer that we will write to from javascript.
    //! \note Input.
    [[nodiscard]] emscripten::val get_input_image_buffer() const;

    //! Get boxes tensor output.
    //! \note Output.
    [[nodiscard]] emscripten::val get_output_boxes() const;

    //! Get classes tensor output.
    //! \note Output.
    [[nodiscard]] emscripten::val get_output_classes() const;

    //! Get features tensor output.
    //! \note Output.
    [[nodiscard]] emscripten::val get_output_features() const;

private:
    void warm_up() const;

private:
    std::shared_ptr<Ort::Env>           m_environment;
    std::unique_ptr<InferenceSession>   m_yolo_pose_session;
    std::unique_ptr<InferenceSession>   m_yolo_nms_session;

    // Image input buffer
    std::tuple<size_t, size_t, size_t>  m_input_buffer_size;
    std::vector<uint8_t>                m_input_buffer;

    // Todo: have better outputs - just hard code these now
    std::vector<float> m_output_boxes;
    std::vector<float> m_output_classes;
    std::vector<float> m_output_features;
};