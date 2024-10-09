#pragma once

#include <base/inference.hpp>
#include <base/inference_session.hpp>

#include <type/image_size.hpp>

//! Main inference class - this will hold all the "black box" logic that our front end will interact with
class Inference_Yolo: public Inference<Inference_Yolo, uint8_t, float> {
public:
    Inference_Yolo(size_t num_threads_intra, size_t num_threads_inter);

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
    [[nodiscard]] type::image_size_t get_input_image_size() const;

    //! General purpose input buffer we write to in javascript.
    [[nodiscard]] emscripten::val get_input_buffer_val();

    //! General purpose output buffer we read from in javascript.
    [[nodiscard]] emscripten::val get_output_buffer_val();

private:
    std::unique_ptr<InferenceSession>   m_yolo_pose_session;
    std::unique_ptr<InferenceSession>   m_yolo_nms_session;
    std::unique_ptr<InferenceSession>   m_pose_classifier_session;

    type::image_size_t m_image_size;
};
