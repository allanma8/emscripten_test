#pragma once

#include <base/inference.hpp>
#include <base/inference_session.hpp>

#include <type/image_size.hpp>

class Inference_Mediapipe: public Inference<Inference_Mediapipe, uint8_t, float> {
public:
    using Base = Inference;

    Inference_Mediapipe(size_t num_threads_intra, size_t num_threads_inter, bool use_lite);

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

    //! General purpose input buffer we write to in javascript.
    [[nodiscard]] emscripten::val get_input_buffer_val();

    //! General purpose output buffer we read from in javascript.
    [[nodiscard]] emscripten::val get_output_buffer_val();

private:
    std::unique_ptr<InferenceSession>   m_session;
    image_size_t                        m_image_size;
};