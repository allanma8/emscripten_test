#pragma once

#include <emscripten/val.h>

#include <onnxruntime_cxx_api.h>

#include <vector>

class YoloModel {
public:
    YoloModel(size_t num_threads_intra, size_t num_threads_inter);
    ~YoloModel();

    // Used to change how large the input buffer is
    void update_input_image_size(size_t width, size_t height);
    void update_input_buffer_size(size_t bytes);

    void run();

    [[nodiscard]] emscripten::val get_input_buffer()    const;
    [[nodiscard]] emscripten::val get_output_buffer()   const;

private:
    [[nodiscard]] std::vector<int64_t> warm_up() const;

public:
    // Rule of one coz I CBF figuring out lifetimes and shit properly
    YoloModel(const YoloModel&) = delete;
    YoloModel(YoloModel&&) = delete;

    YoloModel& operator=(const YoloModel&) = delete;
    YoloModel&& operator=(YoloModel&&) = delete;

private:
    std::optional<Ort::Env> m_environment   = {};
    std::unique_ptr<Ort::Session> m_session = {};

    std::vector<uint8_t> m_input_buffer     = {};
    std::vector<uint8_t> m_output_buffer    = {};

    std::pair<size_t, size_t> m_input_image_size    = {};
    std::vector<int64_t> m_output_tensor_dim        = {};

    std::vector<const char*> m_input_node_names  = {};
    std::vector<const char*> m_output_node_names = {};
};