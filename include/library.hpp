#pragma once

#include <emscripten/val.h>
#include <onnxruntime_cxx_api.h>

#include <optional>
#include <vector>

class YoloModel {
public:
    YoloModel() = default;
    ~YoloModel();

    void load_model();
    [[nodiscard]] emscripten::val model_data_handle(size_t numBytes);

private:
    void warm_up() const;

public:
    // Rule of one coz I CBF figuring out lifetimes and shit properly
    YoloModel(const YoloModel&) = delete;
    YoloModel(YoloModel&&) = delete;

    YoloModel& operator=(const YoloModel&) = delete;
    YoloModel&& operator=(YoloModel&&) = delete;

private:
    std::unique_ptr<Ort::Session> m_session = {};

    std::vector<uint8_t> m_modelBinary      = {};
    std::vector<const char*> m_inputNodeNames  = {};
    std::vector<const char*> m_outputNodeNames = {};
};