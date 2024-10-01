#pragma once

#include <emscripten/val.h>

#include <memory>
#include <span>

struct IOBuffer {
    IOBuffer() = default;
    IOBuffer(
        const size_t x,
        const size_t y,
        const size_t z
    )
        : m_x(x)
        , m_y(y)
        , m_z(z)
        , m_buffer(std::make_unique<uint8_t[]>(m_x * m_y * m_z)) {

    }

    [[nodiscard]] size_t size_bytes() const {
        return m_x * m_y * m_z;
    }

    size_t m_x = 0;
    size_t m_y = 0;
    size_t m_z = 0;
    std::unique_ptr<uint8_t[]> m_buffer = nullptr;
};

class YoloModel {
public:
    YoloModel() = default;

    void load_model();

    void create_input_buffer(size_t x, size_t y, size_t z);
    void create_output_buffer(size_t x, size_t y, size_t z);

    [[nodiscard]] emscripten::val get_input_buffer_raw() const;
    [[nodiscard]] emscripten::val get_output_buffer_raw() const;

public:
    // Rule of one coz I CBF figuring out lifetimes and shit properly
    YoloModel(const YoloModel&) = delete;
    YoloModel(YoloModel&&) = delete;

    YoloModel& operator=(const YoloModel&) = delete;
    YoloModel&& operator=(YoloModel&&) = delete;

private:
    IOBuffer m_inputBuffer  = {};
    IOBuffer m_outputBuffer = {};
};