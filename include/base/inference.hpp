#pragma once

#include <emscripten.h>
#include <emscripten/bind.h>

#include <base/inference_session.hpp>

template<
    typename Base,
    typename I,
    typename O>
class Inference {
public:
    Inference(const size_t num_threads_intra, const size_t num_threads_inter) {
        Ort::ThreadingOptions threading_options;

        // Note: some of these options depend on parallel execution to be enabled in session options
        threading_options.SetGlobalIntraOpNumThreads(static_cast<int>(num_threads_intra));
        threading_options.SetGlobalInterOpNumThreads(static_cast<int>(num_threads_inter));

        m_environment = std::make_shared<Ort::Env>(threading_options, OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING);
    }

protected:
    //! Create another reference to the shared ptr that hold environment
    [[nodiscard]] std::shared_ptr<Ort::Env> get_environment() const {
        return m_environment;
    }

    //! Set the size of input buffer.
    //! \note total number of bytes for buffer == `sizeof(I) * size`
    //! \param size number of elements that can be saved in the input buffer.
    void set_input_buffer_size(const size_t size) {
        m_input_buffer.clear();
        m_input_buffer.resize(size);
    }

    //! Get a reference to the input buffer
    [[nodiscard]] std::vector<I>& get_input_buffer() {
        return m_input_buffer;
    }

    //! Set the size of output buffer.
    //! \note total number of bytes for buffer == `sizeof(O) * size`
    //! \param size number of elements that can be saved in the output buffer.
    void set_output_buffer_size(const size_t size) {
        m_output_buffer.clear();
        m_output_buffer.resize(size);
    }

    //! Get a reference to the output buffer
    [[nodiscard]] std::vector<O>& get_output_buffer() {
        return m_output_buffer;
    }

private:
    std::shared_ptr<Ort::Env> m_environment;

    std::vector<I> m_input_buffer;
    std::vector<O> m_output_buffer;
};
