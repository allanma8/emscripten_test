#pragma once

#include <onnxruntime_cxx_api.h>

//! A modular InferenceSession that is used in our black box inference class.
class InferenceSession {
public:
    InferenceSession(const std::shared_ptr<Ort::Env>& environment, const std::string& model_file);
    ~InferenceSession();

    //! Get name of input nodes.
    [[nodiscard]] const std::vector<const char*>& get_input_node_names()    const;

    //! Get name of output nodes.
    [[nodiscard]] const std::vector<const char*>& get_output_node_names()   const;

    //! Perform inference on the model
    //! \note this overload assumes you know the input and output shape.
    //! \note if you don't know the output shape, you can pass nullptr to each element in `output_tensor`
    //! \param input_tensors input tensors.
    //! \param output_tensors output_tensors.
    template<typename ContainerIn, typename ContainerOut>
    void run(const ContainerIn& input_tensors, ContainerOut& output_tensors) const {
        if (input_tensors.size() != m_input_node_names.size()) {
            throw std::runtime_error("input tensor size does not match input node size");
        }

        if (output_tensors.size() != m_output_node_names.size()) {
            throw std::runtime_error("output tensor size does not match output node size");
        }

        const Ort::RunOptions run_options;

        // Inference here - how simple right?
        m_session->Run(
            run_options,
            m_input_node_names.data(), input_tensors.data(), input_tensors.size(),
            m_output_node_names.data(), output_tensors.data(), output_tensors.size()
        );
    }

public:
    InferenceSession(const InferenceSession&)   = delete;
    InferenceSession(InferenceSession&&)        = delete;

    InferenceSession& operator=(const InferenceSession&) = delete;
    InferenceSession&& operator=(InferenceSession &&)    = delete;

private:
    std::shared_ptr<Ort::Env>       m_environment;
    std::unique_ptr<Ort::Session>   m_session;

    // I hate this so fucking much - we HAVE to pass an array of pointers that point to the input/output
    // names. This means we can't use std::string or anything like that since `sizeof != sizeof(uintptr_t)`
    std::vector<const char*> m_input_node_names;
    std::vector<const char*> m_output_node_names;
};