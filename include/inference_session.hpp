#pragma once

#include <onnxruntime_cxx_api.h>

// TODO: make `run` take array / template it out.

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
    void run(const std::vector<Ort::Value>& input_tensors, std::vector<Ort::Value>& output_tensors) const;

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