#pragma once

#include <onnxruntime_cxx_api.h>

// TODO: make `run` take array / template it out.

//! A modular InferenceSession that is used in our black box inference class.
class InferenceSession {
public:
    InferenceSession(const std::shared_ptr<Ort::Env>& environment, const std::string& model_file);
    ~InferenceSession();

    //! Get dimensions of input tensors we first passed to the session.
    //! \note Value is not available until we run the model at least once.
    [[nodiscard]] const std::vector<std::vector<int64_t>>& get_input_tensor_dimension()     const;

    //! Get dimension of output tensors we got from infering the model at least once.
    //! \note Value is not available until we run the model at least once.
    [[nodiscard]] const std::vector<std::vector<int64_t>>& get_output_tensor_dimensions()   const;

    //! Get name of input nodes.
    [[nodiscard]] const std::vector<const char*>& get_input_node_names()    const;

    //! Get name of output nodes.
    [[nodiscard]] const std::vector<const char*>& get_output_node_names()   const;

    //! Perform inference on the model
    //! \note first call MAY be very slow since we need to calculate/save some data.
    //! \param input_tensors input tensors.
    [[nodiscard]] std::vector<Ort::Value> run(const std::vector<Ort::Value>& input_tensors);

    //! Perform inference on the model
    //! \note this overload assumes you know the input and output tensors and shapes
    //! \param input_tensors input tensors.
    //! \param output_tensors output_tensors.
    void run(const std::vector<Ort::Value>& input_tensors, std::vector<Ort::Value>& output_tensors);

public:
    InferenceSession(const InferenceSession&)   = delete;
    InferenceSession(InferenceSession&&)        = delete;

    InferenceSession& operator=(const InferenceSession&) = delete;
    InferenceSession&& operator=(InferenceSession &&)    = delete;

private:
    void set_input_tensor_shape(const std::vector<Ort::Value>& tensors);
    void set_output_tensor_shape(const std::vector<Ort::Value>& tensors);

private:
    std::shared_ptr<Ort::Env>       m_environment;
    std::unique_ptr<Ort::Session>   m_session;

    std::vector<std::vector<int64_t>> m_input_tensor_dimension;
    std::vector<std::vector<int64_t>> m_output_tensor_dimension;

    // I hate this so fucking much - we HAVE to pass an array of pointers that point to the input/output
    // names. This means we can't use std::string or anything like that since `sizeof != sizeof(uintptr_t)`
    std::vector<const char*> m_input_node_names;
    std::vector<const char*> m_output_node_names;
};