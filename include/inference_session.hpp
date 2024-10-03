#pragma once

#include <onnxruntime_cxx_api.h>

//! A modular InferenceSession that is used in our black box inference class.
class InferenceSession {
public:
    InferenceSession(const std::shared_ptr<Ort::Env>& environment, const std::string& model_file);
    ~InferenceSession();

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
    std::vector<const char*> m_input_names;
    std::vector<const char*> m_output_names;
};