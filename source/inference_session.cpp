#include <inference_session.hpp>

#include <emscripten.h>

InferenceSession::InferenceSession(const std::shared_ptr<Ort::Env>& environment, const std::string& model_file)
   : m_environment(environment) {

   Ort::SessionOptions session_options;

   // Using `-pthreads` means we use a global thread pool which we configure
   // via ThreadingOptions in environment.
   session_options.DisablePerSessionThreads();

   // Parallel must be set in order to set and use "inter" threads
   session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
   session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

   m_session = std::make_unique<Ort::Session>(*m_environment, model_file.c_str(), session_options);
}

InferenceSession::~InferenceSession() {
   for (const auto ptr: m_input_names) {
      delete[] ptr;
   }
   for (const auto ptr: m_output_names) {
      delete[] ptr;
   }
}
