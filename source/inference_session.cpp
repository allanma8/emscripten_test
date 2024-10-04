#include <inference_session.hpp>

InferenceSession::InferenceSession(const std::shared_ptr<Ort::Env> &environment, const std::string &model_file)
    : m_environment(environment) {
    Ort::SessionOptions session_options;

    // Using `-pthreads` means we use a global thread pool which we configure
    // via ThreadingOptions in environment.
    session_options.DisablePerSessionThreads();

    // Parallel must be set in order to set and use "inter" threads
    session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    m_session = std::make_unique<Ort::Session>(*m_environment, model_file.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    // Save the input/output names of our models - this is the part I hate coz `new`
    const size_t num_input_nodes = m_session->GetInputCount();

    for (size_t i = 0; i < num_input_nodes; i++) {
        const auto buff = new char[50];
        const auto input_node_name = m_session->GetInputNameAllocated(i, allocator);

        std::strcpy(buff, input_node_name.get());
        m_input_node_names.push_back(buff);
    }

    const size_t num_output_nodes = m_session->GetOutputCount();

    for (size_t i = 0; i < num_output_nodes; i++) {
        const auto buff = new char[50];
        const auto output_node_name = m_session->GetOutputNameAllocated(i, allocator);

        std::strcpy(buff, output_node_name.get());
        m_output_node_names.push_back(buff);
    }
}

InferenceSession::~InferenceSession() {
    for (const auto ptr: m_input_node_names) {
        delete[] ptr;
    }
    for (const auto ptr: m_output_node_names) {
        delete[] ptr;
    }
}

const std::vector<const char *> &InferenceSession::get_input_node_names() const {
    return m_input_node_names;
}

const std::vector<const char *> &InferenceSession::get_output_node_names() const {
    return m_output_node_names;
}

void InferenceSession::run(const std::vector<Ort::Value> &input_tensors, std::vector<Ort::Value> &output_tensors) const {
    if (input_tensors.size() != m_input_node_names.size()) {
        throw std::runtime_error("input tensor size does not match input node size");
    }

    if (output_tensors.size() != m_output_node_names.size()) {
        throw std::runtime_error("output tensor size does not match output node size");
    }

    Ort::RunOptions run_options;

    // Inference here - how simple right?
    m_session->Run(
        run_options,
        m_input_node_names.data(), input_tensors.data(), input_tensors.size(),
        m_output_node_names.data(), output_tensors.data(), output_tensors.size()
    );
}
