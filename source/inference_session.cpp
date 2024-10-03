#include <inference_session.hpp>

namespace {
    // Helper ?? Maybe move this somewhere else IDK
    void set_tensor_shape(std::vector<std::vector<int64_t>>& output, const std::vector<Ort::Value> &tensors) {
        output.clear();
        output.reserve(tensors.size());

        for (const auto &tensor: tensors) {
            const auto type_info = tensor.GetTypeInfo();
            const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

            output.emplace_back(tensor_info.GetShape());
        }
    }
}

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

const std::vector<std::vector<int64_t>> &InferenceSession::get_input_tensor_dimension() const {
    return m_input_tensor_dimension;
}

const std::vector<std::vector<int64_t>> &InferenceSession::get_output_tensor_dimensions() const {
    return m_output_tensor_dimension;
}

const std::vector<const char *> &InferenceSession::get_input_node_names() const {
    return m_input_node_names;
}

const std::vector<const char *> &InferenceSession::get_output_node_names() const {
    return m_output_node_names;
}

void InferenceSession::run(const std::vector<Ort::Value> &input_tensors, std::vector<Ort::Value> &output_tensors) {
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

    // Only set dimensions AFTER we have successful inference
    if (m_input_tensor_dimension.empty()) {
        set_tensor_shape(m_input_tensor_dimension, input_tensors);
    }

    if (m_output_tensor_dimension.empty()) {
        set_tensor_shape(m_output_tensor_dimension, output_tensors);
    }
}
