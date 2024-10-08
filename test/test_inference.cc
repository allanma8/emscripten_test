// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>

#include <onnxruntime_cxx_api.h>

#include <library.hpp>

TEST(WebAssemblyTest, test) {

    Ort::ThreadingOptions threading_opt;
    threading_opt.SetGlobalIntraOpNumThreads(0); // determine automatically

    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL); // max optimisation?

    Ort::Env ort_env(threading_opt, ORT_LOGGING_LEVEL_WARNING);
    Ort::Session session{ort_env, "data/mul_1.onnx", session_options};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    std::array<float, 6> input_data{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    std::array<int64_t, 2> input_shape{3, 2};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                              input_data.data(), input_data.size(),
                                                              input_shape.data(), input_shape.size());

    std::array<float, 6> output_data{};
    std::array<int64_t, 2> output_shape{3, 2};

    Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                               output_data.data(), output_data.size(),
                                                               output_shape.data(), output_shape.size());

    const char* input_names[]   = {"X"};
    const char* output_names[]  = {"Y"};

    session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, &output_tensor, 1);

    std::array<float, 6> expected_data{1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f};
    std::vector<int64_t> expected_shape{3, 2};

    auto type_info = output_tensor.GetTensorTypeAndShapeInfo();
    ASSERT_EQ(type_info.GetShape(), expected_shape);

    auto total_len = type_info.GetElementCount();
    ASSERT_EQ(total_len, expected_data.size());

    float* result = output_tensor.GetTensorMutableData<float>();

    for (size_t i = 0; i != total_len; ++i) {
        ASSERT_EQ(expected_data[i], result[i]);
    }
}

TEST(WebAssemblyTest, yolo) {

    constexpr size_t IMAGE_WIDTH    = 640;
    constexpr size_t IMAGE_HEIGHT   = 640;

    Ort::ThreadingOptions threading_opt;
    threading_opt.SetGlobalIntraOpNumThreads(0); // determine automatically

    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL); // max optimisation?

    Ort::Env ort_env(threading_opt, ORT_LOGGING_LEVEL_WARNING);
    Ort::Session session{ort_env, "data/yolov8n-pose.onnx", session_options};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    std::array<float, IMAGE_HEIGHT * IMAGE_WIDTH * 3> input_data = {};
    std::array<int64_t, 4> input_shape = {1, 3, IMAGE_HEIGHT, IMAGE_WIDTH};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                              input_data.data(), input_data.size(),
                                                              input_shape.data(), input_shape.size());

    const char* input_names[]   = {"images"};
    const char* output_names[]  = {"output0"};

    std::vector<Ort::Value> output;

    for (size_t i = 0; i < 10; i++) {
        output = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
    }

    ASSERT_EQ(output.empty(), false);
}

TEST(WebAssemblyTest, library) {

    YoloModel model("data/yolov8n-pose.onnx");
    model.update_input_buffer_size(1920, 1080, 3);

}
