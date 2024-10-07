#pragma once

#include <onnxruntime_cxx_api.h>

#include <vector>

namespace helper {
    //! Get the shape parameters of an `Ort::Value` tensor.
    inline std::vector<int64_t> get_tensor_shape(const Ort::Value &tensor) {
        const auto type_info = tensor.GetTypeInfo();
        const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

        return tensor_info.GetShape();
    }
}
