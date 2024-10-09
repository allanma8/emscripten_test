#pragma once

#include <onnxruntime_cxx_api.h>

#include <vector>

namespace helper {
    //! Get the shape parameters of an `Ort::Value` tensor.
    std::vector<int64_t> tensor_shape(const Ort::Value &tensor);

    //! Get the size of a tensor from it's shape parameters
    template<typename T>
    [[nodiscard]] constexpr int64_t tensor_data_size(const T &shape) {
        int64_t out = 1;
        for (const auto i: shape) {
            if (i == 0) {
                continue;
            }
            out *= i;
        }
        return out;
    }
}
