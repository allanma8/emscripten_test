#include <helper/tensor.hpp>

using namespace helper;

std::vector<int64_t> helper::tensor_shape(const Ort::Value& tensor) {
    const auto type_info = tensor.GetTypeInfo();
    const auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    return tensor_info.GetShape();
}
