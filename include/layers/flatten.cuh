#pragma once

#include "../tensor.cuh"
#include <vector>

class FlattenLayer {
private:
    std::vector<size_t> input_shape;  // store original shape for backprop

public:
    FlattenLayer() = default;
    ~FlattenLayer() = default;
    
    // converts N-D tensor to 2D
    Tensor<float>* forward(const Tensor<float>* input);
    
    // reshapes gradients back to original shape
    Tensor<float>* backward(const Tensor<float>* gradient_output);
}; 