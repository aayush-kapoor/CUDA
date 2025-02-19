#pragma once

#include "../tensor.cuh"

class ReLULayer {
private:
    // store mask for backpropagation to keep track of which elements were positive
    Tensor<bool>* mask;

public:
    ReLULayer();
    ~ReLULayer();
    
    Tensor<float>* forward(const Tensor<float>* input);
    Tensor<bool>* get_mask() const { return mask; }
    Tensor<float>* backward(const Tensor<float>* gradient_output);
}; 