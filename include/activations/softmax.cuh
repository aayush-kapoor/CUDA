#pragma once

#include "../tensor.cuh"

class SoftmaxLayer {
private:
    // store intermediate values for backpropagation
    Tensor<float>* output_values;

public:
    SoftmaxLayer();
    ~SoftmaxLayer();
    
    // forward pass (computes exp(x)/sum(exp(x)))
    Tensor<float>* forward(const Tensor<float>* input);
    Tensor<float>* backward(const Tensor<float>* gradient_output);
    
    const Tensor<float>* get_output() const { return output_values; }
}; 