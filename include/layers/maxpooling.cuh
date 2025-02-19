#pragma once

#include "../tensor.cuh"
#include <vector>

class MaxPoolingLayer {
private:
    // pooling parameters
    size_t kernel_size;
    size_t stride;
    
    // helper function to calculate output dimensions
    std::vector<size_t> calculate_output_shape(const std::vector<size_t>& input_shape) const;
    
    // store indices of max elements for backpropagation
    Tensor<int>* max_indices;

public:
    MaxPoolingLayer(size_t kernel_size, size_t stride = 0);
    ~MaxPoolingLayer();
    
    // forward pass
    Tensor<float>* forward(const Tensor<float>* input);
    
    // get function for max indices (used in backpropagation)
    Tensor<int>* get_max_indices() const { return max_indices; }
}; 