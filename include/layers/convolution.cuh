#pragma once

#include "../tensor.cuh"
#include <vector>

class ConvolutionLayer {
private:
    // layer parameters
    size_t in_channels;
    size_t out_channels;
    size_t kernel_size;
    size_t stride;
    size_t padding;
    
    // weights and biases
    Tensor<float>* weights;
    Tensor<float>* bias;
    
    // helper function to calculate output dimensions
    std::vector<size_t> calculate_output_shape(const std::vector<size_t>& input_shape) const;

public:
    ConvolutionLayer(size_t in_channels, 
                     size_t out_channels,
                     size_t kernel_size,
                     size_t stride = 1,
                     size_t padding = 0);
    
    ~ConvolutionLayer();
    
    // forward pass
    Tensor<float>* forward(const Tensor<float>* input);
    
    // getters for layer parameters
    Tensor<float>* get_weights() { return weights; }
    Tensor<float>* get_bias() { return bias; }
}; 