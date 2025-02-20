#include "../../include/layers/flatten.cuh"
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// we dont need kernel for forward pass as it's just a reshape operation

//backward pass kernel (if needed for complex reshaping)
__global__ void flatten_backward_kernel(const float* gradient_output,
                                      float* gradient_input,
                                      const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        gradient_input[idx] = gradient_output[idx];
    }
}

Tensor<float>* FlattenLayer::forward(const Tensor<float>* input) {
    input_shape = input->get_shape();
    
    // calculate flattened shape
    const size_t batch_size = input_shape[0];
    size_t flattened_size = 1;
    for (size_t i = 1; i < input_shape.size(); ++i) {
        flattened_size *= input_shape[i];
    }
    
    // create output tensor with shape (batch_size, flattened_size)
    std::vector<size_t> output_shape = {batch_size, flattened_size};
    Tensor<float>* output = new Tensor<float>(output_shape);
    
    cudaMemcpy(output->get_device_data(),
               input->get_device_data(),
               input->size() * sizeof(float),
               cudaMemcpyDeviceToDevice);
    
    return output;
}

Tensor<float>* FlattenLayer::backward(const Tensor<float>* gradient_output) {
    Tensor<float>* gradient_input = new Tensor<float>(input_shape);
    
    cudaMemcpy(gradient_input->get_device_data(),
               gradient_output->get_device_data(),
               gradient_output->size() * sizeof(float),
               cudaMemcpyDeviceToDevice);
    
    return gradient_input;
} 