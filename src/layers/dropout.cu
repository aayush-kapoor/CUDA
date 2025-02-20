#include "../../include/layers/dropout.cuh"
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void dropout_forward_kernel(const float* input,
                                     float* output,
                                     bool* mask,
                                     const float dropout_rate,
                                     const unsigned int seed,
                                     const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        unsigned int state = seed + idx; // LCG
        state = state * 1664525 + 1013904223;
        float random = static_cast<float>(state) / static_cast<float>(UINT_MAX);
        
        // we keep neuron with probability (1 - dropout_rate)
        const bool keep = random > dropout_rate;
        mask[idx] = keep;
        
        //scale output by 1 / (1-dropout_rate) to maintain expected sum
        const float scale = 1.0f / (1.0f - dropout_rate);
        output[idx] = keep ? input[idx] * scale : 0.0f;
    }
}

__global__ void dropout_backward_kernel(const float* gradient_output,
                                      const bool* mask,
                                      float* gradient_input,
                                      const float dropout_rate,
                                      const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        const float scale = 1.0f / (1.0f - dropout_rate);
        gradient_input[idx] = mask[idx] ? gradient_output[idx] * scale : 0.0f;
    }
}

DropoutLayer::DropoutLayer(float rate) 
    : dropout_rate(rate), mask(nullptr), is_training(true) {
    if (rate < 0.0f || rate >= 1.0f) {
        throw std::invalid_argument("Dropout rate must be in [0, 1)");
    }
}

DropoutLayer::~DropoutLayer() {
    if (mask) {
        delete mask;
    }
}

Tensor<float>* DropoutLayer::forward(const Tensor<float>* input) {
    const int size = input->size();
    Tensor<float>* output = new Tensor<float>(input->get_shape());
    
    if (is_training) {
        if (mask) {
            delete mask;
        }
        mask = new Tensor<bool>(input->get_shape());
        unsigned int seed = std::random_device{}();
        
        const int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dropout_forward_kernel<<<num_blocks, BLOCK_SIZE>>>(
            input->get_device_data(),
            output->get_device_data(),
            mask->get_device_data(),
            dropout_rate,
            seed,
            size
        );
    } else {
        cudaMemcpy(output->get_device_data(),
                  input->get_device_data(),
                  size * sizeof(float),
                  cudaMemcpyDeviceToDevice);
    }
    
    return output;
}

Tensor<float>* DropoutLayer::backward(const Tensor<float>* gradient_output) {
    const int size = gradient_output->size();
    Tensor<float>* gradient_input = new Tensor<float>(gradient_output->get_shape());
    
    const int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dropout_backward_kernel<<<num_blocks, BLOCK_SIZE>>>(
        gradient_output->get_device_data(),
        mask->get_device_data(),
        gradient_input->get_device_data(),
        dropout_rate,
        size
    );
    
    return gradient_input;
} 