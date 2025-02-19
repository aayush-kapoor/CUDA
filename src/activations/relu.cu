#include "../../include/activations/relu.cuh"
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

// kernel for ReLU activation
__global__ void relu_kernel(const float* input,
                           float* output,
                           bool* mask,
                           const int size) {

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    

    if (idx < size) {
        const float val = input[idx];
        // apply ReLU: max(0, x)
        const bool is_positive = (val > 0);
        output[idx] = is_positive ? val : 0.0f;
        //mask for backpropagation
        mask[idx] = is_positive;
    }
}

// fused kernel for Convolution + ReLU (further optimization)
__global__ void conv_relu_kernel(const float* input,
                                const float* weights,
                                const float* bias,
                                float* output,
                                bool* mask,
                                const int batch_size,
                                const int in_channels,
                                const int out_channels,
                                const int input_height,
                                const int input_width,
                                const int kernel_size,
                                const int stride,
                                const int padding,
                                const int output_height,
                                const int output_width) {
}  // haven't added this optimization; will be implemented later

//kernel for backward pass
__global__ void relu_backward_kernel(const float* gradient_output,
                                   const bool* mask,
                                   float* gradient_input,
                                   const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // flow only through positive inputs (where mask is true)
        gradient_input[idx] = mask[idx] ? gradient_output[idx] : 0.0f;
    }
}

ReLULayer::ReLULayer() : mask(nullptr) {}

ReLULayer::~ReLULayer() {
    if (mask) {
        delete mask;
    }
}

Tensor<float>* ReLULayer::forward(const Tensor<float>* input) {
    const int size = input->size();
        Tensor<float>* output = new Tensor<float>(input->get_shape());
    
    if (mask) {
        delete mask;
    }
    mask = new Tensor<bool>(input->get_shape());
    
    const int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        relu_kernel<<<num_blocks, BLOCK_SIZE>>>(
        input->get_device_data(),
        output->get_device_data(),
        mask->get_device_data(),
        size
    );
    
    return output;
}

Tensor<float>* ReLULayer::backward(const Tensor<float>* gradient_output) {
    const int size = gradient_output->size();
    Tensor<float>* gradient_input = new Tensor<float>(gradient_output->get_shape());
    
    const int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    //backward kernel
    relu_backward_kernel<<<num_blocks, BLOCK_SIZE>>>(
        gradient_output->get_device_data(),
        mask->get_device_data(),
        gradient_input->get_device_data(),
        size
    );
    
    return gradient_input;
} 