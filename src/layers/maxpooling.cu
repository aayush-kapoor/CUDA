#include "../../include/layers/maxpooling.cuh"
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

// cuda kernel for max pooling operation
__global__ void maxpool_kernel(const float* input,
                              float* output,
                              int* max_indices,
                              const int batch_size,
                              const int channels,
                              const int input_height,
                              const int input_width,
                              const int kernel_size,
                              const int stride,
                              const int output_height,
                              const int output_width) {


    // calculate output position
    const int batch_idx = blockIdx.z / channels;
    const int channel = blockIdx.z % channels;
    
    const int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // check bounds
    if (out_row >= output_height || out_col >= output_width || batch_idx >= batch_size)
        return;
        

    const int in_row_start = out_row * stride;
    const int in_col_start = out_col * stride;
    
    // shared memory for input tile
    __shared__ float s_input[BLOCK_SIZE * 2][BLOCK_SIZE * 2];
    
    // initializing max value and index
    float max_val = -INFINITY;
    int max_idx = -1;
    

    // max pooling logic here
    for (int k_row = 0; k_row < kernel_size; ++k_row) {
        for (int k_col = 0; k_col < kernel_size; ++k_col) {
            const int in_row = in_row_start + k_row;
            const int in_col = in_col_start + k_col;
            
            if (in_row < input_height && in_col < input_width) {
                const int input_idx = ((batch_idx * channels + channel) * 
                                      input_height + in_row) * 
                                      input_width + in_col;
                const float val = input[input_idx];
                
                if (val > max_val) {
                    max_val = val;
                    max_idx = input_idx;
                }
            }
        }
    }
    
    // write output
    const int output_idx = ((batch_idx * channels + channel) * 
                            output_height + out_row) * 
                            output_width + out_col;
    output[output_idx] = max_val;
    max_indices[output_idx] = max_idx;
}

MaxPoolingLayer::MaxPoolingLayer(size_t k_size, size_t s)
    : kernel_size(k_size), stride(s), max_indices(nullptr) {
    // if stride not specified, use kernel size
    if (stride == 0) {
        stride = kernel_size;
    }
}

MaxPoolingLayer::~MaxPoolingLayer() {
    if (max_indices) {
        delete max_indices;
    }
}

Tensor<float>* MaxPoolingLayer::forward(const Tensor<float>* input) {
    const auto& input_shape = input->get_shape();
    auto output_shape = calculate_output_shape(input_shape);
    
    // create output tensor and indices tensor
    Tensor<float>* output = new Tensor<float>(output_shape);
    if (max_indices) {
        delete max_indices;
    }
    max_indices = new Tensor<int>(output_shape);
    
    // setup grid and block dimensions
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim(
        (output_shape[3] + block_dim.x - 1) / block_dim.x,
        (output_shape[2] + block_dim.y - 1) / block_dim.y,
        output_shape[0] * output_shape[1]  // batch_size * channels
    );
    
    // launch kernel
    maxpool_kernel<<<grid_dim, block_dim>>>(
        input->get_device_data(),
        output->get_device_data(),
        max_indices->get_device_data(),
        input_shape[0],   // batch size
        input_shape[1],   // channels
        input_shape[2],   // input height
        input_shape[3],   // input width
        kernel_size,
        stride,
        output_shape[2],  // output height
        output_shape[3]   // output width
    );
    
    return output;
}

std::vector<size_t> MaxPoolingLayer::calculate_output_shape(
    const std::vector<size_t>& input_shape) const {
    
    size_t output_height = (input_shape[2] - kernel_size) / stride + 1;
    size_t output_width = (input_shape[3] - kernel_size) / stride + 1;
    
    return {input_shape[0], input_shape[1], output_height, output_width};
} 