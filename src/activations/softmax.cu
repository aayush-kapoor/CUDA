#include "../../include/activations/softmax.cuh"
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define WARP_SIZE 32

// kernel to find maximum value in each batch
__global__ void find_max_kernel(const float* input,
                               float* max_values,
                               const int batch_size,
                               const int features) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    // shared mem
    __shared__ float shared_max[WARP_SIZE];
    
    float max_val = -INFINITY;
    
    for (int i = tid; i < features; i += WARP_SIZE) {
        const float val = input[batch_idx * features + i];
        max_val = max(max_val, val);
    }
    
    // warp reduction to find maximum
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        max_val = max(max_val, __shfl_down_sync(0xffffffff, max_val, offset));
    }
    
    if (tid == 0) {
        max_values[batch_idx] = max_val;
    }
}

//compute softmax forward pass
__global__ void softmax_forward_kernel(const float* input,
                                     const float* max_values,
                                     float* output,
                                     float* sum_values,
                                     const int batch_size,
                                     const int features) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    __shared__ float shared_sum[WARP_SIZE];
    
    // compute exp(x - max) and sum (softmax formula)
    float sum = 0.0f;
    const float max_val = max_values[batch_idx];
    
    for (int i = tid; i < features; i += WARP_SIZE) {
        const int idx = batch_idx * features + i;
        const float val = exp(input[idx] - max_val);
        output[idx] = val;
        sum += val;
    }
    
    // warp reduction for sum
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // first thread in warp writes sum
    if (tid == 0) {
        sum_values[batch_idx] = sum;
    }
    
    __syncthreads();
    
    // normalize by sum
    const float final_sum = sum_values[batch_idx];
    for (int i = tid; i < features; i += WARP_SIZE) {
        const int idx = batch_idx * features + i;
        output[idx] /= final_sum;
    }
}

// kernel for backward pass
__global__ void softmax_backward_kernel(const float* gradient_output,
                                      const float* output_values,
                                      float* gradient_input,
                                      const int batch_size,
                                      const int features) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    for (int i = tid; i < features; i += WARP_SIZE) {
        const int row_idx = batch_idx * features + i;
        float sum = 0.0f;
        
        // compute Jacobian-vector product
        for (int j = 0; j < features; j++) {
            const int col_idx = batch_idx * features + j;
            const float y_i = output_values[row_idx];
            const float y_j = output_values[col_idx];
            const float grad = gradient_output[col_idx];
            
            if (i == j) {
                sum += grad * y_i * (1 - y_i);
            } else {
                sum -= grad * y_i * y_j;
            }
        }
        
        gradient_input[row_idx] = sum;
    }
}

SoftmaxLayer::SoftmaxLayer() : output_values(nullptr) {}

SoftmaxLayer::~SoftmaxLayer() {
    if (output_values) {
        delete output_values;
    }
}

Tensor<float>* SoftmaxLayer::forward(const Tensor<float>* input) {
    const auto& shape = input->get_shape();
    const int batch_size = shape[0];
    const int features = shape[1];
    
    if (output_values) {
        delete output_values;
    }
    output_values = new Tensor<float>(shape);
    
    float *d_max_values, *d_sum_values;
    cudaMalloc(&d_max_values, batch_size * sizeof(float));
    cudaMalloc(&d_sum_values, batch_size * sizeof(float));
    
    find_max_kernel<<<batch_size, WARP_SIZE>>>(
        input->get_device_data(),
        d_max_values,
        batch_size,
        features
    );
    
    softmax_forward_kernel<<<batch_size, WARP_SIZE>>>(
        input->get_device_data(),
        d_max_values,
        output_values->get_device_data(),
        d_sum_values,
        batch_size,
        features
    );
    
    cudaFree(d_max_values);
    cudaFree(d_sum_values);
    
    return output_values;
}

Tensor<float>* SoftmaxLayer::backward(const Tensor<float>* gradient_output) {
    const auto& shape = gradient_output->get_shape();
    const int batch_size = shape[0];
    const int features = shape[1];
    
    Tensor<float>* gradient_input = new Tensor<float>(shape);
    
    softmax_backward_kernel<<<batch_size, WARP_SIZE>>>(
        gradient_output->get_device_data(),
        output_values->get_device_data(),
        gradient_input->get_device_data(),
        batch_size,
        features
    );
    
    return gradient_input;
} 