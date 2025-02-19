#include "../../include/initializers/initializer.cuh"
#include <cuda_runtime.h>


__global__ void uniform_init_kernel(float* data, 
                                  const int size,
                                  const float min_val,
                                  const float max_val,
                                  const unsigned int seed) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {

        // random number generation on GPU
        unsigned int state = seed + idx;
        state = state * 1664525 + 1013904223; // LCG
        float random = static_cast<float>(state) / static_cast<float>(UINT_MAX);
        data[idx] = min_val + (max_val - min_val) * random;

    }
}

//
__global__ void normal_init_kernel(float* data,
                                 const int size,
                                 const float mean,
                                 const float std,
                                 const unsigned int seed) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {

        unsigned int state = seed + idx;
        state = state * 1664525 + 1013904223;
        float u1 = static_cast<float>(state) / static_cast<float>(UINT_MAX);
        
        state = state * 1664525 + 1013904223;
        float u2 = static_cast<float>(state) / static_cast<float>(UINT_MAX);
        
        float z = sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI * u2);
        data[idx] = mean + std * z;
    }
}

void UniformInitializer::initialize(Tensor<float>* tensor) {
    const int size = tensor->size();
    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;
    
    unsigned int seed = std::random_device{}();
    
    uniform_init_kernel<<<num_blocks, block_size>>>(
        tensor->get_device_data(),
        size,
        min_val,
        max_val,
        seed
    );
}

void XavierInitializer::initialize(Tensor<float>* tensor) {
    const auto& shape = tensor->get_shape();
    const int fan_in = shape[shape.size() - 2];  
    const int fan_out = shape[shape.size() - 1];
    
    if (uniform) {
        // uniform distribution uses the formula limit = sqrt(6/(fan_in + fan_out))
        float limit = sqrt(6.0f / (fan_in + fan_out));
        UniformInitializer unif(-limit, limit);
        unif.initialize(tensor);
    } else {
        //normal distribution with std = sqrt(2 / (fan_in + fan_out))
        float std = sqrt(2.0f / (fan_in + fan_out));
        const int size = tensor->size();
        const int block_size = 256;
        const int num_blocks = (size + block_size - 1) / block_size;
        
        unsigned int seed = std::random_device{}();
        
        normal_init_kernel<<<num_blocks, block_size>>>(
            tensor->get_device_data(),
            size,
            0.0f, 
            std,
            seed
        );
    }
}

void HeInitializer::initialize(Tensor<float>* tensor) {
    const auto& shape = tensor->get_shape();
    const int fan_in = shape[shape.size() - 2];  
    
    if (uniform) {
        
        // uniform distribution
        float limit = sqrt(6.0f / fan_in);
        UniformInitializer unif(-limit, limit);
        unif.initialize(tensor);

    } else {
        
        //normal distribution with std = sqrt(2 / fan_in)
        float std = sqrt(2.0f / fan_in);
        const int size = tensor->size();
        const int block_size = 256;
        const int num_blocks = (size + block_size - 1) / block_size;
        
        unsigned int seed = std::random_device{}();
        
        normal_init_kernel<<<num_blocks, block_size>>>(
            tensor->get_device_data(),
            size,
            0.0f,  // mean
            std,
            seed
        );
    }
} 