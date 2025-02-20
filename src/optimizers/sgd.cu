#include "../../include/optimizers/sgd.cuh"
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void sgd_update_kernel(float* params,
                                 const float* gradients,
                                 float* velocity,
                                 const float learning_rate,
                                 const float momentum,
                                 const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        // update velocity with the formula v = momentum * v - learning_rate * gradient
        velocity[idx] = momentum * velocity[idx] - learning_rate * gradients[idx];
        
        // update parameters with formula params = params + v
        params[idx] += velocity[idx];
    }
}

SGDOptimizer::SGDOptimizer(float lr, float mom) 
    : learning_rate(lr), momentum(mom) {
    if (lr <= 0.0f) {
        throw std::invalid_argument("Learning rate must be positive");
    }
    if (mom < 0.0f || mom >= 1.0f) {
        throw std::invalid_argument("Momentum must be in [0, 1)");
    }
}

SGDOptimizer::~SGDOptimizer() {
    clear_state();
}

void SGDOptimizer::update(Tensor<float>* params, const Tensor<float>* gradients) {
    const int size = params->size();
    
    // creating velocity tensor
    Tensor<float>* vel;
    auto it = velocity.find(params);
    if (it == velocity.end()) {
        vel = new Tensor<float>(params->get_shape());
        vel->fill(0.0f);  // initialize velocity to zeros
        velocity[params] = vel;
    } else {
        vel = it->second;
    }
    
    const int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sgd_update_kernel<<<num_blocks, BLOCK_SIZE>>>(
        params->get_device_data(),
        gradients->get_device_data(),
        vel->get_device_data(),
        learning_rate,
        momentum,
        size
    );
}

void SGDOptimizer::clear_state() {
    for (auto& pair : velocity) {
        delete pair.second;
    }
    velocity.clear();
} 