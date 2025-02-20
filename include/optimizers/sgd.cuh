#pragma once

#include "../tensor.cuh"
#include <vector>
#include <unordered_map>

class SGDOptimizer {
private:
    float learning_rate;
    float momentum;
    std::unordered_map<Tensor<float>*, Tensor<float>*> velocity;  // store momentum

public:
    SGDOptimizer(float lr = 0.01f, float mom = 0.0f);
    ~SGDOptimizer();
    
    // update params using their gradients
    void update(Tensor<float>* params, const Tensor<float>* gradients);
    
    void clear_state(); // to clear velocity state
}; 