#pragma once

#include "../tensor.cuh"
#include <random>

class DropoutLayer {
private:
    float dropout_rate;  // probability of dropping a neuron
    Tensor<bool>* mask;  // stores which neurons were dropped
    bool is_training;    // whether layer is in training mode
    
public:
    explicit DropoutLayer(float rate = 0.5f);
    ~DropoutLayer();
    
    Tensor<float>* forward(const Tensor<float>* input);
    Tensor<float>* backward(const Tensor<float>* gradient_output);
    
    // training/inference mode switching
    void set_training(bool training) { is_training = training; }
}; 