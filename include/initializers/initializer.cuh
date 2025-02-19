#pragma once

#include "../tensor.cuh"
#include <random>
#include <cmath>

class Initializer {
public:
    virtual void initialize(Tensor<float>* tensor) = 0;
    virtual ~Initializer() = default;
};

// constant initialization (used for bias)
class ConstantInitializer : public Initializer {
private:
    float value;

public:
    explicit ConstantInitializer(float val = 0.0f) : value(val) {}
    
    void initialize(Tensor<float>* tensor) override {
        tensor->fill(value);
    }
};

//uniform random initialization
class UniformInitializer : public Initializer {
private:
    float min_val;
    float max_val;
    std::random_device rd;
    std::mt19937 gen;

public:
    UniformInitializer(float min = -0.05f, float max = 0.05f) 
        : min_val(min), max_val(max), gen(rd()) {}
    
    void initialize(Tensor<float>* tensor) override;
};

// Xavier/Glorot initialization
class XavierInitializer : public Initializer {
private:
    bool uniform;
    std::random_device rd;
    std::mt19937 gen;

public:
    explicit XavierInitializer(bool use_uniform = true) 
        : uniform(use_uniform), gen(rd()) {}
    
    void initialize(Tensor<float>* tensor) override;
};

// He initialization
class HeInitializer : public Initializer {
private:
    bool uniform;
    std::random_device rd;
    std::mt19937 gen;

public:
    explicit HeInitializer(bool use_uniform = true) 
        : uniform(use_uniform), gen(rd()) {}
    
    void initialize(Tensor<float>* tensor) override;
}; 