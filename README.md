# CUDA U-Net Implementation

A from-scratch implementation of U-Net architecture using CUDA for GPU acceleration. This implementation includes custom layers, optimizers, activation functions, and loss functions optimized for GPU computation.

## Features

### Layers
- Convolution (CUDA-optimized)
- MaxPooling
- Flatten
- Dropout
- Up- and DownScaling

### Activation Functions
- ReLU
- SoftMax

### Optimizers
- SGD
- SGD with Momentum
- Adam

### Initializers
- Constant
- Uniform Random
- Xavier/Glorot
- He

### Loss Functions
- Cross Entropy Loss

## Optimization Techniques
- Memory Coalescing
- Layer Merging (Conv+ReLU, Conv+Softmax)
- SM Occupancy Optimization

## Dependencies
- CUDA Toolkit
- C++17 or later
- CMake 3.10 or later

## Building the Project