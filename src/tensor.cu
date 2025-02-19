#include "/Users/aayush/Desktop/CUDA/include/tensor.cuh"

// kernel for element-wise addition
template<typename T>
__global__ void add_kernel(T* c, const T* a, const T* b, size_t size) {
    // calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // check if thread is within bounds
    if (idx < size) {
        c[idx] = a[idx] + b[idx];
    }
}

// kernel for element-wise multiplication
template<typename T>
__global__ void multiply_kernel(T* c, const T* a, const T* b, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        c[idx] = a[idx] * b[idx];
    }
}

// kernel for scalar multiplication
template<typename T>
__global__ void scalar_multiply_kernel(T* out, const T* in, T scalar, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = in[idx] * scalar;
    }
} 