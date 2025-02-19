#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <stdexcept>

template<typename T>
class Tensor {
private:
    // device pointer to store data on GPU
    T* d_data;
    // host pointer for temporary CPU operations
    T* h_data;
    // dimensions of the tensor
    std::vector<size_t> shape;
    // total number of elements
    size_t total_size;
    
    // private helper to calculate total size from shape
    size_t calculate_size() const {
        size_t size = 1;
        for (const auto& dim : shape) {
            size *= dim;
        }
        return size;
    }

public:
    // constructor with shape initialization
    Tensor(const std::vector<size_t>& dimensions) 
        : shape(dimensions), total_size(calculate_size()) {
        // allocate memory on device
        cudaMalloc(&d_data, total_size * sizeof(T));
        // allocate memory on host
        h_data = new T[total_size];
    }

    // destructor to free memory
    ~Tensor() {
        if (d_data) cudaFree(d_data);
        if (h_data) delete[] h_data;
    }

    // copy constructor
    Tensor(const Tensor& other) 
        : shape(other.shape), total_size(other.total_size) {
        cudaMalloc(&d_data, total_size * sizeof(T));
        h_data = new T[total_size];
        cudaMemcpy(d_data, other.d_data, total_size * sizeof(T), cudaMemcpyDeviceToDevice);
        memcpy(h_data, other.h_data, total_size * sizeof(T));
    }

    // get raw device pointer
    T* get_device_data() { return d_data; }
    const T* get_device_data() const { return d_data; }

    // get raw host pointer
    T* get_host_data() { return h_data; }
    const T* get_host_data() const { return h_data; }

    // copy data from host to device
    void to_device() {
        cudaMemcpy(d_data, h_data, total_size * sizeof(T), cudaMemcpyHostToDevice);
    }

    // copy data from device to host
    void to_host() {
        cudaMemcpy(h_data, d_data, total_size * sizeof(T), cudaMemcpyDeviceToHost);
    }

    // get tensor dimensions
    const std::vector<size_t>& get_shape() const { return shape; }
    
    // get total number of elements
    size_t size() const { return total_size; }

    // reshape tensor (doesn't change data)
    void reshape(const std::vector<size_t>& new_shape) {
        size_t new_size = 1;
        for (const auto& dim : new_shape) {
            new_size *= dim;
        }
        if (new_size != total_size) {
            throw std::runtime_error("new shape must have same total size");
        }
        shape = new_shape;
    }

    // fill tensor with a value on host
    void fill(T value) {
        std::fill(h_data, h_data + total_size, value);
        to_device();
    }
}; 