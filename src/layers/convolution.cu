#include "../../include/layers/convolution.cuh"
#include <cuda_runtime.h>

// define block and kernel sizes (shared memory optimization)
#define BLOCK_SIZE 16
#define MAX_KERNEL_SIZE 7  // support up to 7x7 kernels


// cuda kernel for convolution operation with shared memory
__global__ void conv2d_kernel(const float* input, 
                             const float* weights,
                             const float* bias,
                             float* output,
                             const int batch_size,
                             const int in_channels,
                             const int out_channels,
                             const int input_height,
                             const int input_width,
                             const int kernel_size,
                             const int stride,
                             const int padding,
                             const int output_height,
                             const int output_width) {
    
    // shared memory for input tile and weights
    __shared__ float s_input[BLOCK_SIZE + MAX_KERNEL_SIZE - 1][BLOCK_SIZE + MAX_KERNEL_SIZE - 1];
    __shared__ float s_weights[MAX_KERNEL_SIZE][MAX_KERNEL_SIZE];
    
    //calculating output position 
    const int batch_idx = blockIdx.z / out_channels;
    const int out_channel = blockIdx.z % out_channels;
    const int out_row = blockIdx.y * blockDim.y + threadIdx.y;
    const int out_col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // early exit if outside output bounds
    if (out_row >= output_height || out_col >= output_width || batch_idx >= batch_size)
        return;
        
    // load bias value
    float sum = bias[out_channel];
    
    // input starting position
    const int in_row_start = out_row * stride - padding;
    const int in_col_start = out_col * stride - padding;
    
    // load input tile into shared memory
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    
    // each thread loads its corresponding input value
    for (int i = 0; i < (BLOCK_SIZE + kernel_size - 1); i += BLOCK_SIZE) {
        for (int j = 0; j < (BLOCK_SIZE + kernel_size - 1); j += BLOCK_SIZE) {
            const int row_in = in_row_start + ty + i;
            const int col_in = in_col_start + tx + j;
            
            if (row_in >= 0 && row_in < input_height && 
                col_in >= 0 && col_in < input_width) {
                s_input[ty + i][tx + j] = input[
                    ((batch_idx * in_channels) * input_height + row_in) * 
                    input_width + col_in
                ];
            } else {
                s_input[ty + i][tx + j] = 0.0f;
            }
        }
    }
    
    // load weights into shared memory
    // only threads within kernel size bounds load weights
    if (ty < kernel_size && tx < kernel_size) {
        s_weights[ty][tx] = weights[
            (out_channel * kernel_size + ty) * kernel_size + tx
        ];
    }
    
    // to ensure all threads have loaded their data
    __syncthreads();
    
    // we perform convolution using shared memory
    #pragma unroll
    for (int in_c = 0; in_c < in_channels; ++in_c) {
        #pragma unroll
        for (int k_row = 0; k_row < kernel_size; ++k_row) {
            #pragma unroll
            for (int k_col = 0; k_col < kernel_size; ++k_col) {
                sum += s_input[ty + k_row][tx + k_col] * 
                       s_weights[k_row][k_col];
            }
        }
    }
    
    // write output with coalesced memory access
    const int output_idx = ((batch_idx * out_channels + out_channel) * 
                            output_height + out_row) * 
                            output_width + out_col;
    output[output_idx] = sum;
}

ConvolutionLayer::ConvolutionLayer(size_t in_c, size_t out_c, 
                                 size_t k_size, size_t s, size_t p)
    : in_channels(in_c), out_channels(out_c), 
      kernel_size(k_size), stride(s), padding(p) {
    
    if (kernel_size > MAX_KERNEL_SIZE) {
        throw std::runtime_error("Kernel size exceeds maximum supported size");
    }
    
    // initialize weights and biases
    std::vector<size_t> weight_shape = {
        out_channels, in_channels, kernel_size, kernel_size
    };
    weights = new Tensor<float>(weight_shape);
    
    std::vector<size_t> bias_shape = {out_channels};
    bias = new Tensor<float>(bias_shape);
    
    // initialize with small random values
    // TODO: implement proper weight initialization
    weights->fill(0.01f);
    bias->fill(0.0f);
}

ConvolutionLayer::~ConvolutionLayer() {
    delete weights;
    delete bias;
}

Tensor<float>* ConvolutionLayer::forward(const Tensor<float>* input) {
    const auto& input_shape = input->get_shape();
    auto output_shape = calculate_output_shape(input_shape);
    
    // create output tensor
    Tensor<float>* output = new Tensor<float>(output_shape);
    
    // setup grid and block dimensions (thread block optimization)
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_dim(
        (output_shape[3] + block_dim.x - 1) / block_dim.x,
        (output_shape[2] + block_dim.y - 1) / block_dim.y,
        output_shape[0] * output_shape[1]
    );
    
    // launch kernel
    conv2d_kernel<<<grid_dim, block_dim>>>(
        input->get_device_data(),
        weights->get_device_data(),
        bias->get_device_data(),
        output->get_device_data(),
        output_shape[0],  // batch size
        in_channels,
        out_channels,
        input_shape[2],   // input height
        input_shape[3],   // input width
        kernel_size,
        stride,
        padding,
        output_shape[2],  // output height
        output_shape[3]   // output width
    );
    
    return output;
}

std::vector<size_t> ConvolutionLayer::calculate_output_shape(
    const std::vector<size_t>& input_shape) const {
    
    size_t output_height = (input_shape[2] + 2 * padding - kernel_size) / stride + 1;
    size_t output_width = (input_shape[3] + 2 * padding - kernel_size) / stride + 1;
    
    return {input_shape[0], out_channels, output_height, output_width};
} 