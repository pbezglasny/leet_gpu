//
// Created by pavel on 11/7/25.
//
#include <cuda_runtime.h>



__global__ void tanh_kernel(const float *input, float *output, size_t n, size_t m) {
    const size_t i_x = blockDim.x * blockIdx.x + threadIdx.x;
    if (const size_t i_y = blockDim.y * blockIdx.y + threadIdx.y; i_x < n && i_y < m) {
        const size_t idx = i_y * n + i_x;
        output[idx] = tanh(input[idx]);
    }
}

// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float *input, float *output, size_t n, size_t m) {
    dim3 thread_per_block(16, 16);
    dim3 num_blocks((n + thread_per_block.x - 1) / thread_per_block.x,
                    (m + thread_per_block.y - 1) / thread_per_block.y);

    tanh_kernel<<<num_blocks, thread_per_block>>>(input, output, n, m);
    cudaDeviceSynchronize();
}
