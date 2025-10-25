//
// Created by pavel on 10/12/25.
//
#include <cstdio>
#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float *input, const float *kernel, float *output,
                                      int input_size, int kernel_size) {
    size_t idx= blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < input_size - kernel_size + 1) {
        float sum = 0.0f;
        for (int k = 0; k < kernel_size; k++) {
            sum += input[idx + k] * kernel[k];
        }
        output[idx] = sum;
    }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *input, const float *kernel, float *output, int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid = (output_size + threadsPerBlock - 1) / threadsPerBlock;

    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, kernel, output, input_size, kernel_size);
    cudaDeviceSynchronize();
}

int main() {
    const float input[5] = {1, 2, 3, 4, 5};
    const float kernel[3] = {1, 0, -1};
    float output[5];
    const int input_size = 5;
    int kernel_size = 3;
    solve(input, kernel, output, input_size, kernel_size);
    printf("output: [");
    for (int i = 0; i < input_size - kernel_size + 1; i++) {
        printf(" %f ", output[i]);
    }
    printf("]\n");
}
