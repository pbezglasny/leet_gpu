//
// Created by pavel on 10/26/25.
//
#include <cuda_runtime.h>

const static float e = 2.71828f;

__global__ void silu_kernel(const float *input, float *output, int N) {
    const size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx<N){
        float sigma = 1.0f / (1.0f + pow(e, -input[idx]));
        output[idx]=input[idx]*sigma;
    }
}

// input, output are device pointers
extern "C" void solve(const float *input, float *output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    cudaDeviceSynchronize();
}
