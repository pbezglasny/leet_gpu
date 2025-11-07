//
// Created by pavel on 10/27/25.
//
#include <cuda_runtime.h>

__global__ void reduction(const float *input, float *output, int N) {
    extern __shared__ float sdata[];
    const size_t tid = threadIdx.x;
    const size_t idx = blockDim.x * blockIdx.x + tid;
    const size_t stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    for (size_t i = idx; i < N; i += stride) sum += input[i];

    sdata[tid] = sum;
    __syncthreads();

    for (size_t s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s)sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid < 32) {
        volatile float *vsm = sdata;
        vsm[tid] += vsm[tid + 32];
        vsm[tid] += vsm[tid + 16];
        vsm[tid] += vsm[tid + 8];
        vsm[tid] += vsm[tid + 4];
        vsm[tid] += vsm[tid + 2];
        vsm[tid] += vsm[tid + 1];
    }


    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

// input, output are device pointers
extern "C" void solve(const float *input, float *output, int N) {
    const int threads_per_block = 256;
    const int block_per_grid = (N + threads_per_block - 1) / threads_per_block;
    const size_t shared_bytes = threads_per_block * sizeof(float);

    reduction<<<block_per_grid, threads_per_block, shared_bytes>>>(input, output, N);
    cudaDeviceSynchronize();
}
