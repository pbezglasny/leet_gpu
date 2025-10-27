//
// Created by pavel on 10/26/25.
//
#include <cuda_runtime.h>

__global__ void count_2d_equal_kernel(const int *input, int *output, int N, int M, int K) {
    const size_t i_x = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t i_y = blockDim.y * blockIdx.y + threadIdx.y;
    if (i_x <N && i_y < M && input[i_x * M + i_y] == K) {
        atomicAdd(output, 1);
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const int *input, int *output, int N, int M, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((M + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    count_2d_equal_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, M, K);
    cudaDeviceSynchronize();
}
