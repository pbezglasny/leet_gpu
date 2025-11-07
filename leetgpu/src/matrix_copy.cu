//
// Created by pavel on 10/26/25.
//
#include <cuda_runtime.h>

__global__ void copy_matrix_kernel(const float *A, float *B, int N) {
    const size_t idx_x = blockDim.x * blockIdx.x + threadIdx.x;
    const size_t idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    if (idx_x < N && idx_y < N) {
        B[idx_x*N+idx_y] = A[idx_x*N+idx_y];
    }
}

// A, B are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *A, float *B, int N) {
    int total = N * N;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total + threadsPerBlock - 1) / threadsPerBlock;
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
    cudaDeviceSynchronize();
}
