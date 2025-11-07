//
// Created by pavel on 10/12/25.
//
#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float *input, float *output, int rows, int cols) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < cols && idy < rows) {
        output[idx * rows + idy] = input[idy * cols + idx];
    }
}

// input, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float *input, float *output, int rows, int cols) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, rows, cols);
    cudaDeviceSynchronize();
}

int main() {
}
