#include <cuda_runtime.h>

__global__ void swiglu_kernel(const float *input, float *output, int halfN) {
    const size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < halfN) {
        float left = input[idx] * (1.0f / (1.0f + expf(-input[idx])));
        float right = input[idx + halfN];
        output[idx]=left*right;
    }
}

// input, output are device pointers
extern "C" void solve(const float *input, float *output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    cudaDeviceSynchronize();
}

