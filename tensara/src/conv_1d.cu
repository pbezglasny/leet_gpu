#include <cstdio>
#include <cuda_runtime.h>



__global__ void conv_1d(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, size_t N,
                        size_t K) {
    const int half = static_cast<int>(K) / 2;
    if (const size_t idx = blockDim.x * blockIdx.x + threadIdx.x; idx < N) {
        float sum = 0.0f;

        for (int k = 0; k < K; ++k) {
            const int offset = k - half;
            const int src_idx = static_cast<int>(idx) + offset;

            if (src_idx >= 0 && src_idx < N) {
                sum += A[src_idx] * B[k];
            }
        }
        C[idx] = sum;
    }
}

// Note: input, output are all device pointers to float32 arrays
extern "C" void solution(const float *A, const float *B, float *C, size_t N, size_t K) {
    size_t block_size = 128;
    size_t blocksPerGrid = (N + block_size - 1) / block_size;

    conv_1d<<<blocksPerGrid, block_size>>>(A, B, C, N, K);
    cudaDeviceSynchronize();
}


int main() {
    size_t n = 16;
    float *res_ptr = nullptr;
    cudaMallocManaged(&res_ptr, n * sizeof(float));
    cudaMemset(res_ptr, 0, n * sizeof(float));

    size_t k = 3;
    float *kernel_ptr = nullptr;
    cudaMallocManaged(&kernel_ptr, k * sizeof(float));
    kernel_ptr[0] = 1;
    kernel_ptr[1] = 2;
    kernel_ptr[2] = 1;

    float *input_ptr = nullptr;
    cudaMallocManaged(&input_ptr, n * sizeof(float));
    for (int i = 0; i < n; ++i) {
        input_ptr[i] = static_cast<float>(i + 1);
    }

    solution(input_ptr, kernel_ptr, res_ptr, n, k);
    for (size_t i = 0; i < n; ++i) {
        printf("%f ", res_ptr[i]);
    }
    cudaFree(input_ptr);
    cudaFree(kernel_ptr);
    cudaFree(res_ptr);
}
