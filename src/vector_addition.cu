//
// Created by pavel on 10/8/25.
//

#include <cstdio>
__global__ void vector_add(const float *A, const float *B, float *C, int N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}


int main() {
    const int vector_size = 1 << 20;
    const int block_size = 128;

    float *a, *b, *result;
    cudaMallocManaged(&a, vector_size * sizeof(int));
    cudaMallocManaged(&b, vector_size * sizeof(int));
    cudaMallocManaged(&result, vector_size * sizeof(int));

    for (int i = 0; i < vector_size; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i) * 2;
    }

    vector_add<<<(vector_size + block_size - 1) / block_size, block_size>>>(a, b, result, vector_size);

    cudaDeviceSynchronize();

    for (int i = 0; i < 10; i++) {
        printf("%f + %f = %f\n", a[i], b[i], result[i]);
    }

    cudaFree(a);
    cudaFree(b);
    cudaFree(result);
}
