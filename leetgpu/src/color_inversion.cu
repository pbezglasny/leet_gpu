//
// Created by pavel on 10/12/25.
//

#include <cstdio>
#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char *image, int width, int height) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < width * height) {
        image[4*idx]=255-image[4*idx];
        image[4*idx+1]=255-image[4*idx+1];
        image[4*idx+2]=255-image[4*idx+2];
    }
}

// image_input, image_output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(unsigned char *image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (width * height + threadsPerBlock - 1) / threadsPerBlock;

    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    cudaDeviceSynchronize();
}

int main() {
    unsigned char image[8] = {255, 0, 128, 255, 0, 255, 0, 255};
    int width = 1;
    int height = 2;
    solve(image, width, height);
    printf("image: [");
    for (const unsigned char i: image) {
        printf(" %u ", i);
    }
    printf("]\n");
}
