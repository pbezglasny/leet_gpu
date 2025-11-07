cmake_minimum_required(VERSION 3.20)


set(CMAKE_CUDA_ARCHITECTURES "75;80;86;87;89;90;100;110")
set(CMAKE_CUDA_COMPILER /usr/local/cuda-13.0/bin/nvcc)


add_executable(tanh src/tanh.cu)