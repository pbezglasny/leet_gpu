cmake_minimum_required(VERSION 3.20)


set(CMAKE_CUDA_ARCHITECTURES "75;80;86;87;89;90;100;110")
set(CMAKE_CUDA_COMPILER /usr/local/cuda-13.0/bin/nvcc)

project(leet-gpu LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_subdirectory(tensara)
add_subdirectory(leetgpu)