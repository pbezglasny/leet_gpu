cmake_minimum_required(VERSION 3.20)


# You can now use these environment variables in your CMake project
# For example, to print one of them:
message(STATUS "Value of MY_VARIABLE: $ENV{MY_VARIABLE}")

set(CMAKE_CUDA_ARCHITECTURES "75;80;86;87;89;90;100;110")
set(CMAKE_CUDA_COMPILER /usr/local/cuda-13.0/bin/nvcc)

project(leet-gpu LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if (ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else ()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif ()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")

add_executable(vector_addition src/vector_addition.cu)

add_executable(matrix_multi src/matrix_multiplication.cu)

add_executable(transpose src/transpose_matrix.cu)

add_executable(color_inversion src/color_inversion.cu)

add_executable(convolution_1d src/convolution_1d.cu)

add_executable(reverse_arr src/reverse_arr.cu)

add_executable(relu src/relu.cu)


add_executable(leaky_relu src/leaky_relu.cu)

add_executable(rainbow_table src/rainbow_table.cu)

add_executable(matrix_copy src/matrix_copy.cu)

add_executable(count_elements src/count_elements.cu)

add_executable(count_2d src/count_2d_elements.cu)

add_executable(swiglu src/swish_gated.cu)

add_executable(sigmoid_linear_unit src/sigmoid_linear_unit.cu)

add_executable(reduction src/reduction.cu)

add_executable(delivery src/delivery.cpp)